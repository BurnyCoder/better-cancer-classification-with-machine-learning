"""Stage C: LoRA fine-tuning of Hibou-L (S5), weight-space post-processing (S5b), the
augmentation-matched frozen control (H4), and the end-to-end ConvNeXt-T baseline.

Global context: pre-registration section 6. LoRA trains on train folds 1-4 and selects checkpoints
and hyperparameters on held-out train fold 0 (a fixed 10k subsample for checkpointing, all of fold 0
for the step decision). The step decision compares LoRA with the incumbent's OOF predictions on the
same fold-0 patches, whose heads were also trained on folds 1-4, plus the one-shot valid gate.
"""

import json
import logging

import numpy as np
import torch
from scipy.special import expit
from sklearn.model_selection import train_test_split

from pcam_hc import experiments as ex
from pcam_hc.backbones import REGISTRY, load_model
from pcam_hc.config import PATHS, SUBSET_SIZE
from pcam_hc.data import PCamImages, make_subset
from pcam_hc.finetune import LoraClassifier, load_trainable, predict_with_state, train_lora, wise_state
from pcam_hc.ladder import read_ladder, write_ladder
from pcam_hc.stage_b import _fit_cfg, base_arm, features, incumbent
from pcam_hc.stats import paired_comparison

log = logging.getLogger(__name__)
LORA_DIR = PATHS.ckpt / "lora"
SWEEP = [(lr, r) for lr in (5e-5, 1e-4, 3e-4) for r in (8, 16)]


def splits(ctx) -> dict:
    """Index sets: LoRA training (folds 1-4, minus cleaned labels if S3b was adopted), fold 0,
    a stratified 10k checkpointing subsample of fold 0, and train-64k restricted to folds 1-4."""
    _, _, _, cfg = incumbent()
    train = np.flatnonzero(ctx.folds != 0)
    if "clean_threshold" in cfg:  # drop patches flagged by the incumbent's own OOF predictions
        p = expit(ex.load_preds(base_arm(read_ladder()["incumbent"]))["oof"].astype(np.float64)).mean(0)
        train = train[np.abs(ctx.y_train[train] - p[train]) <= cfg["clean_threshold"]]
    fold0 = np.flatnonzero(ctx.folds == 0)
    ckpt, _ = train_test_split(fold0, train_size=10_000, stratify=ctx.y_train[fold0], random_state=0)
    sub = make_subset(ctx.folds, ctx.y_train, SUBSET_SIZE, seed=0)
    return {"train": train, "fold0": fold0, "ckpt": np.sort(ckpt), "sweep_train": np.intersect1d(sub, train)}


def lp_head(ctx, idx) -> dict:
    """LP-FT initial head: the incumbent's frozen-probe head fit on the LoRA training patches only."""
    _, rep, views, cfg = incumbent()
    X = features("train", rep, views)
    head = _fit_cfg(cfg, X[idx], ctx.y_train[idx], seed=0)
    return {"module": {k: v.cpu() for k, v in head.module.state_dict().items()},
            "mu": None if head.mu is None else head.mu.cpu(), "sd": None if head.sd is None else head.sd.cpu()}


def new_model(head_state, rank: int) -> LoraClassifier:
    _, rep, views, _ = incumbent()
    spec = REGISTRY["hibou_l"]
    return LoraClassifier(load_model(spec).to(torch.bfloat16), spec, head_state, rep, views, rank=rank)


def sweep(ctx, sp: dict, head_state: dict) -> dict:
    """Successive halving over lr x rank on train-64k (folds 1-4): every config gets 1/4 epoch of the
    subset, the best half continues to 1/2, the best of those to the full subset epoch."""
    images = PCamImages("train")
    alive, results = list(SWEEP), {}
    for budget in (0.25, 0.5, 1.0):
        scores = {}
        for lr, r in alive:
            model = new_model(head_state, r)
            run = train_lora(model, images, sp["sweep_train"], ctx.y_train, sp["ckpt"], ctx.y_train[sp["ckpt"]],
                             lr=lr, epochs=budget, seed=0, eval_every=10**9)
            scores[(lr, r)] = run["ba"]
            results[f"lr={lr},r={r},budget={budget}"] = {"ckpt_ba": run["ba"], "img_per_s": run["img_per_s"]}
            log.info("sweep lr=%g r=%d budget=%.2f: fold-0 ckpt BA %.4f (%.1f img/s)", lr, r, budget, run["ba"],
                     run["img_per_s"])
            del model
            torch.cuda.empty_cache()
        alive = sorted(alive, key=scores.get, reverse=True)[: max(1, len(alive) // 2)]
        if len(alive) == 1:
            break
    best = alive[0]
    out = {"best": {"lr": best[0], "rank": best[1]}, "runs": results}
    (PATHS.results / "lora_sweep.json").write_text(json.dumps(out, indent=2))
    return out


def final_runs(ctx, sp: dict, head_state: dict, lr: float, rank: int, seeds=(0, 1, 2)) -> dict:
    """3 seeds x 1 epoch on folds 1-4; EMA checkpoint chosen on the fold-0 10k subsample."""
    images = PCamImages("train")
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in seeds:
        path = LORA_DIR / f"seed{seed}.pt"
        if path.exists():
            runs[seed] = torch.load(path, weights_only=False)
            continue
        model = new_model(head_state, rank)
        run = train_lora(model, images, sp["train"], ctx.y_train, sp["ckpt"], ctx.y_train[sp["ckpt"]], lr=lr,
                         epochs=1.0, seed=seed, eval_every=500)
        keep = {k: run[k] for k in ("ba", "step", "state", "history", "img_per_s")}
        torch.save(keep, path)
        runs[seed] = keep
        del model
        torch.cuda.empty_cache()
    return runs


def predict_runs(runs: dict, head_state: dict, rank: int, split: str, idx, wise_alpha: float = 1.0) -> np.ndarray:
    """(n_seeds, n) logits of each LoRA run on images[idx] of a split (identity view)."""
    images = PCamImages(split)
    model = new_model(head_state, rank).cuda()
    lp = {k: v for k, v in head_state["module"].items()}
    out = [predict_with_state(model, wise_state(run["state"], lp, wise_alpha), images, idx) for run in runs.values()]
    return np.stack(out)


def step_s5(ctx) -> dict:
    """LoRA arm vs the incumbent on fold 0 (paired, slide-clustered) and the valid gate."""
    sp = splits(ctx)
    head_state = lp_head(ctx, sp["train"])
    sw = sweep(ctx, sp, head_state) if not (PATHS.results / "lora_sweep.json").exists() else \
        json.loads((PATHS.results / "lora_sweep.json").read_text())
    lr, rank = sw["best"]["lr"], sw["best"]["rank"]
    runs = final_runs(ctx, sp, head_state, lr, rank)
    f0 = predict_runs(runs, head_state, rank, "train", sp["fold0"])
    val = predict_runs(runs, head_state, rank, "valid", np.arange(len(ctx.y_valid)))
    name = "S5_lora"
    y0, s0 = ctx.y_train[sp["fold0"]], ctx.slides_train[sp["fold0"]]
    rec = {"name": name, "description": f"LoRA (r={rank}, lr={lr}) on all 24 blocks, LP-FT, D4+HED, EMA",
           "head": "lora", "backbones": ["hibou_l"], "publishable": True, "oof_subset": "fold0",
           "lora": {"lr": lr, "rank": rank, "runs": {s: {k: r[k] for k in ("ba", "step", "history", "img_per_s")}
                                                     for s, r in runs.items()}},
           "oof": ex.summarise(y0, f0, s0), "valid": ex.summarise(ctx.y_valid, val, ctx.slides_valid)}
    ex.save_arm(name, rec, {"oof": f0, "valid": val, "fold0_idx": sp["fold0"]})
    inc = read_ladder()["incumbent"]
    inc_preds = ex.load_preds(base_arm(inc))
    inc_f0 = ex.seed_average(inc_preds["oof"])[sp["fold0"]]
    cmp = {"oof": paired_comparison(y0, inc_f0, ex.seed_average(f0), s0),
           "valid": paired_comparison(ctx.y_valid, ex.seed_average(ex.load_preds(inc)["valid"]), ex.seed_average(val),
                                      ctx.slides_valid)}
    from pcam_hc.stage_b import _record_decision

    _, rep, views, cfg = incumbent()
    ladder = read_ladder()
    ladder["configs"][name] = {**cfg, "lora": {"lr": lr, "rank": rank}, "base_arm": name}
    write_ladder(ladder)
    return _record_decision("S5_lora", inc, name, cmp, cfg, rep, views, lora={"lr": lr, "rank": rank})
