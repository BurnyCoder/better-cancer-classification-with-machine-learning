"""Stage C: LoRA fine-tuning of Hibou-L (S5), weight-space post-processing (S5b), the
augmentation-matched frozen control (H4), and the end-to-end ConvNeXt-T baseline.

Global context: pre-registration section 6. LoRA adapts the Hibou-L-only pipeline that S1-S3b
produced (same representation, views and LP-FT head). It trains on train folds 1-4 and selects
hyperparameters and checkpoints on held-out train fold 0: a fixed stratified 10k subsample for
checkpointing, and all of fold 0 for the S5 decision. The decision compares LoRA with the frozen
incumbent's OOF predictions on the same fold-0 patches (whose heads were also trained on folds 1-4),
plus the one-shot valid gate. S5b choices use fold 0 minus the checkpointing subsample.
"""

import json
import logging

import numpy as np
import torch
from scipy.special import expit
from sklearn.model_selection import train_test_split

from pcam_hc import experiments as ex
from pcam_hc.backbones import REGISTRY, load_model
from pcam_hc.config import PATHS, SEEDS, SUBSET_SIZE
from pcam_hc.data import PCamImages, make_subset
from pcam_hc.finetune import (
    ConvNeXtClassifier, LoraClassifier, predict_with_state, soup_state, train_lora, wise_state,
)
from pcam_hc.ladder import read_ladder, write_ladder
from pcam_hc.stage_b import _fit_cfg, _record_decision, _seeds_for, base_arm, features, incumbent, nested_keep
from pcam_hc.stats import holm, paired_comparison

log = logging.getLogger(__name__)
LORA_DIR = PATHS.ckpt / "lora"
SWEEP = [(lr, r) for lr in (5e-5, 1e-4, 3e-4) for r in (8, 16)]
WISE_ALPHAS = (0.25, 0.5, 0.75)


# ----------------------------------------------------------------------------- shared pieces
def clean_keep_fold0(ctx, cfg: dict, rep, views) -> np.ndarray:
    """LoRA training patches kept by S3b's cleaning: its nested mask for outer fold 0 (inner CV on folds
    1-4 only), so no fold-0 label decides which patches LoRA trains on. Cached on the data disk."""
    path = PATHS.data / "masks" / f"keep_fold0_{cfg.get('clean_source', 'S3b')}_{cfg['clean_threshold']}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, nested_keep(ctx, features("train", rep, views), cfg, 0, cfg["clean_threshold"]))
    return np.load(path)


def splits(ctx, clean: bool = True) -> dict:
    """Index sets: LoRA training patches (folds 1-4, minus labels flagged by S3b's nested fold-0 mask if
    S3b was adopted and `clean`), fold 0, its 10k checkpointing subsample, the fold-0 selection set (fold 0
    minus checkpointing patches), and the sweep set (train-64k within the LoRA training patches).

    The S5 run of 2026-11-01 (and its H4 control) trained on an earlier mask built from S3b's full-train
    OOF predictions, whose fold 1-4 entries come from heads that saw fold-0 labels; `mask_audit` records
    the difference (results/lora_mask_audit.json)."""
    train = np.flatnonzero(ctx.folds != 0)
    cfg = {}
    if clean and read_ladder().get("incumbent"):
        _, rep, views, cfg = incumbent(hibou_only=True)
    if "clean_threshold" in cfg:
        train = clean_keep_fold0(ctx, cfg, rep, views)
    fold0 = np.flatnonzero(ctx.folds == 0)
    ckpt, _ = train_test_split(fold0, train_size=10_000, stratify=ctx.y_train[fold0], random_state=0)
    ckpt = np.sort(ckpt)
    sub = make_subset(ctx.folds, ctx.y_train, SUBSET_SIZE, seed=0)
    return {"train": train, "fold0": fold0, "ckpt": ckpt, "select": np.setdiff1d(fold0, ckpt),
            "sweep_train": np.intersect1d(sub, train)}


def lp_head(ctx, idx) -> dict:
    """LP-FT initial head: the incumbent's frozen-probe recipe fit on the LoRA training patches only."""
    _, rep, views, cfg = incumbent(hibou_only=True)
    X = features("train", rep, views)
    head = _fit_cfg(cfg, X[idx], ctx.y_train[idx], seed=0)
    return {"module": {k: v.cpu() for k, v in head.module.state_dict().items()}, "kind": cfg["head"],
            "mu": None if head.mu is None else head.mu.cpu(), "sd": None if head.sd is None else head.sd.cpu()}


def new_model(head_state, rank: int, seed: int = 0) -> LoraClassifier:
    """Hibou-L + LoRA; `seed` fixes the adapter initialisation (peft draws lora_A from the global RNG)."""
    _, rep, views, _ = incumbent(hibou_only=True)
    spec = REGISTRY["hibou_l"]
    torch.manual_seed(seed)
    return LoraClassifier(load_model(spec).to(torch.bfloat16), spec, head_state, rep, views, rank=rank)


def mask_audit(ctx) -> dict:
    """Compare the cleaning mask the 2026-11-01 S5 run trained on (S3b full-train OOF, fold-0 labels inside)
    with the nested fold-0 mask that `splits` now uses; written to results/lora_mask_audit.json."""
    _, rep, views, cfg = incumbent(hibou_only=True)
    tr, y, thr = np.flatnonzero(ctx.folds != 0), ctx.y_train, cfg["clean_threshold"]
    p = expit(ex.load_preds(cfg.get("clean_source", "S3b_oof_label_cleaning"))["oof"].astype(np.float64)).mean(0)
    used = set(tr[np.abs(y[tr] - p[tr]) > thr])
    nested = set(tr) - set(clean_keep_fold0(ctx, cfg, rep, views))
    out = {"lora_train_candidates": int(len(tr)), "dropped_by_used_mask": len(used), "dropped_by_nested_mask": len(nested),
           "dropped_by_both": len(used & nested), "jaccard_dropped": len(used & nested) / max(1, len(used | nested)),
           "patches_trained_differently": len(used ^ nested)}
    (PATHS.results / "lora_mask_audit.json").write_text(json.dumps(out, indent=2))
    return out


def predict_states(model, states: list, split: str, idx) -> np.ndarray:
    """(n_states, n) logits of one model architecture under several trainable-weight states."""
    images = PCamImages(split)
    model.cuda()
    return np.stack([predict_with_state(model, st, images, idx) for st in states])


# ----------------------------------------------------------------------------- S5 LoRA
def sweep(ctx, sp: dict, head_state: dict) -> dict:
    """Successive halving over lr x rank on the sweep set: every config gets 1/4 epoch, the best half
    1/2 epoch, the best of those a full epoch of the sweep set; scored on the fold-0 checkpoint subsample."""
    images = PCamImages("train", in_memory=True)
    alive, results = list(SWEEP), {}
    for budget in (0.25, 0.5, 1.0):
        scores = {}
        for lr, r in alive:
            run = train_lora(new_model(head_state, r), images, sp["sweep_train"], ctx.y_train, sp["ckpt"],
                             ctx.y_train[sp["ckpt"]], lr=lr, epochs=budget, seed=0, eval_every=10**9)
            scores[(lr, r)] = run["ba"]
            results[f"lr={lr},r={r},budget={budget}"] = {"ckpt_ba": run["ba"], "img_per_s": run["img_per_s"]}
            log.info("sweep lr=%g r=%d budget=%.2f: ckpt BA %.4f (%.1f img/s)", lr, r, budget, run["ba"], run["img_per_s"])
            torch.cuda.empty_cache()
        alive = sorted(alive, key=scores.get, reverse=True)[: max(1, len(alive) // 2)]
        if len(alive) == 1:
            break
    out = {"best": {"lr": alive[0][0], "rank": alive[0][1]}, "runs": results}
    (PATHS.results / "lora_sweep.json").write_text(json.dumps(out, indent=2))
    return out


def final_runs(ctx, sp: dict, head_state: dict, lr: float, rank: int, seeds=SEEDS) -> dict:
    """One run per seed (1 seed; seed sensitivity postponed) x 1 epoch on folds 1-4; best EMA checkpoint on the fold-0 10k subsample (resumable)."""
    images = PCamImages("train", in_memory=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in seeds:
        path = LORA_DIR / f"seed{seed}.pt"
        if not path.exists():
            run = train_lora(new_model(head_state, rank, seed=seed), images, sp["train"], ctx.y_train, sp["ckpt"],
                             ctx.y_train[sp["ckpt"]], lr=lr, epochs=1.0, seed=seed, eval_every=500)
            torch.save({k: run[k] for k in ("ba", "step", "state", "history", "img_per_s")}, path)
            torch.cuda.empty_cache()
        runs[seed] = torch.load(path, weights_only=False)
    torch.save(head_state, LORA_DIR / "lp_head.pt")
    return runs


def step_s5(ctx) -> dict:
    sp = splits(ctx)
    head_state = lp_head(ctx, sp["train"])
    sweep_path = PATHS.results / "lora_sweep.json"
    sw = json.loads(sweep_path.read_text()) if sweep_path.exists() else sweep(ctx, sp, head_state)
    lr, rank = sw["best"]["lr"], sw["best"]["rank"]
    runs = final_runs(ctx, sp, head_state, lr, rank)
    model = new_model(head_state, rank)
    states = [r["state"] for r in runs.values()]
    f0 = predict_states(model, states, "train", sp["fold0"])
    val = predict_states(model, states, "valid", np.arange(len(ctx.y_valid)))
    name = "S5_lora"
    y0, s0 = ctx.y_train[sp["fold0"]], ctx.slides_train[sp["fold0"]]
    rec = {"name": name, "description": f"LoRA r={rank} lr={lr}, all 24 blocks, LP-FT, D4+HED, drop-path 0.1, EMA",
           "head": "lora", "backbones": ["hibou_l"], "publishable": True, "oof_subset": "fold0",
           "lora": {"lr": lr, "rank": rank, "runs": {str(s): {k: r[k] for k in ("ba", "step", "history", "img_per_s")}
                                                     for s, r in runs.items()}},
           "oof": ex.summarise(y0, f0, s0), "valid": ex.summarise(ctx.y_valid, val, ctx.slides_valid)}
    ex.save_arm(name, rec, {"oof": f0, "valid": val, "fold0_idx": sp["fold0"]})
    hib, rep, views, cfg = incumbent(hibou_only=True)
    base = ex.load_preds(base_arm(hib))
    cmp = {"oof": paired_comparison(y0, ex.seed_average(base["oof"])[sp["fold0"]], ex.seed_average(f0), s0),
           "valid": paired_comparison(ctx.y_valid, ex.seed_average(base["valid"]), ex.seed_average(val),
                                      ctx.slides_valid)}
    overall = read_ladder()["incumbent"]
    decision = _record_decision("S5_lora", hib, name, cmp, cfg, rep, views, lora={"lr": lr, "rank": rank})
    ladder = read_ladder()  # S5 changes only the Hibou-L component; the overall incumbent is recombined in S6
    ladder["configs"].setdefault(name, {**cfg, "lora": {"lr": lr, "rank": rank}})
    ladder["configs"][name]["base_arm"] = name
    ladder["hibou_incumbent"] = name if decision["adopted"] else hib
    ladder["incumbent"] = overall
    write_ladder(ladder)
    return decision


# ----------------------------------------------------------------------------- S5b weight space
def step_s5b(ctx) -> dict:
    """WiSE-style scaling alpha in {0.25, 0.5, 0.75} (and, with several LoRA runs, a uniform soup) vs
    the LoRA model (alpha = 1), on fold 0 minus the checkpointing subsample, Holm over the candidates."""
    sp = splits(ctx)
    head_state = torch.load(LORA_DIR / "lp_head.pt", weights_only=False)
    lora = read_ladder()["configs"]["S5_lora"]["lora"]
    runs = [torch.load(p, weights_only=False)["state"] for p in sorted(LORA_DIR.glob("seed*.pt"))]
    lp = {k: v for k, v in head_state["module"].items()}
    sel, y_sel, s_sel = sp["select"], ctx.y_train[sp["select"]], ctx.slides_train[sp["select"]]
    base = ex.load_preds("S5_lora")
    pos = np.searchsorted(base["fold0_idx"], sel)
    inc_sel, inc_val = ex.seed_average(base["oof"][:, pos]), ex.seed_average(base["valid"])
    model = new_model(head_state, lora["rank"])
    cands, preds = {}, {}
    for a in WISE_ALPHAS:
        states = [wise_state(r, lp, a) for r in runs]
        preds[f"S5b_wise_{a}"] = (predict_states(model, states, "train", sel),
                                  predict_states(model, states, "valid", np.arange(len(ctx.y_valid))))
    if len(runs) > 1:  # a soup needs several runs; with one seed it is postponed with the seed study
        soup_model = new_model(head_state, len(runs) * lora["rank"])
        soup = soup_state(runs)
        preds["S5b_soup"] = (predict_states(soup_model, [soup], "train", sel),
                             predict_states(soup_model, [soup], "valid", np.arange(len(ctx.y_valid))))
    for name, (o, v) in preds.items():
        rec = {"name": name, "description": name.replace("_", " "), "head": "lora", "backbones": ["hibou_l"],
               "publishable": True, "oof_subset": "fold0_minus_ckpt", "oof": ex.summarise(y_sel, o, s_sel),
               "valid": ex.summarise(ctx.y_valid, v, ctx.slides_valid)}
        ex.save_arm(name, rec, {"oof": o, "valid": v, "select_idx": sel})
        cands[name] = {"oof": paired_comparison(y_sel, inc_sel, ex.seed_average(o), s_sel),
                       "valid": paired_comparison(ctx.y_valid, inc_val, ex.seed_average(v), ctx.slides_valid)}
    names = list(cands)
    p_adj = dict(zip(names, holm([cands[n]["oof"]["p_signflip"] for n in names])))
    winner = max(names, key=lambda n: cands[n]["oof"]["delta_ba"])
    oof_ok = cands[winner]["oof"]["delta_ba"] > 0 and p_adj[winner] < 0.05
    valid_ok = cands[winner]["valid"]["delta_ba"] >= 0 and cands[winner]["valid"]["delta_auroc"] >= 0
    decision = {"step": "S5b_weight_space", "incumbent": "S5_lora", "candidates": names, "winner": winner,
                "holm_p": p_adj, "comparisons": cands, "oof_test_passed": bool(oof_ok),
                "valid_gate_passed": bool(valid_ok), "adopted": bool(oof_ok and valid_ok)}
    decision["new_incumbent"] = winner if decision["adopted"] else "S5_lora"
    (PATHS.results / "steps" / "S5b_weight_space.json").write_text(json.dumps(decision, indent=2, default=float))
    ladder = read_ladder()
    if decision["adopted"]:
        ladder["configs"][winner] = {**ladder["configs"]["S5_lora"], "post": winner}
        if ladder.get("hibou_incumbent") == "S5_lora":
            ladder["hibou_incumbent"] = winner
    ladder["history"].append({"step": "S5b_weight_space", "winner": winner, "adopted": decision["adopted"],
                              "incumbent_after": decision["new_incumbent"]})
    write_ladder(ladder)
    log.info("S5b: winner %s, adopted %s", winner, decision["adopted"])
    return decision


# ----------------------------------------------------------------------------- H4 control
def h4_control(ctx) -> dict:
    """Frozen Hibou-L head trained on the LoRA training patches, each seen once under one seeded random
    D4 + HED-light view (the augmentation LoRA trains with); scored on identity views of fold 0 / valid."""
    from pcam_hc.embed import embed_backbone

    sp = splits(ctx)
    _, rep, views, cfg = incumbent(hibou_only=True)
    for base in views:
        view = "aug_s0" if base == "full" else f"aug_s0+{base}"
        embed_backbone("hibou_l", "train", view=view, indices=sp["train"], subset="loratrain")
    parts = [ex_load_aug(rep, base) for base in views]
    X_aug = parts[0] if len(parts) == 1 else np.hstack(parts)
    X_full = features("train", rep, views)
    X_val = features("valid", rep, views)
    f0, val = [], []
    (ex.HEAD_DIR / "H4_aug_control").mkdir(parents=True, exist_ok=True)
    for seed in _seeds_for(cfg):
        head = _fit_cfg(cfg, X_aug, ctx.y_train[sp["train"]], seed)
        f0.append(head.logits(X_full[sp["fold0"]]))
        val.append(head.logits(X_val))
        torch.save({"module": head.module.state_dict(), "mu": head.mu, "sd": head.sd, "kind": cfg["head"]},
                   ex.HEAD_DIR / "H4_aug_control" / f"seed{seed}.pt")  # kept for test scoring
    f0, val = np.stack(f0).astype(np.float32), np.stack(val).astype(np.float32)
    y0, s0 = ctx.y_train[sp["fold0"]], ctx.slides_train[sp["fold0"]]
    rec = {"name": "H4_aug_control", "description": "frozen Hibou-L head on one random D4+HED view per LoRA training patch",
           "head": cfg["head"], "backbones": ["hibou_l"], "publishable": True, "oof_subset": "fold0",
           "oof": ex.summarise(y0, f0, s0), "valid": ex.summarise(ctx.y_valid, val, ctx.slides_valid)}
    ex.save_arm("H4_aug_control", rec, {"oof": f0, "valid": val, "fold0_idx": sp["fold0"]})
    return rec


def ex_load_aug(rep, base: str) -> np.ndarray:
    from pcam_hc.features import load

    view = "aug_s0" if base == "full" else f"aug_s0+{base}"
    return load("hibou_l", "train", rep, view=view, subset="loratrain")


# ----------------------------------------------------------------------------- ConvNeXt-T baseline
def convnext_baseline(ctx, seeds=SEEDS, lr: float = 1e-4, epochs: float = 2.0) -> dict:
    """End-to-end ImageNet ConvNeXt-T, all weights trained, D4 + HED-light, same splits as LoRA.
    Fixed recipe (not tuned): AdamW lr 1e-4, 2 epochs, batch 64, cosine schedule, EMA, drop-path 0.1,
    gradient checkpointing (batch 128 exceeded the 8 GB card)."""
    sp = splits(ctx, clean=False)  # a baseline: plain train folds 1-4, no ladder components
    images = PCamImages("train", in_memory=True)
    out_dir = PATHS.ckpt / "convnext"
    out_dir.mkdir(parents=True, exist_ok=True)
    f0, val, runs = [], [], {}
    for seed in seeds:
        path = out_dir / f"seed{seed}.pt"
        if not path.exists():
            run = train_lora(ConvNeXtClassifier(), images, sp["train"], ctx.y_train, sp["ckpt"],
                             ctx.y_train[sp["ckpt"]], lr=lr, epochs=epochs, batch=64, seed=seed, eval_every=2000)
            torch.save({k: run[k] for k in ("ba", "step", "state", "history", "img_per_s")}, path)
        run = torch.load(path, weights_only=False)
        runs[seed] = {k: run[k] for k in ("ba", "step", "history", "img_per_s")}
        model = ConvNeXtClassifier().cuda()
        f0.append(predict_with_state(model, run["state"], images, sp["fold0"]))
        val.append(predict_with_state(model, run["state"], PCamImages("valid"), np.arange(len(ctx.y_valid))))
    f0, val = np.stack(f0), np.stack(val)
    y0, s0 = ctx.y_train[sp["fold0"]], ctx.slides_train[sp["fold0"]]
    rec = {"name": "baseline_convnext_t", "description": "end-to-end ImageNet ConvNeXt-T, D4+HED, 2 epochs",
           "head": "end-to-end", "backbones": [], "publishable": True, "oof_subset": "fold0", "runs": runs,
           "oof": ex.summarise(y0, f0, s0), "valid": ex.summarise(ctx.y_valid, val, ctx.slides_valid)}
    ex.save_arm("baseline_convnext_t", rec, {"oof": f0, "valid": val, "fold0_idx": sp["fold0"]})
    return rec
