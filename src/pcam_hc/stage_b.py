"""Stage B: the frozen-feature ladder steps S1-S4 on Hibou-L (pre-registration section 6).

Global context: each step builds candidate arms on cached embeddings, scores them out-of-fold (OOF)
on the five slide-grouped train folds plus a full-train refit on valid, and hands them to
`ladder.decide_step`, which applies the pre-registered adoption rule. The incumbent arm (name,
representation, head) is tracked in results/ladder.json together with the config needed to
rebuild it, so the frozen ladder can later be scored on the sealed test split.
"""

import json
import logging

import numpy as np
import torch
from scipy.special import expit

from pcam_hc import experiments as ex
from pcam_hc.config import PATHS, SEEDS, SUBSET_SIZE
from pcam_hc.data import make_subset
from pcam_hc.features import CLS, S1_HIBOU, Representation, load
from pcam_hc.heads import fit_head
from pcam_hc.ladder import arm_record, decide_step, read_ladder, write_ladder
from pcam_hc.metrics import balanced_accuracy
from pcam_hc.stats import paired_comparison

log = logging.getLogger(__name__)
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)  # logistic-regression inverse regularisation grid


# ----------------------------------------------------------------------------- incumbent config
def _rep_to_json(rep: Representation, views=("full",)) -> dict:
    return {"parts": [list(p) for p in rep.parts], "center": rep.center, "views": list(views)}


def recipe(rep: Representation, views) -> dict:
    """How an arm's Hibou-L features are built (stored in the arm record for later test scoring)."""
    return {"backbone": "hibou_l", "representation": _rep_to_json(rep, views)}


def _rep_from_json(d: dict) -> tuple:
    return Representation(tuple(tuple(p) for p in d["parts"]), d["center"]), tuple(d["views"])


def set_incumbent_config(arm: str, rep: Representation, views, head: str, **head_kw) -> None:
    """Remember how an arm is built (used by later steps and by test scoring after unsealing)."""
    ladder = read_ladder()
    ladder.setdefault("configs", {})[arm] = {"representation": _rep_to_json(rep, views), "head": head, **head_kw}
    write_ladder(ladder)


def incumbent(hibou_only: bool = False) -> tuple:
    """(name, representation, views, config) of the current incumbent; with `hibou_only`, of the
    Hibou-L-only pipeline that S4 fused with other backbones (used by S5 LoRA)."""
    ladder = read_ladder()
    name = ladder.get("hibou_incumbent", ladder["incumbent"]) if hibou_only else ladder["incumbent"]
    cfg = ladder["configs"][name]
    rep, views = _rep_from_json(cfg["representation"])
    return name, rep, views, cfg


def view_name(base: str, d4: int) -> str:
    """Cache view name of base view `base` under D4 element `d4` (0 = identity), e.g. 'center64+d4_3'."""
    if d4 == 0:
        return base
    return f"d4_{d4}" if base == "full" else f"{base}+d4_{d4}"


def features(split: str, rep: Representation, views=("full",), subset=None, d4: int = 0) -> np.ndarray:
    """Hibou-L features for a representation; several `views` (e.g. full patch + centre crop)
    contribute the same columns side by side. `d4` selects the rotated/flipped copy of every view."""
    parts = [load("hibou_l", split, rep, view=view_name(v, d4), subset=subset) for v in views]
    return parts[0] if len(parts) == 1 else np.hstack(parts)


# ----------------------------------------------------------------------------- S1 representation
def step_s1(ctx) -> dict:
    base = "S0_hibou_l_cls_linear"
    ladder = read_ladder()
    if ladder["incumbent"] is None:
        ladder["incumbent"] = base
        write_ladder(ladder)
        set_incumbent_config(base, CLS, ("full",), "linear")
    names = {"a_cls24": base}
    for key, rep in S1_HIBOU.items():
        if key == "a_cls24":
            continue
        name = f"S1_{key}"
        names[key] = name
        set_incumbent_config(name, rep, ("full",), "linear")
        if not (PATHS.results / "arms" / f"{name}.json").exists():
            ex.run_arm(ctx, name, features("train", rep), features("valid", rep), description=f"Hibou-L {key}, linear",
                       recipe=recipe(rep, ("full",)))
    best_key = max(names, key=lambda k: arm_record(names[k])["oof"]["seed_avg"]["ba"])
    best_rep = S1_HIBOU[best_key]
    extra = {
        "S1_e_best_plus_centre_tokens": (Representation(best_rep.parts, center=True), ("full",)),
        "S1_f_best_plus_centre_view": (best_rep, ("full", "center64")),
    }
    for name, (rep, views) in extra.items():
        set_incumbent_config(name, rep, views, "linear")
        if not (PATHS.results / "arms" / f"{name}.json").exists():
            ex.run_arm(ctx, name, features("train", rep, views), features("valid", rep, views),
                       description=f"best of S1 a-d ({best_key}) + {name.split('plus_')[1]}, linear",
                       recipe=recipe(rep, views))
    return decide_step(ctx, "S1_representation", base, [n for k, n in names.items() if k != "a_cls24"] + list(extra))


# ----------------------------------------------------------------------------- S2 heads
def _logreg_oof(X, y, folds, Cs=C_GRID) -> tuple:
    """Nested CV: in each outer fold, C is chosen by leave-one-inner-fold-out BA on the other folds."""
    oof = np.zeros(len(y), dtype=np.float32)
    chosen = []
    for f in range(folds.max() + 1):
        inner = np.flatnonzero(folds != f)
        inner_folds = folds[inner]
        scores = {C: [] for C in Cs}
        for g in np.unique(inner_folds):
            tr, va = inner[inner_folds != g], inner[inner_folds == g]
            head = None
            for C in Cs:  # warm start along the regularisation path
                head = fit_head("logreg", X[tr], y[tr], C=C, init=head)
                scores[C].append(balanced_accuracy(y[va], head.logits(X[va]) > 0))
        best_C = max(Cs, key=lambda C: np.mean(scores[C]))
        chosen.append(best_C)
        oof[folds == f] = fit_head("logreg", X[inner], y[inner], C=best_C).logits(X[folds == f])
        log.info("logreg outer fold %d: C=%g (inner BA %.4f)", f, best_C, np.mean(scores[best_C]))
    return oof, chosen


def step_s2(ctx) -> dict:
    inc, rep, views, cfg = incumbent()
    Xtr, Xva = features("train", rep, views), features("valid", rep, views)
    name_lr, name_mlp = "S2_logreg", "S2_mlp"
    if not (PATHS.results / "arms" / f"{name_lr}.json").exists():
        oof, chosen = _logreg_oof(Xtr, ctx.y_train, ctx.folds)
        # Valid refit: C by full 5-fold OOF BA on train, then fit on all of train.
        per_c = {}
        for C in C_GRID:
            per_c[C] = balanced_accuracy(ctx.y_train, ex.cv_oof("logreg", Xtr, ctx.y_train, ctx.folds, seeds=(0,), C=C)[0] > 0)
        C_star = max(C_GRID, key=per_c.get)
        val = ex.refit("logreg", Xtr, ctx.y_train, Xva, seeds=(0,), save_as=name_lr, C=C_star)
        record = {"name": name_lr, "description": "z-scored features + L2 logistic regression, nested-CV C",
                  "head": "logreg", "backbones": ["hibou_l"], "publishable": True, "dim": int(Xtr.shape[1]),
                  "chosen_C_per_fold": chosen, "valid_refit_C": C_star, "cv_ba_per_C": per_c,
                  "oof": ex.summarise(ctx.y_train, oof[None], ctx.slides_train),
                  "valid": ex.summarise(ctx.y_valid, val, ctx.slides_valid), "recipe": recipe(rep, views)}
        ex.save_arm(name_lr, record, {"oof": oof[None], "valid": val})
        set_incumbent_config(name_lr, rep, views, "logreg", C=C_star)
    if not (PATHS.results / "arms" / f"{name_mlp}.json").exists():
        ex.run_arm(ctx, name_mlp, Xtr, Xva, kind="mlp", description="z-scored features + MLP(512, GELU, dropout 0.1)",
                   recipe=recipe(rep, views))
        set_incumbent_config(name_mlp, rep, views, "mlp")
    return decide_step(ctx, "S2_head", inc, [name_lr, name_mlp])


# ----------------------------------------------------------------------------- S3 D4 TTA (train-64k)
def subset_indices(ctx) -> np.ndarray:
    return make_subset(ctx.folds, ctx.y_train, SUBSET_SIZE, seed=0)


def _fit_cfg(cfg, X, y, seed):
    kw = {"C": cfg["C"]} if cfg["head"] == "logreg" else {}
    return fit_head(cfg["head"], X, y, seed=seed, **kw)


def _seeds_for(cfg):
    return (0,) if cfg["head"] == "logreg" else SEEDS


def step_s3(ctx) -> dict:
    """TTA vs no TTA with identical heads: heads fit on the other folds' full-view features predict
    the held-out fold's train-64k patches under the identity view or the mean logit of 8 D4 views."""
    inc, rep, views, cfg = incumbent()
    sub = subset_indices(ctx)
    Xtr = features("train", rep, views)
    seeds, n_folds = _seeds_for(cfg), ctx.folds.max() + 1
    # Fit every head once (fold heads for OOF, full-train heads for valid), then stream the views one
    # at a time so at most one rotated copy of the features is in memory.
    fold_heads = {(si, f): _fit_cfg(cfg, Xtr[ctx.folds != f], ctx.y_train[ctx.folds != f], seed)
                  for si, seed in enumerate(seeds) for f in range(n_folds)}
    full_heads = [_fit_cfg(cfg, Xtr, ctx.y_train, seed) for seed in seeds]
    for seed, h in zip(seeds, full_heads):  # kept for scoring the TTA arm on test after unsealing
        (ex.HEAD_DIR / "S3_d4_tta").mkdir(parents=True, exist_ok=True)
        torch.save({"module": h.module.state_dict(), "mu": h.mu, "sd": h.sd, "kind": cfg["head"]},
                   ex.HEAD_DIR / "S3_d4_tta" / f"seed{seed}.pt")
    sub_folds = ctx.folds[sub]
    logit_sum = np.zeros((len(seeds), len(sub)))
    val_sum = np.zeros((len(seeds), len(ctx.y_valid)))
    for k in range(8):
        Xs = Xtr[sub] if k == 0 else features("train", rep, views, subset="train64k", d4=k)
        Xv = features("valid", rep, views, d4=k)
        for si in range(len(seeds)):
            for f in range(n_folds):
                m = sub_folds == f
                logit_sum[si, m] += fold_heads[(si, f)].logits(Xs[m])
        val_k = np.stack([h.logits(Xv) for h in full_heads])
        val_sum += val_k
        if k == 0:  # identity view = the no-TTA arm, scored by the very same heads
            no_tta, val_no = logit_sum.astype(np.float32).copy(), val_k.astype(np.float32)
        del Xs, Xv
    tta = (logit_sum / 8).astype(np.float32)
    val_tta = (val_sum / 8).astype(np.float32)
    y_sub, s_sub = ctx.y_train[sub], ctx.slides_train[sub]
    records = {}
    for name, oof, val in [("S3_no_tta_train64k", no_tta, val_no), ("S3_d4_tta", tta, val_tta)]:
        rec = {"name": name, "description": "incumbent head; " + ("8-view D4 TTA" if "d4" in name else "identity view"),
               "head": cfg["head"], "backbones": ["hibou_l"], "publishable": True, "dim": int(Xtr.shape[1]),
               "recipe": {**recipe(rep, views), "tta": "d4" in name, "heads_from": "S3_d4_tta"},
               "oof_subset": "train64k", "oof": ex.summarise(y_sub, oof, s_sub),
               "valid": ex.summarise(ctx.y_valid, val, ctx.slides_valid)}
        ex.save_arm(name, rec, {"oof": oof, "valid": val})
        records[name] = rec
    cmp = {"oof": paired_comparison(y_sub, ex.seed_average(no_tta), ex.seed_average(tta), s_sub),
           "valid": paired_comparison(ctx.y_valid, ex.seed_average(val_no), ex.seed_average(val_tta), ctx.slides_valid)}
    return _record_decision("S3_d4_tta", inc, "S3_d4_tta", cmp, cfg, rep, views, tta=True)


def _record_decision(step, inc, challenger, cmp, cfg, rep, views, **flags) -> dict:
    """Decision for steps whose OOF comparison is not on the full train set (S3) or nested (S3b)."""
    oof_ok = cmp["oof"]["delta_ba"] > 0 and cmp["oof"]["p_signflip"] < 0.05  # single candidate: Holm = raw p
    valid_ok = cmp["valid"]["delta_ba"] >= 0 and cmp["valid"]["delta_auroc"] >= 0
    decision = {"step": step, "incumbent": inc, "candidates": [challenger], "winner": challenger, "comparisons":
                {challenger: cmp}, "oof_test_passed": bool(oof_ok), "valid_gate_passed": bool(valid_ok),
                "adopted": bool(oof_ok and valid_ok)}
    decision["new_incumbent"] = challenger if decision["adopted"] else inc
    (PATHS.results / "steps").mkdir(parents=True, exist_ok=True)
    (PATHS.results / "steps" / f"{step}.json").write_text(json.dumps(decision, indent=2, default=float))
    ladder = read_ladder()
    if decision["adopted"]:
        ladder["configs"][challenger] = {**ladder["configs"][inc], **flags, "base_arm": base_arm(inc)}
        ladder["incumbent"] = challenger
    ladder["history"].append({"step": step, "winner": challenger, "adopted": decision["adopted"],
                              "incumbent_after": decision["new_incumbent"]})
    write_ladder(ladder)
    log.info("step %s: delta OOF BA %.4f (p %.3g), valid gate %s -> %s", step, cmp["oof"]["delta_ba"],
             cmp["oof"]["p_signflip"], valid_ok, "ADOPTED" if decision["adopted"] else "rejected")
    return decision


# ----------------------------------------------------------------------------- S3b OOF label cleaning
def base_arm(name: str) -> str:
    """The arm holding full-train OOF predictions for a config (TTA arms only have train-64k OOF)."""
    return read_ladder()["configs"].get(name, {}).get("base_arm", name)


def step_s3b(ctx, threshold: float = 0.9) -> dict:
    """PANDA-1st-place-style cleaning, nested so evaluation labels are never altered: inside each outer
    training set, inner slide-grouped OOF predictions flag patches with |y - p| > threshold; the head
    is refit without them and scored on the untouched outer fold."""
    inc, rep, views, cfg = incumbent()
    inc_full = base_arm(inc)
    name = "S3b_oof_label_cleaning"
    Xtr, Xva = features("train", rep, views), features("valid", rep, views)
    y, folds, seeds = ctx.y_train, ctx.folds, _seeds_for(cfg)
    head_kw = {"C": cfg["C"]} if cfg["head"] == "logreg" else {}
    oof = np.zeros((len(seeds), len(y)), np.float32)
    removed = {}
    for f in range(folds.max() + 1):
        tr = np.flatnonzero(folds != f)
        inner_folds = folds[tr] - (folds[tr] > f)  # renumber the four remaining folds 0..3
        inner = ex.cv_oof(cfg["head"], Xtr[tr], y[tr], inner_folds, seeds=seeds, **head_kw)
        p = expit(inner.astype(np.float64)).mean(0)
        keep = tr[np.abs(y[tr] - p) <= threshold]
        removed[f] = float(1 - len(keep) / len(tr))
        for si, seed in enumerate(seeds):
            oof[si, folds == f] = _fit_cfg(cfg, Xtr[keep], y[keep], seed).logits(Xtr[folds == f])
        log.info("S3b fold %d: removed %.3f%% of training patches", f, 100 * removed[f])
    # Valid refit: flags from the incumbent's own full-train OOF predictions.
    p_full = expit(ex.load_preds(inc_full)["oof"].astype(np.float64)).mean(0)
    keep = np.flatnonzero(np.abs(y - p_full) <= threshold)
    val = ex.refit(cfg["head"], Xtr[keep], y[keep], Xva, seeds=seeds, save_as=name, **head_kw)  # heads kept for test
    rec = {"name": name, "description": f"nested OOF label cleaning, |y-p| > {threshold} dropped",
           "head": cfg["head"], "backbones": ["hibou_l"], "publishable": True, "dim": int(Xtr.shape[1]),
           "removed_fraction_per_fold": removed, "removed_fraction_valid_refit": float(1 - len(keep) / len(y)),
           "oof": ex.summarise(y, oof, ctx.slides_train), "valid": ex.summarise(ctx.y_valid, val, ctx.slides_valid),
           "recipe": recipe(rep, views)}
    ex.save_arm(name, rec, {"oof": oof, "valid": val})
    ladder = read_ladder()
    # clean_source: the arm whose full-train OOF predictions define the mask; configs derived later (S5) inherit it
    ladder["configs"][name] = {**ladder["configs"][inc_full], "clean_threshold": threshold, "clean_source": name}
    write_ladder(ladder)
    decision = decide_step(ctx, "S3b_label_cleaning", inc_full, [name])
    if inc != inc_full:  # a TTA incumbent stays the incumbent; cleaning (if adopted) is recorded on it
        ladder = read_ladder()
        if decision["adopted"]:
            ladder["configs"][inc]["clean_threshold"] = threshold
            ladder["configs"][inc]["clean_source"] = name
            ladder["configs"][inc]["base_arm"] = name
        ladder["incumbent"] = inc
        ladder["history"][-1]["incumbent_after"] = inc
        write_ladder(ladder)
    return decision


# ----------------------------------------------------------------------------- S4 fusion
def _greedy_weights(member_logits: dict, y, start: str, rounds: int = 20) -> dict:
    """Caruana et al. (2004) forward selection with replacement on mean logits, initialised with `start`;
    stops when no member improves BA."""
    counts = {k: 0 for k in member_logits}
    counts[start] = 1
    current = member_logits[start].astype(np.float64).copy()
    best = balanced_accuracy(y, current > 0)
    for _ in range(rounds):
        n = sum(counts.values())
        trial = {k: balanced_accuracy(y, (current * n + v) / (n + 1) > 0) for k, v in member_logits.items()}
        k = max(trial, key=trial.get)
        if trial[k] <= best:
            break
        best, counts[k] = trial[k], counts[k] + 1
        current = (current * n + member_logits[k]) / (n + 1)
    return {k: c for k, c in counts.items() if c}


def _apply_weights(weights: dict, member_logits: dict) -> np.ndarray:
    total = sum(weights.values())
    return sum(member_logits[k].astype(np.float64) * w for k, w in weights.items()) / total


def _backbone_of(arm: str, inc_full: str) -> str:
    return "hibou_l" if arm == inc_full else arm.replace("S0_scan_", "")


def step_s4(ctx, candidates: list[str]) -> dict:
    """Fusion of the Hibou-L incumbent with the S0-scan backbones: nested greedy logit averaging,
    nested logistic-regression stacking, and embedding concatenation with B*. Licence-clean variants
    (publishable backbones only) are always computed alongside and reported, not decided on."""
    from pcam_hc.backbones import REGISTRY

    inc, rep, views, cfg = incumbent()
    inc_full = base_arm(inc)
    pool = {inc_full: ex.load_preds(inc_full), **{f"S0_scan_{b}": ex.load_preds(f"S0_scan_{b}") for b in candidates}}
    oof = {k: ex.seed_average(v["oof"]) for k, v in pool.items()}
    val = {k: ex.seed_average(v["valid"]) for k, v in pool.items()}
    y, folds = ctx.y_train, ctx.folds
    decision_pool = []
    for clean in (False, True):
        keys = [k for k in pool if not clean or REGISTRY[_backbone_of(k, inc_full)].publishable]
        tag = "_licence_clean" if clean else ""
        sub_oof, sub_val = {k: oof[k] for k in keys}, {k: val[k] for k in keys}
        g_oof = np.zeros(len(y))
        for f in range(folds.max() + 1):  # nested: weights from the other folds only
            w = _greedy_weights({k: v[folds != f] for k, v in sub_oof.items()}, y[folds != f], inc_full)
            g_oof[folds == f] = _apply_weights(w, {k: v[folds == f] for k, v in sub_oof.items()})
        w_full = _greedy_weights(sub_oof, y, inc_full)
        g_val = _apply_weights(w_full, sub_val)
        Z, Zv = np.stack([sub_oof[k] for k in keys], 1), np.stack([sub_val[k] for k in keys], 1)
        s_oof = np.zeros(len(y))
        for f in range(folds.max() + 1):
            s_oof[folds == f] = fit_head("logreg", Z[folds != f].astype(np.float32), y[folds != f]).logits(Z[folds == f])
        s_val = ex.refit("logreg", Z.astype(np.float32), y, Zv.astype(np.float32), seeds=(0,),
                         save_as=f"S4_stacking{tag}", C=1.0)[0]  # meta-learner kept for test
        arms_out = [(f"S4_greedy{tag}", g_oof, g_val, sorted({_backbone_of(k, inc_full) for k in w_full}),
                     {"weights_full_train": w_full}),
                    (f"S4_stacking{tag}", s_oof, s_val, sorted({_backbone_of(k, inc_full) for k in keys}),
                     {"members": keys})]
        for name, o, v, used, extra in arms_out:
            rec = {"name": name, "description": name.replace("_", " "), "head": "ensemble", "backbones": used,
                   "publishable": ex.lineage_publishable(used), "oof": ex.summarise(y, o[None], ctx.slides_train),
                   "valid": ex.summarise(ctx.y_valid, v[None], ctx.slides_valid), **extra}
            ex.save_arm(name, rec, {"oof": o[None].astype(np.float32), "valid": v[None].astype(np.float32)})
            if not clean:
                decision_pool.append(name)
    b_star = max(candidates, key=lambda b: arm_record(f"S0_scan_{b}")["oof"]["seed_avg"]["ba"])
    name = "S4_concat_bstar"
    if not (PATHS.results / "arms" / f"{name}.json").exists():
        Xtr = np.hstack([features("train", rep, views), load(b_star, "train", CLS)])
        Xva = np.hstack([features("valid", rep, views), load(b_star, "valid", CLS)])
        kw = {"C": cfg["C"]} if cfg["head"] == "logreg" else {}
        ex.run_arm(ctx, name, Xtr, Xva, kind=cfg["head"], backbones=("hibou_l", b_star), seeds=_seeds_for(cfg),
                   description=f"Hibou-L incumbent features concatenated with B* ({b_star}) CLS",
                   recipe={**recipe(rep, views), "concat_cls": b_star}, **kw)
    decision_pool.append(name)
    ladder = read_ladder()  # remember the Hibou-L-only pipeline: S5 adapts it, S6 recombines components
    ladder["hibou_incumbent"] = inc
    write_ladder(ladder)
    return decide_step(ctx, "S4_fusion", inc_full, decision_pool)
