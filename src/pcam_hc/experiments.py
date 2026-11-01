"""Ladder experiments on frozen features: CV out-of-fold predictions, valid refits, arm records.

Global context: implements the selection protocol of the pre-registration (sections 3, 6, 7).
  * Every candidate is scored by out-of-fold (OOF) predictions from 5 slide-grouped folds of train.
  * A full-train refit gives valid predictions for the one-shot valid gate.
  * Each arm stores per-seed predictions locally (outside git; licence rules forbid publishing
    per-patch outputs of non-redistributable backbones) and an aggregate JSON record in results/arms/.
Test predictions are produced only after the ladder is frozen (see seal.py).
"""

import json
import logging

import numpy as np
import torch
from scipy.special import expit, logit

from pcam_hc.backbones import REGISTRY
from pcam_hc.config import EVA, N_FOLDS, PATHS, SEEDS
from pcam_hc.data import load_labels, load_meta, make_folds
from pcam_hc.heads import fit_head
from pcam_hc.metrics import binary_metrics
from pcam_hc.stats import paired_comparison, slide_bootstrap_ci

log = logging.getLogger(__name__)
PRED_DIR = PATHS.data / "preds"  # per-patch predictions: local only, never committed
HEAD_DIR = PATHS.data / "heads"


# ----------------------------------------------------------------------------- data context
class Context:
    """Labels, slides and folds shared by all arms (computed once per process)."""

    def __init__(self):
        meta_tr, meta_va = load_meta("train"), load_meta("valid")
        self.y_train, self.y_valid = load_labels("train"), load_labels("valid")
        self.slides_train, self.slides_valid = meta_tr["wsi"].to_numpy(), meta_va["wsi"].to_numpy()
        self.folds = make_folds(meta_tr, N_FOLDS, seed=0)


def seed_average(logits_by_seed: np.ndarray) -> np.ndarray:
    """Pre-registered arm prediction: mean probability over seeds, returned as a logit."""
    p = expit(np.asarray(logits_by_seed, dtype=np.float64)).mean(0)
    return logit(np.clip(p, 1e-7, 1 - 1e-7))


# ----------------------------------------------------------------------------- fitting
def cv_oof(kind: str, X, y, folds, seeds=SEEDS, **kw) -> np.ndarray:
    """OOF logits (n_seeds, n): for each fold, fit on the other folds (fixed steps, no early
    stopping, so no information from the held-out fold) and predict the held-out fold."""
    out = np.zeros((len(seeds), len(y)), dtype=np.float32)
    for si, seed in enumerate(seeds):
        for f in range(folds.max() + 1):
            tr, te = folds != f, folds == f
            head = fit_head(kind, X[tr], y[tr], seed=seed, **kw)
            out[si, te] = head.logits(X[te])
        log.info("cv %s seed %d done", kind, seed)
    return out


def refit(kind: str, X, y, X_eval, seeds=SEEDS, X_val=None, y_val=None, save_as: str | None = None, **kw):
    """Fit on all of train (optionally eva early stopping on valid) and predict X_eval.
    Heads are saved so the frozen ladder can later score the test split without refitting."""
    out = np.zeros((len(seeds), len(X_eval)), dtype=np.float32)
    for si, seed in enumerate(seeds):
        head = fit_head(kind, X, y, seed=seed, X_val=X_val, y_val=y_val, **kw)
        out[si] = head.logits(X_eval)
        if save_as:
            (HEAD_DIR / save_as).mkdir(parents=True, exist_ok=True)
            torch.save({"module": head.module.state_dict(), "mu": head.mu, "sd": head.sd, "kind": kind},
                       HEAD_DIR / save_as / f"seed{seed}.pt")
    return out


# ----------------------------------------------------------------------------- records
def summarise(y, logits_by_seed, slides, n_boot: int = 10_000) -> dict:
    """Per-seed metric spread plus metrics and slide-bootstrap CIs of the seed-averaged prediction."""
    per_seed = [binary_metrics(y, l) for l in logits_by_seed]
    avg = seed_average(logits_by_seed)
    rec = {"seed_mean": {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]},
           "seed_sd": {k: float(np.std([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0
                       for k in per_seed[0]},
           "seed_avg": binary_metrics(y, avg)}
    rec["seed_avg"].update(slide_bootstrap_ci(y, avg, slides, n_boot=n_boot))
    return rec


def save_arm(name: str, record: dict, preds: dict) -> None:
    """Aggregate record -> results/arms/<name>.json (committed); predictions -> local npz."""
    (PATHS.results / "arms").mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (PATHS.results / "arms" / f"{name}.json").write_text(json.dumps(record, indent=2, default=float))
    np.savez_compressed(PRED_DIR / f"{name}.npz", **preds)
    log.info("arm %s: %s", name, json.dumps({k: record[k]["seed_avg"]["ba"] for k in ("oof", "valid") if k in record}))


def load_preds(name: str) -> dict:
    return dict(np.load(PRED_DIR / f"{name}.npz"))


def lineage_publishable(backbones) -> bool:
    """Licence guard: per-patch outputs may be exported only if every backbone allows derivatives."""
    return all(REGISTRY[b].publishable for b in backbones)


def run_arm(ctx: Context, name: str, X_train, X_valid, kind: str = "linear", backbones=("hibou_l",),
            description: str = "", y_train=None, recipe: dict | None = None, **kw) -> dict:
    """Standard CV-protocol arm: OOF on train + full-train refit scored on valid. `recipe` records how
    the features were built (backbone, representation, views) so the saved heads can score new splits."""
    y_fit = ctx.y_train if y_train is None else y_train
    oof = cv_oof(kind, X_train, y_fit, ctx.folds, **kw)
    val = refit(kind, X_train, y_fit, X_valid, save_as=name, **kw)
    record = {"name": name, "description": description, "head": kind, "backbones": list(backbones),
              "publishable": lineage_publishable(backbones), "dim": int(X_train.shape[1]),
              "oof": summarise(ctx.y_train, oof, ctx.slides_train),
              "valid": summarise(ctx.y_valid, val, ctx.slides_valid), "recipe": recipe or {}}
    save_arm(name, record, {"oof": oof, "valid": val})
    return record


def compare_arms(ctx: Context, incumbent: str, challenger: str, n_boot: int = 10_000) -> dict:
    """Paired OOF and valid comparisons of two arms' seed-averaged predictions (challenger - incumbent)."""
    a, b = load_preds(incumbent), load_preds(challenger)
    return {
        "incumbent": incumbent, "challenger": challenger,
        "oof": paired_comparison(ctx.y_train, seed_average(a["oof"]), seed_average(b["oof"]), ctx.slides_train,
                                 n_boot=n_boot, n_flips=n_boot),
        "valid": paired_comparison(ctx.y_valid, seed_average(a["valid"]), seed_average(b["valid"]),
                                   ctx.slides_valid, n_boot=n_boot, n_flips=n_boot),
    }


def eva_exact(ctx: Context, name: str, X_train, X_valid, backbones=("hibou_l",)) -> dict:
    """S0 / H2: eva protocol (linear head, early stopping on valid BA, 5 runs)."""
    val = refit("linear", X_train, ctx.y_train, X_valid, X_val=X_valid, y_val=ctx.y_valid, save_as=name)
    record = {"name": name, "description": "eva-exact linear probe, early stopping on valid", "head": "linear",
              "backbones": list(backbones), "publishable": lineage_publishable(backbones),
              "dim": int(X_train.shape[1]), "valid": summarise(ctx.y_valid, val, ctx.slides_valid),
              "eva_protocol": {"lr": EVA.lr, "weight_decay": EVA.weight_decay, "batch_size": EVA.batch_size,
                               "max_steps": EVA.max_steps, "patience": EVA.patience}}
    save_arm(name, record, {"valid": val})
    return record
