"""Stage D: final combination (S6), ladder freeze, and one-shot scoring of every arm on the sealed
PCam test split and the Camelyon17-WILDS subsample.

Global context: pre-registration sections 3, 5 and 7.
  * S6 combines the adopted components: the Hibou-L component (frozen head or LoRA, with TTA if S3
    adopted it) and the S4 fusion if adopted. Candidates are equal-weight logit averaging and
    logistic-regression stacking, compared by slide-grouped inner CV on train fold 0.
  * `freeze` hashes results/ladder.json into frozen_ladder.json, commits it and tags `ladder-frozen`.
    Only then do the test split and WILDS become readable (seal.py).
  * `score` rebuilds every arm's predictions on a new split from its saved heads, LoRA states or
    member arms, then runs the confirmatory tests (H1, H1b, H4, H6; Holm), the secondary metrics and
    the pre-registered sensitivity analysis without the duplicated slide camelyon16_test_049.
"""

import json
import logging
import subprocess

import numpy as np
import torch
from torch import nn

from pcam_hc import experiments as ex
from pcam_hc.config import PATHS, PROJECT_ROOT
from pcam_hc.data import load_labels, load_meta
from pcam_hc.features import CLS, load
from pcam_hc.heads import Head
from pcam_hc.ladder import arm_record, read_ladder
from pcam_hc.metrics import binary_metrics
from pcam_hc.seal import FROZEN_FILE, TAG, assert_test_unsealed, config_hash, frozen_arms
from pcam_hc.stage_b import _rep_from_json, features
from pcam_hc.stats import holm, paired_comparison, slide_bootstrap_ci

log = logging.getLogger(__name__)
DUPLICATE_SLIDE = "camelyon16_test_049"  # pixel-identical to camelyon16_train_tumor_036 (deviation log)


# ----------------------------------------------------------------------------- heads on new splits
def load_heads(arm: str) -> list:
    """Saved heads of an arm (one per seed) as `Head` objects on the training device."""
    from pcam_hc.heads import DEVICE

    heads = []
    for path in sorted((ex.HEAD_DIR / arm).glob("seed*.pt")):
        st = torch.load(path, weights_only=False)
        sd = st["module"]
        if st["kind"] == "mlp":
            d, h = sd["0.weight"].shape[1], sd["0.weight"].shape[0]
            module = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))
        else:
            module = nn.Linear(sd["weight"].shape[1], 1)
        module.load_state_dict(sd)
        heads.append(Head(module.to(DEVICE), st["mu"], st["sd"]))
    return heads


def recipe_features(recipe: dict, split: str, d4: int = 0) -> np.ndarray:
    """Rebuild an arm's input features on `split` from its recorded recipe."""
    rep, views = _rep_from_json(recipe["representation"])
    if recipe["backbone"] == "hibou_l":
        X = features(split, rep, views, d4=d4)
    else:
        X = load(recipe["backbone"], split, rep)
    if "concat_cls" in recipe:
        X = np.hstack([X, load(recipe["concat_cls"], split, CLS)])
    return X


# Stage-A arms were created before recipes were recorded; their inputs are fixed by definition.
LEGACY_RECIPES = {
    "S0_eva_exact_hibou_l": {"backbone": "hibou_l", "representation": {"parts": [[-1, "cls"]], "center": False, "views": ["full"]}},
    "S0_hibou_l_cls_linear": {"backbone": "hibou_l", "representation": {"parts": [[-1, "cls"]], "center": False, "views": ["full"]}},
    "control_shuffled_labels": {"backbone": "hibou_l", "representation": {"parts": [[-1, "cls"]], "center": False, "views": ["full"]}},
}


def head_arm_logits(arm: str, split: str) -> np.ndarray:
    """(n_seeds, n) logits of a head-based arm on a new split (TTA arms average the 8 D4 views)."""
    rec = arm_record(arm)
    recipe = rec.get("recipe") or LEGACY_RECIPES[arm]
    heads = load_heads(recipe.get("heads_from", arm))
    views = range(8) if recipe.get("tta") else (0,)
    out = 0
    for k in views:
        X = recipe_features(recipe, split, d4=k)
        out = out + np.stack([h.logits(X) for h in heads])
    return (out / len(views)).astype(np.float32)


# ----------------------------------------------------------------------------- freeze / unseal
def freeze(push: bool = True) -> str:
    """Hash results/ladder.json, commit it with the hash file, tag `ladder-frozen` (and push)."""
    ladder = read_ladder()
    digest = config_hash(ladder)
    (PROJECT_ROOT / FROZEN_FILE).write_text(digest)
    run = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, check=True)  # noqa: E731
    run("add", str(PATHS.results / "ladder.json"), FROZEN_FILE)
    run("commit", "-m", f"Freeze the ladder before unsealing test (config hash {digest[:12]})")
    run("tag", "-a", TAG, "-m", "Ladder frozen; test split and WILDS may now be scored once")
    if push:
        run("push")
        run("push", "origin", TAG)
    assert_test_unsealed(frozen_arms())
    return digest


def labels_slides(split: str) -> tuple:
    if split == "wilds":
        from pcam_hc.wilds import WildsImages

        meta = WildsImages().meta
        return meta["label"].to_numpy(), meta["slide"].astype(str).to_numpy(), meta["center"].to_numpy()
    meta = load_meta(split)
    return load_labels(split), meta["wsi"].to_numpy(), None


def evaluate(split: str, preds: dict, pairs: dict, mask=None) -> dict:
    """Metrics with slide-bootstrap CIs for every arm, and paired comparisons for named contrasts."""
    y, slides, _ = labels_slides(split)
    m = np.ones(len(y), bool) if mask is None else mask
    out = {"arms": {}, "contrasts": {}}
    for name, logits in preds.items():
        avg = ex.seed_average(logits)[m]
        rec = binary_metrics(y[m], avg)
        rec.update(slide_bootstrap_ci(y[m], avg, slides[m]))
        rec["seed_ba"] = [float(binary_metrics(y[m], l[m])["ba"]) for l in logits]
        out["arms"][name] = rec
    for label, (a, b) in pairs.items():
        if a in preds and b in preds:
            out["contrasts"][label] = paired_comparison(y[m], ex.seed_average(preds[a])[m],
                                                        ex.seed_average(preds[b])[m], slides[m])
    return out


def confirmatory(test: dict, wilds: dict) -> dict:
    """H1, H1b, H4 on test; H6 on the WILDS primary pool; Holm across the four sign-flip p-values."""
    rows = {"H1": test["contrasts"].get("H1"), "H1b": test["contrasts"].get("H1b"),
            "H4": test["contrasts"].get("H4"), "H6": wilds["contrasts"].get("H6")}
    present = {k: v for k, v in rows.items() if v is not None}
    adj = dict(zip(present, holm([v["p_signflip"] for v in present.values()])))
    return {k: {**v, "p_holm": float(adj[k]), "supported": bool(v["delta_ba"] > 0 and adj[k] < 0.05 and v["ci_ba"][0] > 0)}
            for k, v in present.items()}


def write(name: str, obj: dict) -> None:
    (PATHS.results / "final").mkdir(parents=True, exist_ok=True)
    (PATHS.results / "final" / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))


def sensitivity_mask() -> np.ndarray:
    return load_meta("test")["wsi"].to_numpy() != DUPLICATE_SLIDE
