"""Paired comparison of every ladder candidate with the original Hibou-L probe (S0).

Global context: each trained classifier is released as soon as it is evaluated, together with its
difference from the original model (the baseline in the author's sense). The comparison is paired on
the same patches: full-train out-of-fold predictions for CV arms, the train-64k subset for the S3 arms
(against the identity-view arm scored by the same heads), and fold 0 for LoRA arms (against S0's
fold-0 OOF predictions, whose heads were also trained on folds 1-4). Validation is always paired on
the full validation split. Results are cached in results/comparisons/ (aggregates only).
"""

import json
import logging

import numpy as np

from pcam_hc import experiments as ex
from pcam_hc.config import PATHS
from pcam_hc.stats import paired_comparison

log = logging.getLogger(__name__)
BASELINE = "S0_hibou_l_cls_linear"
H4 = "H4_aug_control"
LADDER_PREFIXES = ("S1_", "S2_", "S3_", "S3b_", "S5_", "S5b_")


def ladder_arms() -> list[str]:
    return sorted(p.stem for p in (PATHS.results / "arms").glob("*.json") if p.stem.startswith(LADDER_PREFIXES))


def compare_with_baseline(arm: str, n_boot: int = 10_000, ref: str = BASELINE) -> dict:
    """Paired (arm - ref) comparison, ref = S0 by default; cached on disk. A fold-0 `ref` other than S0
    (the H4 control) must hold predictions for the same fold-0 patches as `arm`."""
    out_dir = PATHS.results / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"{arm}_vs_baseline.json" if ref == BASELINE else f"{arm}_vs_{ref}.json")
    if path.exists():
        return json.loads(path.read_text())
    ctx = ex.Context()
    rec = json.loads((PATHS.results / "arms" / f"{arm}.json").read_text())
    a, b = ex.load_preds(ref), ex.load_preds(arm)
    base_oof = ex.seed_average(a["oof"])
    subset = rec.get("oof_subset")
    if subset is None:  # full-train OOF
        idx, ref_name, ref_oof = np.arange(len(ctx.y_train)), BASELINE, base_oof
    elif subset == "train64k":  # S3: same heads, identity view vs D4 TTA
        from pcam_hc.config import SUBSET_SIZE
        from pcam_hc.data import make_subset

        idx = make_subset(ctx.folds, ctx.y_train, SUBSET_SIZE, seed=0)
        ref_name = "S3_no_tta_train64k" if arm != "S3_no_tta_train64k" else BASELINE
        ref_oof = ex.seed_average(ex.load_preds(ref_name)["oof"]) if ref_name != BASELINE else base_oof[idx]
    else:  # fold0 / fold0_minus_ckpt (LoRA arms and the H4 control)
        idx = b.get("fold0_idx", b.get("select_idx"))
        if ref == BASELINE:
            ref_oof = base_oof[idx]
        else:  # both arms scored the same fold-0 patches
            assert np.array_equal(a["fold0_idx"], idx), f"{ref} and {arm} cover different patches"
            ref_oof = base_oof
        ref_name = ref
    cmp = {"arm": arm, "reference_oof": ref_name, "oof_patches": int(len(idx)),
           "oof": paired_comparison(ctx.y_train[idx], ref_oof, ex.seed_average(b["oof"]), ctx.slides_train[idx],
                                    n_boot=n_boot, n_flips=n_boot),
           "valid": paired_comparison(ctx.y_valid, ex.seed_average(a["valid"]), ex.seed_average(b["valid"]),
                                      ctx.slides_valid, n_boot=n_boot, n_flips=n_boot)}
    path.write_text(json.dumps(cmp, indent=2, default=float))
    log.info("%s vs baseline: dOOF %.4f (p %.3g), dValid %.4f", arm, cmp["oof"]["delta_ba"], cmp["oof"]["p_signflip"],
             cmp["valid"]["delta_ba"])
    return cmp


def all_comparisons() -> dict:
    out = {arm: compare_with_baseline(arm) for arm in ladder_arms()}
    if (PATHS.results / "arms" / f"{H4}.json").exists():  # H4: frozen head with LoRA's images and augmentation
        out[H4] = compare_with_baseline(H4)
        out[f"S5_lora_vs_{H4}"] = compare_with_baseline("S5_lora", ref=H4)
    return out
