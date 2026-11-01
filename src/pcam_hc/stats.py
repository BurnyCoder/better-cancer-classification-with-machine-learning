"""Slide-clustered inference for patch-level metrics (pre-registration section 7).

Global context: PCam patches from one whole-slide image (WSI) are correlated, so the slide is the
unit of resampling. Provides
  * a stratified slide-cluster bootstrap (tumour and normal slides resampled separately, so each
    replicate has both classes) with BCa intervals (Efron 1987, https://doi.org/10.1080/01621459.1987.10478410)
    using a leave-one-slide-out jackknife for the acceleration,
  * a paired sign-flip randomisation test on exact per-slide contributions to delta-BA,
  * Holm's step-down correction (Holm 1979, https://www.jstor.org/stable/4615733).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from pcam_hc.config import THRESHOLD

AUROC_BOOT = 2000  # replicates for AUROC intervals (BA intervals use the full n_boot, default 10,000)


@dataclass
class SlideIndex:
    """Slide bookkeeping shared by all resampling routines."""

    codes: np.ndarray  # slide code per patch, 0..n_slides-1
    tumour: np.ndarray  # codes of slides with at least one positive patch
    normal: np.ndarray  # codes of slides without positives
    members: list  # patch indices per slide code

    @classmethod
    def build(cls, slides: np.ndarray, y: np.ndarray) -> "SlideIndex":
        _, codes = np.unique(np.asarray(slides), return_inverse=True)
        has_pos = np.bincount(codes, weights=np.asarray(y)) > 0
        order = np.argsort(codes, kind="stable")
        members = np.split(order, np.cumsum(np.bincount(codes))[:-1])
        return cls(codes, np.flatnonzero(has_pos), np.flatnonzero(~has_pos), members)

    @property
    def n(self) -> int:
        return len(self.members)

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """Multiplicity of each slide in one stratified bootstrap replicate."""
        picks = np.concatenate([rng.choice(g, g.size, replace=True) for g in (self.tumour, self.normal) if g.size])
        return np.bincount(picks, minlength=self.n)


def stratified_slide_resample(slides: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Patch indices of one stratified slide-cluster bootstrap replicate."""
    idx = SlideIndex.build(slides, y)
    w = idx.draw(rng)
    return np.concatenate([np.repeat(idx.members[s], w[s]) for s in np.flatnonzero(w)])


def _slide_counts(y: np.ndarray, pred: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    """Per-slide [TP, P, TN, N] matrix; BA of any slide multiset follows from weighted sums."""
    y, pred = y.astype(bool), pred.astype(bool)
    cols = [pred & y, y, ~pred & ~y, ~y]
    return np.stack([np.bincount(codes, weights=c, minlength=n) for c in cols], axis=1)


def _ba_from_counts(c: np.ndarray) -> np.ndarray:
    """BA = (TP/P + TN/N) / 2 from summed counts; works on (..., 4) arrays."""
    return 0.5 * (c[..., 0] / c[..., 1] + c[..., 2] / c[..., 3])


def bca_interval(theta: float, boot: np.ndarray, jack: np.ndarray, alpha: float = 0.05) -> tuple:
    """Bias-corrected and accelerated percentile interval (Efron 1987; Efron & Tibshirani 1993, ch. 14)."""
    boot = np.asarray(boot)
    # Bias correction z0 from the share of replicates below the estimate (ties split evenly).
    frac = (boot < theta).mean() + 0.5 * (boot == theta).mean()
    z0 = norm.ppf(np.clip(frac, 1e-6, 1 - 1e-6))
    # Acceleration from the skewness of the leave-one-cluster-out jackknife values.
    d = np.mean(jack) - np.asarray(jack)
    denom = 6.0 * (d**2).sum() ** 1.5
    a = (d**3).sum() / denom if denom > 0 else 0.0
    qs = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = norm.ppf(q)
        qs.append(norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq))))
    return tuple(float(v) for v in np.quantile(boot, qs))


def _auroc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < y.size else np.nan


def slide_bootstrap_ci(y, logits, slides, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Point estimates and BCa 95% CIs for BA and AUROC of one arm."""
    y, logits = np.asarray(y).astype(int), np.asarray(logits, dtype=np.float64)
    idx, rng = SlideIndex.build(slides, y), np.random.default_rng(seed)
    pred = expit(logits) >= THRESHOLD
    counts = _slide_counts(y, pred, idx.codes, idx.n)
    weights = np.stack([idx.draw(rng) for _ in range(n_boot)])  # (n_boot, n_slides)
    boot_ba = _ba_from_counts(weights @ counts)
    jack_ba = _ba_from_counts(counts.sum(0) - counts)  # leave one slide out
    ba = float(_ba_from_counts(counts.sum(0)))
    # AUROC needs patch-level re-ranking per replicate, so it uses the first 2,000 replicates (BA uses all).
    boot_auc = np.array([_auroc(y[i], logits[i]) for i in (_patches(idx, w) for w in weights[:AUROC_BOOT])])
    jack_auc = np.array([_auroc(np.delete(y, m), np.delete(logits, m)) for m in idx.members])
    auc = float(_auroc(y, logits))
    return {"ba": ba, "ci_ba": bca_interval(ba, boot_ba, jack_ba),
            "auroc": auc, "ci_auroc": bca_interval(auc, boot_auc[~np.isnan(boot_auc)], jack_auc)}


def _patches(idx: SlideIndex, w: np.ndarray) -> np.ndarray:
    return np.concatenate([np.repeat(idx.members[s], w[s]) for s in np.flatnonzero(w)])


def per_slide_ba_contributions(y, pred_a, pred_b, slides) -> np.ndarray:
    """Exact decomposition: delta-BA (b - a) = sum_s [dTP_s / 2P + dTN_s / 2N]."""
    y = np.asarray(y).astype(bool)
    a, b = np.asarray(pred_a).astype(bool), np.asarray(pred_b).astype(bool)
    _, codes = np.unique(np.asarray(slides), return_inverse=True)
    d_tp = np.bincount(codes, weights=(b & y).astype(float) - (a & y))
    d_tn = np.bincount(codes, weights=(~b & ~y).astype(float) - (~a & ~y))
    return d_tp / (2 * y.sum()) + d_tn / (2 * (~y).sum())


def sign_flip_pvalue(contrib: np.ndarray, n_flips: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired randomisation test: under H0 each slide's contribution is sign-symmetric."""
    rng = np.random.default_rng(seed)
    obs = abs(contrib.sum())
    null = np.abs((rng.choice([-1.0, 1.0], (n_flips, contrib.size)) * contrib).sum(1))
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + n_flips))


def paired_comparison(y, logits_a, logits_b, slides, n_boot: int = 10_000, n_flips: int = 10_000, seed: int = 0) -> dict:
    """Delta (b - a) in BA and AUROC with paired stratified slide-bootstrap BCa CIs and sign-flip p."""
    y = np.asarray(y).astype(int)
    la, lb = np.asarray(logits_a, np.float64), np.asarray(logits_b, np.float64)
    idx, rng = SlideIndex.build(slides, y), np.random.default_rng(seed)
    pa, pb = expit(la) >= THRESHOLD, expit(lb) >= THRESHOLD
    ca, cb = _slide_counts(y, pa, idx.codes, idx.n), _slide_counts(y, pb, idx.codes, idx.n)
    weights = np.stack([idx.draw(rng) for _ in range(n_boot)])
    boot = _ba_from_counts(weights @ cb) - _ba_from_counts(weights @ ca)  # same slides for both arms
    jack = _ba_from_counts(cb.sum(0) - cb) - _ba_from_counts(ca.sum(0) - ca)
    delta = float(_ba_from_counts(cb.sum(0)) - _ba_from_counts(ca.sum(0)))
    boot_auc = []
    for w in weights[:AUROC_BOOT]:  # AUROC bootstrap is costlier; 2,000 replicates suffice for a 95% CI
        i = _patches(idx, w)
        boot_auc.append(_auroc(y[i], lb[i]) - _auroc(y[i], la[i]))
    jack_auc = [_auroc(np.delete(y, m), np.delete(lb, m)) - _auroc(np.delete(y, m), np.delete(la, m)) for m in idx.members]
    d_auc = float(_auroc(y, lb) - _auroc(y, la))
    contrib = per_slide_ba_contributions(y, pa, pb, slides)
    return {"delta_ba": delta, "ci_ba": bca_interval(delta, boot, jack),
            "delta_auroc": d_auc, "ci_auroc": bca_interval(d_auc, np.array(boot_auc), np.array(jack_auc)),
            "p_signflip": sign_flip_pvalue(contrib, n_flips, seed), "n_slides": idx.n}


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values: sorted p_(i) * (m - i), cumulative max, capped at 1."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate(p[order] * (p.size - np.arange(p.size)))
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out
