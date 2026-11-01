"""Slide-clustered statistics: exactness of the per-slide decomposition, bootstrap coverage, tests.

Global context: the pre-registration (reports/preregistration.md, section 7) makes the slide the
unit of resampling. These tests check the decomposition is exact and the intervals are calibrated
on synthetic clustered data where the truth is known.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from pcam_hc.stats import (
    bca_interval,
    holm,
    paired_comparison,
    per_slide_ba_contributions,
    slide_bootstrap_ci,
    stratified_slide_resample,
)


def _clustered(n_slides=60, per_slide=200, effect=0.0, seed=0):
    """Synthetic PCam-like data: tumour slides contain positives, slide-level random effects."""
    rng = np.random.default_rng(seed)
    slides, ys, la, lb = [], [], [], []
    for s in range(n_slides):
        tumour = s % 2 == 0
        y = rng.integers(0, 2, per_slide) if tumour else np.zeros(per_slide, int)
        slide_shift = rng.normal(0, 1.0)  # shared within slide -> within-slide correlation
        base = 2.0 * (2 * y - 1) + slide_shift + rng.normal(0, 1.5, per_slide)
        slides += [f"s{s}"] * per_slide
        ys.append(y)
        la.append(base)
        lb.append(base + effect * (2 * y - 1))  # arm B is better by `effect` logits
    return np.array(slides), np.concatenate(ys), np.concatenate(la), np.concatenate(lb)


def test_contributions_sum_to_delta_ba():
    slides, y, la, lb = _clustered(effect=0.5)
    c = per_slide_ba_contributions(y, la >= 0, lb >= 0, slides)
    delta = balanced_accuracy_score(y, lb >= 0) - balanced_accuracy_score(y, la >= 0)
    assert np.isclose(c.sum(), delta)


def test_stratified_resample_keeps_both_classes():
    slides, y, _, _ = _clustered()
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = stratified_slide_resample(slides, y, rng)
        assert set(np.unique(y[idx])) == {0, 1}


def test_identical_arms_give_null_result():
    slides, y, la, _ = _clustered()
    r = paired_comparison(y, la, la.copy(), slides, n_boot=500, n_flips=500, seed=0)
    assert r["delta_ba"] == 0 and r["p_signflip"] == 1.0


def test_strong_effect_is_detected():
    slides, y, la, lb = _clustered(effect=1.5)
    r = paired_comparison(y, la, lb, slides, n_boot=1000, n_flips=1000, seed=0)
    assert r["delta_ba"] > 0 and r["ci_ba"][0] > 0 and r["p_signflip"] < 0.01


def test_bootstrap_coverage_is_near_nominal():
    """95% CI for BA should cover the population BA in roughly 95% of synthetic datasets."""
    # Population BA estimated once from a very large draw of the same generative process.
    s_big, y_big, l_big, _ = _clustered(n_slides=4000, per_slide=50, seed=999)
    truth = balanced_accuracy_score(y_big, l_big >= 0)
    hits = 0
    n_sims = 60
    for sim in range(n_sims):
        slides, y, la, _ = _clustered(n_slides=60, per_slide=50, seed=sim)
        lo, hi = slide_bootstrap_ci(y, la, slides, n_boot=400, seed=sim)["ci_ba"]
        hits += lo <= truth <= hi
    assert 0.83 <= hits / n_sims <= 1.0


def test_bca_reduces_to_percentile_without_bias_or_skew():
    rng = np.random.default_rng(0)
    boot = rng.normal(0, 1, 20000)
    jack = rng.normal(0, 1, 200)  # symmetric jackknife -> acceleration ~0
    lo, hi = bca_interval(0.0, boot, jack)
    assert np.isclose(lo, -1.96, atol=0.1) and np.isclose(hi, 1.96, atol=0.1)


def test_holm_matches_textbook_example():
    # Holm (1979): sorted p * (m - rank), cumulative max, capped at 1.
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
