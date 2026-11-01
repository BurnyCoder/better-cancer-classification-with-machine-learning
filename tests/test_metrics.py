"""Metric functions must agree with scikit-learn, the reference implementation.

Global context: every number in the reports flows through `pcam_hc.metrics.binary_metrics`,
so it is pinned against sklearn (https://scikit-learn.org/stable/api/sklearn.metrics.html).
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score

from pcam_hc.metrics import binary_metrics, expected_calibration_error


def _toy(n=2000, seed=0):
    # Labels plus noisy logits correlated with the labels, so all metrics are non-trivial.
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    logits = 1.5 * (2 * y - 1) + rng.normal(0, 1.5, n)
    return y, logits


def test_binary_metrics_match_sklearn():
    y, logits = _toy()
    p = 1 / (1 + np.exp(-logits))
    pred = (p >= 0.5).astype(int)
    m = binary_metrics(y, logits)
    assert np.isclose(m["ba"], balanced_accuracy_score(y, pred))
    assert np.isclose(m["auroc"], roc_auc_score(y, p))
    assert np.isclose(m["acc"], (pred == y).mean())
    assert np.isclose(m["f1"], f1_score(y, pred))
    assert np.isclose(m["nll"], log_loss(y, p))


def test_ece_extremes():
    # Constant confidence 1.0 on a 50/50 problem is off by exactly 0.5 in its only bin.
    y = np.array([0, 1] * 500)
    assert np.isclose(expected_calibration_error(y, np.ones(1000)), 0.5)
    # Labels drawn from the predicted probabilities are calibrated, so ECE is small.
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
