"""Binary classification metrics computed from logits at the fixed 0.5 threshold.

Global context: primary metric is balanced accuracy (BA), as in eva's `BinaryBalancedAccuracy`;
the others are secondary (pre-registration section 4). sklearn is the reference implementation.
"""

import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from pcam_hc.config import THRESHOLD


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean of sensitivity (TP/P) and specificity (TN/N)."""
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return 0.5 * ((pred & y).sum() / y.sum() + (~pred & ~y).sum() / (~y).sum())


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width confidence bins on the positive-class probability (Guo et al. 2017,
    https://arxiv.org/abs/1706.04599): sum over bins of |mean(y) - mean(p)| weighted by bin size."""
    bins = np.minimum((np.asarray(p) * n_bins).astype(int), n_bins - 1)  # p=1.0 goes to last bin
    ece = 0.0
    for b in np.unique(bins):
        m = bins == b
        ece += m.mean() * abs(np.mean(y[m]) - np.mean(p[m]))
    return float(ece)


def binary_metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    """All pre-registered metrics for one prediction vector."""
    y = np.asarray(y).astype(int)
    p = expit(np.asarray(logits, dtype=np.float64))  # numerically stable sigmoid
    pred = p >= THRESHOLD
    return {
        "ba": float(balanced_accuracy(y, pred)),
        "auroc": float(roc_auc_score(y, p)),
        "acc": float((pred == y).mean()),
        "f1": float(f1_score(y, pred)),
        "nll": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "ece": expected_calibration_error(y, p),
        "n": int(y.size),
    }
