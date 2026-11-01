"""Heads on frozen embeddings: eva linear probe, GPU logistic regression (parity with sklearn), MLP.

Global context: S0/S2 of the pre-registered ladder compare these heads on Midnight-12k embeddings.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from pcam_hc.heads import fit_head


def _blobs(n=4000, d=32, seed=0, sep=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    w = rng.normal(0, 1, d)
    X = rng.normal(0, 1, (n, d)) + sep * np.outer(2 * y - 1, w) / np.linalg.norm(w)
    return X.astype(np.float32), y


def test_linear_eva_head_learns_separable_data():
    X, y = _blobs(sep=4.0)
    head = fit_head("linear", X[:3000], y[:3000], seed=0, X_val=X[3000:], y_val=y[3000:], max_steps=600)
    assert ((head.logits(X[3000:]) > 0) == y[3000:]).mean() > 0.97


def test_logreg_matches_sklearn():
    X, y = _blobs(sep=1.0)
    C = 0.1
    head = fit_head("logreg", X, y, seed=0, C=C)
    # Reference: sklearn on the same standardised features (the head standardises internally).
    Xs = (X - X.mean(0)) / X.std(0)
    ref = LogisticRegression(C=C, max_iter=5000, tol=1e-10).fit(Xs, y)
    assert np.allclose(head.logits(X), ref.decision_function(Xs), atol=2e-3)


def test_mlp_head_solves_xor():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (4000, 2)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    head = fit_head("mlp", X[:3000], y[:3000], seed=0, max_steps=3000)
    assert ((head.logits(X[3000:]) > 0) == y[3000:]).mean() > 0.9
