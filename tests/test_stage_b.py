"""Stage-B helpers: Caruana greedy ensemble selection and cache view naming.

Global context: S4 fusion uses greedy forward selection with replacement (Caruana et al. 2004);
S3 TTA reads rotated copies of every base view from the embedding cache.
"""

import numpy as np

from pcam_hc.embed import apply_view
from pcam_hc.stage_b import _apply_weights, _greedy_weights, view_name


def test_greedy_adds_a_complementary_member_and_ignores_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    signal = 2 * y - 1
    a = signal + rng.normal(0, 1.5, y.size)  # incumbent
    b = signal + rng.normal(0, 1.5, y.size)  # independent errors: averaging helps
    noise = rng.normal(0, 1, y.size)  # useless member
    w = _greedy_weights({"a": a, "b": b, "noise": noise}, y, start="a")
    assert "b" in w and "noise" not in w
    assert np.allclose(_apply_weights({"a": 1, "b": 1}, {"a": a, "b": b}), (a + b) / 2)


def test_view_names_compose_and_apply():
    import torch

    assert view_name("full", 0) == "full" and view_name("full", 3) == "d4_3"
    assert view_name("center64", 5) == "center64+d4_5"
    x = torch.randint(0, 256, (1, 3, 96, 96), dtype=torch.uint8)
    assert apply_view(x, "center64+d4_5").shape == (1, 3, 64, 64)
