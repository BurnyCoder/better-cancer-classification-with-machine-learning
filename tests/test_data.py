"""Data layer: D4 group views, eva-equivalent preprocessing, slide-grouped folds, subsets, leakage.

Global context: PCam (https://github.com/basveeling/pcam) is slide-disjoint across splits; the
pre-registration requires CV folds grouped by slide and automated leakage checks on the real files.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms import v2

from pcam_hc import config
from pcam_hc.data import (
    center_crop,
    d4_view,
    make_folds,
    make_subset,
    preprocess,
)


def test_d4_views_form_the_dihedral_group():
    x = torch.randint(0, 256, (2, 3, 96, 96), dtype=torch.uint8)
    views = [d4_view(x, k) for k in range(8)]
    assert torch.equal(views[0], x)  # view 0 is the identity
    flat = {v.numpy().tobytes() for v in views}
    assert len(flat) == 8  # all eight dihedral images are distinct for a generic image
    for v in views:  # closure: every composition of two views is again one of the eight
        for k in range(8):
            assert d4_view(v, k).numpy().tobytes() in flat


def test_center_crop_takes_the_middle():
    x = torch.arange(96 * 96).reshape(1, 1, 96, 96)
    c = center_crop(x, 64)
    assert c.shape == (1, 1, 64, 64) and c[0, 0, 0, 0] == x[0, 0, 16, 16]


def test_preprocess_matches_eva_resize_and_crop():
    """eva's ResizeAndCrop = v2.Resize -> v2.CenterCrop -> ToDtype(scale) -> Normalize."""
    x = torch.randint(0, 256, (4, 3, 96, 96), dtype=torch.uint8)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    ref = v2.Compose([v2.Resize(224), v2.CenterCrop(224), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])
    expected = torch.stack([ref(img) for img in x])
    assert torch.allclose(preprocess(x, mean, std), expected, atol=1e-6)


def _fake_meta(n_slides=40, per_slide=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_slides):
        tumour = s < n_slides // 2
        for _ in range(per_slide):
            rows.append({"wsi": f"slide_{s}", "center_tumor_patch": bool(tumour and rng.random() < 0.6)})
    return pd.DataFrame(rows)


def test_folds_are_grouped_by_slide_and_stratified():
    meta = _fake_meta()
    folds = make_folds(meta, n_folds=5, seed=0)
    assert (meta.assign(f=folds).groupby("wsi")["f"].nunique() == 1).all()  # no slide in two folds
    slide_tumour = meta.groupby("wsi")["center_tumor_patch"].any()
    slide_fold = meta.assign(f=folds).groupby("wsi")["f"].first()
    for f in range(5):  # every fold receives tumour and normal slides
        assert slide_tumour[slide_fold == f].nunique() == 2
    assert np.array_equal(folds, make_folds(meta, n_folds=5, seed=0))  # deterministic


def test_subset_is_stratified_and_deterministic():
    meta = _fake_meta(n_slides=60, per_slide=100)
    y = meta["center_tumor_patch"].to_numpy().astype(int)
    folds = make_folds(meta, n_folds=5, seed=0)
    idx = make_subset(folds, y, size=1000, seed=0)
    assert len(idx) == 1000 and np.all(np.diff(idx) > 0)
    assert abs(y[idx].mean() - y.mean()) < 0.01
    assert np.array_equal(idx, make_subset(folds, y, size=1000, seed=0))


@pytest.mark.data
def test_real_splits_are_slide_disjoint():
    from pcam_hc.data import load_meta

    if not (config.PATHS.data / "raw" / config.PCAM_FILES_TEMPLATE.format(split="test", kind="meta.csv")).exists():
        pytest.skip("PCam not downloaded")
    slides = {s: set(load_meta(s)["wsi"]) for s in config.SPLITS}
    assert not slides["train"] & slides["valid"]
    assert not slides["train"] & slides["test"]
    assert not slides["valid"] & slides["test"]
