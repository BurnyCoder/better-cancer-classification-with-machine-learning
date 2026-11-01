"""PCam data layer: download + md5, HDF5 access, slide-grouped folds, subsets, image views, leakage checks.

Global context: PatchCamelyon (https://github.com/basveeling/pcam) ships gzipped HDF5 files
(x: N x 96 x 96 x 3 uint8, y: N x 1 x 1 x 1) and meta CSVs whose `wsi` column names the source slide.
Everything downstream (embeddings, heads, LoRA, statistics) reads images and labels through here,
so the eva-equivalent preprocessing and the slide-level split logic live in one place.
"""

import gzip
import hashlib
import logging
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from pcam_hc.config import PATHS, PCAM_FILES_TEMPLATE, PCAM_MD5, SPLITS, ZENODO_URL

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- download / files
def _md5(path: Path, chunk: int = 1 << 24) -> str:
    """Streaming md5 so multi-GB files never sit in memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def raw_file(split: str, kind: str) -> Path:
    """Path of a downloaded Zenodo file, e.g. kind='x.h5.gz' or 'meta.csv'."""
    return PATHS.data / "raw" / PCAM_FILES_TEMPLATE.format(split=split, kind=kind)


def h5_file(split: str, kind: str) -> Path:
    """Path of the decompressed HDF5 file (kind 'x' or 'y')."""
    return PATHS.data / "h5" / PCAM_FILES_TEMPLATE.format(split=split, kind=f"{kind}.h5")


def download_pcam() -> None:
    """Download every PCam file from Zenodo (skipping md5-valid ones) and gunzip the HDF5 files."""
    (PATHS.data / "raw").mkdir(parents=True, exist_ok=True)
    (PATHS.data / "h5").mkdir(parents=True, exist_ok=True)
    for name, md5 in PCAM_MD5.items():
        path = PATHS.data / "raw" / name
        if not (path.exists() and _md5(path) == md5):
            log.info("downloading %s", name)
            urllib.request.urlretrieve(ZENODO_URL.format(name=name), path)  # noqa: S310 - fixed https URL
            if _md5(path) != md5:
                raise IOError(f"md5 mismatch for {name}")
        log.info("md5 ok: %s", name)
        if name.endswith(".h5.gz"):
            target = PATHS.data / "h5" / name[:-3]
            if not target.exists():
                log.info("decompressing %s", name)
                with gzip.open(path, "rb") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 24)


def _guard(split: str, audit: bool) -> None:
    """Test images and labels stay sealed until the ladder is frozen (seal.py). The only exception is the
    pre-registered data audit (`verify_splits`: hashes and label/meta agreement), which passes audit=True."""
    if split == "test" and not audit:
        from pcam_hc.seal import assert_test_unsealed, frozen_arms

        assert_test_unsealed(frozen_arms())


def load_labels(split: str, audit: bool = False) -> np.ndarray:
    """Binary labels (int8) of a split; stored as y[N, 1, 1, 1] in the HDF5 file."""
    _guard(split, audit)
    with h5py.File(h5_file(split, "y"), "r") as f:
        return f["y"][:].reshape(-1).astype(np.int8)


def load_meta(split: str, audit: bool = False) -> pd.DataFrame:
    """Meta CSV: coord_y, coord_x, tumor_patch, center_tumor_patch, wsi (one row per patch). Holds the labels, so sealed for test."""
    _guard(split, audit)
    return pd.read_csv(raw_file(split, "meta.csv"), index_col=0)


class PCamImages:
    """Random access to a split's images. Uses a numpy memmap when the HDF5 dataset is stored
    contiguously and uncompressed (fast fancy indexing), otherwise falls back to h5py."""

    def __init__(self, split: str, in_memory: bool = False, audit: bool = False):
        _guard(split, audit)
        self.file = h5py.File(h5_file(split, "x"), "r")
        dset = self.file["x"]
        self.shape = dset.shape
        offset = dset.id.get_offset()
        if dset.chunks is None and dset.compression is None and offset is not None:
            self.array = np.memmap(h5_file(split, "x"), dtype=np.uint8, mode="r", offset=offset, shape=dset.shape)
        else:
            self.array = dset
        if in_memory:  # random-access training: one sequential read (~7 GB for train) beats ~1.4 s per random batch from disk
            self.array = np.asarray(self.array[:])

    def __len__(self) -> int:
        return self.shape[0]

    def read(self, idx: np.ndarray) -> np.ndarray:
        """uint8 images (B, 96, 96, 3) for sorted indices."""
        return np.asarray(self.array[np.asarray(idx)])


def images_for(split: str):
    """Image reader for a PCam split or the Camelyon17-WILDS subsample ('wilds')."""
    if split == "wilds":
        from pcam_hc.wilds import WildsImages

        return WildsImages()
    return PCamImages(split)


def prefetch(read_fn, batches):
    """Yield (batch_indices, read_fn(batch_indices)); the next batch is read in a background
    thread while the caller processes the current one (overlaps disk I/O with GPU compute)."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for b in batches:
            fut = pool.submit(read_fn, b)  # start reading batch b now
            if pending is not None:
                yield pending[0], pending[1].result()  # hand out the previous batch
            pending = (b, fut)
        if pending is not None:
            yield pending[0], pending[1].result()


# ----------------------------------------------------------------------------- image views
def d4_view(x: torch.Tensor, k: int) -> torch.Tensor:
    """Element k of the dihedral group D4 on the last two dims: rotate by 90*(k % 4) degrees,
    then mirror horizontally if k >= 4. k = 0 is the identity."""
    out = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(out, dims=(-1,)) if k >= 4 else out


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Central size x size crop of (..., H, W)."""
    top, left = (x.shape[-2] - size) // 2, (x.shape[-1] - size) // 2
    return x[..., top : top + size, left : left + size]


def preprocess(x: torch.Tensor, mean, std, size: int = 224) -> torch.Tensor:
    """eva's ResizeAndCrop on a uint8 batch: v2.Resize(size) -> CenterCrop(size) -> ToDtype(scale)
    -> Normalize (https://github.com/kaiko-ai/eva/blob/main/src/eva/vision/data/transforms/common/resize_and_crop.py).
    Inputs are square, so the centre crop after resizing is the identity and is skipped."""
    x = TF.resize(x, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    x = TF.to_dtype(x, torch.float32, scale=True)
    return TF.normalize(x, mean=list(mean), std=list(std))


# ----------------------------------------------------------------------------- folds / subsets
def make_folds(meta: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per patch. Slides (not patches) are assigned to folds, stratified by whether the
    slide contributes any centre-tumour patch (pre-registration section 3)."""
    slide_tumour = meta.groupby("wsi")["center_tumor_patch"].any()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    slide_fold = pd.Series(0, index=slide_tumour.index, dtype=np.int8)
    for f, (_, test_idx) in enumerate(skf.split(slide_tumour.index, slide_tumour.to_numpy())):
        slide_fold.iloc[test_idx] = f
    return meta["wsi"].map(slide_fold).to_numpy(dtype=np.int8)


def make_subset(folds: np.ndarray, y: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """Sorted indices of a fixed subset stratified by (fold, label) cells ("train-64k")."""
    strata = folds.astype(int) * 2 + np.asarray(y).astype(int)
    idx, _ = train_test_split(np.arange(len(y)), train_size=size, stratify=strata, random_state=seed)
    return np.sort(idx)


# ----------------------------------------------------------------------------- colour-histogram control
def color_histograms(split: str, bins: int = 32, batch: int = 8192) -> np.ndarray:
    """Per-image normalised RGB histograms (3 x `bins` features): the stain/colour-shortcut control.
    Vectorised with one bincount per batch: pixel value // (256/bins) + channel and image offsets."""
    imgs = images_for(split)
    out = np.empty((len(imgs), 3 * bins), dtype=np.float32)
    for start in range(0, len(imgs), batch):
        x = imgs.read(np.arange(start, min(len(imgs), start + batch)))  # (B, 96, 96, 3) uint8
        b = x.shape[0]
        codes = (x.astype(np.int64) // (256 // bins)) + np.arange(3) * bins  # channel offset
        codes = codes.reshape(b, -1) + (np.arange(b) * 3 * bins)[:, None]  # image offset
        counts = np.bincount(codes.ravel(), minlength=b * 3 * bins).reshape(b, 3 * bins)
        out[start : start + b] = counts / (x.shape[1] * x.shape[2])
    return out


# ----------------------------------------------------------------------------- leakage checks
def verify_splits(check_duplicates: bool = True, batch: int = 8192) -> dict:
    """Slide disjointness, label/meta consistency, and exact-duplicate images across splits."""
    report = {}
    slides = {s: set(load_meta(s, audit=True)["wsi"]) for s in SPLITS}  # pre-registered audit: may read test
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            report[f"shared_slides_{a}_{b}"] = len(slides[a] & slides[b])
    for s in SPLITS:
        meta, y = load_meta(s, audit=True), load_labels(s, audit=True)
        report[f"{s}_n"] = int(len(y))
        report[f"{s}_slides"] = len(slides[s])
        report[f"{s}_tumour_slides"] = int(meta.groupby("wsi")["center_tumor_patch"].any().sum())
        report[f"{s}_positive_rate"] = float(y.mean())
        report[f"{s}_label_matches_meta"] = bool((meta["center_tumor_patch"].to_numpy() == y.astype(bool)).all())
    if check_duplicates:
        hashes, metas = {}, {s: load_meta(s, audit=True) for s in SPLITS}
        for s in SPLITS:
            imgs, hs = PCamImages(s, audit=True), []
            for start in range(0, len(imgs), batch):
                block = imgs.read(np.arange(start, min(len(imgs), start + batch)))
                hs += [hashlib.blake2b(im.tobytes(), digest_size=16).digest() for im in block]
            hashes[s] = hs
            report[f"{s}_within_split_duplicates"] = len(hs) - len(set(hs))
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1 :]:
                shared = set(hashes[a]) & set(hashes[b])
                report[f"cross_duplicates_{a}_{b}"] = len(shared)
                # Which slide pairs the shared images come from (e.g. a re-scanned slide).
                ia = [k for k, h in enumerate(hashes[a]) if h in shared]
                ib = [k for k, h in enumerate(hashes[b]) if h in shared]
                report[f"cross_duplicate_slides_{a}_{b}"] = sorted(
                    {*metas[a]["wsi"].iloc[ia].unique().tolist(), *metas[b]["wsi"].iloc[ib].unique().tolist()})
    return report
