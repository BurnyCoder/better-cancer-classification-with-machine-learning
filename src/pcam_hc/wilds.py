"""Camelyon17-WILDS out-of-distribution evaluation set (pre-registration section 3, H6).

Global context: WILDS Camelyon17 (Koh et al. 2021, https://arxiv.org/abs/2012.07421) has 96x96
patches from 50 slides of 5 hospitals, labelled with the same rule as PCam (any tumour pixel in
the centre 32x32) at the same ~0.97 um/px resolution. The official CodaLab download had an expired
TLS certificate on 2026-11-01, so the data come from the Hugging Face parquet mirror
`wltjr1007/Camelyon17-WILDS` (pinned revision), and its per-centre counts are checked against the
WILDS paper. Nothing is fit on these patches. Like the PCam test split, they are embedded and
scored only after the ladder is frozen.

Centre IDs: 0 = CWZ, 1 = RST, 4 = LPON (arXiv:2404.11161); 2 and 3 are the Camelyon16 source
hospitals (RUMC, UMCU). The primary OOD pool is the hospitals Camelyon16 did not use. That mapping
is cross-checked empirically by `hospital_colour_check` against PCam slides of known hospital.
"""

import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pcam_hc.config import PATHS

log = logging.getLogger(__name__)
WILDS_DIR = Path(str(PATHS.data).replace("pcam", "wilds"))  # D:/AI/wilds
MIRROR = WILDS_DIR / "hf_mirror"
# Split sizes reported in the WILDS paper, Table 9 (train / ID val / OOD val / test).
EXPECTED = {"train": 302_436, "id_val": 33_560, "ood_val": 34_904, "test": 85_054}
CENTER_HOSPITAL = {0: "CWZ", 1: "RST", 2: "RUMC or UMCU", 3: "RUMC or UMCU", 4: "LPON"}
PRIMARY_OOD_CENTERS = (0, 1, 4)  # hospitals that did not contribute to Camelyon16 / PCam
META_COLS = ["label", "center", "image_id", "patient", "node", "x_coord", "y_coord", "slide"]


def parquet_files() -> list:
    return sorted(MIRROR.rglob("*.parquet"))


def load_metadata() -> pd.DataFrame:
    """All rows' metadata (no pixels), with the mirror split name and the parquet file/row of each."""
    import pyarrow.parquet as pq

    frames = []
    for f in parquet_files():
        t = pq.read_table(f, columns=META_COLS).to_pandas()
        t["mirror_split"] = f.name.split("-")[0]
        t["file"], t["row"] = str(f), np.arange(len(t))
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def check_counts(meta: pd.DataFrame) -> dict:
    """Mirror split sizes vs. the WILDS paper; per-centre and per-slide counts."""
    sizes = meta.groupby("mirror_split").size().to_dict()
    return {"mirror_split_sizes": sizes,
            "per_center": meta.groupby("center").size().to_dict(),
            "slides_per_center": meta.groupby("center")["slide"].nunique().to_dict(),
            "n_slides": int(meta["slide"].nunique()),
            "test_matches_paper": sizes.get("test") == EXPECTED["test"],
            "train_matches_paper": sizes.get("train") == EXPECTED["train"],
            "val_matches_paper": sizes.get("validation") == EXPECTED["id_val"] + EXPECTED["ood_val"]}


def subsample(meta: pd.DataFrame, per_slide: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Fixed subsample: up to `per_slide` random patches from every slide (pre-registered, seed 0)."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in meta.groupby("slide", sort=True):
        take = g if len(g) <= per_slide else g.iloc[np.sort(rng.choice(len(g), per_slide, replace=False))]
        parts.append(take)
    return pd.concat(parts, ignore_index=True)


def extract_images(sub: pd.DataFrame, out: Path = WILDS_DIR / "subsample") -> Path:
    """Decode the subsample's PNGs into one uint8 array (N, 96, 96, 3) + metadata CSV."""
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out / "x.npy", mode="w+", dtype=np.uint8, shape=(len(sub), 96, 96, 3))
    pos = 0
    order = sub.sort_values(["file", "row"]).index.to_numpy()
    sub = sub.loc[order].reset_index(drop=True)
    for f, g in sub.groupby("file", sort=False):
        images = pq.read_table(f, columns=["image"]).column("image").to_pylist()
        for row in g["row"].to_numpy():
            arr[pos] = np.asarray(Image.open(io.BytesIO(images[row]["bytes"])).convert("RGB"))
            pos += 1
        log.info("decoded %s (%d patches so far)", Path(f).name, pos)
    arr.flush()
    sub.drop(columns=["file", "row"]).to_csv(out / "meta.csv", index=False)
    return out


class WildsImages:
    """Same interface as data.PCamImages, over the decoded subsample."""

    def __init__(self, root: Path = WILDS_DIR / "subsample"):
        self.array = np.load(root / "x.npy", mmap_mode="r")
        self.meta = pd.read_csv(root / "meta.csv")

    def __len__(self) -> int:
        return len(self.array)

    def read(self, idx) -> np.ndarray:
        return np.asarray(self.array[np.asarray(idx)])


def hospital_colour_check(wilds_root: Path = WILDS_DIR / "subsample", per_slide: int = 200, seed: int = 0) -> dict:
    """Partial empirical check of the centre-to-hospital mapping by colour signature (L1 distance
    between mean RGB histograms). RUMC, CWZ and RST were scanned on the same 3DHistech scanner, UMCU on
    a Hamamatsu and LPON on a Philips scanner (Litjens et al. 2018, GigaScience), so the check can
    identify the UMCU-like and LPON-like centres but cannot separate RUMC from CWZ/RST.

    PCam slides are assigned to hospitals by the ASSUMED Camelyon16 numbering (RUMC: tumor_001-070 and
    normal_001-100, i.e. 70 + 100 slides as published; UMCU: the rest). The assumption is tested by
    comparing the between-group distance with a random slide split of the same sizes: a scanner
    difference should make the assumed split far more separated than chance."""
    from pcam_hc.data import PCamImages, load_meta

    def hist(x):
        return np.stack([np.bincount(x[..., c].ravel() // 8, minlength=32) for c in range(3)]).ravel() / x[..., 0].size

    rng = np.random.default_rng(seed)
    meta = load_meta("train").reset_index(drop=True)
    num = meta["wsi"].str.extract(r"_(\d+)$")[0].astype(int)
    is_tumour = meta["wsi"].str.contains("tumor")
    hospital = np.where((is_tumour & (num <= 70)) | (~is_tumour & (num <= 100)), "RUMC", "UMCU")
    imgs = PCamImages("train")
    ref = {}
    for h in ("RUMC", "UMCU"):
        idx = np.sort(rng.choice(np.flatnonzero(hospital == h), 4000, replace=False))
        ref[h] = hist(imgs.read(idx))
    # Null: random slide split with the same number of slides per group.
    slides = meta["wsi"].unique()
    n_rumc = len(set(meta["wsi"][hospital == "RUMC"]))
    shuffled = rng.permutation(slides)
    fake = np.where(meta["wsi"].isin(shuffled[:n_rumc]), "A", "B")
    null = [hist(imgs.read(np.sort(rng.choice(np.flatnonzero(fake == g), 4000, replace=False)))) for g in ("A", "B")]
    w = WildsImages(wilds_root)
    out = {}
    for c, g in w.meta.groupby("center"):
        idx = np.sort(rng.choice(g.index.to_numpy(), min(len(g), 50 * per_slide // 10), replace=False))
        hc = hist(w.read(idx))
        out[int(c)] = {h: float(np.abs(hc - r).sum()) for h, r in ref.items()}
    return {"l1_distance_to_pcam_hospital": out,
            "pcam_assumed_hospital_split_l1": float(np.abs(ref["RUMC"] - ref["UMCU"]).sum()),
            "pcam_random_slide_split_l1": float(np.abs(null[0] - null[1]).sum()),
            "reference_rule": "assumed Camelyon16 numbering (RUMC: tumor<=70, normal<=100)"}


def prepare() -> dict:
    """Metadata checks, the fixed subsample, decoded pixels, and the hospital cross-check."""
    meta = load_metadata()
    report = check_counts(meta)
    sub = subsample(meta)
    report["subsample_per_center"] = sub.groupby("center").size().to_dict()
    report["subsample_positive_rate_per_center"] = sub.groupby("center")["label"].mean().round(4).to_dict()
    extract_images(sub)
    report["hospital_check"] = hospital_colour_check()
    report["center_hospital"] = CENTER_HOSPITAL
    report["primary_ood_centers"] = list(PRIMARY_OOD_CENTERS)
    (PATHS.results / "wilds_data.json").write_text(json.dumps(report, indent=2, default=str))
    return report
