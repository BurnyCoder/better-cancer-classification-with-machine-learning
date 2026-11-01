"""Embedding-cache layout and column selection for representation candidates (ladder S1).

Global context: one forward pass of each backbone fills a cache whose columns are laid out as
[CLS_l, mean_l] for each cached block l, then (optionally) the centre-token mean of the last block.
A `meta.json` next to each cache records that layout. Representation candidates (pre-registration
section 6, S1) are column subsets of the cache, so no backbone is re-run to compare them.
"""

import json
from dataclasses import dataclass

import numpy as np

from pcam_hc.config import PATHS

# Hibou-L caches the blocks needed by S1 (c: last four, d: every sixth) plus centre tokens;
# all other backbones cache the last block only (eva protocol, B* scan, fusion).
CACHE_LAYOUT = {"hibou_l": {"layers": (6, 12, 18, 21, 22, 23, 24), "center": True}}
LAST = -1  # placeholder for "the network's last block" in a Representation


@dataclass(frozen=True)
class Representation:
    """Which pooled vectors to concatenate: tuples of (block, 'cls' | 'mean'), plus centre tokens."""

    parts: tuple
    center: bool = False

    def columns(self, meta: dict) -> np.ndarray:
        layers, dim = list(meta["layers"]), meta["dim"]
        cols = []
        for layer, kind in self.parts:
            pos = layers.index(layers[-1] if layer == LAST else layer)
            base = pos * 2 * dim + (0 if kind == "cls" else dim)
            cols.append(np.arange(base, base + dim))
        if self.center:
            if not meta["center"]:
                raise ValueError("centre tokens were not cached for this backbone")
            cols.append(np.arange(2 * dim * len(layers), 2 * dim * len(layers) + dim))
        return np.concatenate(cols)


CLS = Representation(((LAST, "cls"),))  # eva protocol embedding
CLS_MEAN = Representation(((LAST, "cls"), (LAST, "mean")))

# S1 candidates for Hibou-L (24 blocks). e/f extend the best of a-d and are built at run time.
S1_HIBOU = {
    "a_cls24": CLS,
    "b_cls24_mean24": CLS_MEAN,
    "c_dinov2_last4": Representation(((21, "cls"), (22, "cls"), (23, "cls"), (24, "cls"), (24, "mean"))),
    "d_spread_6_12_18_24": Representation(tuple((l, k) for l in (6, 12, 18, 24) for k in ("cls", "mean"))),
}


def cache_dir(backbone: str, split: str, subset: str | None = None):
    return PATHS.emb / backbone / (f"{split}_{subset}" if subset else split)


def read_meta(backbone: str, split: str, subset: str | None = None) -> dict:
    return json.loads((cache_dir(backbone, split, subset) / "meta.json").read_text())


def load(backbone: str, split: str, rep: Representation, view: str = "full", subset: str | None = None) -> np.ndarray:
    """Select a representation's columns from the float16 cache into RAM (chunked reads)."""
    arr = np.load(cache_dir(backbone, split, subset) / f"{view}.npy", mmap_mode="r")
    cols = rep.columns(read_meta(backbone, split, subset))
    out = np.empty((arr.shape[0], len(cols)), dtype=np.float16)
    for s in range(0, arr.shape[0], 32768):
        out[s : s + 32768] = arr[s : s + 32768][:, cols]
    return out
