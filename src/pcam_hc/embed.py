"""Resumable fp16 embedding cache: one `.npy` memmap per (backbone, split[/subset], view).

Global context: frozen-feature steps (S0-S4) never touch the GPU backbone again after this cache
is filled. A `.progress` file records how many rows are final, so an interrupted multi-hour pass
resumes where it stopped instead of restarting.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch

from pcam_hc.data import center_crop, d4_view, prefetch

log = logging.getLogger(__name__)


def apply_view(x: torch.Tensor, view: str) -> torch.Tensor:
    """Image view of a uint8 (B,3,H,W) batch: 'full', 'center64', or 'd4_k' (k = 0..7)."""
    if view == "full":
        return x
    if view == "center64":
        return center_crop(x, 64)
    if view.startswith("d4_"):
        return d4_view(x, int(view[3:]))
    raise ValueError(view)


def embed_images(encoder, read_fn, n: int, path: Path, batch_size: int = 128, view: str = "full") -> np.ndarray:
    """Embed `n` images given by `read_fn(local_indices) -> uint8 (B,96,96,3)` into `path`.

    Rows [0, progress) are trusted; work restarts at the first unfinished batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = Path(str(path) + ".progress")
    mode = "r+" if path.exists() else "w+"
    out = np.lib.format.open_memmap(path, mode=mode, dtype=np.float16, shape=(n, encoder.out_dim))
    done = int(progress.read_text()) if progress.exists() and mode == "r+" else 0
    batches = [np.arange(s, min(n, s + batch_size)) for s in range(done, n, batch_size)]
    t0, seen = time.time(), 0
    for idx, imgs in prefetch(read_fn, batches):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2)  # NHWC -> NCHW, uint8
        feats = encoder(apply_view(x, view))
        out[idx[0] : idx[-1] + 1] = feats.cpu().numpy().astype(np.float16)
        seen += len(idx)
        if idx[-1] + 1 == n or (idx[0] // batch_size) % 50 == 0:  # flush + log every 50 batches
            out.flush()
            progress.write_text(str(idx[-1] + 1))
            log.info("%s: %d/%d rows, %.1f img/s", path.name, idx[-1] + 1, n, seen / (time.time() - t0))
    out.flush()
    progress.write_text(str(n))
    del out
    return np.load(path, mmap_mode="r")


def embed_backbone(name: str, split: str, view: str = "full", indices=None, subset: str | None = None,
                   batch_size: int = 128) -> Path:
    """Fill the cache for one (backbone, split[/subset], view), writing the layout to meta.json.

    Test images are sealed: embedding the test split requires the frozen-ladder tag."""
    import json

    from pcam_hc.backbones import REGISTRY, Encoder, load_model
    from pcam_hc.data import PCamImages
    from pcam_hc.features import CACHE_LAYOUT, cache_dir

    if split == "test":
        from pcam_hc.seal import assert_test_unsealed, frozen_arms

        assert_test_unsealed(frozen_arms())
    spec = REGISTRY[name]
    lay = CACHE_LAYOUT.get(name, {})
    encoder = Encoder(load_model(spec), spec, layers=lay.get("layers"), center=lay.get("center", False))
    images = PCamImages(split)
    idx = np.arange(len(images)) if indices is None else np.asarray(indices)
    out_dir = cache_dir(name, split, subset)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"backbone": name, "repo": spec.repo, "revision": spec.revision, "layers": list(encoder.layers),
            "center": encoder.center, "dim": spec.dim, "n": int(len(idx)), "n_prefix": encoder.n_prefix}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("embedding %s %s%s view=%s n=%d dim=%d", name, split, f"/{subset}" if subset else "", view, len(idx),
             encoder.out_dim)
    embed_images(encoder, lambda local: images.read(idx[local]), len(idx), out_dir / f"{view}.npy", batch_size, view)
    return out_dir
