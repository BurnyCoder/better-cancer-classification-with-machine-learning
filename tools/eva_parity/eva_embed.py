"""Reference Hibou-L embeddings computed by kaiko-eva itself (H2 parity check).

Global context: the pre-registration requires our Hibou-L CLS embeddings to match eva's
(mean cosine > 0.999 on 256 patches). eva loads `histai/hibou-L` via its backbone registry entry
`pathology/histai_hibou_l` (Hugging Face remote code) and preprocesses with `ResizeAndCrop` using
the Hibou mean/std from eva's replicate-evaluations recipe. This script runs in its own pinned
environment (transformers 4.x) and writes `eva_hibou_l_cls.npy`; the main project compares it with
its native-transformers encoder (`main.py parity`).

Usage: uv run python eva_embed.py <path to camelyonpatch_level_2_split_valid_x.h5> <out_dir>
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from eva.vision.data.transforms.common import ResizeAndCrop
from eva.vision.models import ModelFromRegistry

N_PATCHES = 256  # first 256 validation patches, as fixed in the pre-registration
HIBOU_MEAN, HIBOU_STD = (0.7068, 0.5755, 0.722), (0.195, 0.2316, 0.1816)  # eva replicate recipe


def main(h5_path: str, out_dir: str) -> None:
    with h5py.File(h5_path, "r") as f:
        imgs = torch.from_numpy(f["x"][:N_PATCHES]).permute(0, 3, 1, 2)  # uint8 NCHW
    transform = ResizeAndCrop(size=224, mean=HIBOU_MEAN, std=HIBOU_STD)
    batch = torch.stack([transform(img) for img in imgs])
    model = ModelFromRegistry("pathology/histai_hibou_l").eval()  # remote-code Hibou, CLS features
    with torch.inference_mode():
        feats = torch.cat([model(batch[i : i + 32]) for i in range(0, N_PATCHES, 32)])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "eva_hibou_l_cls.npy", feats.float().numpy())
    print("saved", feats.shape, "to", out / "eva_hibou_l_cls.npy")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
