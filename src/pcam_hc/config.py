"""Central configuration: paths from `.env`, PCam file manifest, eva protocol constants.

Global context: imported first by every entry point so that `HF_HUB_CACHE` (Hugging Face model
cache on a large disk) is set before `transformers`/`timm` are imported. `HF_HOME` is deliberately
not overridden: it would hide the stored login token that licence-accepted models need. All
hyperparameters that the pre-registration fixes (reports/preregistration.md) live here, each with
its source.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")  # sets HF_HOME etc. before any HF import


@dataclass(frozen=True)
class Paths:
    """Filesystem layout; large artifacts default to `D:/AI/pcam` via `.env`."""

    data: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    emb: Path = Path(os.getenv("EMB_DIR", PROJECT_ROOT / "data" / "embeddings"))
    ckpt: Path = Path(os.getenv("CKPT_DIR", PROJECT_ROOT / "data" / "checkpoints"))
    results: Path = PROJECT_ROOT / os.getenv("RESULTS_DIR", "results")
    logs: Path = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")


PATHS = Paths()
SPLITS = ("train", "valid", "test")

# PCam files and md5 checksums from the Zenodo API (record 2546921). The GitHub README lists the
# valid_x / test_x checksums swapped, so the Zenodo values are authoritative.
ZENODO_URL = "https://zenodo.org/records/2546921/files/{name}?download=1"
PCAM_FILES_TEMPLATE = "camelyonpatch_level_2_split_{split}_{kind}"
PCAM_MD5 = {
    "camelyonpatch_level_2_split_train_x.h5.gz": "1571f514728f59376b705fc836ff4b63",
    "camelyonpatch_level_2_split_train_y.h5.gz": "35c2d7259d906cfc8143347bb8e05be7",
    "camelyonpatch_level_2_split_train_meta.csv": "5a3dd671e465cfd74b5b822125e65b0a",
    "camelyonpatch_level_2_split_valid_x.h5.gz": "d5b63470df7cfa627aeec8b9dc0c066e",
    "camelyonpatch_level_2_split_valid_y.h5.gz": "2b85f58b927af9964a4c15b8f7e8f179",
    "camelyonpatch_level_2_split_valid_meta.csv": "67589e00a4a37ec317f2d1932c7502ca",
    "camelyonpatch_level_2_split_test_x.h5.gz": "d8c2d60d490dbd479f8199bdfa0cf6ec",
    "camelyonpatch_level_2_split_test_y.h5.gz": "60a7035772fbdb7f34eb86d4420cf66a",
    "camelyonpatch_level_2_split_test_meta.csv": "3455fd69135b66734e1008f3af684566",
}


@dataclass(frozen=True)
class EvaProtocol:
    """eva offline PCam head protocol.

    Source: https://github.com/kaiko-ai/eva/blob/main/configs/vision/pathology/offline/classification/patch_camelyon.yaml
    AdamW is instantiated with only `lr`, so PyTorch's default weight_decay=0.01 applies
    (https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
    """

    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    max_steps: int = 12_500
    patience: int = 3  # in validation epochs (Lightning validates once per epoch)
    n_runs: int = 5
    resize: int = 224


EVA = EvaProtocol()
# One seed per arm from Stage B on (author's instruction, 2026-11-01; seed sensitivity postponed).
# The eva-exact reproduction and the Stage-A arms used EVA.n_runs = 5 seeds.
SEEDS = (0,)
N_FOLDS = 5
SUBSET_SIZE = 65_536  # "train-64k" subset from the pre-registration
THRESHOLD = 0.5  # fixed decision threshold on the sigmoid probability
