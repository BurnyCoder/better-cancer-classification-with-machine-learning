"""Published PCam baselines that need no training (context only, never protocol-matched claims).

Global context: the user's order of work asks for baselines that do not need training wherever
they exist. Two kinds are recorded in results/published_baselines.json:
  * the kaiko eva leaderboard (frozen backbone + linear probe, balanced accuracy on PCam val and test),
    fetched from a pinned commit of https://github.com/kaiko-ai/eva so the numbers come from the source
    file rather than being typed in;
  * literature results on the official PCam test split, each with its paper and table.
"""

import csv
import io
import json
import urllib.request

from pcam_hc.config import PATHS

EVA_COMMIT = "5710f2df0342e6f2b85fa658a0b2a519829b75e8"  # last commit touching pathology.csv (2025-09-29)
EVA_CSV = f"https://raw.githubusercontent.com/kaiko-ai/eva/{EVA_COMMIT}/tools/data/leaderboards/pathology.csv"
EVA_NAMES = {"histai_hibou_l": "Hibou-L", "mahmood_uni2_h": "UNI2-h", "prov_gigapath": "Prov-GigaPath",
             "bioptimus_h_optimus_0": "H-optimus-0", "paige_virchow2": "Virchow2", "mahmood_uni": "UNI",
             "kaiko_midnight_12k": "Midnight-12k", "owkin_phikon": "Phikon", "kaiko_vitb8": "kaiko ViT-B/8",
             "kaiko_vitl14": "kaiko ViT-L/14", "kaiko_vitb16": "kaiko ViT-B/16", "kaiko_vits16": "kaiko ViT-S/16",
             "owkin_phikon_v2": "Phikon-v2", "lunit_vits16": "Lunit ViT-S/16", "kaiko_vits8": "kaiko ViT-S/8"}
ACCESSIBLE = {"Hibou-L", "Prov-GigaPath", "H-optimus-0", "Virchow2", "Midnight-12k", "Phikon", "Phikon-v2"}

# Literature results on the official PCam test split (metric, value, source, location in source).
LITERATURE = [
    {"model": "P4M-DenseNet (rotation-equivariant CNN, trained from scratch)", "metric": "accuracy", "value": 0.898,
     "source": "https://arxiv.org/abs/1806.03962", "where": "Veeling et al. 2018, Table 1"},
    {"model": "P4M-DenseNet", "metric": "AUROC", "value": 0.963,
     "source": "https://arxiv.org/abs/1806.03962", "where": "Veeling et al. 2018, Table 1"},
    {"model": "Hibou-L, SGD linear probe", "metric": "accuracy", "value": 0.953,
     "source": "https://arxiv.org/abs/2406.05074", "where": "Nechaev et al. 2024, Table 1"},
    {"model": "Hibou-B, SGD linear probe", "metric": "accuracy", "value": 0.946,
     "source": "https://arxiv.org/abs/2406.05074", "where": "Nechaev et al. 2024, Table 1"},
    {"model": "UNI, linear probe", "metric": "accuracy", "value": 0.9160,
     "source": "https://arxiv.org/abs/2410.16038", "where": "Lee et al. 2024, PCam results"},
    {"model": "UNI, full fine-tuning", "metric": "accuracy", "value": 0.9040,
     "source": "https://arxiv.org/abs/2410.16038", "where": "Lee et al. 2024, PCam results"},
    {"model": "UNI, LoRA", "metric": "accuracy", "value": 0.9424,
     "source": "https://arxiv.org/abs/2410.16038", "where": "Lee et al. 2024, PCam results"},
]


def fetch() -> dict:
    """Download the pinned eva leaderboard, keep the PCam columns, add the literature table."""
    with urllib.request.urlopen(EVA_CSV) as r:  # noqa: S310 - fixed https URL at a pinned commit
        rows = list(csv.DictReader(io.StringIO(r.read().decode())))
    eva = [{"model": EVA_NAMES.get(row["model"], row["model"]), "eva_id": row["model"],
            "val_ba": float(row["patch_camelyon"]), "test_ba": float(row["patch_camelyon/test"]),
            "accessible_here": EVA_NAMES.get(row["model"], row["model"]) in ACCESSIBLE} for row in rows]
    eva.sort(key=lambda r: -r["test_ba"])
    out = {"eva_leaderboard": {"source": EVA_CSV, "protocol": "frozen backbone, linear head, eva PCam config, 5 runs",
                               "rows": eva},
           "literature": LITERATURE}
    PATHS.results.mkdir(parents=True, exist_ok=True)
    (PATHS.results / "published_baselines.json").write_text(json.dumps(out, indent=2))
    return out
