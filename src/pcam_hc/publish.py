"""Hugging Face publication of licence-clean artifacts (heads, LoRA adapters) into one collection.

Global context: pre-registration section 9. Only artifacts whose entire backbone lineage allows
redistribution are uploaded. Virchow2 (CC-BY-NC-ND-4.0) and Phikon / Phikon-v2 (non-commercial) never
leave this machine. `assert_publishable` is the single gate every upload passes through, and
tests/test_licence.py covers it.
"""

import json
import logging
from pathlib import Path

from pcam_hc.backbones import REGISTRY
from pcam_hc.config import PATHS, PROJECT_ROOT

log = logging.getLogger(__name__)
REPO_ID = "BurnyCoder/pcam-hibou-hillclimb"
COLLECTION_TITLE = "PCam hill-climbing of Hibou-L"


class LicenceError(RuntimeError):
    """Raised when an artifact derives from a backbone whose licence forbids redistribution."""


def assert_publishable(record: dict) -> None:
    bad = [b for b in record.get("backbones", []) if not REGISTRY[b].publishable]
    if bad:
        raise LicenceError(f"{record.get('name')} derives from non-redistributable backbone(s): {bad}")


def model_card(records: list[dict]) -> str:
    """Card with attribution, licence notices and per-arm metrics read from the arm records."""
    notices = sorted({REGISTRY[b].notice for r in records for b in r["backbones"]})
    rows = "\n".join(
        f"| `{r['name']}` | {', '.join(r['backbones'])} | {r['valid']['seed_avg']['ba']:.4f} | "
        f"{r['oof']['seed_avg']['ba']:.4f} |" if "oof" in r else
        f"| `{r['name']}` | {', '.join(r['backbones'])} | {r['valid']['seed_avg']['ba']:.4f} | — |"
        for r in records)
    return f"""---
license: apache-2.0
tags: [pathology, histopathology, patchcamelyon, hibou, linear-probe, lora]
base_model: histai/hibou-L
datasets: [1aurent/PatchCamelyon]
---

# PCam hill-climbing of Hibou-L: heads and adapters

Classification heads and LoRA adapters trained on top of the **Hibou-L** pathology foundation model
([Nechaev et al. 2024](https://arxiv.org/abs/2406.05074)) for PatchCamelyon metastasis detection.
They come from the pre-registered study at
https://github.com/BurnyCoder/better-cancer-classification-with-machine-learning, where you will find the
pre-registration, the code to rebuild every artifact, and all results with slide-level confidence intervals.
The current paper release is included as `paper.pdf`.

**Research use only. Not a medical device. Not for clinical decision-making.**

## Artifacts

| Arm | Backbone(s) | Valid BA (seed-averaged) | Train OOF BA |
|---|---|---|---|
{rows}

Each `heads/<arm>/seed<k>.pt` holds a `torch` state dict (`module`), the z-scoring `mu`/`sd` (or None), and
the head `kind`. The input is the representation documented in the arm's JSON record (`arms/<arm>.json`).

## Licences and attribution
- The base model weights are **not** included. Accept the Hibou-L licence on its Hugging Face page and
  download it yourself.
{chr(10).join('- ' + n for n in notices)}
- These derived artifacts are released under Apache-2.0. Changes: newly trained classification heads / LoRA adapters.
"""


def upload(arm_names: list[str], token=None) -> str:
    """Create/update the model repo with the heads of `arm_names`, and add it to the collection."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    records = []
    for name in arm_names:
        rec = json.loads((PATHS.results / "arms" / f"{name}.json").read_text())
        assert_publishable(rec)
        records.append(rec)
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    staging = PATHS.data / "publish"
    (staging / "arms").mkdir(parents=True, exist_ok=True)
    for rec in records:
        (staging / "arms" / f"{rec['name']}.json").write_text(json.dumps(rec, indent=2))
    (staging / "README.md").write_text(model_card(records), encoding="utf-8")
    (staging / "LICENSE").write_text((PROJECT_ROOT / "LICENSE").read_text())
    paper = PROJECT_ROOT / "reports" / "paper" / "paper.pdf"
    if paper.exists():  # the current release of the paper travels with the artifacts
        (staging / "paper.pdf").write_bytes(paper.read_bytes())
    api.upload_folder(folder_path=str(staging), repo_id=REPO_ID, commit_message="Update card and arm records")
    from pcam_hc.experiments import HEAD_DIR

    for rec in records:
        api.upload_folder(folder_path=str(HEAD_DIR / rec["name"]), path_in_repo=f"heads/{rec['name']}",
                          repo_id=REPO_ID, commit_message=f"Upload heads of {rec['name']}")
    collection = next((c for c in api.list_collections(owner=REPO_ID.split("/")[0], limit=100)
                       if c.title == COLLECTION_TITLE), None)
    if collection is None:
        collection = api.create_collection(COLLECTION_TITLE, description=
                                           "Pre-registered hill-climbing of Hibou-L on PatchCamelyon: heads, adapters, results.")
    try:
        api.add_collection_item(collection.slug, item_id=REPO_ID, item_type="model")
    except Exception as e:  # already present
        log.info("collection item: %s", e)
    return f"https://huggingface.co/{REPO_ID}"


def publishable_arms() -> list[str]:
    return [p.stem for p in sorted((PATHS.results / "arms").glob("*.json"))
            if all(REGISTRY[b].publishable for b in json.loads(p.read_text()).get("backbones", []))
            and (Path(PATHS.data) / "heads" / p.stem).exists()]
