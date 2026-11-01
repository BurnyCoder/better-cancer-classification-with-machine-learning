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
LORA_ARM = "S5_lora"  # the fine-tuned model; its weights live in the LoRA checkpoint folder, not in heads/


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
        + (f"{r['oof']['seed_avg']['ba']:.4f} ({r.get('oof_subset', 'all train')}) |" if "oof" in r else "— |")
        for r in records)
    lora = next((r for r in records if r["name"] == LORA_ARM), None)
    lora_md = (f"""
## Fine-tuned model (`lora/{LORA_ARM}/`)

The adopted LoRA model ({lora['description']}). Files: `seed*.safetensors` hold the trainable weights (LoRA A/B
matrices of the attention projections in all 24 blocks plus the linear head; best EMA checkpoint), `lp_head.safetensors`
holds the LP-FT start head, and `config.json` holds the pinned Hibou-L revision, PEFT settings and preprocessing. Rebuild it
with the repository's `pcam_hc.stage_c.new_model` and `load_state_dict(..., strict=False)`. Hibou-L itself is not
redistributed: accept its licence on Hugging Face first.

Known issues of this run (found by code review after training): its training-set cleaning mask used predictions that
saw fold-0 labels, its stain augmentation is stronger than "HED-light" suggests, and its lr/rank choice is close to
arbitrary. Its validation result is unaffected by the first issue. Details are in the repository's
`reports/experiments/02_ladder_hibou_l.md`.
""" if lora else "")
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
Train OOF BA is out-of-fold over all training slides, or over train fold 0 for the fine-tuned model and its H4 control.
{lora_md}
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
    if LORA_ARM in arm_names:
        stage_lora(staging / "lora" / LORA_ARM)
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
        if (HEAD_DIR / rec["name"]).exists():  # frozen-probe heads; the LoRA model was staged above
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


def stage_lora(out: Path) -> None:
    """Stage the S5 LoRA model: per run, the best EMA trainable weights (LoRA A/B of the attention
    projections in all 24 Hibou-L blocks plus the linear head) as safetensors; the LP-FT start head;
    and the configuration that rebuilds it on the pinned Hibou-L revision."""
    import torch
    from safetensors.torch import save_file

    from pcam_hc.finetune import LORA_TARGETS
    from pcam_hc.stage_c import LORA_DIR

    rec = json.loads((PATHS.results / "arms" / f"{LORA_ARM}.json").read_text())
    assert_publishable(rec)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(LORA_DIR.glob("seed*.pt")):
        save_file({k: v.contiguous().cpu() for k, v in torch.load(p, weights_only=False)["state"].items()},
                  str(out / f"{p.stem}.safetensors"))
    head = torch.load(LORA_DIR / "lp_head.pt", weights_only=False)
    save_file({k: v.contiguous().cpu() for k, v in head["module"].items()}, str(out / "lp_head.safetensors"))
    spec = REGISTRY["hibou_l"]
    cfg = {"base_model": spec.repo, "revision": spec.revision, "base_licence": spec.licence,
           "peft": {"r": rec["lora"]["rank"], "lora_alpha": rec["lora"]["rank"], "lora_dropout": 0.0, "bias": "none",
                    "target_modules": LORA_TARGETS},
           "preprocess": {"input": "96x96 uint8 RGB", "resize": "bilinear to 224x224", "mean": spec.mean, "std": spec.std},
           "head": {"kind": head["kind"], "features": rec["recipe"] if "recipe" in rec else "block-24 CLS"},
           "training": {"lr": rec["lora"]["lr"], "runs": rec["lora"]["runs"], "description": rec["description"]},
           "rebuild": "pcam_hc.stage_c.new_model(lp_head_state, rank), then load_state_dict(seed*.safetensors, strict=False)"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=list))


def publishable_arms() -> list[str]:
    """Arms whose whole lineage is redistributable and that have weights to upload (heads or the LoRA model)."""
    from pcam_hc.stage_c import LORA_DIR

    return [p.stem for p in sorted((PATHS.results / "arms").glob("*.json"))
            if all(REGISTRY[b].publishable for b in json.loads(p.read_text()).get("backbones", []))
            and ((Path(PATHS.data) / "heads" / p.stem).exists() or (p.stem == LORA_ARM and any(LORA_DIR.glob("seed*.pt"))))]
