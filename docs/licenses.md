# Model and data licences

This study is **non-commercial academic research**. It makes **no clinical or diagnostic claims**, and none of its outputs are intended for patient care.

**What the code enforces:**
- Every backbone in `src/pcam_hc/backbones.py` carries its licence and a `publishable` flag.
- Every arm in `results/arms/*.json` lists its backbone lineage.
- Nothing derived from a non-publishable backbone is exported: no weights, embeddings, heads or per-patch predictions. The guard is tested in `tests/test_licence.py`.

**Access:** several models require accepting their licence on Hugging Face (a click-through, sometimes with name and email). The pool used is the set of models for which `huggingface_hub.auth_check` passed on 2026-11-01 (`results/model_pool.json`).

## Models

| Model (pinned revision) | Licence | Additional terms (from the Hugging Face gate / model card) | What this project publishes |
|---|---|---|---|
| [Hibou-L](https://huggingface.co/histai/hibou-L) `31a3b9f`, [Hibou-B](https://huggingface.co/histai/hibou-b) `7fdc8cc` | Apache-2.0 | Click-through to share contact information | Heads and LoRA adapters under Apache-2.0, with attribution, a licence copy and a statement of changes. Users must accept the base model's licence themselves. |
| [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) `b145cc1` | Apache-2.0 | Provided as-is; not reviewed or approved by any regulatory body; healthcare use requires regulatory compliance and independent validation; users are responsible for GDPR/HIPAA compliance | Derived artifacts with Apache-2.0 attribution and this disclaimer reproduced |
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) `64f9e26` | Apache-2.0 | Intended solely for research on pathology foundation models and for reproducibility; not for clinical care or clinical decision-making; deployed use is out of scope | Derived research artifacts with this intended-use statement reproduced |
| [Virchow](https://huggingface.co/paige-ai/Virchow) `19eebc8` | Apache-2.0 | Non-binding guidelines: attribution, responsible use, not for medical diagnosis or treatment of real patients, privacy, no malicious use, disclosure of use in products | Derived artifacts with Apache-2.0 attribution and the guidelines reproduced |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2) `3158645` | **CC-BY-NC-ND-4.0** | Academic non-commercial research only. The model may not be distributed, published or reproduced. No diagnostic, clinical, IUO or RUO use, and no re-identification. Commercial use of derivatives is prohibited, including models trained on its outputs and datasets created from it. | **Nothing derived.** Aggregate metrics and figures only, with attribution. |
| [Midnight-12k](https://huggingface.co/kaiko-ai/midnight) `adc6b15` | MIT | — | Derived artifacts with the MIT notice |
| [OpenMidnight-1.1](https://huggingface.co/medarc/OpenMidnight-1.1) `a9ed8ab` | Apache-2.0 | — | Derived artifacts with attribution |
| [Phikon](https://huggingface.co/owkin/phikon) `057cc02`, [Phikon-v2](https://huggingface.co/owkin/phikon-v2) `2ae989a` | Owkin non-commercial licence | Non-commercial research use | **Nothing derived.** Aggregate metrics only. |
| [DINOv2-L with registers](https://huggingface.co/facebook/dinov2-with-registers-large) `e4c89a4` | Apache-2.0 | — | Control only |
| [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | CC-BY-NC-ND-4.0 | Institutional email required; licence not accepted | Not used |

## Data

| Dataset | Licence | Source |
|---|---|---|
| PatchCamelyon | CC0 (GitHub README); Zenodo lists MIT | [basveeling/pcam](https://github.com/basveeling/pcam), [Zenodo 2546921](https://zenodo.org/records/2546921) |
| Camelyon17-WILDS | CC0 | [WILDS](https://wilds.stanford.edu/datasets/) |

## This repository
The repository's own code is MIT-licensed. It contains no third-party model weights. Weights are downloaded from Hugging Face into the local cache by each user, under their own acceptance of each licence.
