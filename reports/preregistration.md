# Pre-registration: Hill-climbing Hibou-L on PatchCamelyon

- **Author:** Libor "Burny" Burian (implementation by Claude Code under the author's specification)
- **Version:** 2. Registered 2026-10-31 before any data was loaded, any embedding computed or any model trained. See the [version history](#version-history).
- **Repository:** https://github.com/BurnyCoder/better-cancer-classification-with-machine-learning

> After this version is pushed and tagged `preregistration-v2`, any change goes into the [Deviations log](#deviations-log) with the date and reason. Results obtained before a deviation are never deleted.

---

## 1. Domain and question

**Domain.** Patch-level detection of lymph-node metastases in H&E histopathology. The task is binary: "does the centre of this 96×96 px patch contain tumour?". It is the building block of slide-level metastasis detection for breast-cancer staging ([Camelyon16; Ehteshami Bejnordi et al. 2017, JAMA](https://jamanetwork.com/journals/jama/fullarticle/2665774)).

**Question.** How far can the best-performing pathology foundation model on the official PatchCamelyon test split, **Hibou-L** ([Nechaev et al. 2024, arXiv:2406.05074](https://arxiv.org/abs/2406.05074)), be improved? The candidate improvements are well-evidenced interventions transferred from the literature and from Kaggle solutions, run on one 8 GB laptop GPU. Each is measured with statistics that respect slide-level clustering, against a sealed test split.

## 2. Existing evidence

### 2.1 Benchmark
PatchCamelyon (PCam) ([Veeling et al. 2018](https://arxiv.org/abs/1806.03962); [README](https://github.com/basveeling/pcam)):
- **Patches:** 327,680 RGB patches of 96×96 px from Camelyon16 slides. The slides were scanned at 40× (0.243 µm/px) and the patches undersampled to 10×.
- **Label:** a patch is positive iff its **centre 32×32 px region** contains at least one tumour pixel.
- **Splits:** train 262,144 / valid 32,768 / test 32,768, each roughly 50/50. "There is no overlap in WSIs between the splits." Test comes from the Camelyon16 test slides; valid from 20% of the Camelyon16 train slides.
- **Meta CSVs:** give the source slide (`wsi`) of every patch.
- **Licence:** CC0 (README). Zenodo record [2546921](https://zenodo.org/records/2546921) lists MIT.

### 2.2 Why Hibou-L
The candidate pool is the models whose licences the author has accepted on Hugging Face (§9). Among them Hibou-L is best on both the validation and the test split of the standard leaderboard (kaiko **eva** frozen linear probe, [pathology.csv](https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv)):

| Model | eva PCam **test** BA | eva PCam val BA | Midnight Table A1 (CLS / CLS+mean) |
|---|---|---|---|
| **Hibou-L** | **0.955** | **0.939** | 0.952 / 0.949 |
| Prov-GigaPath | 0.945 | 0.935 | — / 0.944 |
| H-optimus-0 | 0.943 | 0.932 | 0.942 / 0.943 |
| Virchow2 | 0.938 | 0.933 | 0.936 / 0.938 |
| Midnight-12k | 0.929 | 0.929 | 0.930 / 0.931 |

- **Midnight Table A1** is from [Karasikov et al. 2025, arXiv:2504.05186](https://arxiv.org/abs/2504.05186).
- **Hibou paper:** Hibou-L reaches 0.953 PCam accuracy with an SGD linear probe (Table 1).
- **Virchow (v1):** not on eva. It scores 93.4 weighted F1 in [Virchow2, arXiv:2408.00738](https://arxiv.org/abs/2408.00738).
- **UNI2-h:** 0.950 test / 0.944 val. Its licence is not accepted (§9), so it is context only.
- **Target rule, fixed now:** the climb target is the accessible model with the highest eva test BA.

**Hibou-L facts:**
- **Architecture:** DINOv2 ViT-L/14 with 4 register tokens, 24 blocks, d = 1024, SwiGLU, about 304M parameters. Apache-2.0; revision `31a3b9fee16554c8552cb643d311733bce8faf7c`.
- **Pretraining:** "trained exclusively on a private dataset": 936,441 H&E + 202,464 non-H&E slides, human and veterinary.
  - Augmentation included random rotations and flips.
  - Hibou-L is the most rotation-invariant pathology foundation model measured in [arXiv:2412.11938](https://arxiv.org/abs/2412.11938).
- **eva preprocessing:** resize to 224; mean (0.7068, 0.5755, 0.722), std (0.195, 0.2316, 0.1816); CLS token.

### 2.3 eva PCam protocol (reproduced in S0)
Source: [`patch_camelyon.yaml`](https://github.com/kaiko-ai/eva/blob/main/configs/vision/pathology/offline/classification/patch_camelyon.yaml).
- Input: resize 96 → 224 (`v2.Resize` → `CenterCrop` → `ToDtype(scale)` → `Normalize`); no augmentation; embeddings precomputed with the backbone frozen.
- Head: `nn.Linear(d, 1)` with `BCEWithLogitsLoss`.
- Optimiser: AdamW at lr 3e-4. Weight decay is the PyTorch default, 0.01.
- Training: batch 256, at most 12,500 steps; early stopping (patience 3) and checkpoint selection on val `BinaryBalancedAccuracy`.
- Evaluation: 5 runs.

### 2.4 Methods transferred from neighbouring work
Each method below becomes one pre-specified candidate in §6.

| Method | Evidence | Step |
|---|---|---|
| Multi-layer features: CLS + average-pooled patch tokens of intermediate blocks | DINOv2 linear protocol (`n_last_blocks` ∈ {1,4}, avgpool) in [`eval/linear.py`](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py). Large average PCam gains from all-layer linear probing of 9 frozen ViTs in [arXiv:2601.09322](https://arxiv.org/abs/2601.09322) (v1 reports +5.3 pp). | S1 |
| Centre-region token pooling, which matches the centre-32 px label rule | Region/attentive pooling gains on PCam in the same paper; label rule from the PCam README. | S1 |
| z-scored features + logistic regression; seed ensembles of heads | UNI and [Virchow2](https://arxiv.org/abs/2408.00738) probe recipes; UBC-OCEAN 1st place (Phikon features, many-head averaging, logistic-regression calibration; [Owkin](https://www.owkin.com/blog/ubc-ocean-kaggle-competition-meet-the-winners)). | S2 |
| D4 test-time augmentation (TTA) | Kaggle Histopathologic Cancer Detection: TTA plus ensemble 0.9816 vs. 0.9794 for the best single model ([arXiv:1906.09587](https://arxiv.org/abs/1906.09587)). | S3 |
| Out-of-fold (OOF) label-noise cleaning | PANDA 1st place dropped samples with a large OOF-vs-label gap: private QWK 0.916 → 0.932 ([code](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution)). PCam label errors are estimated at about 0.2% ([SelfClean, arXiv:2305.17048](https://arxiv.org/abs/2305.17048)). | S3b |
| Stacking OOF predictions with a meta-learner | ISIC 2024 top solutions (image-model OOF predictions fed into a second-level model on the same grouped folds). | S4, S6 |
| LoRA fine-tuning | UNI (ViT-L) on PCam: linear probe 91.60 → LoRA 94.24 accuracy; full fine-tuning 90.40 ([arXiv:2410.16038](https://arxiv.org/abs/2410.16038)). | S5 |
| LP-FT: fine-tune starting from the probe head | [Kumar et al. 2022, arXiv:2202.10054](https://arxiv.org/abs/2202.10054); Camelyon17-WILDS OOD SGD 86.8 → 94.1 with LP-FT ([arXiv:2211.09359](https://arxiv.org/abs/2211.09359)). | S5 |
| HED stain augmentation | WILDS Camelyon17 leaderboard: ERM 70.3 vs. H&E jitter 91.6 OOD ([WILDS](https://wilds.stanford.edu/leaderboard/)); [Tellez et al. 2019, arXiv:1902.06543](https://arxiv.org/abs/1902.06543). | S5 |
| Drop-path > 0; EMA of trainable weights | Removing drop-path degrades PEFT ([arXiv:2409.16434](https://arxiv.org/abs/2409.16434)). | S5 |
| WiSE-FT-style scaling of the LoRA update; model soups | [arXiv:2109.01903](https://arxiv.org/abs/2109.01903); [arXiv:2203.05482](https://arxiv.org/abs/2203.05482); WiSE-style interpolation helps PEFT ([arXiv:2409.16434](https://arxiv.org/abs/2409.16434)). | S5b |

**Considered and rejected before any result:**
- **Pseudo-labelling on test images or on unlabelled Camelyon16 slides:** this would leak.
- **Inputs above 224 px:** they hurt frozen foundation models (UNI-2 0.951 → 0.929 at 392 px, Midnight Table A1).
- **Centre-32 px-only training and focal loss:** no gain in the Kaggle HCD write-up above.
- **DoRA, LoRA+, SAM:** no ViT-classification evidence; SAM is not complementary to weight averaging.
- **Full-token attentive probe:** about 134 GB token cache, and only small gains for DINOv2-L.
- **OUSM and mixup:** thin evidence.

## 3. Data roles (fixed before any result)

| Split | Role |
|---|---|
| **train** (262,144) | Model fitting. **5 slide-grouped folds**, stratified by whether a slide contributes any centre-tumour patch (seed 0). Every step-selection decision uses **OOF** predictions on these folds. |
| **train-64k** | A fixed subset of 65,536 train patches, stratified by (fold, label), seed 0. Used for D4-TTA selection, the LoRA sweep and the augmentation-matched control. |
| **valid** (32,768) | (a) The H2 reproduction gate. (b) A **one-shot gate** per ladder step: the CV-selected candidate is adopted only if valid ΔBA ≥ 0 and valid AUROC does not drop. (c) Early stopping, **only** inside the eva-exact S0. |
| **test** (32,768) | **Sealed.** Test images and embeddings are touched only after every arm's config is frozen: config hashes committed and the commit git-tagged `ladder-frozen`. Then all arms are scored once, in one batch, and **every** result is reported. Code enforces the seal. |
| **Camelyon17-WILDS** ([Koh et al. 2021](https://arxiv.org/abs/2012.07421)) | Out-of-distribution evaluation only; nothing is fit on it. It uses the same label rule and resolution. Fixed subsample of ≤ 1,000 patches per slide (seed 0), all 50 slides, reported per hospital. The **primary pool** excludes the two Camelyon16 source hospitals (RUMC, UMCU). The centre-to-hospital mapping is verified and documented before unsealing. |

## 4. Metrics

- **Primary:** balanced accuracy (BA) at a **fixed 0.5 probability threshold**, for every arm.
- **Secondary:** ROC-AUC, accuracy, F1, NLL, ECE (15 equal-width bins).

## 5. Hypotheses

**Gate (not a claim):**
- **H2:** our eva-protocol Hibou-L probe (CLS) reaches **valid BA 0.939 ± 0.010** (mean of 5 seeds).
- Our Hibou-L embeddings match `kaiko-eva`'s on 256 patches (mean cosine > 0.999).
- Our native-`transformers` loading matches Hibou's own remote code (cosine > 0.9999).
- If any check fails, we debug against valid and eva only; test stays sealed.

**Confirmatory** (sealed test; Holm–Bonferroni across H1, H1b, H4, H6; α = 0.05):

| ID | Hypothesis | Prediction | Falsified if |
|---|---|---|---|
| **H1** | The final pipeline S6 beats the reproduced Hibou-L baseline S0 on test BA. | Δ ≥ +0.5 pt | The paired 95% CI of ΔBA includes 0, or ΔBA < 0. A significant Δ below +0.5 pt falsifies the effect-size prediction and is reported as such. |
| **H1b** | S6 beats B\*, the best single non-Hibou-L backbone probe chosen by CV: Prov-GigaPath, H-optimus-0, Virchow2, Virchow, Hibou-B, Midnight-12k, OpenMidnight-1.1, Phikon or Phikon-v2. | Δ > 0 | The CI includes 0, so the gain is explainable by a backbone swap. |
| **H4** | LoRA-adapted Hibou-L beats a frozen Hibou-L head trained on the **same images with the same D4 + HED augmentation**. The head is trained on embeddings of one seeded random augmented view per image. | Δ > 0 | The CI includes 0, so the gain is augmentation, not adaptation. |
| **H6** | The S6 − S0 gain transfers to Camelyon17-WILDS (primary OOD pool). | Δ > 0 | The CI includes 0, or Δ ≤ 0: the gain is in-distribution only. |

Each arm's prediction is its **seed-averaged probability**, so S0 is itself a 5-seed probe ensemble and serves as the ensemble-matched baseline.

**Exploratory predictions** (descriptive, labelled as such):
- S1: CLS+mean ≈ CLS for Hibou-L. Multi-layer and centre-token features gain > 0 but far less than +5 pp, because the baseline is near the ceiling.
- S3: D4 TTA gain ≤ 0.3 pt, because Hibou-L is near rotation-invariant.
- S3b: |Δ| ≤ 0.2 pt from label cleaning.
- S5/S5b: LP-FT, HED augmentation and WiSE scaling help OOD (H6) more than the in-distribution test.

## 6. Hill-climb ladder (cumulative, pre-specified order)

**Adoption rule for each step:**
- The winning candidate is the one with the highest **mean OOF BA**.
- It is adopted only if it beats the incumbent on OOF predictions under a paired slide-cluster test, Holm-corrected within the step (p < 0.05), **and** passes the valid gate.
- Otherwise the incumbent stays and the step is reported as a negative result.

**S0. Baseline.**
- eva-exact Hibou-L probe: CLS, linear head, early stopping on valid, 5 seeds. This is the H2 gate.
- The same probe under CV is run on each candidate backbone (B\* scan) and on the controls.
- The single Hibou-L forward pass caches CLS and patch-mean (registers excluded) at blocks {6, 12, 18, 21, 22, 23, 24}, plus the mean of the centre 6×6 patch tokens at block 24.

**S1. Representation.**
- a: CLS₂₄
- b: CLS₂₄ + mean₂₄
- c: CLS₂₁₋₂₄ + mean₂₄ (DINOv2 protocol)
- d: CLS+mean at {6, 12, 18, 24}
- e: best of a–d + centre-token mean
- f: best of a–d + centre view (centre 64 px crop resized to 224)

**S2. Head.**
- eva linear (12,500 steps under CV)
- L2 logistic regression on z-scored features, C ∈ {1e-3 … 1e2} chosen by inner grouped CV
- one-hidden-layer MLP (512 units, dropout 0.1)

**S3. D4 TTA.** Average logits over the 8 dihedral views, selected on OOF predictions over train-64k.

**S3b. OOF label cleaning** (nested, so evaluation labels are never altered):
- Within each outer training set, inner grouped-CV OOF predictions flag patches with |y − p| > 0.9.
- Flagged patches are dropped and the head refit.
- The held-out fold is scored on its original labels.

**S4. Fusion** over the S0 candidates combined with Hibou-L:
- greedy forward selection with logit averaging ([Caruana et al. 2004](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf))
- embedding concatenation
- logistic-regression stacking on OOF logits

A licence-clean variant restricted to publishable backbones (§9) is always reported alongside.

**S5. LoRA** ([Hu et al. 2021](https://arxiv.org/abs/2106.09685); [`peft`](https://github.com/huggingface/peft)) on Hibou-L:
- **Architecture:** all 24 blocks, attention query/key/value/output projections.
- **Training setup:** bf16, gradient checkpointing, micro-batch 64; AdamW with cosine schedule and warm-up.
- **Fixed recipe:**
  - LP-FT: the head is initialised from the S2 probe.
  - D4 + HED-light augmentation (σ = 0.05).
  - Drop-path 0.1.
  - EMA of the trainable weights (decay 0.999).
- **Sweep:** lr ∈ {5e-5, 1e-4, 3e-4} × rank ∈ {8, 16}, by successive halving on train-64k, selected on held-out train fold 0.
- **Final:** 3 seeds on folds 1–4 for 1 epoch.
- **Throughput fallback**, triggered only if measured speed is below 20 img/s: the last 12 blocks.
- Hibou-B is used only for 1k-patch smoke tests of this pipeline, never for selection.

**S5b. Weight-space post-processing** (no training), chosen on fold 0:
- LoRA scale α ∈ {0.25, 0.5, 0.75, 1.0}
- 3-seed logit ensemble vs. uniform soup of the LoRA deltas

**S6. Final.** Combination of all adopted components: equal-weight logit averaging vs. logistic-regression stacking, fitted with slide-grouped inner CV on fold 0.

**Ablations** (valid and, after unsealing, test): remove one adopted component at a time from S6.

**Controls** (sanity checks, not gates):
- shuffled labels (expected BA ≈ 0.5)
- majority class
- colour-histogram logistic regression (stain shortcut)
- randomly initialised ViT-L/14-reg4 (Hibou-L architecture)
- DINOv2 ViT-L/14-reg4 trained on natural images (`facebook/dinov2-with-registers-large`)

**Baselines:**
- S0, the reproduced Hibou-L.
- B\*.
- A trained end-to-end ImageNet ConvNeXt-T with D4 augmentation ([Liu et al. 2022](https://arxiv.org/abs/2201.03545)).
- The published P4M-DenseNet (89.8 accuracy / 0.963 AUC; [Veeling 2018](https://arxiv.org/abs/1806.03962)), cited only.

**Sensitivity:** seed spread for every arm; the full LoRA sweep table.

## 7. Statistical analysis plan

- **Unit of resampling: the slide** (`wsi`). Patches from one slide are correlated (test has 130 slides).
- **Confidence intervals:** stratified slide-cluster bootstrap.
  - Tumour-containing and normal slides are resampled separately, 10,000 replicates.
  - BCa intervals ([Efron 1987](https://doi.org/10.1080/01621459.1987.10478410)), with acceleration from a leave-one-slide-out jackknife.
  - Paired differences resample the same slides for both arms.
- **p-values:** paired sign-flip randomisation test (10,000 flips) on exact per-slide contributions, ΔBA = Σₛ[ΔTPₛ/(2P) + ΔTNₛ/(2N)].
- **ΔAUROC:** paired slide-cluster bootstrap CI.
- **Seeds:** 5 per head arm and 3 per LoRA arm, reported as mean ± SD. Tests use the seed-averaged probability.
- **Multiplicity:** Holm within each ladder step and across the four confirmatory tests.
- **No iteration on test:** nothing is chosen after unsealing.

## 8. Leakage and contamination controls

**Automated tests:**
- Slide IDs are disjoint across splits.
- No exact pixel-hash duplicates across splits.
- Folds are grouped by slide.
- The test seal is enforced.
- The licence guard (§9) is enforced.

**Pretraining data** (authors' statements; unverifiable claims are flagged):

| Backbone | Pretraining data | Camelyon16/17 |
|---|---|---|
| Hibou-L, Hibou-B | Private HistAI slides only ([2406.05074](https://arxiv.org/abs/2406.05074)) | not included per authors |
| Prov-GigaPath | Providence slides, 171k WSIs ([Xu et al. 2024](https://doi.org/10.1038/s41586-024-07441-w)) | not included per authors |
| Virchow2 / Virchow | MSKCC slides, 3.1M / 1.5M ([2408.00738](https://arxiv.org/abs/2408.00738); [card](https://huggingface.co/paige-ai/Virchow)) | not stated |
| H-optimus-0 | Proprietary Bioptimus data ([card](https://huggingface.co/bioptimus/H-optimus-0)) | not stated |
| Midnight-12k | TCGA only ([card](https://huggingface.co/kaiko-ai/midnight)) | no |
| Phikon / Phikon-v2 | TCGA / TCGA, CPTAC, GTEx, TCIA ([card](https://huggingface.co/owkin/phikon-v2)) | no |
| OpenMidnight-1.1 | not stated ([card](https://huggingface.co/medarc/OpenMidnight-1.1)) | **unverified** |

H1 and H1b are reported with and without backbones whose data is unverified.

## 9. Model access and licence compliance

**Access:**
- **Pool rule:** the candidate pool is every registry model for which `huggingface_hub.auth_check` passes when Stage A starts. The check is logged and the pool frozen before any embedding.
- **Accepted on 2026-10-31:** Hibou-L, Hibou-B, H-optimus-0, Prov-GigaPath, Virchow, Virchow2.
- **Ungated:** Midnight-12k, OpenMidnight-1.1, Phikon, Phikon-v2, DINOv2.
- UNI2-h, UNI and CONCH are excluded unless their licences are accepted before Stage A.

**Licences:**
- **Apache-2.0 or MIT backbones** (Hibou-L/B, H-optimus-0, Prov-GigaPath, Virchow, Midnight-12k, OpenMidnight-1.1): derived heads and adapters may be published with attribution and the upstream notices.
- **CC-BY-NC-ND-4.0 (Virchow2) and Owkin non-commercial (Phikon, Phikon-v2):** used locally for non-commercial academic research only. **Nothing derived from them is published** (no weights, embeddings, heads, fused models or per-patch predictions), only aggregate metrics.
- **No clinical claims:** the project makes no clinical or diagnostic claims.
- **Enforcement:** code refuses to export any artifact whose lineage includes a non-publishable backbone.
- **Full terms:** recorded in `docs/licenses.md`.

## 10. Claims policy

- Comparisons are made only against matched baselines under the same protocol.
- Published leaderboard numbers, including 0.955, are context. A LoRA, TTA or fusion pipeline is not protocol-matched to a frozen linear probe, so **no state-of-the-art claim** is made.

## 11. Compute budget (estimate)

- Hardware: RTX 5070 Laptop GPU, 8 GB.
- Hibou-L inference: 150–250 img/s.
- LoRA: 40–60 img/s, about 1–1.5 h per epoch-seed.
- ViT-g candidates: 2–3 h per full pass.
- Total: about 45–55 GPU-hours.

## Version history

| Version | Date | Change |
|---|---|---|
| 1 (tag `preregistration`) | 2026-10-31 | Targeted Midnight-12k under an "ungated models only" constraint. |
| 2 (tag `preregistration-v2`) | 2026-10-31 | Superseded v1 **before any data was loaded or any result computed**. The author accepted the Hugging Face licences of stronger models and dropped the "ungated only" constraint. The target became the leaderboard-best Hibou-L, and transferred methods were added (§2.4). |

## Deviations log

| Date | Change | Reason | Results affected |
|---|---|---|---|
| 2026-11-01 | **Added a pre-specified sensitivity analysis: every test metric is also reported with slide `camelyon16_test_049` excluded** (443 test patches, 269 positive). The primary metric stays the full official test split, for comparability with published numbers. | The automated leakage check (§8) found 33 distinct test images that are pixel-identical copies of train images. All of them come from `camelyon16_test_049` and `camelyon16_train_tumor_036`, at identical slide coordinates. CAMELYON16 dataset notes list test_049 as a "duplicate slide" that was removed from the official release ([notes](https://yuanpinz.github.io/blog/machine%20learning/2021/09/30/camelyon.html)), yet PCam's test split contains it. The decision was made with test still sealed: no test embedding or prediction exists. | Test-split reporting only (sealed). |
| 2026-11-01 | Corrected the counts: PCam test covers **129** slides (49 with tumour); valid 54 (22); train 216 (88). | Measured by `main.py verify`; §7 said 130. | None. |
| 2026-11-01 | Reported: within-split exact duplicates (train 42,119 extra copies in 15,235 groups, none with conflicting labels; valid 4,660; test 3,385). No action, because slide-grouped folds keep identical patches from the same slide in one fold. | Data-quality transparency. | None. |
| 2026-11-01 | Hibou-L embeddings are computed in **bf16**. The parity gate passed: mean cosine to eva 0.9994, fp32 native vs. eva 1.0000. The bf16 minimum cosine over the 256 patches is 0.934, which is reported. | Precision/throughput trade-off on 8 GB, documented. | None. |
| 2026-11-01 | AUROC and ΔAUROC bootstrap intervals use 2,000 slide-cluster replicates. BA and ΔBA intervals keep 10,000. | Each AUROC replicate re-ranks up to 262k patches, about 10 min per arm at 10,000; 2,000 replicates are ample for 95% intervals. Applies to every arm computed after the first Stage-A run. The arms of that run (`S0_eva_exact_hibou_l`, `S0_hibou_l_cls_linear`, `control_*`) used 10,000 replicates for AUROC too. | Secondary metric CIs only. |
| 2026-11-01 | **Postponed to a later study day (author's instruction):** (a) ablations that require training a new model, e.g. LoRA without LP-FT or without HED augmentation; (b) the separate seed/hyperparameter sensitivity analysis. Kept: ablations that only recombine existing predictions (remove one adopted component from S6), the pre-registered seeds inside every arm (5 per head, 3 per LoRA), and the LoRA lr x rank selection sweep, which is part of the method. | Compute and time budget, as the author decided. | Ablation and sensitivity sections are reported as postponed, not omitted. |
| 2026-11-01 | **Milestone order:** results are released in stages, as instructed: (1) the reproduced Hibou-L probe with controls and no-training baselines, pushed with paper, reports, README, website and HF; then (2) the ladder, (3) LoRA and trained baselines, (4) unsealed test/WILDS. The remaining ladder steps were paused until milestone 1 was published. | Author's order of work. | None (no analysis changes). |
| 2026-11-01 | **Release order revised (author's instruction), superseding the previous row:** (1) reproduced Hibou-L; (2) the Hibou-L-only improvement ladder (S1, S2, S3, S3b, S5, S5b) with the existing cheap controls and no-training baselines; (3) control models (DINOv2-L, random ViT-L, H4 augmented-feature control); (4) baseline models (the other foundation models, i.e. the B* scan, and ConvNeXt-T); (5) ablations that need no training; (6) more models (S4 fusion, S6 final). The test split and WILDS stay sealed until the whole ladder, including S4/S6, is frozen. The adoption rules are unchanged; only execution order changes. S4 runs after S5 and fuses with the Hibou-L-only incumbent at that point. | Author's order of work. | Order of steps only. |

