# Pre-registration: Hill-climbing Midnight-12k on PatchCamelyon

- **Author:** Libor "Burny" Burian (implementation by Claude Code under the author's specification)
- **Date registered:** 2026-10-31. Committed and pushed before any embedding, training run or test score exists.
- **Repository:** https://github.com/BurnyCoder/better-cancer-classification-with-machine-learning

> Any change to this document after the first push is logged in the [Deviations log](#deviations-log) at the bottom, with the date and the reason. Results produced before a deviation are never deleted.

## 1. Domain and question

**Domain.** Patch-level detection of lymph-node metastases in H&E histopathology. This is the binary task "does the centre of this patch contain tumour?". It is the building block of slide-level metastasis detection in breast-cancer staging ([Camelyon16, Ehteshami Bejnordi et al. 2017](https://jamanetwork.com/journals/jama/fullarticle/2665774)).

**Question.** How much can the strongest openly licensed (MIT), ungated pathology foundation model with a published PatchCamelyon score, **Midnight-12k** ([Karasikov et al. 2025, arXiv:2504.05186](https://arxiv.org/abs/2504.05186), [weights](https://huggingface.co/kaiko-ai/midnight)), be improved on the official PCam test split? The improvements tried are cheap, well-motivated interventions run on one 8 GB laptop GPU, and each one is measured with statistics that respect the fact that patches are clustered within slides.

## 2. Existing evidence (literature review)

**Benchmark.** PatchCamelyon (PCam) ([Veeling et al. 2018, arXiv:1806.03962](https://arxiv.org/abs/1806.03962); [data and README](https://github.com/basveeling/pcam)):
- **Size:** 327,680 RGB patches, 96×96 px, from Camelyon16 slides scanned at 40× (0.243 µm/px) and undersampled to 10×.
- **Label:** positive if the centre 32×32 px region contains at least one tumour pixel; tumour in the outer region does not count.
- **Splits:** train 262,144 / valid 32,768 / test 32,768. "There is no overlap in WSIs between the splits." Test = the Camelyon16 test slides; valid = 20% of the Camelyon16 train slides. The splits are roughly balanced 50/50.
- **Licence:** CC0 (README). The Zenodo record [2546921](https://zenodo.org/records/2546921) lists MIT.

**Published reference numbers on the official test split.** BA = balanced accuracy.

| Model | Protocol | PCam test | Source |
|---|---|---|---|
| P4M-DenseNet (rotation-equivariant CNN, trained from scratch) | supervised | Acc 89.8, AUC 96.3 | [Veeling 2018](https://arxiv.org/abs/1806.03962) |
| UNI (ViT-L) | linear probe / full FT / LoRA | Acc 91.60 / 90.40 / 94.24 | [Lee et al. 2024](https://arxiv.org/abs/2410.16038) |
| **Midnight-12k** (ViT-g/14, MIT, ungated) | eva frozen linear probe, CLS | **BA 0.929** (val 0.929) | [eva leaderboard CSV](https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv) |
| Midnight-12k | eva frozen linear probe, CLS+mean | BA 0.931 | [Midnight model card](https://huggingface.co/kaiko-ai/midnight) |
| DINOv2 ViT-g/14 (natural images) | eva linear probe | BA 0.881 | Midnight model card |
| UNI2-h (gated, CC-BY-NC-ND) | eva linear probe | BA 0.950 | eva leaderboard |
| Hibou-L (gated) | eva linear probe | BA 0.955 | [eva leaderboard](https://kaiko-ai.github.io/eva/main/leaderboards/) |

The gated models are context only. The project constraint, set by the author, is to use **ungated models and datasets only**.

**Evidence behind the interventions.**
- **LoRA fine-tuning.** It beat linear probing by +2.6 accuracy points for UNI on PCam ([Lee 2024](https://arxiv.org/abs/2410.16038)). Their probe baseline (91.6) was weaker than eva-style probes, so we expect a smaller gain.
- **CLS+mean pooling** vs. CLS alone for Midnight: 0.931 vs. 0.930 ([Midnight Table A1](https://arxiv.org/abs/2504.05186)).
- **Fusing several foundation models** with a mixture of experts: about +1.4 F1 on PCam ([Shazam, arXiv:2503.00736](https://arxiv.org/abs/2503.00736)).
- **Rotation sensitivity.** Pathology foundation models are measurably not rotation-invariant ([arXiv:2412.11938](https://arxiv.org/abs/2412.11938)). We found no published number for D4 test-time augmentation on a foundation model on PCam, which is why H3 is exploratory.
- **Stain normalisation and augmentation** gives roughly no gain on frozen self-supervised features ([arXiv:2311.11772](https://arxiv.org/abs/2311.11772)), so it is not part of the ladder.

**The eva PCam protocol** ([config](https://github.com/kaiko-ai/eva/blob/main/configs/vision/pathology/offline/classification/patch_camelyon.yaml)):
- Resize 96→224, no augmentation, embeddings precomputed with the backbone frozen.
- Head `torch.nn.Linear(d, 1)` with `BCEWithLogitsLoss`, AdamW at lr 3e-4.
- Batch 256, at most 12,500 steps, early stopping on val `BinaryBalancedAccuracy`.
- Mean of 5 runs.

## 3. Data roles (fixed before any result)

| Split | Role |
|---|---|
| **train** (262,144) | Model fitting. Split into **5 slide-grouped folds**, stratified by whether a slide contributes any centre-tumour patch (seed 0). Every step-selection decision uses **out-of-fold (OOF)** predictions on these folds. |
| **train-64k** | A fixed subset of 65,536 patches, sampled within the folds with stratification by slide and label (seed 0). Used where full-train cost is prohibitive: D4 test-time augmentation selection, the LoRA sweep, and the augmentation-matched control. |
| **valid** (32,768) | (a) The H2 reproduction gate. (b) A **one-shot gate** per ladder step: a step selected on CV passes only if its valid ΔBA ≥ 0 and its valid AUROC does not drop. (c) Early stopping, only inside the eva-exact baseline reproduction. |
| **test** (32,768) | **Sealed.** Test logits are computed only after the ladder and every arm config are frozen, i.e. their hashes are committed and git-tagged `ladder-frozen`. All pre-registered arms are then scored once, in one batch, and **every** test number is reported. Code enforces the seal: test inference refuses to run unless the tag exists. |
| **Camelyon17-WILDS** ([Koh et al. 2021](https://arxiv.org/abs/2012.07421)) | Out-of-distribution (OOD) check only; nothing is fit on it. Same label rule (centre 32×32) and resolution (about 0.97 µm/px). Fixed slide-stratified subsample of ≤1,000 patches per slide, seed 0, all 50 slides, reported per hospital. **Primary OOD pool** = hospitals that are not Camelyon16 sources (RUMC, UMCU). The centre→hospital mapping is verified and documented before unsealing. |

## 4. Metrics

- **Primary:** balanced accuracy (BA) at a **fixed 0.5 probability threshold**, used for every arm, as in eva.
- **Secondary:** ROC-AUC, accuracy, F1, negative log-likelihood (NLL), expected calibration error (ECE, 15 equal-width bins).

## 5. Hypotheses and falsification criteria

**Confirmatory** (tested on sealed test; Holm–Bonferroni across H1, H1b-i, H1b-ii, H4, H6; α = 0.05):

| ID | Hypothesis | Prediction | Falsified if |
|---|---|---|---|
| **H1** | The final ladder pipeline (S6) beats the reproduced Midnight-12k eva baseline (S0) on test BA. | Δ ≥ +1.0 pt | The paired 95% CI of ΔBA includes 0, or ΔBA < 0. (If Δ is significant but < +1.0 pt, the effect-size prediction is falsified and reported as such.) |
| **H1b-i** | S6 beats the best single ungated backbone probe B\*, chosen on CV among the candidate backbones. | Δ > 0 | The CI includes 0. That would make the gain explainable by a backbone swap. |
| **H1b-ii** | S6 beats a 5-seed ensemble of Midnight-12k probes (ensemble-matched baseline). | Δ > 0 | The CI includes 0. That would make the gain explainable by plain ensembling. |
| **H4** | LoRA-adapted Midnight-12k beats a frozen Midnight-12k head trained on the **same images with the same D4 augmentation** (augmented-feature control). | Δ > 0 | The CI includes 0. That would make the gain augmentation, not adaptation. |
| **H6** | The S6 − S0 gain transfers to Camelyon17-WILDS (primary OOD pool). | Δ > 0 | The CI includes 0, or Δ ≤ 0. That would make the improvement in-distribution only. |

**Gate (not a claim):**
- **H2 (reproduction).** Our eva-protocol Midnight-12k probe (CLS) reaches **valid BA within 0.929 ± 0.010** (mean of 5 seeds), and our embeddings match `kaiko-eva`'s on 256 patches (mean cosine > 0.999).
- If either check fails, we debug against valid and eva only, test stays sealed, and the climb does not start until H2 passes.

**Exploratory** (descriptive, labelled as such in every report):
- Per-step CV and valid gains for S1–S5.
- H3: D4 TTA helps.
- H5: fusion helps.
- Calibration, per-slide error analysis, centre-crop view.

## 6. Hill-climb ladder (pre-specified order, cumulative)

Each step lists its candidates. The winner is the candidate with the highest **mean OOF CV BA over 5 seeds**. The step is adopted only if that winner beats the incumbent with a paired, Holm-corrected slide-cluster test on OOF predictions (p < 0.05) **and** passes the valid gate. Otherwise the incumbent is kept and the step is reported as a negative result.

- **S0. Baseline (eva-exact).**
  - Midnight-12k, CLS embedding, linear head, eva hyperparameters, early stopping on valid, 5 seeds. This is the H2 gate.
  - The same probe, using CV, is run on the candidate backbones (OpenMidnight-1.1, Phikon, Phikon-v2) and on the controls. B\* is the best candidate by CV.
- **S1. Embedding:**
  - CLS
  - CLS+mean of patch tokens
  - CLS+mean plus the **centre view**: the centre 64×64 crop resized to 224, embedded and concatenated with the full-patch embedding
- **S2. Head:**
  - eva linear (fixed 12,500 steps under CV; no early stopping)
  - L2 logistic regression (`sklearn`, C ∈ {1e-3 … 1e2} by inner grouped CV)
  - one-hidden-layer MLP (hidden 512, dropout 0.1)
- **S3. D4 TTA:** average logits over the 8 dihedral views (4 rotations × optional horizontal flip). Selected on OOF predictions over train-64k.
- **S4. Fusion:** greedy forward ensemble selection with replacement ([Caruana et al. 2004](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)) over the S0 candidate backbones, using OOF logits. Logit averaging is compared with embedding concatenation.
- **S5. LoRA** ([Hu et al. 2021](https://arxiv.org/abs/2106.09685); [`peft`](https://github.com/huggingface/peft)) on Midnight-12k.
  - Adapters on the attention projections of the last 12 of 40 blocks; earlier blocks are frozen and run without gradients.
  - bf16, gradient checkpointing, micro-batch 32, D4 augmentation only.
  - **Sweep:** lr ∈ {5e-5, 1e-4, 3e-4} × rank ∈ {8, 16}, by successive halving on train-64k, selected on held-out train fold 0.
  - **Final:** 3 seeds on train folds 1–4 for 1 epoch, checkpoint chosen on a fixed 10k subsample of fold 0.
  - **Throughput fallback**, pre-registered and triggered only by measured throughput below 8 img/s, never by a score: the last 6 blocks and train-64k.
- **S6. Final:** the cumulative combination of all adopted steps.

**Ablations** (on test after unsealing, and on valid): remove one adopted component at a time from S6.

**Controls** (reported; sanity checks, not gates):
- Shuffled-label probe (expected BA ≈ 0.5).
- Majority class (BA = 0.5).
- Colour-histogram logistic regression (stain-shortcut check).
- Randomly initialised ViT-g/14.
- DINOv2 ViT-g/14 with natural-image weights (published 0.881).

**Baselines:**
- S0 (reproduced Midnight-12k).
- B\* (best single ungated backbone).
- The 5-seed probe ensemble.
- A trained end-to-end ImageNet ConvNeXt-T with D4 augmentation ([Liu et al. 2022](https://arxiv.org/abs/2201.03545)).
- Published P4M-DenseNet, cited only.

**Sensitivity:** seed spread for every arm; the full LoRA sweep table.

## 7. Statistical analysis plan

- **Unit of resampling: the slide (`wsi` in the PCam meta CSVs).** Patches from one slide are correlated, and test has 130 slides; patch-level tests would be anti-conservative.
- **Confidence intervals:** stratified slide-cluster bootstrap. Tumour-containing and normal slides are resampled separately, 10,000 replicates, BCa intervals with jackknife over slides. For paired differences, both arms are resampled with the same slides.
- **p-values:** paired sign-flip randomisation test (10,000 flips) on per-slide contributions to ΔBA = Σₛ [ΔTPₛ / (2P) + ΔTNₛ / (2N)], where P and N are the total positive and negative patch counts.
- **ΔAUROC:** paired slide-cluster bootstrap CI.
- **Seeds:** 5 per probe arm, 3 per LoRA arm. Reported as mean ± SD across seeds. The statistical tests use the **seed-averaged probability** per patch (one prediction per arm).
- **Multiplicity:** Holm across the candidates within a ladder step, and Holm across the five confirmatory tests.
- **No test-set iteration.** No hyperparameter, threshold, step or arm is chosen after unsealing.

## 8. Leakage and contamination controls

Automated tests:
- Slide IDs are disjoint across train/valid/test.
- No exact pixel-hash duplicate images across splits.
- CV folds are grouped by slide.
- The test seal is enforced in code.

Pretraining data of each backbone, documented and cited in `reports/`:

| Backbone | Pretraining data | Camelyon? |
|---|---|---|
| Midnight-12k | TCGA FFPE only ([card](https://huggingface.co/kaiko-ai/midnight)) | no |
| Phikon | TCGA ([card](https://huggingface.co/owkin/phikon)) | no |
| Phikon-v2 | TCGA, CPTAC, GTEx, TCIA ([card](https://huggingface.co/owkin/phikon-v2)) | no Camelyon listed |
| OpenMidnight-1.1 | not stated on its card ([card](https://huggingface.co/medarc/OpenMidnight-1.1)) | **unverified** |

Because OpenMidnight-1.1 is unverified, H1/H1b are reported both with and without it.

## 9. Claims policy

- Results are compared only against matched baselines under the same protocol.
- Leaderboard numbers from frozen probes of gated models are context.
- **No state-of-the-art claim** is made against them, because a LoRA, TTA or fusion pipeline is not protocol-matched to a frozen linear probe.

## 10. Compute budget (estimate)

- Hardware: RTX 5070 Laptop GPU, 8 GB.
- Expected throughput: ViT-g/14 at 30–45 img/s in bf16, so about 2–3 h per backbone pass over all 327k patches; about 2–3 h per LoRA epoch-seed.
- WILDS: ≤ 50k patches.
- Total: about 50–70 GPU-hours.

## Deviations log

| Date | Change | Reason | Results affected |
|---|---|---|---|
| — | — | — | — |
