#set document(title: "Hill-climbing Hibou-L on PatchCamelyon", author: "Libor Burian")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show link: set text(fill: rgb("#2a78d6"))
#align(center)[#text(16pt)[*Hill-climbing Hibou-L on PatchCamelyon:*] \ #text(13pt)[a pre-registered, slide-clustered study. Release 3: improvement ladder and control models]]
#align(center)[Libor "Burny" Burian \ #link("https://github.com/BurnyCoder/better-cancer-classification-with-machine-learning")[github.com/BurnyCoder/better-cancer-classification-with-machine-learning]]

*Abstract.* Pathology foundation models are usually compared on PatchCamelyon (PCam) with a frozen backbone and a linear probe. The best published such result belongs to Hibou-L [4]: 0.955 test balanced accuracy (BA) on the eva leaderboard [5]. We pre-registered a ladder of interventions to improve it, each transferred from neighbouring work: intermediate-layer and centre-region features, head choice, rotation/flip test-time augmentation, out-of-fold label cleaning, multi-model fusion, and LoRA fine-tuning with LP-FT, stain augmentation and weight-space averaging. Every step is selected on slide-grouped cross-validation, gated on validation, and the test split stays sealed until the ladder is frozen. We reproduce the eva Hibou-L probe to the published precision (three decimals): validation BA 0.9394 ± 0.0007 over 5 seeds, against the published 0.939. Our native-transformers implementation matches Hibou's own code to a mean cosine of 1.000000. Under the cross-validation protocol the original probe reaches 0.9498 [0.9366, 0.9600] out-of-fold BA. Controls behave as expected: shuffled labels give 0.520 and a colour-histogram classifier gives 0.787. A pixel-hash audit found 33 test images identical to train images, all from PCam test slide `camelyon16_test_049` and train slide `camelyon16_train_tumor_036`, all at identical slide coordinates, which indicates a duplicate scan. We also find that the official Camelyon17-WILDS test hospital matches the colour signature of one of PCam's own source hospitals. Both were found before any test data was used: the duplicate slide added a sensitivity analysis, and the out-of-hospital pool is being re-decided. Of 6 decided ladder steps, 2 were adopted (`S3b_label_cleaning`, `S5_lora`). No frozen-feature classifier has a validation gain whose 95% CI excludes zero (largest: +0.01 pt [-0.76, +0.40], `S3b_oof_label_cleaning`). LoRA fine-tuning raises validation BA from 0.9367 to 0.9444 (paired Δ +0.77 pt [+0.27, +1.41]). LoRA also beats a frozen head trained on the same patches with the same augmentation recipe (Δ valid +0.99 pt [+0.58, +1.51]). A ViT-L/14-reg4 of Hibou-L's size reaches 0.880 OOF BA with natural-image (DINOv2) weights and 0.797 with random weights, against 0.950 for Hibou-L. The test split and WILDS remain sealed.

= Introduction
Assessing sentinel lymph nodes for metastases is part of breast-cancer staging, and it is tedious and time-consuming [2, 3]. PCam [1, 31] recasts it as 327,680 patches of 96×96 px, labelled positive if the centre 32×32 px contains at least one tumour pixel. Self-supervised pathology foundation models (FMs) trained with DINOv2 [9] now dominate PCam under frozen-probe protocols [5, 6]. Many adaptation techniques are each reported to help elsewhere: LoRA [12, 11], LP-FT [13], WiSE-FT and model soups [14, 15], multi-layer probing [16], ensembles [17], label-noise cleaning [23], and stain augmentation [18]. Their joint value on top of the strongest model is unknown. This study measures it with a pre-registered ladder. It uses statistics that resample whole slides, because patches from one slide are correlated, and a test split sealed in code. This document is release 3 of the study; each release adds the classifiers trained since the last one, and the test split is unsealed only after the ladder is frozen.

= Methods
== Data and audit
We use the official PCam splits from Zenodo record 2546921, md5-verified against the Zenodo API. Note that the GitHub README [31] lists the validation and test checksums swapped for all three file pairs (images, labels and meta CSVs). Train has 262,144 patches from 216 slides (88 with tumour), validation 32,768 / 54 (22), and test 32,768 / 129 (49). The audit checks that no slide is shared between splits (0 train–test, 0 train–valid, 0 valid–test), that the labels agree with the meta CSVs, and it counts exact pixel duplicates within and across splits (BLAKE2b hashes), with the slide coordinates of every cross-split copy.
== Model, access and licences
Hibou-L is a ViT-L/14 with 4 registers [10] and a SwiGLU MLP block, with about 304M parameters [35], pretrained with DINOv2 on 1.2B patches from about 1.14M private slides [4]. It ships custom code; we load its pinned weights into transformers' native `Dinov2WithRegistersModel` (no remote code) and test equivalence against kaiko-eva running the remote code. The candidate pool is every registry model the author's Hugging Face account can load (gated models whose licence the author accepted, plus ungated ones), frozen before any embedding: 12 of 13 registry entries, including the two control backbones (not accessible: MahmoodLab/UNI2-h). Derivatives of non-redistributable models (Virchow2, Phikon) are never published; only aggregate metrics are.
== Features, heads and protocol
Patches are resized 96→224 px (bilinear), scaled and normalised with Hibou's eva mean and std. Embeddings are computed in bf16 and cached in fp16. The *eva-exact* probe is `nn.Linear` with BCE loss, AdamW at lr 0.0003, batch 256, at most 12,500 steps, early stopping on validation BA (patience 3), and 5 seeds [5]. Ladder decisions use a *CV protocol*: five slide-grouped folds of train, stratified by slide tumour status, out-of-fold (OOF) predictions with fixed 12,500 steps, and full-train refits scored on validation.
== Statistics
Every arm's prediction is its seed-averaged probability; BA uses a fixed 0.5 threshold. 95% CIs come from a stratified slide-cluster bootstrap [33] (tumour and normal slides resampled separately; 10,000 replicates for BA; BCa with a leave-one-slide-out jackknife [20]). Paired differences use a per-slide sign-flip randomisation test on the exact decomposition ΔBA = Σ#sub[s] [ΔTP#sub[s]/2P + ΔTN#sub[s]/2N], with Holm correction [21]. The test split and WILDS are sealed: reading them requires a git tag whose committed hash matches the frozen ladder configuration.

#figure(image("../figures/pipeline.png", width: 100%), caption: [Study pipeline. Statistics resample whole slides; the test split and WILDS are scored once, after the ladder is frozen.])

= Results
== Data audit
No slide is shared between splits. There are many exact duplicates *within* splits (extra copies: train 42,119, valid 4,660, test 3,385; no group has conflicting labels), consistent with PCam's probabilistic sampling [31, 34]. Only 2 duplicate groups span two slides and can straddle CV folds; slide-grouped folds keep every other group inside one fold. Across splits, 33 test images are pixel-identical to train images. All of them come from `camelyon16_test_049` and train slide `camelyon16_train_tumor_036`, all at identical slide coordinates. The official CAMELYON16 README lists `test_049` as a duplicate slide that was removed from the data set [32]; PCam's test split still contains it (443 patches). With the test split still sealed, we added a sensitivity analysis that excludes this slide, through the deviations log of the pre-registration.
== Reproduction of the published probe (H2)
Against eva running Hibou's remote code on 256 validation patches, our fp32 embeddings agree to a mean cosine of 1.000000 (max |Δ| 6.8e-06). In bf16 the mean cosine is 0.9994, but the minimum is 0.9336: bf16 noticeably perturbs a few patches. The eva-exact probe reaches validation BA 0.9394 ± 0.0007 (5 seeds), against the published 0.939. This passes the pre-registered ±0.010 gate; the seed-averaged prediction gives 0.9396 [0.8936, 0.9638].
== The original probe under the CV protocol, and controls
#[#set par(justify: false)
#set text(size: 9pt)
#table(columns: 3, stroke: 0.4pt + luma(200), inset: 4pt, align: left,
  table.header([*Arm*], [*Train OOF BA [95% CI]*], [*Valid BA [95% CI]*]),
  [Reproduced Hibou-L probe, eva-exact], [—], [0.9396 \[0.8936, 0.9638\]],
  [Hibou-L CLS + linear, CV protocol (S0)], [0.9498 \[0.9366, 0.9600\]], [0.9367 \[0.8888, 0.9617\]],
  [Hibou-L, shuffled train labels], [0.5204 \[0.4935, 0.5457\]], [0.4731 \[0.4280, 0.5212\]],
  [Colour histogram + logistic regression], [0.7870 \[0.7580, 0.8106\]], [0.8098 \[0.7530, 0.8472\]],
  [Constant 'tumour' prediction], [—], [0.5000 \[0.5000, 0.5000\]],
  [DINOv2-L/14-reg4 with natural-image weights], [0.8797 \[0.8622, 0.8940\]], [0.8682 \[0.8103, 0.9023\]],
  [ViT-L/14-reg4 with random weights, seed 0], [0.7973 \[0.7692, 0.8201\]], [0.7902 \[0.7232, 0.8334\]])]

The validation CI is 3.1× wider than the OOF CI, because validation has only 54 slides (22 with tumour). This is why validation serves only as a no-harm gate and selection uses train CV. Shuffled labels land at chance, so there is no label-leakage path. Colour alone is strongly informative but far below the foundation model. The two controls of Hibou-L's size separate what the weights contribute: natural-image DINOv2 weights (same size, but a standard MLP block) recover part of the gap to Hibou-L, and random weights in Hibou-L's exact architecture score close to the colour histogram.
#figure(image("../figures/stage_a.png", width: 92%), caption: [Reproduced Hibou-L probe and controls. Dots are seed-averaged BA; bars are 95% stratified slide-bootstrap BCa CIs.])
== No-training baselines
@pub shows the eva leaderboard [5] (pinned commit) with our reproduction. Literature results on PCam, which use different metrics and protocols and are context only: P4M-DenseNet (rotation-equivariant CNN; no pretraining reported): accuracy 0.8980 (Veeling et al. 2018, Table 1); P4M-DenseNet: AUROC 0.9630 (Veeling et al. 2018, Table 1); Hibou-L, SGD linear probe: accuracy 0.9530 (Nechaev et al. 2024, Table 1); Hibou-B, SGD linear probe: accuracy 0.9460 (Nechaev et al. 2024, Table 1); UNI, linear probe: accuracy 0.9160 (Lee et al. 2024, Table 3 (the PEFT column is LoRA)); UNI, full fine-tuning: accuracy 0.9040 (Lee et al. 2024, Table 3 (the PEFT column is LoRA)); UNI, LoRA: accuracy 0.9424 (Lee et al. 2024, Table 3 (the PEFT column is LoRA)).
#figure(image("../figures/published_baselines.png", width: 88%), caption: [Published frozen-probe PCam BA (eva, validation and test) and our reproduced Hibou-L validation BA. Among 15 leaderboard models, Hibou-L ranks 2 on validation and 1 on test. The ±SD of our reproduction is smaller than its marker; \* marks models outside this study's model pool.]) <pub>
== Camelyon17-WILDS preparation
The official WILDS download failed (expired TLS certificate), so we use a Hugging Face mirror (`wltjr1007/Camelyon17-WILDS`, revision `d784d53`). Its split sizes match the WILDS paper [19]. We fixed a subsample of ≤1,000 patches per slide (49,888 patches from 50 slides). PCam train slides grouped by the assumed Camelyon16 numbering (RUMC vs. UMCU) differ in colour (L1 1.30) far more than a random split (0.14); this run put one RUMC slide of the official CAMELYON16 reference list, `normal_144`, in the UMCU group (fixed in code). WILDS centre 2, the official WILDS *test* hospital, is closest to UMCU (0.69 vs. 1.68). It is therefore most likely UMCU, a Camelyon16 source hospital, and not out-of-distribution for PCam-trained models. The planned primary OOD pool is centres 0, 1 and 4, read as CWZ, RST and LPON from [29]; the pre-registration only excludes RUMC and UMCU, and the mapping is an open issue in its deviations log. The headers of the official CAMELYON17 slide files (node 0 of all 100 training patients) give centres 0, 1 and 3 one pixel size (0.2431 µm/px), centre 2 0.2262–0.2269 µm/px and centre 4 a Philips scanner. RUMC's 3DHistech scanner, which also scanned the CWZ and RST slides, gives 0.243 µm/px, UMCU's Hamamatsu 0.226 µm/px, and LPON used a Philips scanner [3, 2]. So centre 2 is UMCU (as the colour check found) and centre 4 is LPON; the statement that centres 0, 3 and 4 share one scanner [26] does not hold; and RUMC, a Camelyon16 source hospital, is one of centres 0, 1 and 3, which no source we found resolves and the colour check does not separate. The pool is therefore re-decided and logged before WILDS is unsealed.

== Improvement ladder
Hibou-L stays frozen in S1–S3b; each candidate trains a new classifier on features read from it (5 slide-grouped folds; 5 seeds for S1 b–e, 1 seed from S1 f on, by the author's decision; seed sensitivity is postponed). Each classifier is compared with the original probe, paired on the same patches. Per-candidate p-values are descriptive; the pre-registered adoption rule applies Holm within a step and the validation no-harm gate.
#[#set par(justify: false)
#set text(size: 9pt)
#table(columns: 7, stroke: 0.4pt + luma(200), inset: 4pt, align: left,
  table.header([*Step*], [*Classifier*], [*OOF BA [95% CI]*], [*OOF reference (patches)*], [*Δ OOF vs reference [95% CI] (pt)*], [*p*], [*Δ valid vs original (pt)*]),
  [H4 control (frozen, LoRA's augmentation)], [H4\_​aug\_​control], [0.9731 \[0.9568, 0.9823\]], [original (51,522)], [+0.20 \[-0.10, +0.84\]], [0.518], [-0.23],
  [S1 representation], [S1\_​b\_​cls24\_​mean24], [0.9491 \[0.9353, 0.9598\]], [original (262,144)], [-0.07 \[-0.24, +0.06\]], [0.426], [-0.14],
  [S1 representation], [S1\_​c\_​dinov2\_​last4], [0.9457 \[0.9303, 0.9573\]], [original (262,144)], [-0.40 \[-0.89, -0.12\]], [0.0284], [-0.51],
  [S1 representation], [S1\_​d\_​spread\_​6\_​12\_​18\_​24], [0.9469 \[0.9318, 0.9585\]], [original (262,144)], [-0.29 \[-0.69, -0.04\]], [0.0931], [-0.41],
  [S1 representation], [S1\_​e\_​best\_​plus\_​centre\_​tokens], [0.9496 \[0.9361, 0.9599\]], [original (262,144)], [-0.01 \[-0.14, +0.11\]], [0.826], [-0.15],
  [S1 representation], [S1\_​f\_​best\_​plus\_​centre\_​view], [0.9497 \[0.9362, 0.9599\]], [original (262,144)], [-0.01 \[-0.15, +0.17\]], [0.944], [-0.13],
  [S2 head], [S2\_​logreg], [0.9510 \[0.9378, 0.9610\]], [original (262,144)], [+0.12 \[+0.04, +0.31\]], [0.0236], [-0.02],
  [S2 head], [S2\_​mlp], [0.9312 \[0.9112, 0.9462\]], [original (262,144)], [-1.85 \[-2.81, -1.17\]], [0.0001], [-3.13],
  [S3 D4 TTA], [S3\_​d4\_​tta], [0.9522 \[0.9396, 0.9617\]], [S3\_​no\_​tta\_​train64k (65,536)], [+0.04 \[-0.02, +0.13\]], [0.247], [-0.24],
  [S3 D4 TTA], [S3\_​no\_​tta\_​train64k], [0.9517 \[0.9391, 0.9615\]], [original (65,536)], [+0.01 \[-0.03, +0.10\]], [0.779], [-0.15],
  [S3b label cleaning], [S3b\_​oof\_​label\_​cleaning], [0.9525 \[0.9394, 0.9624\]], [original (262,144)], [+0.27 \[+0.10, +0.50\]], [0.0032], [+0.01],
  [S5 LoRA], [S5\_​lora], [0.9798 \[0.9654, 0.9883\]], [original (51,522)], [+0.87 \[+0.50, +1.56\]], [0.0001], [+0.77],
  [S5b weight space], [S5b\_​wise\_​0.25], [0.9782 \[0.9638, 0.9872\]], [original (41,522)], [+0.72 \[+0.45, +1.22\]], [0.0001], [+0.21],
  [S5b weight space], [S5b\_​wise\_​0.5], [0.9789 \[0.9644, 0.9879\]], [original (41,522)], [+0.80 \[+0.48, +1.33\]], [0.0001], [+0.49],
  [S5b weight space], [S5b\_​wise\_​0.75], [0.9798 \[0.9655, 0.9885\]], [original (41,522)], [+0.88 \[+0.52, +1.49\]], [0.0001], [+0.60])]

#figure(image("../figures/ladder_deltas.png", width: 92%), caption: [Every trained ladder classifier: paired Δ BA with 95% slide-bootstrap CIs, on train OOF against the reference and patch set named in each row label (the original probe, except S3 D4 TTA against its no-TTA twin) and on validation against the original probe.])
- *S1_representation*: best candidate `S1_f_best_plus_centre_view`; OOF test passed False, valid gate passed False → *rejected*.
- *S2_head*: best candidate `S2_logreg`; OOF test passed True, valid gate passed False → *rejected*.
- *S3_d4_tta*: best candidate `S3_d4_tta`; OOF test passed False, valid gate passed False → *rejected*.
- *S3b_label_cleaning*: best candidate `S3b_oof_label_cleaning`; OOF test passed True, valid gate passed True → *adopted*.
- *S5_lora*: best candidate `S5_lora`; OOF test passed True, valid gate passed True → *adopted*.
- *S5b_weight_space*: best candidate `S5b_wise_0.75`; OOF test passed False, valid gate passed False → *rejected*.

== H4 augmentation-matched control
LoRA trains on augmented images, so its gain could come from the augmentation rather than from adapting the backbone [18]. The H4 control trains a frozen Hibou-L linear head on the same LoRA training patches, each embedded under one seeded random D4 + HED view (σ = 0.05). LoRA − H4: +0.67 pt [+0.48, +0.97] on fold 0 and +0.99 pt [+0.58, +1.51] on validation (paired slide bootstrap). These differences are descriptive; the pre-registered H4 test is scored on the sealed test split.

#figure(image("../figures/lora_training.png", width: 100%), caption: [S5 LoRA final run. Left: single-minibatch BCE loss every 50 steps (log scale) and its running mean. Right: BA of the EMA weights on the 10k fold-0 checkpoint subsample (zoomed y-axis); the best checkpoint is kept.])

#figure(image("../figures/examples.png", width: 88%), caption: [Example outputs on validation patches with each model's tumour probability: the original probe (baseline), LoRA (new method), and two controls. Rows are outcome groups of original vs. LoRA (LoRA fixes an error of the original: 468; LoRA introduces an error: 217; both wrong: 1,605; both right: 30,478 patches); 3 patches per group, uniform without replacement, numpy default_rng(0), so the examples are not hand-picked.])

= Related work
*Frozen-probe benchmarks.* Pathology FMs are ranked on PCam with frozen backbones and linear probes [5, 6]; Hibou-L [4] has the highest PCam test BA on the eva leaderboard (0.955; rank 2 on validation), ahead of models trained on a larger cohort (Virchow2, 3.1M slides [7]) or a different one (Prov-GigaPath, 171k Providence slides [8]). *Adapting FMs.* A benchmark of adaptation strategies reports LoRA above both linear probing and full fine-tuning for UNI on PCam [11]; LP-FT [13] and weight-space interpolation or averaging [14, 15] improve fine-tuned models in and out of distribution. *Data-centric fixes.* Out-of-fold label cleaning was part of the PANDA-winning solution [23], SelfClean estimates PCam's label-error rate [22], TTA was part of strong ensembles on Kaggle's PCam data [24]; rotation invariance varies widely across pathology FMs, and Hibou-L, trained with rotations and flips [4], was the most invariant by cosine distance [28]. *Representations.* Multi-layer and attentive probes improve frozen natural-image ViTs on average, including on PCam [16]; DINOv2's linear evaluation searches over the last 1 or 4 layers [9]. This study tests these ideas on the strongest frozen model under one pre-registered, slide-clustered protocol.

= Discussion
*Why frozen-feature tricks did not transfer.* Hibou-L starts near the frozen-probe ceiling, its pretraining already included rotations and flips, and the reported multi-layer gains were measured on natural-image ViTs. No frozen-feature change improved validation BA beyond its confidence interval; 4 have validation bootstrap CIs below zero (`S1_c_dinov2_last4` p = 0.19, `S1_d_spread_6_12_18_24` p = 0.068, `S2_mlp` p = 0.0012, `S3_d4_tta` p = 0.037; unadjusted sign-flip p, and one-seed arms include the seed term in Limitations). Label cleaning was adopted on cross-validation, but it removed 11–15 times more patches per fold than SelfClean's PCam label-error estimate (589 images, 0.2%) [22]; if that estimate holds, it also drops hard, correctly labelled patches.
*Why LoRA helped.* The H4 control targets the augmentation: a frozen head trained on the same patches with the same augmentation recipe is no better than the original, while LoRA is clearly better. This suggests that the gain comes from adapting the backbone's attention layers rather than from the augmentation, although the two arms also differ in optimiser and schedule, and the stain augmentation is strong (Limitations), which may handicap a frozen head scored on clean images more than LoRA. Shrinking the LoRA update (WiSE) lowered validation BA, more so the further it was shrunk; WiSE-FT reported its largest gains under distribution shift and smaller ones in distribution (+1.6 pp on ImageNet) [14]; here even the in-distribution effect is negative, and the shift case can only be tested on the sealed WILDS pool.
*Strength of the evidence.* All results are on cross-validation and the 54-slide validation split, with one seed per arm from S1 f on and one LoRA run. They are hedged findings until the sealed test split and the out-of-hospital WILDS pool are scored once.

= Planned and postponed analyses
The pre-registered ladder follows. Steps decided so far: 6.
- S1 representation: last-layer CLS, CLS+mean, DINOv2 last-4-layer [9], spread layers [16], centre tokens and a centre-crop view.
- S2 heads: linear, nested-CV logistic regression, MLP.
- S3 D4 TTA [24, 28]; S3b OOF label cleaning [23, 22].
- S4 fusion with other FMs [17, 8, 7].
- S5 LoRA [12, 11] with LP-FT [13], HED augmentation [30, 18], drop-path [27] and weight averaging (EMA) [25]; S5b WiSE scaling and soups [14, 15].
- S6 final combination.
Then the confirmatory tests on the sealed test split (H1, H1b, H4) and on WILDS (H6). By the author's decision, ablations that require training new models and the separate seed/hyperparameter sensitivity analysis are postponed to a later release.

*Release order* (author's instruction; the adoption rules are unchanged):
+ Reproduced Hibou-L with cheap controls and no-training baselines (release 1).
+ The Hibou-L-only improvement ladder: S1, S2, S3, S3b, S5, S5b (release 2).
+ Control models: DINOv2-L with natural-image weights, random-init ViT-L, and the H4 augmentation-matched control (release 3).
Postponed to a later study day: the baseline models (the B\* scan of 9 other foundation models; ConvNeXt-T), the ablations, S4 fusion and S6, and the sensitivity study. H1b depends on B\* and is deferred with them.

= Limitations
- Hibou-L and most candidates were pretrained on private data; their exclusion of Camelyon cannot be verified.
- bf16 embeddings deviate from fp32 for a few patches.
- Validation is small, so the no-harm gate has low power.
- The WILDS centre-to-hospital mapping is partly unresolved: slide metadata and colour identify centre 2 (UMCU) and centre 4 (LPON), but not which of centres 0, 1 and 3 is RUMC; the WILDS data come from a community mirror.
- Results so far are from cross-validation and validation only.
- From S1 f on, every arm uses one seed, and LoRA is a single run: seed variance of the ladder arms is unmeasured (sensitivity study postponed).
- Seed-count confounder: the original's validation BA averages 5 seeds (per-seed SD 0.15 pt), and the same recipe with seed 0 alone scores -0.15 pt [-0.35, -0.02] on validation. One-seed head arms are compared with that 5-seed average, so their Δ valid vs. the original includes a seed term of this size; no adoption decision depends on it.
- S2's logistic regression chose the smallest C on the pre-registered grid in every fold, so the optimum may lie outside the grid.
- Cleaning mask: S5 and H4 trained on folds 1–4 minus 7,131 patches flagged by S3b's full-train OOF predictions, whose heads also saw fold-0 labels. S3b's nested fold-0 mask drops 5,412; 2,687 of 210,622 patches differ. This is a small label leak into the fold-0 comparisons (S5 vs. the original or S3b, S5b selection); the validation comparisons use no validation label and are unaffected.
- Stain augmentation strength: σ = 0.05 in scikit-image stain units (optical density divided by ln(10⁶)) shifts each stain by up to 0.69 in natural-log optical density (0.30 in base-10 OD); random draws change pixels by 36/255 on average, and a shift of β = +σ on all three stains darkens mean RGB from 162 to 65. Tellez et al. 2018 (App. B, Eqs. 5–7) perturb unscaled optical density, where the same σ shifts a stain by at most 0.05, so the additive shift here is ln(10⁶) = 13.8× theirs if their logarithm is natural (6× if it is base 10; the paper does not state the base). That is stronger than the name "HED-light" (Tellez et al. 2019) suggests. H4 is matched to LoRA as pre-registered, but its frozen head sees only recoloured patches and is scored on clean ones, which may widen the LoRA − H4 gap.
- LoRA sweep: the EMA (decay 0.999, no warm-up) still held most of the starting weights at the short rungs; 3 of 6 first-rung configs scored identically, and the final pick won by 0.011 pt (about one of the 10k checkpoint patches), so the lr/rank choice is close to arbitrary.
- Also fixed: the LoRA adapter initialisation is now seeded, the LoRA head runs in fp32 (logits were bf16; rounding keeps each logit's sign, so BA at the 0.5 threshold is unaffected, but NLL and ECE are coarsened), predictions are order-safe for unsorted indices, and test images, labels and meta are guarded by the seal.

= Conclusion
The published Hibou-L PCam probe is reproducible to within 0.0004 BA with an independent, remote-code-free implementation. Two data issues that would have contaminated evaluation (a duplicated test slide and an in-distribution WILDS test hospital) were found before any test data was used: the first is handled by a sensitivity analysis, the second by re-deciding the out-of-hospital pool before WILDS is unsealed. Of the 2 adopted changes, only `S5_lora` improves validation BA with a 95% CI that excludes zero. Whether the gains hold on the test split and across hospitals is the next, sealed, question.

= References
[1] B. S. Veeling et al. Rotation equivariant CNNs for digital pathology. MICCAI 2018. #link("https://arxiv.org/abs/1806.03962")[https://arxiv.org/abs/1806.03962] \
[2] B. Ehteshami Bejnordi et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. JAMA 2017. #link("https://jamanetwork.com/journals/jama/fullarticle/2665774")[https://jamanetwork.com/journals/jama/fullarticle/2665774] \
[3] G. Litjens et al. 1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset. GigaScience 2018. #link("https://academic.oup.com/gigascience/article/7/6/giy065/5026175")[https://academic.oup.com/gigascience/article/7/6/giy065/5026175] \
[4] D. Nechaev, A. Pchelnikov, E. Ivanova. Hibou: A family of foundational vision transformers for pathology. 2024. #link("https://arxiv.org/abs/2406.05074")[https://arxiv.org/abs/2406.05074] \
[5] kaiko.ai. eva: evaluation framework for pathology foundation models (leaderboard and PCam config). #link("https://kaiko-ai.github.io/eva/main/leaderboards/")[https://kaiko-ai.github.io/eva/main/leaderboards/] \
[6] M. Karasikov et al. Training state-of-the-art pathology foundation models with orders of magnitude less data. 2025. #link("https://arxiv.org/abs/2504.05186")[https://arxiv.org/abs/2504.05186] \
[7] E. Zimmermann et al. Virchow2: Scaling self-supervised mixed magnification models in pathology. 2024. #link("https://arxiv.org/abs/2408.00738")[https://arxiv.org/abs/2408.00738] \
[8] H. Xu et al. A whole-slide foundation model for digital pathology from real-world data. Nature 2024. #link("https://doi.org/10.1038/s41586-024-07441-w")[https://doi.org/10.1038/s41586-024-07441-w] \
[9] M. Oquab et al. DINOv2: Learning robust visual features without supervision. TMLR 2024. #link("https://arxiv.org/abs/2304.07193")[https://arxiv.org/abs/2304.07193] \
[10] T. Darcet et al. Vision transformers need registers. ICLR 2024. #link("https://arxiv.org/abs/2309.16588")[https://arxiv.org/abs/2309.16588] \
[11] J. Lee et al. Benchmarking pathology foundation models: adaptation strategies and scenarios. 2024. #link("https://arxiv.org/abs/2410.16038")[https://arxiv.org/abs/2410.16038] \
[12] E. J. Hu et al. LoRA: Low-rank adaptation of large language models. ICLR 2022. #link("https://arxiv.org/abs/2106.09685")[https://arxiv.org/abs/2106.09685] \
[13] A. Kumar et al. Fine-tuning can distort pretrained features and underperform out-of-distribution. ICLR 2022. #link("https://arxiv.org/abs/2202.10054")[https://arxiv.org/abs/2202.10054] \
[14] M. Wortsman et al. Robust fine-tuning of zero-shot models. CVPR 2022. #link("https://arxiv.org/abs/2109.01903")[https://arxiv.org/abs/2109.01903] \
[15] M. Wortsman et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. ICML 2022. #link("https://arxiv.org/abs/2203.05482")[https://arxiv.org/abs/2203.05482] \
[16] L. Ciernik, M. Morik, L. Thede, L. Eyring, S. Nakajima, Z. Akata, L. Muttenthaler. Attentive multilayer fusion for vision transformers. 2026. #link("https://arxiv.org/abs/2601.09322")[https://arxiv.org/abs/2601.09322] \
[17] R. Caruana, A. Niculescu-Mizil, G. Crew, A. Ksikes. Ensemble selection from libraries of models. ICML 2004. #link("https://doi.org/10.1145/1015330.1015432")[https://doi.org/10.1145/1015330.1015432] \
[18] D. Tellez et al. Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. Medical Image Analysis 58:101544, 2019. #link("https://doi.org/10.1016/j.media.2019.101544")[https://doi.org/10.1016/j.media.2019.101544] \
[19] P. W. Koh et al. WILDS: A benchmark of in-the-wild distribution shifts. ICML 2021. #link("https://arxiv.org/abs/2012.07421")[https://arxiv.org/abs/2012.07421] \
[20] B. Efron. Better bootstrap confidence intervals. JASA 1987. #link("https://doi.org/10.1080/01621459.1987.10478410")[https://doi.org/10.1080/01621459.1987.10478410] \
[21] S. Holm. A simple sequentially rejective multiple test procedure. Scand. J. Stat. 1979. #link("https://www.jstor.org/stable/4615733")[https://www.jstor.org/stable/4615733] \
[22] F. Gröger et al. Intrinsic self-supervision for data quality audits (SelfClean). NeurIPS 2024 Datasets and Benchmarks Track (arXiv v3, App. L.1, Table 16: automatic-mode estimates). #link("https://arxiv.org/abs/2305.17048")[https://arxiv.org/abs/2305.17048] \
[23] Team PND. PANDA challenge 1st-place solution write-up (OOF-based label-noise removal); code at github.com/kentaroy47/Kaggle-PANDA-1st-place-solution. 2020. #link("https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169143")[https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169143] \
[24] A. Jaiswal et al. Semi-supervised learning for cancer detection of lymph node metastases. CVPR Workshops 2019 (Kaggle PCam data; TTA inside a model ensemble). #link("https://arxiv.org/abs/1906.09587")[https://arxiv.org/abs/1906.09587] \
[25] D. Morales-Brotons, T. Vogels, H. Hendrikx. Exponential moving average of weights in deep learning: dynamics and benefits. TMLR 2024. #link("https://arxiv.org/abs/2411.18704")[https://arxiv.org/abs/2411.18704] \
[26] D. Tomar, A. Binder, A. Kleppe. Are nuclear masks all you need for improved out-of-domain generalisation? A closer look at cancer classification in histopathology. NeurIPS 2024 (Sec. 4.1 states that CAMELYON17 centres 0, 3 and 4 share one scanner; the slide files disagree, Sec. 3). #link("https://arxiv.org/abs/2411.09373")[https://arxiv.org/abs/2411.09373] \
[27] Z. Mai et al. Lessons and insights from a unifying study of parameter-efficient fine-tuning (PEFT) in visual recognition. CVPR 2025. #link("https://arxiv.org/abs/2409.16434")[https://arxiv.org/abs/2409.16434] \
[28] M. Elphick, S. Turajlic, G. Yang. Are the latent representations of foundation models for pathology invariant to rotation? 2024. #link("https://arxiv.org/abs/2412.11938")[https://arxiv.org/abs/2412.11938] \
[29] J. Wang, Y. Cui, Y. Mao, N. Guan, C. J. Xue. Pre-processing matters: A segment search method for WSI classification. arXiv:2404.11161v1, 2024 (Sec. 2.1 names CAMELYON17 centres 0, 1, 4). #link("https://arxiv.org/abs/2404.11161v1")[https://arxiv.org/abs/2404.11161v1] \
[30] D. Tellez et al. Whole-slide mitosis detection in H&E breast histology using PHH3 as a reference to train distilled stain-invariant convolutional networks. IEEE TMI 37(9), 2018 (App. B: HED stain augmentation). #link("https://arxiv.org/abs/1808.05896")[https://arxiv.org/abs/1808.05896] \
[31] B. S. Veeling. PatchCamelyon (PCam) dataset README: label rule, sampling and file checksums. #link("https://github.com/basveeling/pcam")[https://github.com/basveeling/pcam] \
[32] CAMELYON16 data set README (lists test_049 as a duplicate slide removed from the data set). #link("https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/README.md")[https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/README.md] \
[33] C. A. Field, A. H. Welsh. Bootstrapping clustered data. JRSS-B 69(3):369–390, 2007. #link("https://doi.org/10.1111/j.1467-9868.2007.00593.x")[https://doi.org/10.1111/j.1467-9868.2007.00593.x] \
[34] Kaggle. Histopathologic Cancer Detection, competition description: "the original PCam dataset contains duplicate images due to its probabilistic sampling". 2018. #link("https://www.kaggle.com/c/histopathologic-cancer-detection")[https://www.kaggle.com/c/histopathologic-cancer-detection] \
[35] HistAI. Hibou-L on Hugging Face: config.json (4 registers, SwiGLU) and released weights (303.7M parameters in the safetensors metadata; the card and paper state no count). #link("https://huggingface.co/histai/hibou-L")[https://huggingface.co/histai/hibou-L] \

= Appendix
== LoRA hyperparameter sweep
Successive halving over learning rate × rank on the train-64k patches of folds 1–4; each rung scores the EMA weights on the 10k fold-0 checkpoint subsample. Selected: lr 0.0001, rank 16.
#[#set par(justify: false)
#set text(size: 9pt)
#table(columns: 4, stroke: 0.4pt + luma(200), inset: 4pt, align: left,
  table.header([*Configuration*], [*Budget (epochs)*], [*Checkpoint BA*], [*img/s*]),
  [lr=5e-05,r=8], [0.25], [0.9783], [28.8],
  [lr=5e-05,r=16], [0.25], [0.9786], [28.8],
  [lr=0.0001,r=8], [0.25], [0.9783], [28.8],
  [lr=0.0001,r=16], [0.25], [0.9783], [28.8],
  [lr=0.0003,r=8], [0.25], [0.9783], [28.8],
  [lr=0.0003,r=16], [0.25], [0.9784], [28.8],
  [lr=5e-05,r=16], [0.5], [0.9785], [31.7],
  [lr=0.0003,r=16], [0.5], [0.9787], [31.7],
  [lr=0.0001,r=16], [0.5], [0.9788], [31.7])]

== All arms
#[#set par(justify: false)
#set text(size: 9pt)
#table(columns: 5, stroke: 0.4pt + luma(200), inset: 4pt, align: left,
  table.header([*Arm*], [*Backbone*], [*OOF BA [95% CI] (patches)*], [*Valid BA [95% CI]*], [*Valid AUROC*]),
  [control\_​color\_​histogram], [—], [0.7870 \[0.7580, 0.8106\] (all train)], [0.8098 \[0.7530, 0.8472\]], [0.8923],
  [control\_​majority], [—], [—], [0.5000 \[0.5000, 0.5000\]], [0.5000],
  [control\_​shuffled\_​labels], [hibou\_​l], [0.5204 \[0.4935, 0.5457\] (all train)], [0.4731 \[0.4280, 0.5212\]], [0.4637],
  [H4\_​aug\_​control], [hibou\_​l], [0.9731 \[0.9568, 0.9823\] (fold0)], [0.9344 \[0.8853, 0.9627\]], [0.9780],
  [S0\_​eva\_​exact\_​hibou\_​l], [hibou\_​l], [—], [0.9396 \[0.8936, 0.9638\]], [0.9833],
  [S0\_​hibou\_​l\_​cls\_​linear], [hibou\_​l], [0.9498 \[0.9366, 0.9600\] (all train)], [0.9367 \[0.8888, 0.9617\]], [0.9823],
  [S0\_​scan\_​dinov2\_​reg\_​l], [dinov2\_​reg\_​l], [0.8797 \[0.8622, 0.8940\] (all train)], [0.8682 \[0.8103, 0.9023\]], [0.9448],
  [S0\_​scan\_​random\_​vitl\_​reg4], [random\_​vitl\_​reg4], [0.7973 \[0.7692, 0.8201\] (all train)], [0.7902 \[0.7232, 0.8334\]], [0.8721],
  [S1\_​b\_​cls24\_​mean24], [hibou\_​l], [0.9491 \[0.9353, 0.9598\] (all train)], [0.9353 \[0.8830, 0.9607\]], [0.9816],
  [S1\_​c\_​dinov2\_​last4], [hibou\_​l], [0.9457 \[0.9303, 0.9573\] (all train)], [0.9316 \[0.8712, 0.9593\]], [0.9796],
  [S1\_​d\_​spread\_​6\_​12\_​18\_​24], [hibou\_​l], [0.9469 \[0.9318, 0.9585\] (all train)], [0.9326 \[0.8786, 0.9594\]], [0.9787],
  [S1\_​e\_​best\_​plus\_​centre\_​tokens], [hibou\_​l], [0.9496 \[0.9361, 0.9599\] (all train)], [0.9352 \[0.8839, 0.9608\]], [0.9817],
  [S1\_​f\_​best\_​plus\_​centre\_​view], [hibou\_​l], [0.9497 \[0.9362, 0.9599\] (all train)], [0.9354 \[0.8889, 0.9609\]], [0.9818],
  [S2\_​logreg], [hibou\_​l], [0.9510 \[0.9378, 0.9610\] (all train)], [0.9365 \[0.8886, 0.9618\]], [0.9827],
  [S2\_​mlp], [hibou\_​l], [0.9312 \[0.9112, 0.9462\] (all train)], [0.9054 \[0.8227, 0.9423\]], [0.9741],
  [S3\_​d4\_​tta], [hibou\_​l], [0.9522 \[0.9396, 0.9617\] (train64k)], [0.9343 \[0.8836, 0.9604\]], [0.9823],
  [S3\_​no\_​tta\_​train64k], [hibou\_​l], [0.9517 \[0.9391, 0.9615\] (train64k)], [0.9352 \[0.8854, 0.9612\]], [0.9822],
  [S3b\_​oof\_​label\_​cleaning], [hibou\_​l], [0.9525 \[0.9394, 0.9624\] (all train)], [0.9368 \[0.8908, 0.9636\]], [0.9833],
  [S5\_​lora], [hibou\_​l], [0.9798 \[0.9654, 0.9883\] (fold0)], [0.9444 \[0.8962, 0.9695\]], [0.9848],
  [S5b\_​wise\_​0.25], [hibou\_​l], [0.9782 \[0.9638, 0.9872\] (fold0\_​minus\_​ckpt)], [0.9388 \[0.8914, 0.9663\]], [0.9827],
  [S5b\_​wise\_​0.5], [hibou\_​l], [0.9789 \[0.9644, 0.9879\] (fold0\_​minus\_​ckpt)], [0.9416 \[0.8935, 0.9681\]], [0.9837],
  [S5b\_​wise\_​0.75], [hibou\_​l], [0.9798 \[0.9655, 0.9885\] (fold0\_​minus\_​ckpt)], [0.9427 \[0.8945, 0.9686\]], [0.9843])]

= AI statement
This codebase and paper were written by Claude Code (Anthropic), executing a specification provided by a human author, Libor Burian. Human cleanup and rewriting are in progress.