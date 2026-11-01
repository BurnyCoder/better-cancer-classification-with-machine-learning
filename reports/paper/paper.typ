#set document(title: "Hill-climbing Hibou-L on PatchCamelyon", author: "Libor Burian")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show link: set text(fill: rgb("#2a78d6"))
#align(center)[#text(16pt)[*Hill-climbing Hibou-L on PatchCamelyon:*] \ #text(13pt)[a pre-registered, slide-clustered study. Release 3: improvement ladder and control models]]
#align(center)[Libor "Burny" Burian \ #link("https://github.com/BurnyCoder/better-cancer-classification-with-machine-learning")[github.com/BurnyCoder/better-cancer-classification-with-machine-learning]]

*Abstract.* Pathology foundation models are usually compared on PatchCamelyon (PCam) with a frozen backbone and a linear probe. The best published such result belongs to Hibou-L [4]: 0.955 test balanced accuracy (BA) on the eva leaderboard [5]. We pre-registered a ladder of interventions to improve it, each transferred from neighbouring work: intermediate-layer and centre-region features, head choice, rotation/flip test-time augmentation, out-of-fold label cleaning, multi-model fusion, and LoRA fine-tuning with LP-FT, stain augmentation and weight-space averaging. Every step is selected on slide-grouped cross-validation, gated on validation, and the test split stays sealed until the ladder is frozen. We reproduce the eva Hibou-L probe exactly: validation BA 0.9394 ± 0.0007 over five seeds, against the published 0.939. Our native-transformers implementation matches Hibou's own code to a mean cosine of 1.000000. Under the cross-validation protocol the incumbent reaches 0.9498 [0.9366, 0.9600] out-of-fold BA. Controls behave as expected: shuffled labels give 0.520 and a colour-histogram classifier gives 0.787. A pixel-hash audit found 33 test images identical to train images, all from PCam test slide `camelyon16_test_049` and train slide `camelyon16_train_tumor_036` at identical coordinates, which indicates a duplicate scan. We also find that the official Camelyon17-WILDS test hospital matches the colour signature of one of PCam's own source hospitals. Both findings changed the pre-registered evaluation before any test data was touched. Of 6 decided ladder steps, 2 were adopted (`S3b_label_cleaning`, `S5_lora`). No frozen-feature classifier has a validation gain whose 95% CI excludes zero (largest: +0.01 pt [-0.76, +0.40], `S3b_oof_label_cleaning`). LoRA fine-tuning raises validation BA from 0.9367 to 0.9444 (paired Δ +0.77 pt [+0.27, +1.41]). LoRA also beats a frozen head trained on the same images with the same augmentation (Δ valid +0.99 pt [+0.58, +1.51]). With Hibou-L's architecture, natural-image weights reach 0.880 and random weights 0.797 OOF BA, against 0.950 for Hibou-L. The test split and WILDS remain sealed.

= Introduction
Sentinel lymph-node metastasis detection is a labour-intensive step in breast-cancer staging [2]. PCam [1] recasts it as 327,680 patches of 96×96 px, labelled positive if the centre 32×32 px contains tumour. Self-supervised pathology foundation models (FMs) trained with DINOv2 [9] now dominate PCam under frozen-probe protocols [5, 6]. Many adaptation techniques are each reported to help elsewhere: LoRA [12, 11], LP-FT [13], WiSE-FT and model soups [14, 15], multi-layer probing [16], ensembles [17], label-noise cleaning [23], and stain augmentation [18]. Their joint value on top of the strongest model is unknown. This study measures it with a pre-registered ladder. It uses statistics that resample whole slides, because patches from one slide are correlated, and a test split sealed in code. This document is release 3 of the study; each release adds the classifiers trained since the last one, and the test split is unsealed only after the ladder is frozen.

= Methods
== Data and audit
We use the official PCam splits from Zenodo record 2546921, md5-verified against the Zenodo API. Note that the GitHub README lists the validation and test image checksums swapped. Train has 262,144 patches from 216 slides (88 with tumour), validation 32,768 / 54 (22), and test 32,768 / 129 (49). The audit checks that no slide is shared between splits (0 train–test, 0 train–valid, 0 valid–test), that the labels agree with the meta CSVs, and that there are no exact pixel duplicates within or across splits (BLAKE2b hashes).
== Model, access and licences
Hibou-L is a DINOv2 ViT-L/14 with 4 registers [10] and 304M parameters, pretrained on private HistAI slides [4]. It ships custom code; we load its pinned weights into transformers' native `Dinov2WithRegistersModel` (no remote code) and test equivalence against kaiko-eva running the remote code. The candidate pool is every model whose Hugging Face licence the author accepted, frozen before any embedding: 12 of 13 registry entries. Derivatives of non-redistributable models (Virchow2, Phikon) are never published; only aggregate metrics are.
== Features, heads and protocol
Patches are resized 96→224 px (bilinear), scaled and normalised with Hibou's eva mean and std. Embeddings are computed in bf16 and cached in fp16. The *eva-exact* probe is `nn.Linear` with BCE loss, AdamW at lr 3e-4, batch 256, at most 12,500 steps, early stopping on validation BA (patience 3), and five seeds. Ladder decisions use a *CV protocol*: five slide-grouped folds of train, stratified by slide tumour status, out-of-fold (OOF) predictions with fixed 12,500 steps, and full-train refits scored on validation.
== Statistics
Every arm's prediction is its seed-averaged probability; BA uses a fixed 0.5 threshold. 95% CIs come from a stratified slide-cluster bootstrap (tumour and normal slides resampled separately; 10,000 replicates for BA; BCa with a leave-one-slide-out jackknife [20]). Paired differences use a per-slide sign-flip randomisation test on the exact decomposition ΔBA = Σ#sub[s] [ΔTP#sub[s]/2P + ΔTN#sub[s]/2N], with Holm correction [21]. The test split and WILDS are sealed: reading them requires a git tag whose committed hash matches the frozen ladder configuration.

#figure(image("../figures/pipeline.png", width: 100%), caption: [Study pipeline. Statistics resample whole slides; the test split and WILDS are scored once, after the ladder is frozen.])

= Results
== Data audit
No slide is shared between splits. There are many exact duplicates *within* splits (train 42,119, valid 4,660, test 3,385), consistent with PCam's probabilistic sampling. Slide-grouped folds keep them inside one fold. Across splits, 33 test images are pixel-identical to train images. All of them come from `camelyon16_test_049` and `camelyon16_train_tumor_036`, at identical slide coordinates. Secondary sources (the eva documentation) report that a duplicate CAMELYON16 test slide was removed from the official release; PCam's test split still contains test_049. With the test split still sealed, we added a pre-registered sensitivity analysis that excludes this slide.
== Exact reproduction of the published probe (H2)
Against eva running Hibou's remote code on 256 validation patches, our fp32 embeddings agree to a mean cosine of 1.000000 (max |Δ| 6.8e-06). In bf16 the mean cosine is 0.9994, but the minimum is 0.9336: bf16 noticeably perturbs a few patches. The eva-exact probe reaches validation BA 0.9394 ± 0.0007 (five seeds), against the published 0.939. This passes the pre-registered ±0.010 gate; the seed-averaged prediction gives 0.9396 [0.8936, 0.9638].
== Incumbent under the CV protocol and controls
#table(columns: 3, stroke: 0.4pt + luma(200), inset: 4pt, align: left,
  table.header([*Arm*], [*Train OOF BA [95% CI]*], [*Valid BA [95% CI]*]),
  [Reproduced Hibou-L probe, eva-exact], [—], [0.9396 \[0.8936, 0.9638\]],
  [Hibou-L CLS + linear, CV protocol (S0)], [0.9498 \[0.9366, 0.9600\]], [0.9367 \[0.8888, 0.9617\]],
  [Hibou-L, shuffled train labels], [0.5204 \[0.4935, 0.5457\]], [0.4731 \[0.4280, 0.5212\]],
  [Colour histogram + logistic regression], [0.7870 \[0.7580, 0.8106\]], [0.8098 \[0.7530, 0.8472\]],
  [Majority class], [—], [0.5000 \[0.5000, 0.5000\]],
  [DINOv2-L/14-reg4 with natural-image weights], [0.8797 \[0.8622, 0.8940\]], [0.8682 \[0.8103, 0.9023\]],
  [ViT-L/14-reg4 with random weights, seed 0], [0.7973 \[0.7692, 0.8201\]], [0.7902 \[0.7232, 0.8334\]])

The validation CI is 3.1× wider than the OOF CI, because validation has only 54 slides (22 with tumour). This is why validation serves only as a no-harm gate and selection uses train CV. Shuffled labels land at chance, so there is no label-leakage path. Colour alone is strongly informative (tumour is hyperchromatic) but far below the foundation model. The two same-architecture controls separate what the weights contribute: natural-image DINOv2 weights recover part of the gap to Hibou-L, and random weights score close to the colour histogram.
#figure(image("../figures/stage_a.png", width: 92%), caption: [Reproduced Hibou-L probe and controls. Dots are seed-averaged BA; bars are 95% stratified slide-bootstrap BCa CIs.])
== No-training baselines
@pub shows the eva leaderboard [5] (pinned commit) with our reproduction. Literature results on the official test split, which use different metrics and protocols and are context only: P4M-DenseNet (rotation-equivariant CNN, trained from scratch): accuracy 0.8980 (Veeling et al. 2018, Table 1); P4M-DenseNet: AUROC 0.9630 (Veeling et al. 2018, Table 1); Hibou-L, SGD linear probe: accuracy 0.9530 (Nechaev et al. 2024, Table 1); Hibou-B, SGD linear probe: accuracy 0.9460 (Nechaev et al. 2024, Table 1); UNI, linear probe: accuracy 0.9160 (Lee et al. 2024, PCam results); UNI, full fine-tuning: accuracy 0.9040 (Lee et al. 2024, PCam results); UNI, LoRA: accuracy 0.9424 (Lee et al. 2024, PCam results).
#figure(image("../figures/published_baselines.png", width: 88%), caption: [Published frozen-probe PCam BA (eva, validation and test) and our reproduced Hibou-L validation BA. Among 15 leaderboard models, Hibou-L ranks 2 on validation and 1 on test.]) <pub>
== Camelyon17-WILDS preparation
The official WILDS download failed (expired TLS certificate), so we use a Hugging Face mirror. Its split sizes match the WILDS paper [19]. We fixed a subsample of ≤1,000 patches per slide (49,888 patches from 50 slides). PCam train slides grouped by the assumed Camelyon16 numbering (RUMC vs. UMCU) differ in colour (L1 1.30) far more than a random split (0.14). WILDS centre 2, the official WILDS *test* hospital, is closest to UMCU (0.69 vs. 1.68). It is therefore a Camelyon16 source hospital and not out-of-distribution for PCam-trained models. The pre-registered primary OOD pool is centres 0, 1 and 4, reported as CWZ, RST and LPON [29]. This mapping is uncertain: centres 0, 3 and 4 share one scanner [26], and the per-hospital scanners in [3] then place RUMC, a Camelyon16 source hospital, among centres 0, 3 and 4. The pool is therefore re-examined and logged before WILDS is unsealed.

== Improvement ladder
Hibou-L stays frozen in S1–S3b; each candidate trains a new classifier on features read from it (5 slide-grouped folds; 5 seeds for S1 b–e, 1 seed from S1 f on, by the author's decision; seed sensitivity is postponed). Each classifier is compared with the original probe, paired on the same patches. Per-candidate p-values are descriptive; the pre-registered adoption rule applies Holm within a step and the validation no-harm gate.
#table(columns: 7, stroke: 0.4pt + luma(200), inset: 4pt, align: left,
  table.header([*Step*], [*Classifier*], [*OOF BA [95% CI]*], [*OOF reference (patches)*], [*Δ OOF vs reference [95% CI] (pt)*], [*p*], [*Δ valid vs original (pt)*]),
  [H4 control (frozen, LoRA's augmentation)], [H4\_aug\_control], [0.9731 \[0.9568, 0.9823\]], [original (51,522)], [+0.20 \[-0.10, +0.84\]], [0.518], [-0.23],
  [S1 representation], [S1\_b\_cls24\_mean24], [0.9491 \[0.9353, 0.9598\]], [original (262,144)], [-0.07 \[-0.24, +0.06\]], [0.426], [-0.14],
  [S1 representation], [S1\_c\_dinov2\_last4], [0.9457 \[0.9303, 0.9573\]], [original (262,144)], [-0.40 \[-0.89, -0.12\]], [0.0284], [-0.51],
  [S1 representation], [S1\_d\_spread\_6\_12\_18\_24], [0.9469 \[0.9318, 0.9585\]], [original (262,144)], [-0.29 \[-0.69, -0.04\]], [0.0931], [-0.41],
  [S1 representation], [S1\_e\_best\_plus\_centre\_tokens], [0.9496 \[0.9361, 0.9599\]], [original (262,144)], [-0.01 \[-0.14, +0.11\]], [0.826], [-0.15],
  [S1 representation], [S1\_f\_best\_plus\_centre\_view], [0.9497 \[0.9362, 0.9599\]], [original (262,144)], [-0.01 \[-0.15, +0.17\]], [0.944], [-0.13],
  [S2 head], [S2\_logreg], [0.9510 \[0.9378, 0.9610\]], [original (262,144)], [+0.12 \[+0.04, +0.31\]], [0.0236], [-0.02],
  [S2 head], [S2\_mlp], [0.9312 \[0.9112, 0.9462\]], [original (262,144)], [-1.85 \[-2.81, -1.17\]], [0.0001], [-3.13],
  [S3 D4 TTA], [S3\_d4\_tta], [0.9522 \[0.9396, 0.9617\]], [S3\_no\_tta\_train64k (65,536)], [+0.04 \[-0.02, +0.13\]], [0.247], [-0.24],
  [S3 D4 TTA], [S3\_no\_tta\_train64k], [0.9517 \[0.9391, 0.9615\]], [original (65,536)], [+0.01 \[-0.03, +0.10\]], [0.779], [-0.15],
  [S3b label cleaning], [S3b\_oof\_label\_cleaning], [0.9525 \[0.9394, 0.9624\]], [original (262,144)], [+0.27 \[+0.10, +0.50\]], [0.0032], [+0.01],
  [S5 LoRA], [S5\_lora], [0.9798 \[0.9654, 0.9883\]], [original (51,522)], [+0.87 \[+0.50, +1.56\]], [0.0001], [+0.77],
  [S5b weight space], [S5b\_wise\_0.25], [0.9782 \[0.9638, 0.9872\]], [original (41,522)], [+0.72 \[+0.45, +1.22\]], [0.0001], [+0.21],
  [S5b weight space], [S5b\_wise\_0.5], [0.9789 \[0.9644, 0.9879\]], [original (41,522)], [+0.80 \[+0.48, +1.33\]], [0.0001], [+0.49],
  [S5b weight space], [S5b\_wise\_0.75], [0.9798 \[0.9655, 0.9885\]], [original (41,522)], [+0.88 \[+0.52, +1.49\]], [0.0001], [+0.60])

#figure(image("../figures/ladder_deltas.png", width: 92%), caption: [Every trained ladder classifier vs. the original Hibou-L probe: paired Δ BA with 95% slide-bootstrap CIs (train OOF and validation).])
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
*Frozen-probe benchmarks.* Pathology FMs are ranked on PCam with frozen backbones and linear probes [5, 6]; Hibou-L [4] leads the eva leaderboard, ahead of models trained on far larger or different cohorts [8, 7]. *Adapting FMs.* A benchmark of adaptation strategies reports LoRA above both linear probing and full fine-tuning for UNI on PCam [11]; LP-FT [13] and weight-space interpolation [14, 15] were proposed to keep pretrained features robust under shift. *Data-centric fixes.* Out-of-fold label cleaning won PANDA [23], SelfClean estimates PCam's label-error rate [22], TTA was part of strong ensembles on Kaggle's PCam data [24]; rotation invariance varies widely across pathology FMs, and Hibou-L, trained with rotations and flips [4], was the most invariant by cosine distance [28]. *Representations.* Multi-layer and attentive probes improve frozen natural-image ViTs [16, 9]. This study tests these ideas on the strongest frozen model under one pre-registered, slide-clustered protocol.

= Discussion
*Why frozen-feature tricks did not transfer.* Hibou-L starts near the frozen-probe ceiling, its pretraining already included rotations and flips, and the reported multi-layer gains were measured on natural-image ViTs. No frozen-feature change improved validation BA beyond its confidence interval; 4 have validation bootstrap CIs below zero (`S1_c_dinov2_last4` p = 0.19, `S1_d_spread_6_12_18_24` p = 0.068, `S2_mlp` p = 0.0012, `S3_d4_tta` p = 0.037; unadjusted sign-flip p, and one-seed arms include the seed term in Limitations). Label cleaning was adopted on cross-validation, but it removed about ten times more patches than SelfClean's label-error estimate; if that estimate holds, it also drops hard, correctly labelled patches.
*Why LoRA helped.* The H4 control targets the augmentation: a frozen head trained on the same augmented images is no better than the original, while LoRA is clearly better. This suggests that the gain comes from adapting the backbone's attention layers rather than from the augmentation, although the two arms also differ in optimiser and schedule, and the stain augmentation is strong (Limitations), which may handicap a frozen head scored on clean images more than LoRA. Shrinking the LoRA update (WiSE) lowered validation BA, more so the further it was shrunk; WiSE-FT targets robustness under distribution shift, which only the sealed WILDS evaluation can test.
*Strength of the evidence.* All results are on cross-validation and the 54-slide validation split, with one seed per arm and one LoRA run. They are hedged findings until the sealed test split and the out-of-hospital WILDS pool are scored once.

= Planned and postponed analyses
The pre-registered ladder follows. Steps decided so far: 6.
- S1 representation: last-layer CLS, CLS+mean, DINOv2 last-4-layer [9], spread layers [16], centre tokens and a centre-crop view.
- S2 heads: linear, nested-CV logistic regression, MLP.
- S3 D4 TTA [24, 28]; S3b OOF label cleaning [23, 22].
- S4 fusion with other FMs [17, 8, 7].
- S5 LoRA [12, 11] with LP-FT [13], HED augmentation [18], drop-path [27] and weight averaging (EMA) [25]; S5b WiSE scaling and soups [14, 15].
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
- The WILDS centre-to-hospital mapping rests on one paper plus a colour-signature check, and the WILDS data come from a community mirror.
- Results so far are from cross-validation and validation only.
- From S1 f on, every arm uses one seed, and LoRA is a single run: seed variance of the ladder arms is unmeasured (sensitivity study postponed).
- Seed-count confounder: the original's validation BA averages 5 seeds (per-seed SD 0.15 pt), and the same recipe with seed 0 alone scores -0.15 pt [-0.35, -0.02] on validation. One-seed linear-head arms are compared with that 5-seed average, so their Δ valid vs. the original includes a seed term of this size; no adoption decision depends on it.
- S2's logistic regression chose the smallest C on the pre-registered grid in every fold, so the optimum may lie outside the grid.
- Cleaning mask: S5 and H4 trained on folds 1–4 minus 7,131 patches flagged by S3b's full-train OOF predictions, whose heads also saw fold-0 labels. S3b's nested fold-0 mask drops 5,412; 2,687 of 210,622 patches differ. This is a small label leak into the fold-0 comparisons (S5 vs. the original or S3b, S5b selection); the validation comparisons use no validation label and are unaffected.
- Stain augmentation strength: σ = 0.05 in scikit-image stain units shifts each stain by up to 0.69 optical density; random draws change pixels by 36/255 on average, and the largest shift darkens mean RGB from 162 to 65. This is stronger than the name "HED-light" suggests (Tellez et al. do not state units). H4 is matched to LoRA as pre-registered, but its frozen head sees only recoloured patches and is scored on clean ones, which may widen the LoRA − H4 gap.
- LoRA sweep: the EMA (decay 0.999, no warm-up) still held most of the starting weights at the short rungs; 3 of 6 first-rung configs scored identically, and the final pick won by 0.011 pt (about one of the 10k checkpoint patches), so the lr/rank choice is close to arbitrary.
- Also fixed: the LoRA adapter initialisation is now seeded, the LoRA head runs in fp32 (logits were bf16, which coarsens NLL and ECE but not BA), predictions are order-safe for unsorted indices, and test images, labels and meta are guarded by the seal.

= Conclusion
The published Hibou-L PCam probe is reproducible to within 0.001 BA with an independent, remote-code-free implementation. Two data issues that would have contaminated evaluation (a duplicated test slide and an in-distribution WILDS test hospital) were caught and handled before any test data was used. Of 6 decided ladder steps, 2 were adopted (`S3b_label_cleaning`, `S5_lora`). No frozen-feature classifier has a validation gain whose 95% CI excludes zero (largest: +0.01 pt [-0.76, +0.40], `S3b_oof_label_cleaning`). LoRA fine-tuning raises validation BA from 0.9367 to 0.9444 (paired Δ +0.77 pt [+0.27, +1.41]). LoRA also beats a frozen head trained on the same images with the same augmentation (Δ valid +0.99 pt [+0.58, +1.51]). With Hibou-L's architecture, natural-image weights reach 0.880 and random weights 0.797 OOF BA, against 0.950 for Hibou-L. Whether the gains hold on the test split and across hospitals is the next, sealed, question.

= References
[1] B. S. Veeling et al. Rotation equivariant CNNs for digital pathology. MICCAI 2018. #link("https://arxiv.org/abs/1806.03962")[https://arxiv.org/abs/1806.03962] \
[2] B. Ehteshami Bejnordi et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. JAMA 2017. #link("https://jamanetwork.com/journals/jama/fullarticle/2665774")[https://jamanetwork.com/journals/jama/fullarticle/2665774] \
[3] G. Litjens et al. 1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset. GigaScience 2018. #link("https://academic.oup.com/gigascience/article/7/6/giy065/5026175")[https://academic.oup.com/gigascience/article/7/6/giy065/5026175] \
[4] D. Nechaev, A. Pchelnikov, E. Ivanova. Hibou: A family of foundational vision transformers for pathology. 2024. #link("https://arxiv.org/abs/2406.05074")[https://arxiv.org/abs/2406.05074] \
[5] kaiko.ai. eva: evaluation framework for pathology foundation models (leaderboard and PCam config). #link("https://github.com/kaiko-ai/eva")[https://github.com/kaiko-ai/eva] \
[6] M. Karasikov et al. Training state-of-the-art pathology foundation models with orders of magnitude less data. 2025. #link("https://arxiv.org/abs/2504.05186")[https://arxiv.org/abs/2504.05186] \
[7] E. Zimmermann et al. Virchow2: Scaling self-supervised mixed magnification models in pathology. 2024. #link("https://arxiv.org/abs/2408.00738")[https://arxiv.org/abs/2408.00738] \
[8] H. Xu et al. A whole-slide foundation model for digital pathology from real-world data. Nature 2024. #link("https://doi.org/10.1038/s41586-024-07441-w")[https://doi.org/10.1038/s41586-024-07441-w] \
[9] M. Oquab et al. DINOv2: Learning robust visual features without supervision. 2023. #link("https://arxiv.org/abs/2304.07193")[https://arxiv.org/abs/2304.07193] \
[10] T. Darcet et al. Vision transformers need registers. 2023. #link("https://arxiv.org/abs/2309.16588")[https://arxiv.org/abs/2309.16588] \
[11] J. Lee et al. Benchmarking pathology foundation models: adaptation strategies and scenarios. 2024. #link("https://arxiv.org/abs/2410.16038")[https://arxiv.org/abs/2410.16038] \
[12] E. J. Hu et al. LoRA: Low-rank adaptation of large language models. 2021. #link("https://arxiv.org/abs/2106.09685")[https://arxiv.org/abs/2106.09685] \
[13] A. Kumar et al. Fine-tuning can distort pretrained features and underperform out-of-distribution. ICLR 2022. #link("https://arxiv.org/abs/2202.10054")[https://arxiv.org/abs/2202.10054] \
[14] M. Wortsman et al. Robust fine-tuning of zero-shot models. CVPR 2022. #link("https://arxiv.org/abs/2109.01903")[https://arxiv.org/abs/2109.01903] \
[15] M. Wortsman et al. Model soups. ICML 2022. #link("https://arxiv.org/abs/2203.05482")[https://arxiv.org/abs/2203.05482] \
[16] Attentive multilayer fusion for vision transformers (arXiv:2601.09322). 2026. #link("https://arxiv.org/abs/2601.09322")[https://arxiv.org/abs/2601.09322] \
[17] R. Caruana et al. Ensemble selection from libraries of models. ICML 2004. #link("https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf")[https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf] \
[18] D. Tellez et al. Quantifying the effects of data augmentation and stain color normalization in CNNs for computational pathology. 2019. #link("https://arxiv.org/abs/1902.06543")[https://arxiv.org/abs/1902.06543] \
[19] P. W. Koh et al. WILDS: A benchmark of in-the-wild distribution shifts. ICML 2021. #link("https://arxiv.org/abs/2012.07421")[https://arxiv.org/abs/2012.07421] \
[20] B. Efron. Better bootstrap confidence intervals. JASA 1987. #link("https://doi.org/10.1080/01621459.1987.10478410")[https://doi.org/10.1080/01621459.1987.10478410] \
[21] S. Holm. A simple sequentially rejective multiple test procedure. Scand. J. Stat. 1979. #link("https://www.jstor.org/stable/4615733")[https://www.jstor.org/stable/4615733] \
[22] F. Gröger et al. Intrinsic self-supervision for data quality audits (SelfClean). 2023. #link("https://arxiv.org/abs/2305.17048")[https://arxiv.org/abs/2305.17048] \
[23] PANDA challenge 1st-place solution (PND): OOF-based label-noise removal. #link("https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution")[https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution] \
[24] A. Jaiswal et al. Semi-supervised learning for cancer detection of lymph node metastases. CVPR Workshops 2019 (Kaggle PCam data; TTA inside a model ensemble). #link("https://arxiv.org/abs/1906.09587")[https://arxiv.org/abs/1906.09587] \
[25] B. T. Polyak, A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM J. Control Optim. 1992. #link("https://doi.org/10.1137/0330046")[https://doi.org/10.1137/0330046] \
[26] D. Tomar, A. Binder, A. Kleppe. Are nuclear masks all you need for improved out-of-domain generalisation? NeurIPS 2024 (CAMELYON17 scanners per centre, Sec. 4.1). #link("https://arxiv.org/abs/2411.09373")[https://arxiv.org/abs/2411.09373] \
[27] Z. Mai et al. Lessons and insights from a unifying study of parameter-efficient fine-tuning in visual recognition. 2024. #link("https://arxiv.org/abs/2409.16434")[https://arxiv.org/abs/2409.16434] \
[28] M. Elphick, S. Turajlic, G. Yang. Are the latent representations of foundation models for pathology invariant to rotation? 2024. #link("https://arxiv.org/abs/2412.11938")[https://arxiv.org/abs/2412.11938] \
[29] J. Wang et al. BAHOP: Similarity-based basin hopping for a fast hyper-parameter search in WSI classification. 2024 (v1 lists CAMELYON17 centre names). #link("https://arxiv.org/abs/2404.11161")[https://arxiv.org/abs/2404.11161] \

= AI statement
This codebase and paper were written by Claude Code (Anthropic), executing a specification provided by a human author, Libor Burian. Human cleanup and rewriting are in progress.