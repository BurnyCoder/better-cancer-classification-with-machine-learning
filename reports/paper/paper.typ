#set document(title: "Hill-climbing Hibou-L on PatchCamelyon", author: "Libor Burian")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show link: set text(fill: rgb("#2a78d6"))
#align(center)[#text(16pt)[*Hill-climbing Hibou-L on PatchCamelyon:*] \ #text(13pt)[a pre-registered, slide-clustered study. Release 2 (in progress): Hibou-L improvement ladder]]
#align(center)[Libor "Burny" Burian \ #link("https://github.com/BurnyCoder/better-cancer-classification-with-machine-learning")[github.com/BurnyCoder/better-cancer-classification-with-machine-learning]]

*Abstract.* Pathology foundation models are usually compared on PatchCamelyon (PCam) with a frozen backbone and a linear probe. The best published such result belongs to Hibou-L [4]: 0.955 test balanced accuracy (BA) on the eva leaderboard [5]. We pre-registered a ladder of interventions to improve it, each transferred from neighbouring work: intermediate-layer and centre-region features, head choice, rotation/flip test-time augmentation, out-of-fold label cleaning, multi-model fusion, and LoRA fine-tuning with LP-FT, stain augmentation and weight-space averaging. Every step is selected on slide-grouped cross-validation, gated on validation, and the test split stays sealed until the ladder is frozen. This first release reports the foundation of that study. We reproduce the eva Hibou-L probe exactly: validation BA 0.9394 ± 0.0007 over five seeds, against the published 0.939. Our native-transformers implementation matches Hibou's own code to a mean cosine of 1.000000. Under the cross-validation protocol the incumbent reaches 0.9498 [0.9366, 0.9600] out-of-fold BA. Controls behave as expected: shuffled labels give 0.520 and a colour-histogram classifier gives 0.787. A pixel-hash audit found 33 test images identical to train images, all from PCam test slide `camelyon16_test_049` and train slide `camelyon16_train_tumor_036` at identical coordinates, which indicates a duplicate scan. We also find that the official Camelyon17-WILDS test hospital matches the colour signature of one of PCam's own source hospitals. Both findings changed the pre-registered evaluation before any test data was touched. Improvement ladder so far: 7 classifiers trained on frozen Hibou-L features; the best paired out-of-fold difference from the original probe is +0.12 pt BA; steps decided: 2, adopted: 0.

= Introduction
Sentinel lymph-node metastasis detection is a labour-intensive step in breast-cancer staging [2]. PCam [1] recasts it as 327,680 patches of 96×96 px, labelled positive if the centre 32×32 px contains tumour. Self-supervised pathology foundation models (FMs) trained with DINOv2 [9] now dominate PCam under frozen-probe protocols [5, 6]. Many adaptation techniques are each reported to help elsewhere: LoRA [12, 11], LP-FT [13], WiSE-FT and model soups [14, 15], multi-layer probing [16], ensembles [17], label-noise cleaning [23], and stain augmentation [18]. Their joint value on top of the strongest model is unknown. This study measures it with a pre-registered ladder. It uses statistics that resample whole slides, because patches from one slide are correlated, and a test split sealed in code. This document is release 1: reproduction, controls, no-training baselines and data audit. Later releases add the ladder, fine-tuning and the unsealed test.

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

= Results (release 1)
== Data audit
No slide is shared between splits. There are many exact duplicates *within* splits (train 42,119, valid 4,660, test 3,385), consistent with PCam's probabilistic sampling. Slide-grouped folds keep them inside one fold. Across splits, 33 test images are pixel-identical to train images. All of them come from `camelyon16_test_049` and `camelyon16_train_tumor_036`, at identical slide coordinates. CAMELYON16 documentation lists test_049 as a duplicate slide that was removed from the official release, yet PCam's test split contains it. With the test split still sealed, we added a pre-registered sensitivity analysis that excludes this slide.
== Exact reproduction of the published probe (H2)
Against eva running Hibou's remote code on 256 validation patches, our fp32 embeddings agree to a mean cosine of 1.000000 (max |Δ| 6.8e-06). In bf16 the mean cosine is 0.9994, but the minimum is 0.9336: bf16 noticeably perturbs a few patches. The eva-exact probe reaches validation BA 0.9394 ± 0.0007 (five seeds), against the published 0.939. This passes the pre-registered ±0.010 gate; the seed-averaged prediction gives 0.9396 [0.8936, 0.9638].
== Incumbent under the CV protocol and controls
#table(columns: 3, stroke: 0.4pt + luma(200), inset: 4pt, align: left,
  table.header([*Arm*], [*Train OOF BA [95% CI]*], [*Valid BA [95% CI]*]),
  [Reproduced Hibou-L probe, eva-exact], [—], [0.9396 \[0.8936, 0.9638\]],
  [Hibou-L CLS + linear, CV protocol (S0)], [0.9498 \[0.9366, 0.9600\]], [0.9367 \[0.8888, 0.9617\]],
  [Hibou-L, shuffled train labels], [0.5204 \[0.4935, 0.5457\]], [0.4731 \[0.4280, 0.5212\]],
  [Colour histogram + logistic regression], [0.7870 \[0.7580, 0.8106\]], [0.8098 \[0.7530, 0.8472\]],
  [Majority class], [—], [0.5000 \[0.5000, 0.5000\]])

The validation CI is 3.1× wider than the OOF CI, because validation has only 54 slides (22 with tumour). This is why validation serves only as a no-harm gate and selection uses train CV. Shuffled labels land at chance, so there is no label-leakage path. Colour alone is strongly informative (tumour is hyperchromatic) but far below the foundation model.
#figure(image("../figures/stage_a.png", width: 92%), caption: [Reproduced Hibou-L probe and controls. Dots are seed-averaged BA; bars are 95% stratified slide-bootstrap BCa CIs.])
== No-training baselines
@pub shows the eva leaderboard [5] (pinned commit) with our reproduction. Literature results on the official test split, which use different metrics and protocols and are context only: P4M-DenseNet (rotation-equivariant CNN, trained from scratch): accuracy 0.8980 (Veeling et al. 2018, Table 1); P4M-DenseNet: AUROC 0.9630 (Veeling et al. 2018, Table 1); Hibou-L, SGD linear probe: accuracy 0.9530 (Nechaev et al. 2024, Table 1); Hibou-B, SGD linear probe: accuracy 0.9460 (Nechaev et al. 2024, Table 1); UNI, linear probe: accuracy 0.9160 (Lee et al. 2024, PCam results); UNI, full fine-tuning: accuracy 0.9040 (Lee et al. 2024, PCam results); UNI, LoRA: accuracy 0.9424 (Lee et al. 2024, PCam results).
#figure(image("../figures/published_baselines.png", width: 88%), caption: [Published frozen-probe PCam BA (eva, validation and test) and our reproduced Hibou-L validation BA. Among 15 leaderboard models, Hibou-L ranks 2 on validation and 1 on test.]) <pub>
== Camelyon17-WILDS preparation
The official WILDS download failed (expired TLS certificate), so we use a Hugging Face mirror. Its split sizes match the WILDS paper [19]. We fixed a subsample of ≤1,000 patches per slide (49,888 patches from 50 slides). PCam train slides grouped by the assumed Camelyon16 numbering (RUMC vs. UMCU) differ in colour (L1 1.30) far more than a random split (0.14). WILDS centre 2, the official WILDS *test* hospital, is closest to UMCU (0.69 vs. 1.68). It is therefore a Camelyon16 source hospital and not out-of-distribution for PCam-trained models. Our primary OOD pool is centres 0, 1 and 4 (CWZ, RST, LPON [27, 3]).

== Improvement ladder (release 2, in progress)
Hibou-L stays frozen in S1–S3b; each candidate trains a new classifier on features read from it (5 slide-grouped folds × 5 seeds). Each classifier is compared with the original probe, paired on the same patches. Per-candidate p-values are descriptive; the pre-registered adoption rule applies Holm within a step and the validation no-harm gate.
#table(columns: 6, stroke: 0.4pt + luma(200), inset: 4pt, align: left,
  table.header([*Step*], [*Classifier*], [*OOF BA [95% CI]*], [*Δ OOF vs original [95% CI] (pt)*], [*p*], [*Δ valid (pt)*]),
  [S1 representation], [S1\_b\_cls24\_mean24], [0.9491 \[0.9353, 0.9598\]], [-0.07 \[-0.24, +0.06\]], [0.426], [-0.14],
  [S1 representation], [S1\_c\_dinov2\_last4], [0.9457 \[0.9303, 0.9573\]], [-0.40 \[-0.89, -0.12\]], [0.0284], [-0.51],
  [S1 representation], [S1\_d\_spread\_6\_12\_18\_24], [0.9469 \[0.9318, 0.9585\]], [-0.29 \[-0.69, -0.04\]], [0.0931], [-0.41],
  [S1 representation], [S1\_e\_best\_plus\_centre\_tokens], [0.9496 \[0.9361, 0.9599\]], [-0.01 \[-0.14, +0.11\]], [0.826], [-0.15],
  [S1 representation], [S1\_f\_best\_plus\_centre\_view], [0.9497 \[0.9362, 0.9599\]], [-0.01 \[-0.15, +0.17\]], [0.944], [-0.13],
  [S2 head], [S2\_logreg], [0.9510 \[0.9378, 0.9610\]], [+0.12 \[+0.04, +0.31\]], [0.0236], [-0.02],
  [S2 head], [S2\_mlp], [0.9312 \[0.9112, 0.9462\]], [-1.85 \[-2.81, -1.17\]], [0.0001], [-3.13])

#figure(image("../figures/ladder_deltas.png", width: 92%), caption: [Every trained ladder classifier vs. the original Hibou-L probe: paired Δ BA with 95% slide-bootstrap CIs (train OOF and validation).])
- *S1_representation*: best candidate `S1_f_best_plus_centre_view`; OOF test passed False, valid gate passed False → *rejected*.
- *S2_head*: best candidate `S2_logreg`; OOF test passed True, valid gate passed False → *rejected*.

= Planned and postponed analyses
The pre-registered ladder follows. Steps decided so far: 2.
- S1 representation: last-layer CLS, CLS+mean, DINOv2 last-4-layer [9], spread layers [16], centre tokens and a centre-crop view.
- S2 heads: linear, nested-CV logistic regression, MLP.
- S3 D4 TTA [24, 26]; S3b OOF label cleaning [23, 22].
- S4 fusion with other FMs [17, 8, 7].
- S5 LoRA [12, 11] with LP-FT [13], HED augmentation [18], drop-path and EMA [25]; S5b WiSE scaling and soups [14, 15].
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

= Conclusion (release 1)
The published Hibou-L PCam probe is reproducible to within 0.001 BA with an independent, remote-code-free implementation. The study design is in place: slide-clustered statistics, a sealed test and pre-registered decisions. Two data issues that would have contaminated evaluation (a duplicated test slide and an in-distribution WILDS test hospital) were caught and handled before any test data was used.

= AI statement
This codebase and paper were written by Claude Code (Anthropic), executing a specification provided by a human author, Libor Burian. Human cleanup and rewriting are in progress.

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
[24] Kaggle Histopathologic Cancer Detection solution write-up (TTA and ensembling). 2019. #link("https://arxiv.org/abs/1906.09587")[https://arxiv.org/abs/1906.09587] \
[25] Z. Mai et al. Lessons and insights from a unifying study of parameter-efficient fine-tuning in visual recognition. 2024. #link("https://arxiv.org/abs/2409.16434")[https://arxiv.org/abs/2409.16434] \
[26] Are the latent representations of foundation models for pathology invariant to rotation? 2024. #link("https://arxiv.org/abs/2412.11938")[https://arxiv.org/abs/2412.11938] \
[27] Pre-processing matters: a segment search method for WSI classification (CAMELYON17 centre names). 2024. #link("https://arxiv.org/abs/2404.11161")[https://arxiv.org/abs/2404.11161] \