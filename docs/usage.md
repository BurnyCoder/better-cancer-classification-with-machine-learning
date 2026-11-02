# Usage and data flow

## Phases
Every phase is a subcommand of `main.py`. Each one logs to `logs/<timestamp>_<phase>.log` and writes machine-readable output to `results/`.

| Command | What happens | Main outputs |
|---|---|---|
| `main.py download` | Downloads the 9 PCam files from Zenodo, checks their md5 sums against the Zenodo API, and gunzips the HDF5 files into `DATA_DIR/h5`. | `D:/AI/pcam/raw`, `D:/AI/pcam/h5` |
| `main.py verify` | Runs the leakage checks: no slide is shared between splits, labels match the meta CSV, exact pixel-hash duplicates within splits (copies, groups, label conflicts, groups spanning slides) and across splits (the slides behind them, whether they share slide coordinates, and the duplicated slide's patch counts). | `results/data_verification.json` |
| `main.py access` | Applies the pre-registered pool rule: `auth_check` for every registry model, frozen before any embedding. | `results/model_pool.json` |
| `main.py embed --backbones hibou_l --splits train valid [--views full center64 d4_1 ...] [--subset train64k]` | Runs each frozen backbone over the patches (bf16 on the GPU) into a resumable fp16 cache. Hibou-L caches CLS + patch-mean at blocks 6/12/18/21–24 plus the centre-6×6-token mean. Embedding the test split is refused until the ladder is frozen. | `EMB_DIR/<backbone>/<split>/<view>.npy`, `meta.json` |
| `main.py parity` | H2 parity check: our native-`transformers` Hibou-L against kaiko-eva's remote-code embeddings (computed by `tools/eva_parity/eva_embed.py` in an isolated environment with transformers 4.x). | `results/parity.json` |
| `main.py stage-a --parts h2 s0 controls scan` | Runs the H2 gate (eva-exact probe, early stopping on valid, 5 seeds), S0 (the original probe under 5-fold slide-grouped CV, 5 seeds), the controls, and the backbone scan (B\*, and the two control backbones). | `results/h2_gate.json`, `results/arms/*.json` |
| `main.py published` | Records the no-training baselines: the eva leaderboard CSV at a pinned commit, plus literature numbers. | `results/published_baselines.json` |
| `main.py stage-b --parts s1 s2 s3 s3b s4` | Runs the frozen-feature ladder: representation, head, D4 TTA, OOF label cleaning, fusion. S1, S2, S3b and S4 are decided by `ladder.decide_step`; S3 (the same heads with and without TTA) by `stage_b._record_decision`. Both apply the pre-registered adoption rule. | `results/steps/*.json`, `results/ladder.json` |
| `main.py embed-tta [--batch-size 128]` | Embeds the 7 non-identity D4 views of the incumbent's views for train-64k and valid (needed by S3). | `EMB_DIR/hibou_l/<split>/d4_k.npy` |
| `main.py stage-c --parts s5 s5b h4 convnext` | S5 LoRA (successive-halving sweep over lr × rank, then the final run on folds 1–4), S5b WiSE scaling, the H4 augmentation-matched frozen control, the ConvNeXt-T baseline (postponed). S5 and S5b are decided in `stage_c` with the same rule. | `results/arms/S5*.json`, `results/lora_sweep.json`, LoRA weights in `DATA_DIR/checkpoints/lora` |
| `main.py wilds-prepare` | Reads a local copy of the Camelyon17-WILDS Hugging Face mirror (`wltjr1007/Camelyon17-WILDS`, revision `d784d53`, downloaded with `huggingface_hub` into `D:/AI/wilds/hf_mirror`), checks split sizes against the paper, fixes the ≤1,000-per-slide subsample and runs the hospital colour check (no predictions). | `results/wilds_data.json` |
| `main.py report` | Regenerates every table and figure from `results/`, the experiment reports, README blocks, site data and the paper PDF. | `reports/`, `site/data/results.json` (+ figures), README blocks |
| `main.py publish [--arms ...]` | Uploads licence-clean heads and the LoRA model, with the Apache-2.0 `LICENSE`, a `NOTICE` and the card, to the Hugging Face repo and collection; refuses Virchow2/Phikon lineages. | `huggingface.co/BurnyCoder/pcam-hibou-hillclimb` |
| `main.py release --title "..."` | One full release: paired comparisons, `report`, commit, push, pull request, merge, Hugging Face upload. | GitHub PR, Pages, Hugging Face |
| `main.py release-watch` | Polls `results/` and runs `release` in a fresh process for every newly settled arm or step file. | one release per new classifier |

## Data flow
1. **Raw data.** Raw HDF5 patches (96×96×3 uint8) are read through `data.PCamImages`, which memory-maps the contiguous HDF5 datasets.
2. **Embedding.** The embedding phase applies eva's preprocessing (bilinear resize to 224, scale to [0,1], per-model mean/std normalisation), runs the frozen backbone and pools the tokens, writing float16 rows to disk.
3. **Heads.**
   - Heads train on cached features only.
   - Out-of-fold predictions come from 5 slide-grouped folds of train; the full-train refits predict the validation split.
   - Every arm stores per-seed logits locally (`DATA_DIR/preds`, not in git, because of licence rules) and an aggregate record in `results/arms/`.
4. **Statistics.** `stats.py` resamples whole slides (stratified tumour/normal slides) to get BCa confidence intervals, and computes per-slide sign-flip tests for paired differences.
5. **Ladder.** `ladder.py` applies the pre-registered adoption rule and records the incumbent configuration. Its hash is what unlocks the test split (`seal.py`).

## Reproducing the H2 parity check
```bash
cd tools/eva_parity
uv sync
HF_HUB_CACHE=D:/AI/hf_cache/hub uv run python eva_embed.py D:/AI/pcam/h5/camelyonpatch_level_2_split_valid_x.h5 D:/AI/pcam/parity
cd ../..
uv run python main.py parity
```

## Hardware notes
Development hardware is the author's RTX 5070 Laptop GPU (8 GB). The desktop already holds about 2 GB of that. The rates below are approximate observations from the (uncommitted) run logs, except the LoRA rate, which is recorded in `results/arms/S5_lora.json`.
- **Run one GPU job at a time.** Concurrent jobs page VRAM through WDDM and slow everything down.
- **Measured throughput:** Hibou-L embedding runs at about 160 img/s in bf16 with multi-layer outputs. LoRA training on all 24 blocks (rank 16, gradient checkpointing, micro-batch 64) runs at 25.6 img/s, so one epoch of the 203,491 cleaned fold 1–4 patches takes about 2.2 h (the ladder report computes both from `results/`).
- **Keep training images in RAM.** Random reads from the HDF5 file on disk cost about 1.4 s per 64-patch batch; `PCamImages(split, in_memory=True)` loads train once (about 100 s).
