"""Command-line entry point: one subcommand per pipeline phase.

Global context: `uv run python main.py <phase>`; each phase logs to logs/<timestamp>_<phase>.log
and writes machine-readable outputs to results/, from which `report` regenerates every table and
figure. Phases are thin wrappers around functions in src/pcam_hc/.
"""

import argparse
import json

from pcam_hc import config  # noqa: F401 - loads .env (HF_HUB_CACHE, paths) before any HF import
from pcam_hc.logging_utils import setup_logging


def cmd_download(args):
    from pcam_hc.data import download_pcam

    download_pcam()


def cmd_verify(args):
    from pcam_hc.data import verify_splits

    report = verify_splits(check_duplicates=not args.no_duplicates)
    config.PATHS.results.mkdir(parents=True, exist_ok=True)
    (config.PATHS.results / "data_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_access(args):
    from pcam_hc.backbones import check_access

    print(json.dumps(check_access(), indent=2))


def cmd_embed(args):
    from pcam_hc.embed import embed_backbone

    indices = None
    if args.subset == "train64k":  # pre-registered fixed subset (seed 0), stratified by (fold, label)
        from pcam_hc.data import load_labels, load_meta, make_folds, make_subset

        indices = make_subset(make_folds(load_meta("train"), config.N_FOLDS), load_labels("train"), config.SUBSET_SIZE)
    for backbone in args.backbones:
        for split in args.splits:
            for view in args.views:
                embed_backbone(backbone, split, view=view, batch_size=args.batch_size,
                               indices=indices if split == "train" else None,
                               subset=args.subset if split == "train" else None)


def cmd_parity(args):
    """H2 parity: our native-transformers Hibou-L CLS vs kaiko-eva's remote-code embeddings
    (tools/eva_parity/eva_embed.py) on the first 256 validation patches."""
    import numpy as np
    import torch

    from pcam_hc.backbones import REGISTRY, Encoder, load_model
    from pcam_hc.data import PCamImages

    ref = np.load(config.PATHS.data / "parity" / "eva_hibou_l_cls.npy")
    x = torch.from_numpy(PCamImages("valid").read(np.arange(len(ref)))).permute(0, 3, 1, 2)
    spec = REGISTRY["hibou_l"]

    def cosine(a, b):
        a, b = a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=1, keepdims=True)
        return (a * b).sum(1)

    report = {}
    for label, device, dtype in [("fp32_cpu", "cpu", torch.float32), ("bf16_gpu", "cuda", torch.bfloat16)]:
        enc = Encoder(load_model(spec), spec, device=device, dtype=dtype)
        ours = torch.cat([enc(x[i : i + 32]) for i in range(0, len(x), 32)])[:, : spec.dim].cpu().numpy()
        cos = cosine(ours, ref)
        report[label] = {"mean_cosine": float(cos.mean()), "min_cosine": float(cos.min()),
                         "max_abs_diff": float(np.abs(ours - ref).max())}
    cache = config.PATHS.emb / "hibou_l" / "valid" / "full.npy"
    if cache.exists():  # the production cache (bf16 forward, float16 storage), CLS of block 24 = columns 6*2048...
        from pcam_hc.features import CLS, load

        cached = load("hibou_l", "valid", CLS)[: len(ref)].astype(np.float32)
        report["cache_fp16"] = {"mean_cosine": float(cosine(cached, ref).mean()), "min_cosine": float(cosine(cached, ref).min())}
    report["gate_pass"] = all(v["mean_cosine"] > 0.999 for v in report.values() if isinstance(v, dict))
    (config.PATHS.results / "parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_stage_a(args):
    """Stage A: H2 reproduction gate, S0 incumbent under CV, controls, and the B* backbone scan."""
    import numpy as np

    from pcam_hc import experiments as ex
    from pcam_hc.backbones import REGISTRY
    from pcam_hc.data import color_histograms
    from pcam_hc.features import CLS, cache_dir, load

    ctx = ex.Context()
    out = config.PATHS.results
    if "h2" in args.parts:  # eva-exact reproduction on valid (test stays sealed)
        rec = ex.eva_exact(ctx, "S0_eva_exact_hibou_l", load("hibou_l", "train", CLS), load("hibou_l", "valid", CLS))
        ba = rec["valid"]["seed_mean"]["ba"]
        gate = {"valid_ba_seed_mean": ba, "valid_ba_seed_sd": rec["valid"]["seed_sd"]["ba"], "target": 0.939,
                "tolerance": 0.010, "pass": abs(ba - 0.939) <= 0.010}
        (out / "h2_gate.json").write_text(json.dumps(gate, indent=2))
        print(json.dumps(gate, indent=2))
    if "s0" in args.parts:  # incumbent of the ladder, CV protocol (fixed steps, no valid early stopping)
        ex.run_arm(ctx, "S0_hibou_l_cls_linear", load("hibou_l", "train", CLS), load("hibou_l", "valid", CLS),
                   description="Hibou-L CLS (block 24), eva linear head, CV protocol")
    if "controls" in args.parts:
        Xtr, Xva = load("hibou_l", "train", CLS), load("hibou_l", "valid", CLS)
        perm = np.random.default_rng(0).permutation(len(ctx.y_train))
        ex.run_arm(ctx, "control_shuffled_labels", Xtr, Xva, y_train=ctx.y_train[perm],
                   description="Hibou-L CLS, linear head trained on permuted train labels")
        ex.run_arm(ctx, "control_color_histogram", color_histograms("train"), color_histograms("valid"),
                   kind="logreg", backbones=(), C=1.0, seeds=(0,),
                   description="96-bin RGB histogram + logistic regression (stain shortcut check)")
        const = np.zeros((1, len(ctx.y_valid)), dtype=np.float32) + 1e-3  # always 'tumour' at p ~ 0.5
        rec = {"name": "control_majority", "description": "constant prediction", "backbones": [],
               "publishable": True, "valid": ex.summarise(ctx.y_valid, const, ctx.slides_valid)}
        ex.save_arm("control_majority", rec, {"valid": const})
    if "scan" in args.parts:  # every backbone with a finished cache, eva CLS embedding, CV protocol
        for name in args.backbones if args.backbones != ["hibou_l"] else REGISTRY:
            if name == "hibou_l" or not (cache_dir(name, "valid") / "full.npy.progress").exists():
                continue
            arm = f"S0_scan_{name}"
            if (out / "arms" / f"{arm}.json").exists():
                continue
            ex.run_arm(ctx, arm, load(name, "train", CLS), load(name, "valid", CLS), backbones=(name,),
                       description=f"{name} CLS, eva linear head, CV protocol",
                       recipe={"backbone": name, "representation": {"parts": [[-1, "cls"]], "center": False,
                                                                    "views": ["full"]}})


def cmd_stage_b(args):
    """Stage B: frozen-feature ladder steps S1, S2, S3 (D4 TTA), S3b (label cleaning), S4 (fusion)."""
    import json as _json

    from pcam_hc import stage_b
    from pcam_hc.experiments import Context

    ctx = Context()
    steps = {"s1": stage_b.step_s1, "s2": stage_b.step_s2, "s3": stage_b.step_s3, "s3b": stage_b.step_s3b}
    for part in args.parts:
        if part == "s4":
            pool = _json.loads((config.PATHS.results / "model_pool.json").read_text())
            controls = {"hibou_l", "dinov2_reg_l", "random_vitl_reg4"}
            stage_b.step_s4(ctx, [b for b, ok in pool.items() if ok and b not in controls])
        else:
            steps[part](ctx)


def cmd_report(args):
    from pcam_hc.report import build

    build()


PHASES = {"download": cmd_download, "verify": cmd_verify, "access": cmd_access, "embed": cmd_embed,
          "parity": cmd_parity, "stage-a": cmd_stage_a, "stage-b": cmd_stage_b, "report": cmd_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--no-duplicates", action="store_true", help="verify: skip the pixel-hash duplicate scan")
    parser.add_argument("--backbones", nargs="+", default=["hibou_l"], help="embed: registry names")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], help="embed: splits (test is sealed)")
    parser.add_argument("--views", nargs="+", default=["full"], help="embed: full | center64 | d4_k")
    parser.add_argument("--subset", default=None, help="embed: 'train64k' restricts the train split to the subset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parts", nargs="+", default=["h2", "s0", "controls", "scan"],
                        help="stage-a: h2 s0 controls scan; stage-b: s1 s2 s3 s3b s4")
    args = parser.parse_args()
    setup_logging(args.phase)
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
