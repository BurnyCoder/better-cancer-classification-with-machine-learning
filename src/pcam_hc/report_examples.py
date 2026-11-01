"""Output examples and the LoRA training curve (figures for the README, reports and paper).

Global context: the author asks for example outputs of the new method, the baseline and the controls,
and for loss graphs. Examples are validation patches (PCam is CC0) with each model's predicted
tumour probability; they are drawn by a fixed rule (seed 0, uniformly within outcome groups) so they
are not cherry-picked. Every model shown has a redistributable lineage. The LoRA loss is logged only
per 50 steps in the run's log (logs/ is not committed), so it is extracted once into
results/lora_train_curve.json and the figure reads that file.
"""

import json
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.special import expit

from pcam_hc.config import PATHS, PROJECT_ROOT
from pcam_hc.report import FIG_DIR, GRID, INK, INK2, SURFACE, _style

# Models shown in the example grid: (arm, short label). Baseline, new method, two controls.
EXAMPLE_ARMS = [("S0_hibou_l_cls_linear", "original"), ("S5_lora", "LoRA"),
                ("S0_scan_dinov2_reg_l", "DINOv2-L"), ("control_color_histogram", "colour")]
GROUPS = [("LoRA fixes an error of the original", False, True), ("LoRA introduces an error", True, False),
          ("both wrong", False, False), ("both right", True, True)]
PER_GROUP = 3


def examples_figure() -> dict | None:
    """Grid of validation patches (rows: outcome groups of original vs. LoRA; 3 random patches each)
    with every model's tumour probability. Returns the group sizes, also saved to results/examples.json."""
    from pcam_hc import experiments as ex
    from pcam_hc.data import PCamImages, load_labels

    try:
        probs = {arm: expit(ex.seed_average(ex.load_preds(arm)["valid"])) for arm, _ in EXAMPLE_ARMS}
    except FileNotFoundError:  # predictions live on the data disk; skip on a fresh clone
        return None
    y = load_labels("valid")
    ok0, ok5 = (probs["S0_hibou_l_cls_linear"] > 0.5) == y, (probs["S5_lora"] > 0.5) == y
    rng = np.random.default_rng(0)
    picks, sizes = [], {}
    for name, a, b in GROUPS:
        members = np.flatnonzero((ok0 == a) & (ok5 == b))
        sizes[name] = int(len(members))
        picks.append(np.sort(rng.choice(members, size=min(PER_GROUP, len(members)), replace=False)))
    images = PCamImages("valid")
    fig, axes = plt.subplots(len(GROUPS), PER_GROUP, figsize=(8.4, 3.05 * len(GROUPS)), facecolor=SURFACE)
    for row, ((name, _, _), idx) in enumerate(zip(GROUPS, picks)):
        for col in range(PER_GROUP):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(idx):
                continue
            i = idx[col]
            ax.imshow(images.read(np.array([i]))[0])
            ax.add_patch(Rectangle((31.5, 31.5), 32, 32, fill=False, edgecolor="white", linestyle="--", linewidth=1))
            ax.set_title(f"#{i}  label: {'tumour' if y[i] else 'normal'}", fontsize=8.5, color=INK, loc="left")
            pairs = [f"{lab} {probs[arm][i]:.2f}" for arm, lab in EXAMPLE_ARMS]
            ax.text(0, -0.03, " · ".join(pairs[:2]) + "\n" + " · ".join(pairs[2:]), transform=ax.transAxes,
                    va="top", ha="left", fontsize=7.5, color=INK2, linespacing=1.2)
        axes[row, 0].text(-0.08, 0.5, f"{name}\n(n = {sizes[name]:,})", transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=9, color=INK)
    fig.suptitle("Validation patches with predicted tumour probability (dashed box: the labelled centre 32×32 px)",
                 color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0.03, 0.01, 1, 0.97), h_pad=3.0)
    fig.savefig(FIG_DIR / "examples.png", dpi=150)
    plt.close(fig)
    out = {"groups": sizes, "indices": {n: p.tolist() for (n, _, _), p in zip(GROUPS, picks)},
           "rule": f"{PER_GROUP} patches per group, uniform without replacement, numpy default_rng(0)"}
    (PATHS.results / "examples.json").write_text(json.dumps(out, indent=2))
    return out


def lora_curve() -> dict | None:
    """Per-50-step training loss of the final S5 LoRA run, from results/ or, once, from its log."""
    path, arm = PATHS.results / "lora_train_curve.json", PATHS.results / "arms" / "S5_lora.json"
    if path.exists():
        return json.loads(path.read_text())
    if not arm.exists():
        return None
    total = json.loads(arm.read_text())["lora"]["runs"]["0"]["history"][-1]["step"]  # final run length
    pat = re.compile(rf"lora seed 0 step (\d+)/{total} loss ([\d.]+)")
    for log in sorted((PROJECT_ROOT / "logs").glob("*_stage-c.log")):
        pts = [(int(m[1]), float(m[2])) for m in pat.finditer(log.read_text(encoding="utf-8", errors="ignore"))]
        if pts:
            curve = {"source_log": log.name, "note": "single-minibatch BCE loss logged every 50 steps (batch 64)",
                     "step": [s for s, _ in pts], "loss": [v for _, v in pts]}
            path.write_text(json.dumps(curve, indent=2))
            return curve
    return None


def lora_curve_figure() -> None:
    """Training loss (left) and EMA-checkpoint BA on the fold-0 checkpoint subsample (right)."""
    curve, arm = lora_curve(), PATHS.results / "arms" / "S5_lora.json"
    if curve is None:
        return
    hist = json.loads(arm.read_text())["lora"]["runs"]["0"]["history"]
    steps, loss = np.array(curve["step"]), np.array(curve["loss"])
    k = 10
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=SURFACE)
    for ax in (a, b):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    a.plot(steps, loss, "o", ms=2.5, color="#8a8985", alpha=0.6, label="single minibatch (every 50 steps)")
    a.plot(steps[k - 1:], smooth, color="#2a78d6", linewidth=2, label=f"running mean of {k}")
    a.set_yscale("log")
    a.set_xlabel("training step (batch 64)", color=INK2)
    a.set_ylabel("BCE loss (log scale)", color=INK2)
    a.legend(frameon=False, fontsize=8)
    a.set_title("S5 LoRA: training loss", color=INK, fontsize=10, loc="left")
    b.plot([h["step"] for h in hist], [h["eval_ba"] for h in hist], "o-", color="#2a78d6", linewidth=2, ms=6,
           mec=SURFACE, mew=1.5)
    best = max(hist, key=lambda h: h["eval_ba"])  # the checkpoint train_lora keeps
    b.annotate(f"selected: step {best['step']}, {best['eval_ba']:.4f}", (best["step"], best["eval_ba"]),
               xytext=(0, -28), textcoords="offset points", ha="center", fontsize=8, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    b.set_xlabel("training step", color=INK2)
    b.set_ylabel("BA, 10k fold-0 patches (zoomed y-axis)", color=INK2)
    b.set_title("EMA weights: checkpoint-selection BA", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lora_training.png", dpi=180)
    plt.close(fig)
