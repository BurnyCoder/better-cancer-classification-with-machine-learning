"""Regenerate every results table and figure from results/*.json (every result number is read from results/).

Global context: `uv run python main.py report` is the single command behind the README tables,
the per-experiment reports, the paper figures and the site data. Only aggregate metrics are read
and written here, so outputs are licence-clean for every backbone (pre-registration section 9).
Chart style follows the dataviz skill's reference palette: thin marks, CIs as whiskers, one axis,
recessive grid, identity carried by labels as well as colour.
"""

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pcam_hc.config import PATHS, PROJECT_ROOT  # noqa: E402

FIG_DIR, TAB_DIR = PROJECT_ROOT / "reports" / "figures", PROJECT_ROOT / "reports" / "tables"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUP_COLOR = {"Hibou-L": "#2a78d6", "other foundation model": "#eb6834", "control": "#8a8985"}
PRETTY = {"hibou_l": "Hibou-L", "hibou_b": "Hibou-B", "prov_gigapath": "Prov-GigaPath", "h_optimus_0": "H-optimus-0",
          "virchow2": "Virchow2", "virchow": "Virchow", "midnight": "Midnight-12k", "openmidnight": "OpenMidnight-1.1",
          "phikon": "Phikon", "phikon_v2": "Phikon-v2", "dinov2_reg_l": "DINOv2-L (natural images)",
          "random_vitl_reg4": "ViT-L random init"}
CONTROLS = {"dinov2_reg_l", "random_vitl_reg4"}
EVA_LEADERBOARD = "https://kaiko-ai.github.io/eva/main/leaderboards/"


def arms() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((PATHS.results / "arms").glob("*.json"))}


def fmt_ci(block: dict, key: str = "ba") -> str:
    lo, hi = block[f"ci_{key}"]
    return f"{block[key]:.4f} [{lo:.4f}, {hi:.4f}]"


def arms_table(records: dict) -> str:
    """Markdown table of every arm: seed-averaged metric with slide-bootstrap BCa 95% CI, seed count and seed SD
    ("—" for one seed, where the SD is undefined)."""
    rows = ["| Arm | Backbone(s) | Head | Dim | OOF BA [95% CI] | OOF patches | Valid BA [95% CI] | Seeds | Valid BA seed SD | Valid AUROC | Publishable |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in records.items():
        oof = fmt_ci(r["oof"]["seed_avg"]) if "oof" in r else "—"
        val = r["valid"]
        sd = f"{val['seed_sd']['ba']:.4f}" if val.get("seed_sd") else "—"
        rows.append(f"| `{name}` | {', '.join(PRETTY.get(b, b) for b in r['backbones']) or '—'} | {r.get('head', '—')} | "
                    f"{r.get('dim', '—')} | {oof} | {r.get('oof_subset', 'all train') if 'oof' in r else '—'} | {fmt_ci(val['seed_avg'])} | "
                    f"{val['n_seeds']} | {sd} | {val['seed_avg']['auroc']:.4f} | {'yes' if r.get('publishable') else 'no (aggregates only)'} |")
    return "\n".join(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def published_context_figure() -> None:
    """Published eva PCam BA (val and test) per model, with our reproduced Hibou-L valid BA."""
    pub = PATHS.results / "published_baselines.json"
    gate = PATHS.results / "h2_gate.json"
    if not pub.exists():
        return
    rows = json.loads(pub.read_text())["eva_leaderboard"]["rows"]
    rows = sorted(rows, key=lambda r: r["test_ba"])
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(rows) + 1.4), facecolor=SURFACE)
    y = range(len(rows))
    ax.plot([r["val_ba"] for r in rows], y, "o", mfc=SURFACE, mec=GROUP_COLOR["other foundation model"], mew=2, ms=7,
            linestyle="", label="eva published, PCam valid")
    ax.plot([r["test_ba"] for r in rows], y, "o", color=GROUP_COLOR["Hibou-L"], ms=7, mec=SURFACE, mew=1.5,
            linestyle="", label="eva published, PCam test")
    if gate.exists():
        g = json.loads(gate.read_text())
        i = [r["model"] for r in rows].index("Hibou-L")
        ax.errorbar(g["valid_ba_seed_mean"], i + 0.28, xerr=g["valid_ba_seed_sd"], fmt="D", color=INK, ms=6,
                    elinewidth=1.5, capsize=0, label=f"this study, reproduced Hibou-L valid ({g['n_seeds']} seeds; ±SD hidden by the marker)")
    labels = [r["model"] + ("" if r["accessible_here"] else " *") for r in rows]
    ax.set_yticks(list(y), labels, color=INK)
    ax.set_xlabel("Balanced accuracy (frozen backbone + linear probe)", color=INK2)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")  # the lower right holds no points
    ax.set_title("Published no-training baselines (kaiko eva leaderboard); * = not in this study's model pool",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "published_baselines.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def stage_a_figure(records: dict) -> None:
    """S0 reproduction, the original probe under CV and the controls: OOF (train CV) and valid BA with 95% CIs."""
    order = [("control_majority", "Constant 'tumour' prediction"), ("control_shuffled_labels", "Hibou-L, shuffled labels"),
             ("control_color_histogram", "Colour histogram + logistic regression"),
             ("S0_scan_random_vitl_reg4", "ViT-L, random weights (Hibou-L config)"),
             ("S0_scan_dinov2_reg_l", "DINOv2-L, natural-image weights"),
             ("S0_hibou_l_cls_linear", "Hibou-L CLS + linear (CV protocol)"),
             ("S0_eva_exact_hibou_l", "Hibou-L CLS + linear (eva-exact)")]
    order = [(k, l) for k, l in order if k in records]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(order) + 1.3), facecolor=SURFACE)
    labelled = set()
    for i, (k, _) in enumerate(order):
        r = records[k]
        for split, dy, color, label in (("oof", 0.12, GROUP_COLOR["Hibou-L"], "train OOF (5 slide-grouped folds)"),
                                        ("valid", -0.12, GROUP_COLOR["other foundation model"], "PCam valid")):
            if split not in r:
                continue
            m = r[split]["seed_avg"]
            lo, hi = m["ci_ba"]
            ax.plot([lo, hi], [i + dy] * 2, color=color, linewidth=2)
            ax.plot(m["ba"], i + dy, "o", color=color, ms=7, mec=SURFACE, mew=1.5,
                    label=None if split in labelled else label)
            labelled.add(split)
    ax.set_yticks(range(len(order)), [l for _, l in order], color=INK)
    ax.set_xlabel("Balanced accuracy (95% stratified slide-bootstrap BCa CI)", color=INK2)
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Stage A: reproduced Hibou-L probe vs. controls", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage_a.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def pipeline_figure() -> None:
    """Block diagram of the study pipeline (data -> frozen features -> ladder -> sealed evaluation)."""
    from matplotlib.patches import FancyBboxPatch

    texts = ["PCam HDF5\n(Zenodo, md5)", "Leakage checks\nslide-disjoint,\npixel hashes",
             "Frozen backbones\nHibou-L + 2 controls\n(bf16, cached)", "Heads on cache\nlinear / logreg / MLP\n5 slide-grouped folds",
             "Ladder S1-S6\nOOF winner + Holm\n+ valid no-harm gate", "Freeze + tag,\nthen score test\n& WILDS once"]
    boxes = [(t, 0.01 + i * 0.167) for i, t in enumerate(texts)]
    fig, ax = plt.subplots(figsize=(13, 2.4), facecolor=SURFACE)
    ax.set_axis_off()
    for text, x in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.2), 0.14, 0.6, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor="#eef3fb", edgecolor=GROUP_COLOR["Hibou-L"], linewidth=1.2))
        ax.text(x + 0.07, 0.5, text, ha="center", va="center", fontsize=8, color=INK)
    for (_, x0), (_, x1) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(x1, 0.5), xytext=(x0 + 0.14, 0.5), arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(0.585, 0.05, "LoRA fine-tuning (S5) reads images directly; statistics resample whole slides",
            ha="center", fontsize=8, color=INK2)
    ax.set_xlim(-0.01, 1.0)  # leave room for the first box's rounded padding
    ax.set_ylim(0, 1)
    fig.savefig(FIG_DIR / "pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def ladder_figure() -> None:
    """Paired difference from the original Hibou-L probe for every trained ladder classifier."""
    from pcam_hc.report_experiments import ladder_rows, ref_label

    rows = ladder_rows()
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(rows) + 1.4), facecolor=SURFACE)
    ax.axvline(0, color=INK2, linewidth=1)
    labelled = set()
    for i, r in enumerate(rows):
        for key, dy, color, label in (("oof", 0.12, GROUP_COLOR["Hibou-L"], "train OOF (paired; patch set in the row label)"),
                                      ("valid", -0.12, GROUP_COLOR["other foundation model"], "PCam valid (paired)")):
            d = r["cmp"][key]
            ax.plot([100 * d["ci_ba"][0], 100 * d["ci_ba"][1]], [i + dy] * 2, color=color, linewidth=2)
            ax.plot(100 * d["delta_ba"], i + dy, "o", color=color, ms=7, mec=SURFACE, mew=1.5,
                    label=None if key in labelled else label)
            labelled.add(key)
    ax.set_yticks(range(len(rows)), [f"{r['step']}: {r['arm']}\nOOF {r['subset']} vs {ref_label(r['cmp'])}" for r in rows],
                  color=INK, fontsize=7.5)
    ax.set_xlabel("Δ balanced accuracy (pp), 95% slide-bootstrap CI", color=INK2)
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_title("Improvement ladder: each classifier vs. the original Hibou-L probe", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ladder_deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def site_data(records: dict) -> None:
    """Aggregate-only JSON for the GitHub Pages explorer (licence-clean for every backbone)."""

    def block(b):
        return {"ba": b["ba"], "lo": b["ci_ba"][0], "hi": b["ci_ba"][1], "auroc": b["auroc"]}

    arms_out = {}
    for name, r in records.items():
        b = r["backbones"][0] if r["backbones"] else ""
        group = "Hibou-L" if b == "hibou_l" else ("control" if (b in CONTROLS or name.startswith("control")) else
                                                   "other foundation model")
        arms_out[name] = {"label": PRETTY.get(b, b) if name.startswith("S0_scan_") or name == "S0_hibou_l_cls_linear" else name,
                          "group": group, "backbones": [PRETTY.get(x, x) for x in r["backbones"]],
                          "publishable": r.get("publishable", True), "valid": block(r["valid"]["seed_avg"]),
                          **({"oof": block(r["oof"]["seed_avg"]), "subset": r.get("oof_subset", "all train")} if "oof" in r else {})}
    steps = []
    for p in sorted((PATHS.results / "steps").glob("*.json"), key=lambda p: p.stat().st_mtime):
        s = json.loads(p.read_text())
        steps.append({k: s[k] for k in ("step", "incumbent", "candidates", "winner", "adopted")})
    load = lambda f: json.loads((PATHS.results / f).read_text()) if (PATHS.results / f).exists() else None  # noqa: E731
    final = load("final/summary.json")
    pub = load("published_baselines.json")
    from pcam_hc.report_experiments import ladder_rows, ref_label

    ladder_out = [{"arm": r["arm"], "step": r["step"], "description": r["description"],
                   "oof_ba": r["oof"]["ba"], "delta": r["cmp"]["oof"]["delta_ba"], "lo": r["cmp"]["oof"]["ci_ba"][0],
                   "hi": r["cmp"]["oof"]["ci_ba"][1], "p": r["cmp"]["oof"]["p_signflip"],
                   "valid_ba": r["valid"]["ba"], "valid_delta": r["cmp"]["valid"]["delta_ba"],
                   "valid_lo": r["cmp"]["valid"]["ci_ba"][0], "valid_hi": r["cmp"]["valid"]["ci_ba"][1],
                   "ref": ref_label(r["cmp"])} for r in ladder_rows()]
    out = {"arms": arms_out, "steps": steps, "gate": load("h2_gate.json"), "ladder": load("ladder.json"),
           "ladder_rows": ladder_out,
           "final": final, "published": pub["eva_leaderboard"]["rows"] if pub else [],
           "literature": pub["literature"] if pub else []}
    site = PROJECT_ROOT / "site" / "data"
    site.mkdir(parents=True, exist_ok=True)
    (site / "results.json").write_text(json.dumps(out, indent=1, default=float))
    for fig in ("examples.png", "lora_training.png"):  # Pages serves only site/, so the figures are copied there
        if (FIG_DIR / fig).exists():
            (site / fig).write_bytes((FIG_DIR / fig).read_bytes())


def best_model_block() -> str | None:
    """README top: the best improved Hibou-L pipeline so far vs. the original probe, every number from results/."""
    from pcam_hc.publish import LORA_ARM, REPO_ID

    get = lambda f: json.loads((PATHS.results / f).read_text()) if (PATHS.results / f).exists() else None  # noqa: E731
    lad, base = get("ladder.json"), "S0_hibou_l_cls_linear"
    best = lad and (lad.get("hibou_incumbent") or lad.get("incumbent"))
    b, s0, c = get(f"arms/{best}.json"), get(f"arms/{base}.json"), get(f"comparisons/{best}_vs_baseline.json")
    if not (best and best != base and b and s0 and c):
        return None
    pt = lambda x: f"{100 * x['delta_ba']:+.2f} pt [{100 * x['ci_ba'][0]:+.2f}, {100 * x['ci_ba'][1]:+.2f}]"  # noqa: E731
    v, v0, dv = b["valid"]["seed_avg"], s0["valid"]["seed_avg"], get("data_verification.json")
    name = "Hibou-L fine-tuned with LoRA" if b.get("head") == "lora" else b.get("description", best)
    lines = [f"### 🏆 Best improved model so far: {name} (`{best}`)",
             "", f"| | Original Hibou-L probe | `{best}` |", "|---|---|---|",
             f"| Validation balanced accuracy [95% CI] | {fmt_ci(v0)} | **{fmt_ci(v)}** |",
             f"| Validation AUROC | {v0['auroc']:.4f} | **{v['auroc']:.4f}** |", "",
             f"- **Gain over the original:** {pt(c['valid'])} balanced accuracy on the {dv['valid_slides'] if dv else ''} validation slides, "
             f"paired per-slide sign-flip p = {c['valid']['p_signflip']:.2g}."]
    if h4 := get(f"comparisons/{best}_vs_H4_aug_control.json"):
        lines.append(f"- **Augmentation-matched control:** it beats a frozen Hibou-L head trained on the same patches with the same augmentation recipe by {pt(h4['valid'])}, "
                     "which suggests the gain comes from adapting the network rather than from the augmentation alone.")
    lines.append(f"- **Recipe:** {b['description']}."
                 + (f" Weights: [Hugging Face](https://huggingface.co/{REPO_ID}/tree/main/lora/{LORA_ARM})." if best == LORA_ARM else ""))
    lines.append("- **Caveats:** validation only (the test split and the other-hospital check are still sealed); a single training run; "
                 "trained before the code-review fixes listed in the [ladder report](reports/experiments/02_ladder_hibou_l.md).")
    return "\n".join(lines)


def highlights() -> str | None:
    """README highlights (positive and negative findings), every number read from results/."""
    get = lambda f: json.loads((PATHS.results / f).read_text()) if (PATHS.results / f).exists() else None  # noqa: E731
    dv, gate, parity, s0 = get("data_verification.json"), get("h2_gate.json"), get("parity.json"), get("arms/S0_hibou_l_cls_linear.json")
    sh, ch, wl, pub = (get("arms/control_shuffled_labels.json"), get("arms/control_color_histogram.json"),
                       get("wilds_data.json"), get("published_baselines.json"))
    if not all([dv, gate, parity, s0, sh, ch]):
        return None
    hib = next(r for r in pub["eva_leaderboard"]["rows"] if r["model"] == "Hibou-L") if pub else None
    ci = lambda b: f"{b['ba']:.3f} (95% CI {b['ci_ba'][0]:.3f}–{b['ci_ba'][1]:.3f})"  # noqa: E731
    from pcam_hc.report_experiments import CAMELYON16_README, duplicate_slide, ladder_findings, wilds_pool_md

    out = [
        f"- ✅ **Reproduction to the published precision.** Our Hibou-L probe scores **{gate['valid_ba_seed_mean']:.4f} ± {gate['valid_ba_seed_sd']:.4f}** validation balanced accuracy (BA; {gate['n_seeds']} seeds), against the published **{hib['val_ba']:.3f}** ([eva leaderboard]({EVA_LEADERBOARD})). "
        f"Our remote-code-free model matches Hibou's own code to a mean cosine of {parity['fp32_cpu']['mean_cosine']:.6f} (fp32, {parity['n']} validation patches).",
        f"- ✅ **The original Hibou-L probe under cross-validation** over {dv['train_slides']} training slides: OOF BA {ci(s0['oof']['seed_avg'])}; validation {ci(s0['valid']['seed_avg'])}.",
        f"- ✅ **Controls behave as expected.** Shuffled labels: {sh['oof']['seed_avg']['ba']:.3f} (chance). Colour histogram only: {ci(ch['oof']['seed_avg'])}. Colour carries signal, but far less than Hibou-L.",
    ]
    if dup := duplicate_slide(dv):
        out.append(f"- ⚠️ **Data leak found in PCam.** {dup['n']} distinct test images are pixel-identical to train images. All come from test slide `{dup['test_slide']}` "
                   f"and {dup['train_phrase']}, {dup['coords_phrase']}, which indicates a duplicate scan. "
                   f"The official [CAMELYON16 README]({CAMELYON16_README}) lists test_049 as a duplicate slide intentionally removed from the data set; PCam still contains it "
                   f"({dup['test_slide_n']:,} test patches). Every test result will also be reported without it (decided while test was sealed).")
    out += [
        f"- ⚠️ **Validation split is small.** It has {dv['valid_slides']} slides, {dv['valid_tumour_slides']} with tumour, so the original probe's validation CI is "
        f"{(s0['valid']['seed_avg']['ci_ba'][1] - s0['valid']['seed_avg']['ci_ba'][0]) / (s0['oof']['seed_avg']['ci_ba'][1] - s0['oof']['seed_avg']['ci_ba'][0]):.1f}× wider than its train-CV CI. "
        "Ideas are therefore selected on train CV, and validation only checks that nothing gets worse.",
    ]
    if findings := ladder_findings():  # release 2+ headline results (shared with the paper's abstract)
        out.insert(3, "- 📈 **Improvement ladder and controls** (validation and cross-validation; test still sealed): " + " ".join(findings))
    if wl:
        d = wl["hospital_check"]["l1_distance_to_pcam_hospital"]["2"]
        out.append(f"- ⚠️ **The official Camelyon17-WILDS test hospital is most likely not out-of-distribution for PCam.** Its colour signature matches UMCU, a Camelyon16 source hospital (L1 {d['UMCU']:.2f} vs. {d['RUMC']:.2f} to RUMC). "
                   + wilds_pool_md())
    out.append(f"- **Data:** train {dv['train_n']:,} / valid {dv['valid_n']:,} / test {dv['test_n']:,} patches from {dv['train_slides']} / {dv['valid_slides']} / {dv['test_slides']} slides; no slide is shared between splits.")
    return "\n".join(out)


def inject_readme(markdown: str, marker: str) -> None:
    """Replace the README block between <!-- {marker}:START --> and <!-- {marker}:END -->."""
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- {marker}:START -->)(.*?)(<!-- {marker}:END -->)", re.S)
    if pattern.search(text):
        readme.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{markdown}\n{m.group(3)}", text), encoding="utf-8")


def build() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    records = arms()
    table = arms_table(records)
    (TAB_DIR / "arms.md").write_text(table + "\n", encoding="utf-8")
    published_context_figure()
    stage_a_figure(records)
    pipeline_figure()
    ladder_figure()
    from pcam_hc.report_examples import examples_figure, hed_strength, lora_curve_figure

    hed_strength()
    examples_figure()
    lora_curve_figure()
    site_data(records)
    gate_path = PATHS.results / "h2_gate.json"
    if gate_path.exists():
        g = json.loads(gate_path.read_text())
        gate = (f"H2 reproduction gate: valid BA {g['valid_ba_seed_mean']:.4f} ± {g['valid_ba_seed_sd']:.4f} "
                f"({g['n_seeds']} seeds) vs. eva's published {g['target']:.3f} (pre-registered gate ±{g['tolerance']:.3f}) → **{'pass' if g['pass'] else 'FAIL'}**.")
        (TAB_DIR / "h2_gate.md").write_text(gate + "\n", encoding="utf-8")
        from pcam_hc.report_experiments import ladder_rows, ladder_table_md

        rows = ladder_rows()
        ladder_md = ("\n\n**Improvement ladder: every trained classifier vs. the original Hibou-L probe** (paired on the same "
                     "patches; per-candidate p-values are descriptive, and adoption uses the pre-registered Holm + valid-gate rule):\n\n"
                     + ladder_table_md(rows) + "\n\n![Ladder](reports/figures/ladder_deltas.png)") if rows else ""
        inject_readme(gate + "\n\n" + table + ladder_md, "RESULTS")
    hl = highlights()
    if hl:
        inject_readme(hl, "HIGHLIGHTS")
    if top := best_model_block():
        inject_readme(top, "BEST")
    from pcam_hc.paper import compile_paper
    from pcam_hc.report_experiments import build_experiment_reports

    build_experiment_reports()
    compile_paper()
