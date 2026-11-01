"""Regenerate every results table and figure from results/*.json (no hand-typed numbers).

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


def arms() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((PATHS.results / "arms").glob("*.json"))}


def fmt_ci(block: dict, key: str = "ba") -> str:
    lo, hi = block[f"ci_{key}"]
    return f"{block[key]:.4f} [{lo:.4f}, {hi:.4f}]"


def arms_table(records: dict) -> str:
    """Markdown table of every arm: seed-averaged metric with slide-bootstrap BCa 95% CI, seed SD."""
    rows = ["| Arm | Backbone(s) | Head | Dim | OOF BA [95% CI] | Valid BA [95% CI] | Valid BA seed SD | Valid AUROC | Publishable |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, r in records.items():
        oof = fmt_ci(r["oof"]["seed_avg"]) if "oof" in r else "—"
        val = r["valid"]
        rows.append(f"| `{name}` | {', '.join(PRETTY.get(b, b) for b in r['backbones']) or '—'} | {r.get('head', '—')} | "
                    f"{r.get('dim', '—')} | {oof} | {fmt_ci(val['seed_avg'])} | {val['seed_sd']['ba']:.4f} | "
                    f"{val['seed_avg']['auroc']:.4f} | {'yes' if r.get('publishable') else 'no (aggregates only)'} |")
    return "\n".join(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def backbone_scan_figure(records: dict) -> None:
    """Dot + 95% CI whisker per backbone (eva CLS linear probe, OOF over train slides)."""
    rows = []
    for name, r in records.items():
        if name.startswith("S0_scan_") or name == "S0_hibou_l_cls_linear":
            b = r["backbones"][0]
            group = "Hibou-L" if b == "hibou_l" else ("control" if b in CONTROLS else "other foundation model")
            rows.append((PRETTY.get(b, b), r["oof"]["seed_avg"], group))
    if not rows:
        return
    rows.sort(key=lambda t: t[1]["ba"])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2), facecolor=SURFACE)
    for i, (label, m, group) in enumerate(rows):
        lo, hi = m["ci_ba"]
        ax.plot([lo, hi], [i, i], color=GROUP_COLOR[group], linewidth=2, solid_capstyle="round")
        ax.plot(m["ba"], i, "o", markersize=8, color=GROUP_COLOR[group], markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.002, i, f"{m['ba']:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK)
    ax.set_xlabel("Balanced accuracy, out-of-fold over 5 slide-grouped folds (95% slide-bootstrap CI)", color=INK2)
    _style(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=g) for g, c in GROUP_COLOR.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("S0 backbone scan: frozen CLS embedding + eva linear probe", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s0_backbone_scan.png", dpi=180)
    plt.close(fig)


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
    backbone_scan_figure(records)
    gate_path = PATHS.results / "h2_gate.json"
    if gate_path.exists():
        g = json.loads(gate_path.read_text())
        gate = (f"H2 reproduction gate: valid BA {g['valid_ba_seed_mean']:.4f} ± {g['valid_ba_seed_sd']:.4f} "
                f"(5 seeds) vs. eva's published 0.939 ± 0.010 → **{'pass' if g['pass'] else 'FAIL'}**.")
        (TAB_DIR / "h2_gate.md").write_text(gate + "\n", encoding="utf-8")
        inject_readme(gate + "\n\n" + table, "RESULTS")
    from pcam_hc.report_experiments import build_experiment_reports

    build_experiment_reports()
