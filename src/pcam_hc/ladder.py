"""Step decisions of the pre-registered ladder (section 6 of the pre-registration).

Global context: each ladder step compares candidate arms with the current incumbent. The adoption
rule is fixed:
  1. winner = candidate with the highest out-of-fold (OOF) balanced accuracy of its seed-averaged
     prediction;
  2. the winner must beat the incumbent on OOF predictions under the paired per-slide sign-flip
     test, Holm-corrected over all candidates of the step (p < 0.05, delta > 0);
  3. one-shot valid gate: valid delta-BA >= 0 and valid delta-AUROC >= 0.
The decision record goes to results/steps/<step>.json; results/ladder.json keeps the current
incumbent arm and the history, and is the file whose hash freezes the ladder before unsealing test.
"""

import json
import logging

from pcam_hc.config import PATHS
from pcam_hc.experiments import compare_arms
from pcam_hc.stats import holm

log = logging.getLogger(__name__)
LADDER = PATHS.results / "ladder.json"


def read_ladder() -> dict:
    return json.loads(LADDER.read_text()) if LADDER.exists() else {"incumbent": None, "history": []}


def write_ladder(ladder: dict) -> None:
    LADDER.write_text(json.dumps(ladder, indent=2))


def arm_record(name: str) -> dict:
    return json.loads((PATHS.results / "arms" / f"{name}.json").read_text())


def decide_step(ctx, step: str, incumbent: str, candidates: list[str], alpha: float = 0.05) -> dict:
    """Apply the adoption rule; returns and stores the decision, and advances the ladder if adopted."""
    oof_ba = {c: arm_record(c)["oof"]["seed_avg"]["ba"] for c in candidates}
    comparisons = {c: compare_arms(ctx, incumbent, c) for c in candidates}
    p_adj = dict(zip(candidates, holm([comparisons[c]["oof"]["p_signflip"] for c in candidates])))
    winner = max(candidates, key=oof_ba.get)
    w = comparisons[winner]
    oof_ok = w["oof"]["delta_ba"] > 0 and p_adj[winner] < alpha
    valid_ok = w["valid"]["delta_ba"] >= 0 and w["valid"]["delta_auroc"] >= 0
    decision = {
        "step": step, "incumbent": incumbent, "candidates": candidates, "winner": winner,
        "oof_ba": oof_ba, "incumbent_oof_ba": arm_record(incumbent)["oof"]["seed_avg"]["ba"],
        "holm_p": p_adj, "comparisons": comparisons, "oof_test_passed": bool(oof_ok),
        "valid_gate_passed": bool(valid_ok), "adopted": bool(oof_ok and valid_ok),
        "new_incumbent": winner if (oof_ok and valid_ok) else incumbent,
    }
    (PATHS.results / "steps").mkdir(parents=True, exist_ok=True)
    (PATHS.results / "steps" / f"{step}.json").write_text(json.dumps(decision, indent=2, default=float))
    ladder = read_ladder()
    ladder["incumbent"] = decision["new_incumbent"]
    ladder["history"].append({"step": step, "winner": winner, "adopted": decision["adopted"],
                              "incumbent_after": decision["new_incumbent"]})
    write_ladder(ladder)
    log.info("step %s: winner %s (OOF BA %.4f vs %.4f, Holm p %.3g), valid gate %s -> %s", step, winner,
             oof_ba[winner], decision["incumbent_oof_ba"], p_adj[winner], valid_ok,
             "ADOPTED" if decision["adopted"] else "rejected")
    return decision


def best_of(names: list[str]) -> str:
    """Highest OOF BA among arms (used to build e/f on top of the best of S1 a-d)."""
    return max(names, key=lambda n: arm_record(n)["oof"]["seed_avg"]["ba"])
