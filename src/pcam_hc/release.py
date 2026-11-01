"""Per-classifier releases: every newly trained and evaluated classifier (or step decision) is pushed
with the regenerated paper, reports, README, site and Hugging Face artifacts.

Global context: the author's order of work asks for a full push after each trained classifier,
compared with the original Hibou-L (baseline) and the controls. `release` does one push:
  1. paired comparison of every ladder arm with the original probe (cached; comparisons.py),
  2. `report.build()`: tables, figures, experiment reports, README blocks, site data, paper PDF,
  3. git commit -> push -> pull request -> merge (GitHub Pages redeploys from main),
  4. upload of all licence-clean heads to the Hugging Face collection.
`watch` polls results/ and runs `release` whenever new arm or step files have settled.
"""

import json
import logging
import subprocess
import time

from pcam_hc.config import PATHS, PROJECT_ROOT

log = logging.getLogger(__name__)
STATE = PATHS.data / "release_state.json"
TRACKED = ["README.md", "reports", "results", "site"]


def _git(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=check, capture_output=True, text=True)


def _gh(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)


def release(title: str, body: str = "", hf: bool = True) -> None:
    from pcam_hc.comparisons import all_comparisons
    from pcam_hc.report import build

    all_comparisons()
    build()
    _git("add", *TRACKED)
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        log.info("release %s: nothing new to commit", title)
        return
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git("commit", "-m", title + ("\n\n" + body if body else ""))
    _git("push", "-u", "origin", branch)
    pr = _gh("pr", "create", "--base", "main", "--head", branch, "--title", title,
             "--body", body or "Automatic per-classifier release: results, reports, paper, README and site regenerated "
                                "by `main.py report`.").stdout.strip()
    _gh("pr", "merge", pr, "--merge")
    _git("pull", "--no-rebase", "origin", "main")  # fast-forward the working branch to the merge commit
    log.info("released %s (%s)", title, pr)
    if hf:
        from pcam_hc.publish import publishable_arms, upload

        log.info("Hugging Face: %s", upload(publishable_arms()))


def _settled_results(min_age: float = 60.0) -> set:
    now = time.time()
    files = list((PATHS.results / "arms").glob("*.json")) + list((PATHS.results / "steps").glob("*.json"))
    return {f"{p.parent.name}/{p.stem}" for p in files if now - p.stat().st_mtime > min_age}


def watch(poll: float = 120.0) -> None:
    """Release every new classifier/step decision as soon as its result file has settled."""
    released = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    while True:
        new = sorted(_settled_results() - released)
        if new:
            names = ", ".join(n.split("/")[1] for n in new)
            try:
                release(f"Release: {names}", f"New trained classifier(s) / decision(s): {names}. Compared with the "
                                             "original Hibou-L probe and the controls; paper, reports, README, site "
                                             "and Hugging Face updated.")
                released |= set(new)
                STATE.write_text(json.dumps(sorted(released)))
            except Exception as e:  # keep watching; the next poll retries
                log.exception("release failed: %s", e)
        time.sleep(poll)
