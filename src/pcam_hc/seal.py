"""Test-set seal (pre-registration section 3).

Global context: the PCam test split may only be touched after every arm's configuration is frozen.
Freezing = committing `frozen_ladder.json` (a hash of all arm configs) and tagging that commit
`ladder-frozen`. Any code path that would read test images, embed test patches or score test
predictions calls `assert_test_unsealed(arms)` first.
"""

import hashlib
import json
import subprocess
from pathlib import Path

from pcam_hc.config import PROJECT_ROOT

TAG = "ladder-frozen"
FROZEN_FILE = "frozen_ladder.json"


class TestSealedError(RuntimeError):
    """Raised when test data is requested before the ladder is frozen, or after configs changed."""

    __test__ = False  # not a pytest test class despite the name


def config_hash(arms: dict) -> str:
    """Order-invariant SHA-256 of the arm configurations."""
    return hashlib.sha256(json.dumps(arms, sort_keys=True, default=str).encode()).hexdigest()


def frozen_arms() -> dict:
    """The arm configurations the ladder ended with (written by the experiments when freezing)."""
    path = PROJECT_ROOT / "results" / "ladder.json"
    if not path.exists():
        raise TestSealedError("test split is sealed: results/ladder.json does not exist yet")
    return json.loads(path.read_text())


def assert_test_unsealed(arms: dict, repo: Path = PROJECT_ROOT) -> None:
    """Allow test access only if the `ladder-frozen` tag exists and its frozen hash matches `arms`."""
    tags = subprocess.run(["git", "tag", "-l", TAG], cwd=repo, capture_output=True, text=True).stdout.split()
    if TAG not in tags:
        raise TestSealedError(f"test split is sealed: git tag '{TAG}' does not exist")
    frozen = subprocess.run(["git", "show", f"{TAG}:{FROZEN_FILE}"], cwd=repo, capture_output=True, text=True)
    if frozen.returncode != 0 or frozen.stdout.strip() != config_hash(arms):
        raise TestSealedError("arm configurations differ from the frozen ladder; test access refused")
