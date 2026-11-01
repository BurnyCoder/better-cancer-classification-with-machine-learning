"""The test split is sealed until the ladder is frozen (pre-registration section 3).

Global context: `assert_test_unsealed` is called by every code path that would touch test images
or test embeddings; it requires the git tag `ladder-frozen` and a committed frozen-config hash.
"""

import subprocess

import pytest

from pcam_hc.seal import TestSealedError, assert_test_unsealed, config_hash


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_seal_blocks_until_tag_and_hash_match(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    arms = {"S0": {"backbone": "midnight", "head": "linear"}}
    with pytest.raises(TestSealedError):
        assert_test_unsealed(arms, repo=repo)  # no tag yet
    (repo / "frozen_ladder.json").write_text(config_hash(arms))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "freeze")
    _git(repo, "tag", "ladder-frozen")
    assert_test_unsealed(arms, repo=repo)  # tag + matching hash -> allowed
    with pytest.raises(TestSealedError):
        assert_test_unsealed({"S0": {"backbone": "midnight", "head": "mlp"}}, repo=repo)  # config changed


def test_config_hash_is_order_invariant():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})


def test_test_images_and_labels_are_guarded(monkeypatch):
    """Reading test images or labels outside the pre-registered audit goes through the seal."""
    import pcam_hc.seal as seal
    from pcam_hc.data import PCamImages, load_labels

    def sealed():
        raise TestSealedError("sealed")

    monkeypatch.setattr(seal, "frozen_arms", sealed)
    with pytest.raises(TestSealedError):
        PCamImages("test")
    with pytest.raises(TestSealedError):
        load_labels("test")
