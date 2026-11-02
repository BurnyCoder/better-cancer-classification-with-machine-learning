"""Licence guard: derivatives of non-redistributable backbones must never be exported.

Global context: pre-registration section 9. Virchow2 (CC-BY-NC-ND-4.0) and Phikon / Phikon-v2
(Owkin non-commercial) may be used locally for research, but no weights, embeddings, heads or
per-patch predictions derived from them may be published.
"""

from pcam_hc.backbones import REGISTRY
from pcam_hc.experiments import lineage_publishable


def test_non_redistributable_backbones_are_flagged():
    for name in ("virchow2", "phikon", "phikon_v2", "uni2_h"):
        assert not REGISTRY[name].publishable
    for name in ("hibou_l", "hibou_b", "h_optimus_0", "prov_gigapath", "virchow", "midnight", "openmidnight"):
        assert REGISTRY[name].publishable


def test_any_restricted_backbone_taints_the_lineage():
    assert lineage_publishable(["hibou_l", "prov_gigapath"])
    assert not lineage_publishable(["hibou_l", "virchow2"])
    assert not lineage_publishable(["phikon"])
    assert lineage_publishable([])  # e.g. the colour-histogram control


def test_publish_gate_refuses_restricted_lineage():
    import pytest

    from pcam_hc.publish import LicenceError, assert_publishable

    assert_publishable({"name": "ok", "backbones": ["hibou_l", "virchow"]})
    for bad in (["hibou_l", "virchow2"], ["phikon_v2"]):
        with pytest.raises(LicenceError):
            assert_publishable({"name": "bad", "backbones": bad})


def test_model_card_credits_every_backbone_without_a_dataset_mirror():
    """The card states Apache-2.0, names each backbone's source repo and licence, and does not point at a
    third-party PCam mirror (the data come from Zenodo 2546921)."""
    from pcam_hc.publish import model_card, notice

    recs = [{"name": "a", "backbones": ["hibou_l"], "valid": {"seed_avg": {"ba": 0.9}}},
            {"name": "b", "backbones": ["dinov2_reg_l"], "valid": {"seed_avg": {"ba": 0.8}}},
            {"name": "c", "backbones": [], "valid": {"seed_avg": {"ba": 0.7}}}]
    card = model_card(recs)
    assert "license: apache-2.0" in card and "datasets:" not in card and "zenodo.org/records/2546921" in card
    for repo in ("histai/hibou-L", "facebook/dinov2-with-registers-large"):
        assert repo in card and repo in notice(recs)


def test_apache_licence_text_ships_with_the_artifacts():
    from pcam_hc.publish import ARTIFACT_LICENSE

    assert ARTIFACT_LICENSE.read_text().lstrip().startswith("Apache License")
