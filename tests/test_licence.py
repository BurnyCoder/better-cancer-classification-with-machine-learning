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
