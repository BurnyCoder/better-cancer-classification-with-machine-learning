"""Embedding cache: CLS / mean-patch pooling on a tiny randomly initialised DINOv2, resumability.

Global context: every frozen-feature arm reads embeddings from this cache, so pooling and
resumption after an interruption must be exact.
"""

import numpy as np
import torch
from transformers import Dinov2Config, Dinov2Model

from pcam_hc.backbones import BackboneSpec, Encoder
from pcam_hc.embed import embed_images


def _tiny_encoder():
    torch.manual_seed(0)
    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                       patch_size=14, image_size=224)
    spec = BackboneSpec(name="tiny", repo="", revision="", kind="hf", mean=(0.5,) * 3, std=(0.5,) * 3, dim=32)
    return Encoder(Dinov2Model(cfg).eval(), spec, device="cpu", dtype=torch.float32)


def test_pooling_returns_cls_and_mean_patch():
    enc = _tiny_encoder()
    x = torch.randint(0, 256, (3, 3, 96, 96), dtype=torch.uint8)
    out = enc(x)
    tokens = enc.model(pixel_values=enc.prepare(x)).last_hidden_state
    assert out.shape == (3, 64)
    assert torch.allclose(out[:, :32], tokens[:, 0], atol=1e-5)
    assert torch.allclose(out[:, 32:], tokens[:, 1:].mean(1), atol=1e-5)


def test_embed_images_is_resumable(tmp_path):
    enc = _tiny_encoder()
    imgs = np.random.default_rng(0).integers(0, 256, (10, 96, 96, 3), dtype=np.uint8)
    read = lambda idx: imgs[idx]  # noqa: E731 - stands in for the HDF5 reader
    path = tmp_path / "e.npy"
    full = embed_images(enc, read, n=10, path=path, batch_size=4, view="full")
    # Simulate a crash after the first batch: reset the progress marker and recompute.
    (tmp_path / "e.npy.progress").write_text("4")
    again = embed_images(enc, read, n=10, path=path, batch_size=4, view="full")
    assert np.allclose(full, again, atol=1e-3) and full.shape == (10, 64)
