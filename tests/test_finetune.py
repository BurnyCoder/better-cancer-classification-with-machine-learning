"""LoRA-stage utilities: HED stain jitter must match scikit-image's stain separation, WiSE scaling.

Global context: S5 augments with HED-light jitter (Tellez et al. 2019); the colour transform is
pinned against scikit-image (skimage.color.rgb2hed / hed2rgb), the reference implementation.
"""

import numpy as np
import torch
from skimage.color import hed2rgb, rgb2hed

from pcam_hc.finetune import hed_jitter, wise_state


def test_identity_jitter_matches_skimage_round_trip():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
    ref = np.stack([np.round(np.clip(hed2rgb(rgb2hed(im)), 0, 1) * 255) for im in x]).astype(np.uint8)
    ours = hed_jitter(torch.from_numpy(x).permute(0, 3, 1, 2), 0.0,
                      alpha=torch.ones(2, 1, 3), beta=torch.zeros(2, 1, 3)).permute(0, 2, 3, 1).numpy()
    assert np.abs(ours.astype(int) - ref.astype(int)).max() <= 1


def test_jitter_changes_colours_but_keeps_shape():
    x = torch.randint(0, 256, (4, 3, 96, 96), dtype=torch.uint8)
    y = hed_jitter(x, 0.05, generator=torch.Generator().manual_seed(0))
    assert y.shape == x.shape and y.dtype == torch.uint8 and not torch.equal(x, y)


def test_lora_classifier_starts_at_the_probe_solution():
    """LP-FT: with LoRA B = 0 the fine-tuning model must reproduce the frozen probe's logits exactly."""
    from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

    from pcam_hc.backbones import BackboneSpec, Encoder
    from pcam_hc.features import Representation
    from pcam_hc.finetune import LoraClassifier

    torch.manual_seed(0)
    cfg = Dinov2WithRegistersConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                    num_register_tokens=4, image_size=224, patch_size=14, use_swiglu_ffn=True)
    spec = BackboneSpec("tiny", "", "", "hf", (0.5,) * 3, (0.5,) * 3, 32)
    rep = Representation(((1, "cls"), (2, "cls"), (2, "mean")), center=True)
    backbone = Dinov2WithRegistersModel(cfg).eval()
    x = torch.randint(0, 256, (3, 3, 96, 96), dtype=torch.uint8)
    enc = Encoder(backbone, spec, device="cpu", dtype=torch.float32, layers=(1, 2), center=True)
    full = enc(x)  # cache layout: [cls1, mean1, cls2, mean2, centre]
    feats = torch.cat([full[:, 0:32], full[:, 64:96], full[:, 96:128], full[:, 128:160]], 1)
    head = torch.nn.Linear(feats.shape[1], 1)
    mu, sd = feats.mean(0), feats.std(0) + 1e-3
    probe = head((feats - mu) / sd).squeeze(-1)
    model = LoraClassifier(backbone, spec, {"module": head.state_dict(), "mu": mu, "sd": sd}, rep, rank=4).eval()
    with torch.no_grad():
        assert torch.allclose(model(x), probe, atol=1e-4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n or n.startswith("head.") for n in trainable)


def test_soup_of_lora_runs_is_the_exact_mean_update():
    from pcam_hc.finetune import soup_state

    g = torch.Generator().manual_seed(0)
    runs = [{"x.lora_A.w": torch.randn(4, 8, generator=g), "x.lora_B.w": torch.randn(6, 4, generator=g),
             "head.weight": torch.randn(1, 3, generator=g)} for _ in range(3)]
    soup = soup_state(runs)
    mean_delta = sum(r["x.lora_B.w"] @ r["x.lora_A.w"] for r in runs) / 3
    assert torch.allclose(soup["x.lora_B.w"] @ soup["x.lora_A.w"], mean_delta, atol=1e-5)
    assert torch.allclose(soup["head.weight"], sum(r["head.weight"] for r in runs) / 3)


def test_wise_alpha_endpoints():
    ft = {"lora_B.w": torch.ones(2, 2), "lora_A.w": torch.ones(2, 2), "head.weight": torch.ones(1, 2)}
    lp = {"weight": torch.zeros(1, 2)}
    s0, s1 = wise_state(ft, lp, 0.0), wise_state(ft, lp, 1.0)
    assert torch.equal(s0["lora_B.w"], torch.zeros(2, 2)) and torch.equal(s0["head.weight"], torch.zeros(1, 2))
    assert torch.equal(s1["lora_B.w"], ft["lora_B.w"]) and torch.equal(s1["head.weight"], ft["head.weight"])


def test_training_loop_runs_and_tracks_the_best_ema_checkpoint():
    """End-to-end smoke test of train_lora / predict_with_state on CPU with a tiny backbone."""
    import numpy as np
    from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

    from pcam_hc.backbones import BackboneSpec
    from pcam_hc.features import Representation
    from pcam_hc.finetune import LoraClassifier, predict_with_state, train_lora

    class Images:  # stands in for PCamImages: brighter images are "tumour"
        def __init__(self, n=64):
            rng = np.random.default_rng(0)
            self.y = rng.integers(0, 2, n)
            self.x = (rng.integers(0, 120, (n, 96, 96, 3)) + 120 * self.y[:, None, None, None]).astype(np.uint8)

        def read(self, idx):
            return self.x[np.asarray(idx)]

    torch.manual_seed(0)
    cfg = Dinov2WithRegistersConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                    num_register_tokens=4, image_size=224, patch_size=14, use_swiglu_ffn=True)
    spec = BackboneSpec("tiny", "", "", "hf", (0.5,) * 3, (0.5,) * 3, 32)
    head = torch.nn.Linear(32, 1)
    model = LoraClassifier(Dinov2WithRegistersModel(cfg), spec, {"module": head.state_dict(), "mu": None, "sd": None},
                           Representation(((-1, "cls"),)), rank=2)
    imgs = Images()
    run = train_lora(model, imgs, np.arange(48), imgs.y, np.arange(48, 64), imgs.y[48:], lr=1e-2, epochs=2.0,
                     batch=8, seed=0, eval_every=4, device="cpu", log_every=100)
    assert run["ba"] >= 0 and run["history"] and set(run["state"]) == {k for k, p in model.named_parameters() if p.requires_grad}
    logits = predict_with_state(model, run["state"], imgs, np.arange(48, 64), device="cpu")
    assert logits.shape == (16,) and np.isfinite(logits).all()
