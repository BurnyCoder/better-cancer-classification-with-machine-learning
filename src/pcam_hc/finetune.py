"""LoRA fine-tuning of Hibou-L (ladder S5) with the pre-registered recipe, plus WiSE scaling and soups (S5b).

Global context: pre-registration section 6, S5/S5b.
  * LoRA (Hu et al. 2021, https://arxiv.org/abs/2106.09685) via `peft` on the attention
    query/key/value/output projections of all 24 blocks.
  * LP-FT (Kumar et al. 2022, https://arxiv.org/abs/2202.10054): the classifier starts from the
    S2 probe head (including its z-scoring), so fine-tuning begins at the linear-probe solution.
  * Augmentation: D4 (random rotation/flip) + HED-light stain jitter (Tellez et al. 2019,
    https://arxiv.org/abs/1902.06543): alpha ~ U(1-s, 1+s), beta ~ U(-s, s) per HED channel, s = 0.05.
  * Drop-path 0.1, EMA of the trainable weights (decay 0.999), AdamW + cosine schedule with warm-up,
    bf16 autocast, gradient checkpointing.
  * WiSE-FT-style interpolation (Wortsman et al. 2022, https://arxiv.org/abs/2109.01903): the LoRA
    update is scaled by alpha and the head interpolated between the probe and the fine-tuned head.
"""

import logging
import math
import time

import numpy as np
import torch
from torch import nn

from pcam_hc.data import PCamImages, d4_view, preprocess

log = logging.getLogger(__name__)
# LoRA targets: transformers 5 names the attention projections q_proj/k_proj/v_proj/o_proj.
LORA_TARGETS = r".*attention\.(q_proj|k_proj|v_proj|o_proj)"

# Ruifrok & Johnston stain vectors as used by scikit-image (skimage.color.rgb_from_hed / hed_from_rgb):
# https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=torch.float64)
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)
LOG_ADJUST = math.log(1e-6)


def hed_jitter(x: torch.Tensor, sigma: float, generator: torch.Generator | None = None,
               alpha: torch.Tensor | None = None, beta: torch.Tensor | None = None) -> torch.Tensor:
    """HED colour augmentation on uint8 (B,3,H,W); returns uint8.

    Mirrors skimage's separate_stains/combine_stains: stains = log(max(rgb,1e-6))/log(1e-6) @ hed_from_rgb,
    clipped at 0; perturbed stains s*alpha + beta; rgb = exp(-(stains * -log(1e-6)) @ rgb_from_hed)."""
    b = x.shape[0]
    dev = x.device
    if alpha is None:
        alpha = 1 + (torch.rand(b, 1, 3, generator=generator) * 2 - 1) * sigma
        beta = (torch.rand(b, 1, 3, generator=generator) * 2 - 1) * sigma
    rgb = x.permute(0, 2, 3, 1).reshape(b, -1, 3).double().to(dev) / 255.0
    stains = (torch.log(rgb.clamp_min(1e-6)) / LOG_ADJUST) @ HED_FROM_RGB.to(dev)
    stains = stains.clamp_min(0) * alpha.to(dev).double() + beta.to(dev).double()
    out = torch.exp(-(stains * -LOG_ADJUST) @ RGB_FROM_HED.to(dev)).clamp(0, 1)
    out = (out * 255).round().to(torch.uint8).reshape(b, x.shape[2], x.shape[3], 3).permute(0, 3, 1, 2)
    return out.contiguous()


def augment(x: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    """Random D4 element per image, then HED jitter. Note on strength: in scikit-image >= 0.19 stain units
    sigma = 0.05 shifts each stain by up to 0.05 * ln(1e6) = 0.69 optical density, which changes pixels by
    about 36/255 on average (measured on PCam valid); Tellez et al. 2019 call sigma = 0.05 "HED-light" but
    do not state the units."""
    ks = torch.randint(0, 8, (x.shape[0],), generator=generator)
    x = torch.stack([d4_view(img, int(k)) for img, k in zip(x, ks)])
    return hed_jitter(x, sigma, generator)


def pool_tokens(maps: dict, rep, n_prefix: int, last: int, center_tokens: int = 6) -> torch.Tensor:
    """Pooled features of a `features.Representation` from token maps {block: (B, T, D)}, in the same
    column order as the embedding cache: each (block, cls|mean) part in order, then centre tokens."""
    feats = []
    for layer, kind in rep.parts:
        t = maps[last if layer == -1 else layer]
        feats.append(t[:, 0] if kind == "cls" else t[:, n_prefix:].mean(1))
    if rep.center:
        patches = maps[last][:, n_prefix:]
        side = int(patches.shape[1] ** 0.5)
        lo = (side - center_tokens) // 2
        grid = patches.reshape(patches.shape[0], side, side, -1)
        feats.append(grid[:, lo : lo + center_tokens, lo : lo + center_tokens].mean((1, 2)))
    return torch.cat(feats, -1)


def build_head(head_state: dict) -> nn.Module:
    """Rebuild a probe head (linear / logreg = nn.Linear, mlp = the S2 MLP) from its saved state."""
    sd = head_state["module"]
    if head_state.get("kind") == "mlp":
        d, h = sd["0.weight"].shape[1], sd["0.weight"].shape[0]
        head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))
    else:
        head = nn.Linear(sd["weight"].shape[1], 1)
    head.load_state_dict(sd)
    return head


class LoraClassifier(nn.Module):
    """Hibou-L + LoRA adapters + (z-scored) linear head on the ladder's representation.

    The representation (blocks, pooling, centre tokens) and image views (e.g. full + centre crop)
    are those selected in S1, and the head and z-scoring are the S2 probe's (LP-FT), so fine-tuning
    starts exactly at the frozen-probe solution."""

    def __init__(self, backbone: nn.Module, spec, head_state: dict, rep, views=("full",),
                 rank: int = 8, drop_path: float = 0.1):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        backbone.config.drop_path_rate = drop_path
        for layer in backbone.encoder.layer:  # re-create drop-path modules with the new rate
            if hasattr(layer, "drop_path"):
                from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
                    Dinov2WithRegistersDropPath,
                )

                layer.drop_path = Dinov2WithRegistersDropPath(drop_path) if drop_path > 0 else nn.Identity()
        cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none", target_modules=LORA_TARGETS)
        self.backbone = get_peft_model(backbone, cfg)
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.spec, self.rep, self.views = spec, rep, tuple(views)
        self.n_layers = backbone.config.num_hidden_layers
        self.rep_layers = tuple(sorted({self.n_layers if l == -1 else l for l, _ in rep.parts} | {self.n_layers}))
        self.head = build_head(head_state)
        d = self.head[0].in_features if isinstance(self.head, nn.Sequential) else self.head.in_features
        self.register_buffer("mu", head_state["mu"] if head_state.get("mu") is not None else torch.zeros(d))
        self.register_buffer("sd", head_state["sd"] if head_state.get("sd") is not None else torch.ones(d))
        self.n_prefix = 1 + backbone.config.num_register_tokens

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        need_hidden = self.rep_layers != (self.n_layers,)
        out = self.backbone(pixel_values=pixels, output_hidden_states=need_hidden)
        maps = {self.n_layers: out.last_hidden_state}
        base = self.backbone.base_model.model
        for layer in self.rep_layers:  # intermediate blocks get the final LayerNorm, as in the cache
            if layer != self.n_layers:
                maps[layer] = base.layernorm(out.hidden_states[layer])
        return pool_tokens(maps, self.rep, self.n_prefix, self.n_layers)

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        from pcam_hc.embed import apply_view

        feats = torch.cat([self.features(preprocess(apply_view(x_uint8, v), self.spec.mean, self.spec.std))
                           for v in self.views], -1).float()
        with torch.autocast(device_type=feats.device.type, enabled=False):  # fp32 head: bf16 logits coarsen NLL/ECE
            return self.head((feats - self.mu) / self.sd).squeeze(-1)

    def trainable_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.named_parameters() if v.requires_grad}


class ConvNeXtClassifier(nn.Module):
    """End-to-end baseline: ImageNet ConvNeXt-T (Liu et al. 2022, https://arxiv.org/abs/2201.03545) via
    timm, one logit, same 96 -> 224 resize as the foundation models; all weights trainable."""

    def __init__(self, name: str = "convnext_tiny.fb_in22k_ft_in1k", drop_path: float = 0.1):
        super().__init__()
        import timm

        self.net = timm.create_model(name, pretrained=True, num_classes=1, drop_path_rate=drop_path)
        self.net.set_grad_checkpointing(True)  # full fine-tuning at 224 px must fit next to the desktop in 8 GB
        cfg = self.net.pretrained_cfg
        self.mean, self.std = tuple(cfg["mean"]), tuple(cfg["std"])

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.net(preprocess(x_uint8, self.mean, self.std)).squeeze(-1)

    def trainable_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.named_parameters() if v.requires_grad}


def train_lora(model: nn.Module, images: PCamImages, train_idx, y, eval_idx, y_eval, *, lr: float,
               epochs: float = 1.0, batch: int = 64, seed: int = 0, sigma: float = 0.05, ema_decay: float = 0.999,
               warmup_frac: float = 0.05, eval_every: int = 500, device: str = "cuda", log_every: int = 50) -> dict:
    """One training run of any classifier exposing `trainable_state()` and `forward(uint8 batch)` (the
    LoRA model or the ConvNeXt baseline); returns the best EMA trainable state (by eval BA), its eval
    logits and a history."""
    from pcam_hc.metrics import balanced_accuracy

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total = int(math.ceil(len(train_idx) * epochs / batch))
    warm = max(1, int(warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ema = {k: v.clone() for k, v in model.trainable_state().items()}
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"ba": -1.0}
    history, t0, step = [], time.time(), 0
    order = np.concatenate([np.random.default_rng(seed + e).permutation(train_idx) for e in range(int(math.ceil(epochs)))])
    yt = torch.as_tensor(y, dtype=torch.float32)
    while step < total:
        idx = np.sort(order[step * batch : (step + 1) * batch])
        x = augment(torch.from_numpy(images.read(idx)).permute(0, 3, 1, 2).to(device), sigma, g)
        model.train()
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            loss = loss_fn(model(x), yt[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        # EMA of trainable weights with the num_updates warm-up of tf.train.ExponentialMovingAverage,
        # decay = min(decay, (1 + n) / (10 + n)): short runs (the sweep rungs) are not dominated by the start
        # weights. https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        d = min(ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for k, v in model.trainable_state().items():
                ema[k].mul_(d).add_(v, alpha=1 - d)
        if step % log_every == 0:
            log.info("lora seed %d step %d/%d loss %.4f lr %.2e %.1f img/s", seed, step, total, loss.item(),
                     sched.get_last_lr()[0], step * batch / (time.time() - t0))
        if step % eval_every == 0 or step == total:
            logits = predict_with_state(model, ema, images, eval_idx, device=device)
            ba = float(balanced_accuracy(y_eval, logits > 0))
            history.append({"step": step, "eval_ba": ba, "loss": float(loss.item())})
            log.info("lora seed %d step %d eval BA %.4f (EMA)", seed, step, ba)
            if ba > best["ba"]:
                best = {"ba": ba, "step": step, "state": {k: v.cpu().clone() for k, v in ema.items()},
                        "eval_logits": logits}
    best["history"] = history
    best["img_per_s"] = step * batch / (time.time() - t0)
    return best


@torch.no_grad()
def predict_with_state(model: nn.Module, state: dict | None, images: PCamImages, idx, view: int = 0,
                       batch: int = 256, device: str = "cuda") -> np.ndarray:
    """Logits for images[idx] (D4 view `view`) using `state` for the trainable weights (e.g. EMA)."""
    current = model.trainable_state()
    if state is not None:
        load_trainable(model, state)
    model.eval()
    out, idx = [], np.asarray(idx)
    for s in range(0, len(idx), batch):
        chunk = idx[s : s + batch]
        order = np.argsort(chunk)  # HDF5 reads need increasing indices; results go back in the caller's order
        x = torch.from_numpy(images.read(chunk[order])).permute(0, 3, 1, 2)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            logits = model(d4_view(x, view).to(device)).float().cpu()
        out.append(logits[torch.from_numpy(np.argsort(order))])
    load_trainable(model, current)
    return torch.cat(out).numpy()


def load_trainable(model: nn.Module, state: dict) -> None:
    params = dict(model.named_parameters())
    for k, v in state.items():
        params[k].data.copy_(v.to(params[k].device))


def wise_state(ft_state: dict, lp_head: dict, alpha: float) -> dict:
    """WiSE-style interpolation: LoRA B matrices scaled by alpha (delta-W = alpha * B A), head
    interpolated (1 - alpha) * probe + alpha * fine-tuned."""
    out = {}
    for k, v in ft_state.items():
        if "lora_B" in k:
            out[k] = v * alpha
        elif k.startswith("head."):
            out[k] = (1 - alpha) * lp_head[k.split(".", 1)[1]].to(v) + alpha * v
        else:
            out[k] = v.clone()
    return out


def soup_state(states: list[dict]) -> dict:
    """Uniform soup of LoRA runs (Wortsman et al. 2022, https://arxiv.org/abs/2203.05482) without
    approximation: the mean update (1/n) sum_i B_i A_i equals [B_1 ... B_n]/n @ [A_1; ...; A_n], i.e. one
    LoRA of rank n*r (build the model with rank n*r and lora_alpha n*r so the scaling stays 1). Heads
    are averaged."""
    n, out = len(states), {}
    for k in states[0]:
        if "lora_A" in k:
            out[k] = torch.cat([s[k] for s in states], 0)
        elif "lora_B" in k:
            out[k] = torch.cat([s[k] for s in states], 1) / n
        else:
            out[k] = sum(s[k] for s in states) / n
    return out
