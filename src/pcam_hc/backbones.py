"""Backbone registry and a uniform frozen-feature encoder.

Global context: every foundation model in the study (climb target Hibou-L, fusion/B* candidates,
controls) is described by a `BackboneSpec`: pinned revision, eva-recipe normalisation
(https://kaiko-ai.github.io/eva/main/user-guide/advanced/replicate_evaluations/), and licence
metadata that the publish/site exporters use to refuse non-redistributable derivatives
(pre-registration section 9). `Encoder` turns uint8 PCam patches into pooled features:
CLS and mean-of-patch-tokens (registers excluded) at chosen blocks, optionally the mean of the
centre 6x6 patch tokens, which cover the centre 32 px that defines the PCam label.
"""

import json
import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from pcam_hc.config import PATHS
from pcam_hc.data import preprocess

log = logging.getLogger(__name__)

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HIBOU = ((0.7068, 0.5755, 0.722), (0.195, 0.2316, 0.1816))  # eva recipe / Hibou preprocessor_config.json
H_OPTIMUS = ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))  # eva recipe / H-optimus-0 card
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Midnight card

APACHE_NOTICE = "Licensed under the Apache License 2.0; derived artifacts keep attribution and the upstream notices."
NON_CLINICAL = " Not for clinical use or medical decision-making."


@dataclass(frozen=True)
class BackboneSpec:
    """Static description of a backbone. `kind` selects the loader: 'hf' (transformers AutoModel),
    'hibou' (native Dinov2WithRegistersModel with Hibou weights), 'timm' (timm hf-hub), or
    'random_reg' (Hibou-L architecture, random init)."""

    name: str
    repo: str
    revision: str
    kind: str
    mean: tuple
    std: tuple
    dim: int
    licence: str = ""
    publishable: bool = True
    notice: str = ""
    timm_kwargs: dict = field(default_factory=dict)
    active: bool = True


REGISTRY = {
    s.name: s
    for s in [
        # Climb target (eva PCam test BA 0.955) and its same-family smaller sibling.
        BackboneSpec("hibou_l", "histai/hibou-L", "31a3b9fee16554c8552cb643d311733bce8faf7c", "hibou", *HIBOU, 1024,
                     "Apache-2.0", True, APACHE_NOTICE),
        BackboneSpec("hibou_b", "histai/hibou-b", "7fdc8cce365ebfd9e7e60ee6ee9a8d4d59388f3c", "hibou", *HIBOU, 768,
                     "Apache-2.0", True, APACHE_NOTICE),
        # Fusion / B* candidates.
        BackboneSpec("prov_gigapath", "prov-gigapath/prov-gigapath", "64f9e26c15019f2d4f6d9113c6822f88bb16b01b", "timm",
                     *IMAGENET, 1536, "Apache-2.0", True,
                     APACHE_NOTICE + " Intended for research and reproducibility only; deployment is out of scope." + NON_CLINICAL),
        BackboneSpec("h_optimus_0", "bioptimus/H-optimus-0", "b145cc1e6c6b30d3251aa8b1f844e6974188a743", "timm",
                     *H_OPTIMUS, 1536, "Apache-2.0", True,
                     APACHE_NOTICE + " Provided as-is; not reviewed or approved by any regulatory body." + NON_CLINICAL,
                     {"init_values": 1e-5, "dynamic_img_size": False}),
        BackboneSpec("virchow2", "paige-ai/Virchow2", "3158645804b69e3f3bc4439d4116edddf0840a72", "timm",
                     *IMAGENET, 1280, "CC-BY-NC-ND-4.0", False,
                     "Non-commercial academic research only; no redistribution of the model or derivatives." + NON_CLINICAL,
                     {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"}),
        BackboneSpec("virchow", "paige-ai/Virchow", "19eebc84ae33e79f1b2d866e6ff90ae50e522f9a", "timm",
                     *IMAGENET, 1280, "Apache-2.0", True, APACHE_NOTICE + NON_CLINICAL,
                     {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"}),
        BackboneSpec("midnight", "kaiko-ai/midnight", "adc6b15679c981cce6f9b018bbad09d16eeeda9f", "hf", *HALF, 1536,
                     "MIT", True, "MIT licence; keep the copyright notice."),
        BackboneSpec("openmidnight", "medarc/OpenMidnight-1.1", "a9ed8ab814124bcec8ab46c746b2bd3cc7e35b0e", "timm",
                     *IMAGENET, 1536, "Apache-2.0", True, APACHE_NOTICE),
        BackboneSpec("phikon", "owkin/phikon", "057cc0295895c2df3dd7681a89680da6015cbefe", "hf", *IMAGENET, 768,
                     "Owkin non-commercial", False, "Non-commercial research use only; derivatives not redistributed."),
        BackboneSpec("phikon_v2", "owkin/phikon-v2", "2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd", "hf", *IMAGENET, 1024,
                     "Owkin non-commercial", False, "Non-commercial research use only; derivatives not redistributed."),
        # Controls: same architecture as Hibou-L with natural-image weights, and random weights.
        BackboneSpec("dinov2_reg_l", "facebook/dinov2-with-registers-large", "e4c89a4e05589de9b3e188688a303d0f3c04d0f3",
                     "hf", *IMAGENET, 1024, "Apache-2.0", True, APACHE_NOTICE),
        BackboneSpec("random_vitl_reg4", "histai/hibou-L", "31a3b9fee16554c8552cb643d311733bce8faf7c", "random_reg",
                     *HIBOU, 1024, "Apache-2.0", True, "Random weights; only the architecture config is used."),
        # Click-through licences not accepted on 2026-10-31: inactive unless the pool check admits them.
        BackboneSpec("uni2_h", "MahmoodLab/UNI2-h", "d517a8dd47902dd7c308b3c36f63bce47e7b9a43", "timm", *IMAGENET, 1536,
                     "CC-BY-NC-ND-4.0", False, "Non-commercial academic research only.", active=False),
    ]
}


def _hibou_native(spec: BackboneSpec, random_init: bool = False) -> nn.Module:
    """Hibou weights in transformers' native Dinov2WithRegistersModel (no remote code).

    Hibou ships a custom DINOv2-with-registers implementation (`trust_remote_code`). Its embedding
    order [CLS, 4 registers, patches] and position-embedding handling match the native class, so
    the pinned safetensors are loaded into it; loading must report no missing/unexpected keys, and
    tests/check_parity compare outputs against the remote code."""
    from huggingface_hub import hf_hub_download
    from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

    with open(hf_hub_download(spec.repo, "config.json", revision=spec.revision)) as f:
        raw = json.load(f)
    keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "mlp_ratio", "hidden_act", "layer_norm_eps",
            "image_size", "patch_size", "num_channels", "qkv_bias", "layerscale_value", "use_swiglu_ffn",
            "num_register_tokens", "drop_path_rate", "hidden_dropout_prob", "attention_probs_dropout_prob",
            "initializer_range", "apply_layernorm"]
    cfg = Dinov2WithRegistersConfig(**{k: raw[k] for k in keys if k in raw})
    if random_init:
        torch.manual_seed(0)
        return Dinov2WithRegistersModel(cfg)
    model, info = Dinov2WithRegistersModel.from_pretrained(
        spec.repo, revision=spec.revision, config=cfg, output_loading_info=True, attn_implementation="sdpa")
    bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    if bad:
        raise RuntimeError(f"Hibou weights do not map cleanly onto the native class: {bad}")
    return model


def load_model(spec: BackboneSpec) -> nn.Module:
    """Instantiate the pretrained (or random) network for a spec, in eval mode, on CPU."""
    if spec.kind == "hibou":
        model = _hibou_native(spec)
    elif spec.kind == "random_reg":
        model = _hibou_native(spec, random_init=True)
    elif spec.kind == "hf":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(spec.repo, revision=spec.revision, attn_implementation="sdpa")
    elif spec.kind == "timm":
        import timm
        from timm.layers import SwiGLUPacked

        kwargs = dict(spec.timm_kwargs)
        if kwargs.get("mlp_layer") == "SwiGLUPacked":  # Virchow/Virchow2 cards require these layers
            kwargs["mlp_layer"], kwargs["act_layer"] = SwiGLUPacked, torch.nn.SiLU
        model = timm.create_model(f"hf-hub:{spec.repo}@{spec.revision}", pretrained=True, **kwargs)
    else:
        raise ValueError(spec.kind)
    return model.eval()


class Encoder(nn.Module):
    """Frozen feature extractor: uint8 (B,3,96,96) -> float features.

    Output layout, for each block in `layers` (default: last block only): [CLS, mean(patch tokens)],
    then, if `center`, the mean of the centre `center_tokens` x `center_tokens` patch tokens of the
    last block. Intermediate blocks get the model's final LayerNorm, as in DINOv2's
    `get_intermediate_layers(..., norm=True)` linear-evaluation protocol."""

    def __init__(self, model: nn.Module, spec: BackboneSpec, device="cuda", dtype=torch.bfloat16,
                 layers: tuple | None = None, center: bool = False, center_tokens: int = 6):
        super().__init__()
        self.model, self.spec, self.device, self.dtype = model.to(device=device, dtype=dtype), spec, device, dtype
        self.is_timm = hasattr(model, "forward_features") and hasattr(model, "num_prefix_tokens")
        cfg = getattr(model, "config", None)
        self.n_layers = cfg.num_hidden_layers if cfg is not None else len(model.blocks)
        self.n_prefix = model.num_prefix_tokens if self.is_timm else 1 + getattr(cfg, "num_register_tokens", 0)
        self.layers, self.center, self.center_tokens = layers or (self.n_layers,), center, center_tokens
        if self.is_timm and self.layers != (self.n_layers,):
            raise NotImplementedError("multi-layer features are only implemented for transformers models")
        self.out_dim = 2 * spec.dim * len(self.layers) + (spec.dim if center else 0)

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        """eva preprocessing on the target device, cast to the model dtype."""
        return preprocess(x.to(self.device), self.spec.mean, self.spec.std).to(self.dtype)

    def _token_maps(self, pixels: torch.Tensor) -> dict:
        """Normalised token sequences (B, T, D) for each requested block."""
        if self.is_timm:
            return {self.n_layers: self.model.forward_features(pixels)}
        need_hidden = self.layers != (self.n_layers,)
        out = self.model(pixel_values=pixels, output_hidden_states=need_hidden)
        maps = {self.n_layers: out.last_hidden_state}
        for layer in self.layers:
            if layer != self.n_layers:
                maps[layer] = self.model.layernorm(out.hidden_states[layer])
        return maps

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self._token_maps(self.prepare(x))
        feats = []
        for layer in self.layers:
            t = maps[layer]
            feats += [t[:, 0], t[:, self.n_prefix :].mean(1)]  # CLS, mean of patch tokens (registers excluded)
        if self.center:
            patches = maps[self.n_layers][:, self.n_prefix :]
            side = int(patches.shape[1] ** 0.5)  # 16 for 224 px / patch 14
            lo = (side - self.center_tokens) // 2
            grid = patches.reshape(patches.shape[0], side, side, -1)
            feats.append(grid[:, lo : lo + self.center_tokens, lo : lo + self.center_tokens].mean((1, 2)))
        return torch.cat(feats, dim=-1).float()


def check_access(names=None) -> dict:
    """Pre-registered pool rule: which registry models the logged-in account can load right now."""
    from huggingface_hub import auth_check

    status = {}
    for name in names or REGISTRY:
        spec = REGISTRY[name]
        try:
            auth_check(spec.repo)
            status[name] = True
        except Exception as e:  # GatedRepoError / RepositoryNotFoundError
            log.info("no access to %s: %s", spec.repo, type(e).__name__)
            status[name] = False
    PATHS.results.mkdir(parents=True, exist_ok=True)
    (PATHS.results / "model_pool.json").write_text(json.dumps(status, indent=2))
    return status
