"""Classification heads on frozen embeddings: eva linear probe, logistic regression, MLP.

Global context: S0 reproduces the eva PCam head exactly (nn.Linear + BCE, AdamW lr 3e-4, batch 256,
<= 12,500 steps, early stopping on validation balanced accuracy with patience 3;
https://github.com/kaiko-ai/eva/blob/main/configs/vision/pathology/offline/classification/patch_camelyon.yaml).
S2 compares it with L2 logistic regression on z-scored features (sklearn's objective, solved by
full-batch L-BFGS on the GPU) and a one-hidden-layer MLP. All heads expose `.logits(X)`.
"""

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pcam_hc.config import EVA
from pcam_hc.metrics import balanced_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Head:
    """A fitted head: optional z-scoring (mu, sd) followed by a torch module producing one logit."""

    module: nn.Module
    mu: torch.Tensor | None = None
    sd: torch.Tensor | None = None

    @torch.no_grad()
    def logits(self, X, batch: int = 65536) -> np.ndarray:
        self.module.eval()
        out = []
        for s in range(0, len(X), batch):
            xb = _standardise(torch.as_tensor(np.asarray(X[s : s + batch]), device=DEVICE).float(), self.mu, self.sd)
            out.append(self.module(xb).squeeze(-1).double().cpu())
        return torch.cat(out).numpy()


def _standardise(x, mu, sd):
    return x if mu is None else (x - mu) / sd


def _zscore_stats(X) -> tuple:
    """Feature mean and std (floored), computed in float64 chunks to stay exact for 262k rows."""
    s = torch.zeros(X.shape[1], dtype=torch.float64, device=DEVICE)
    ss = torch.zeros_like(s)
    for i in range(0, len(X), 65536):
        xb = torch.as_tensor(np.asarray(X[i : i + 65536]), device=DEVICE).double()
        s += xb.sum(0)
        ss += (xb**2).sum(0)
    mu = s / len(X)
    sd = (ss / len(X) - mu**2).clamp_min(0).sqrt().clamp_min(1e-6)
    return mu.float(), sd.float()


def _train_sgd(module, X, y, seed, max_steps, lr, wd, batch, X_val=None, y_val=None, patience=EVA.patience,
               mu=None, sd=None):
    """Minibatch AdamW training with per-epoch shuffling; with validation data, eva-style early
    stopping on validation BA (checked every epoch, best checkpoint kept)."""
    g = torch.Generator().manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X), device=DEVICE, dtype=torch.float16)  # fits in 8 GB for <=~2k dims x 262k
    yt = torch.as_tensor(np.asarray(y), device=DEVICE, dtype=torch.float32)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    steps_per_epoch = math.ceil(len(yt) / batch)
    best, best_state, bad, step = -1.0, None, 0, 0
    while step < max_steps:
        perm = torch.randperm(len(yt), generator=g).to(DEVICE)
        module.train()
        for i in range(steps_per_epoch):
            idx = perm[i * batch : (i + 1) * batch]
            loss = loss_fn(module(_standardise(Xt[idx].float(), mu, sd)).squeeze(-1), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= max_steps:
                break
        if X_val is not None:  # validation once per epoch, as Lightning does by default
            ba = balanced_accuracy(y_val, Head(module, mu, sd).logits(X_val) > 0)
            if ba > best:
                best, best_state, bad = ba, copy.deepcopy(module.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        module.load_state_dict(best_state)
    return module


def _fit_logreg(X, y, C: float, max_iter: int = 500, chunk: int = 32768, init: Head | None = None) -> Head:
    """L2 logistic regression with sklearn's objective  C * sum(logloss) + 0.5 * ||w||^2  (intercept
    unpenalised; https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) on
    z-scored features, solved by full-batch L-BFGS with strong-Wolfe line search.

    Features stay float16 on the GPU and are standardised chunk by chunk inside the closure, so
    262k x 8k-dimensional problems fit in 8 GB. `init` warm-starts from another head (C path)."""
    mu, sd = (init.mu, init.sd) if init is not None else _zscore_stats(X)
    Xt = torch.as_tensor(np.asarray(X), device=DEVICE, dtype=torch.float16)
    yt = torch.as_tensor(np.asarray(y), device=DEVICE, dtype=torch.float32)
    lin = nn.Linear(Xt.shape[1], 1).to(DEVICE)
    if init is not None:
        lin.load_state_dict(init.module.state_dict())
    else:
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        # Objective divided by C * n for conditioning; the minimiser is unchanged.
        total = 0.0
        for s in range(0, len(yt), chunk):
            z = lin(_standardise(Xt[s : s + chunk].float(), mu, sd)).squeeze(-1)
            part = nn.functional.binary_cross_entropy_with_logits(z, yt[s : s + chunk], reduction="sum") / len(yt)
            part.backward()
            total += part.item()
        reg = 0.5 * (lin.weight**2).sum() / (C * len(yt))
        reg.backward()
        return torch.tensor(total + reg.item())

    opt.step(closure)
    return Head(lin, mu, sd)


def fit_head(kind: str, X, y, seed: int = 0, X_val=None, y_val=None, max_steps: int = EVA.max_steps,
             C: float = 1.0) -> Head:
    """Fit one head. kind: 'linear' (eva), 'logreg' (needs C), 'mlp'."""
    torch.manual_seed(seed)
    d = X.shape[1]
    if kind == "linear":
        module = _train_sgd(nn.Linear(d, 1).to(DEVICE), X, y, seed, max_steps, EVA.lr, EVA.weight_decay,
                            EVA.batch_size, X_val, y_val)
        return Head(module)
    if kind == "logreg":
        return _fit_logreg(X, y, C)
    if kind == "mlp":
        mu, sd = _zscore_stats(X)
        module = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 1)).to(DEVICE)
        module = _train_sgd(module, X, y, seed, max_steps, EVA.lr, EVA.weight_decay, EVA.batch_size,
                            X_val, y_val, mu=mu, sd=sd)
        return Head(module, mu, sd)
    raise ValueError(kind)
