"""Isolated A-model diagnostics, preserving the original checkpoint contract."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .corruption import SpanMasker
from .model_a import SpatialBiLSTM


class TokenSpatialBiLSTM(SpatialBiLSTM):
    """One learned 64-vector replaces the spatial encoding at artificial times.

    Initialization from an existing A makes this a function-preserving change.
    Original target/quality/feature validity NEVER enter the token path.
    """
    def __init__(self, original=None):
        super().__init__()
        if original is not None:
            self.load_state_dict(original.state_dict(), strict=True)
        with torch.no_grad():
            hidden = torch.zeros(1, 1, 53)
            hidden[..., 52] = 1
            initial = self.spatial(hidden)[0, 0].clone()
        self.mask_token = nn.Parameter(initial)
        self.spatial.register_forward_hook(self._substitute_token)

    def _substitute_token(self, module, args, encoded):
        return torch.where(args[0][..., 52:53] > .5, self.mask_token, encoded)


class SmallTemporal(nn.Module):
    """64-channel, symmetric depthwise temporal model, receptive field 85.

    This is an engineered-feature control, not a Dynamic Graph or Fusion model.
    """
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(53, 64), nn.LayerNorm(64), nn.ELU(),
                                     nn.Linear(64, 64), nn.LayerNorm(64), nn.ELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(64, 64, 5, padding=2*d, dilation=d, groups=64),
                          nn.ELU(), nn.Conv1d(64, 64, 1), nn.ELU())
            for d in (1, 4, 16)])
        self.reconstruction = nn.Sequential(nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 13))

    def forward(self, x, padding_mask=None):
        if x.ndim != 3 or x.shape[-1] != 53 or not torch.isfinite(x).all():
            raise ValueError("SmallTemporal expects finite corrupted [B,T,53]")
        if padding_mask is not None and padding_mask.any():
            raise ValueError("Diagnostic SmallTemporal requires full-context windows")
        h = self.spatial(x).transpose(1, 2)
        for block in self.blocks:
            h = h+block(h)
        h = h.transpose(1, 2)
        return self.reconstruction(h), h


@dataclass(frozen=True)
class MixedSpanMasker:
    seed: int = 2027

    def sample(self, length, *, center_frame, period, split, epoch=0):
        # Same routing-only randomness as before, with an independent choice RNG.
        choice = int(np.random.default_rng(np.random.SeedSequence([
            self.seed, epoch if split == "train" else 0, center_frame, period, 731
        ])).integers(3))
        lo, hi = ((8, 13), (25, 25), (50, 50))[choice]
        return SpanMasker(self.seed, lo, hi).sample(length, center_frame=center_frame,
                                                    period=period, split=split, epoch=epoch)


def context_gradient_probe(model, x, padding, hidden_time):
    """Aggregate input sensitivity at one hidden time, not causal importance."""
    model.eval()
    device = next(model.parameters()).device
    a = torch.as_tensor(x, device=device).clone().detach().requires_grad_(True)
    predicted, _ = model(a, torch.as_tensor(padding, device=device))
    gradient, = torch.autograd.grad(predicted[:, hidden_time].sum(), a)
    visible = (a[..., 52] < .5) & ~torch.as_tensor(padding, device=device)
    g = gradient[..., :24].masked_fill(~visible[..., None], 0)
    return dict(past_visible_value_gradient_norm=float(g[:, :hidden_time].norm().detach().cpu()),
                future_visible_value_gradient_norm=float(g[:, hidden_time+1:].norm().detach().cpu()),
                interpretation="local derivative sensitivity, not a causal explanation")


def covariance_diagnostics(embedding):
    a = np.asarray(embedding, dtype=np.float64)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Need at least two finite embeddings")
    eigen = np.maximum(np.linalg.eigvalsh(np.cov(a, rowvar=False))[::-1], 0)
    ratio = eigen/eigen.sum() if eigen.sum() else np.zeros(a.shape[1])
    positive = ratio[ratio > 0]
    cumulative = ratio.cumsum()
    return dict(frames=len(a), dimension_variance=a.var(axis=0).tolist(),
                explained_variance_ratio=ratio.tolist(), cumulative_explained_variance=cumulative.tolist(),
                effective_rank=float(np.exp(-np.sum(positive*np.log(positive)))) if len(positive) else 0.,
                dimensions_for_90pct=int(np.searchsorted(cumulative, .9)+1) if len(positive) else 0,
                dimensions_for_95pct=int(np.searchsorted(cumulative, .95)+1) if len(positive) else 0)
