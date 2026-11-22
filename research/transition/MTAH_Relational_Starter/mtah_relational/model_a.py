"""Spatial feature-only reconstruction model. No graph, owner or contest head."""
from __future__ import annotations

import torch
from torch import nn


class SpatialBiLSTM(nn.Module):
    input_dim = 53
    target_dim = 13
    embedding_dim = 64

    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(53, 64), nn.LayerNorm(64), nn.ELU(),
                                     nn.Linear(64, 64), nn.LayerNorm(64), nn.ELU())
        self.temporal = nn.LSTM(64, 32, num_layers=1, batch_first=True, bidirectional=True)
        self.reconstruction = nn.Sequential(nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 13))

    def forward(self, corrupted_features, padding_mask=None):
        x = corrupted_features
        if x.ndim != 3 or x.shape[-1] != 53 or x.shape[1] == 0:
            raise ValueError("Model A requires [B,T,53] corrupted feature inputs")
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if padding_mask.shape != x.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("Invalid temporal padding mask")
        if not torch.isfinite(x.masked_fill(padding_mask[..., None], 0)).all():
            raise FloatingPointError("Nonfinite visible model input")
        encoded = self.spatial(x.masked_fill(padding_mask[..., None], 0))
        if not padding_mask.any():
            embedding, _ = self.temporal(encoded)
        else:
            # Real context is contiguous. Do not let left/right zeros drive the
            # recurrent states. This avoids packed-sequence backend dependence.
            embedding = encoded * 0
            for b in range(len(x)):
                times = torch.nonzero(~padding_mask[b], as_tuple=False).flatten()
                if not len(times):
                    continue
                left, right = int(times[0]), int(times[-1])+1
                if right-left != len(times):
                    raise ValueError("Padding cannot bridge an internal record gap")
                state, _ = self.temporal(encoded[b:b+1, left:right])
                embedding[b, left:right] = state[0]
        predicted = self.reconstruction(embedding).masked_fill(padding_mask[..., None], 0)
        embedding = embedding.masked_fill(padding_mask[..., None], 0)
        if not torch.isfinite(predicted).all() or not torch.isfinite(embedding).all():
            raise FloatingPointError("Nonfinite Model A output")
        return predicted, embedding


def save_checkpoint(path, model, **metadata):
    """Only tensors/primitives; load with weights_only=True on the local machine."""
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(version="model-a-v1", model_state=state, metadata=metadata), path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != "model-a-v1":
        raise ValueError("Unrecognized Model A checkpoint")
    model = SpatialBiLSTM()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device), checkpoint["metadata"]
