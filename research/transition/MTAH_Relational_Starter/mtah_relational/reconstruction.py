"""Quality-weighted masked Huber and aggregate reconstruction metrics."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MaskedLoss:
    loss: torch.Tensor | None
    weight_sum: float
    valid_cells: int


def masked_huber(prediction, target, observed, quality_weight, artificial_mask,
                 feature_weights=None, delta=1.0):
    if prediction.shape != target.shape or observed.shape != target.shape \
            or quality_weight.shape != target.shape or artificial_mask.shape != target.shape[:-1] \
            or target.shape[-1] != 13 or delta <= 0:
        raise ValueError("Invalid reconstruction loss shapes or Huber delta")
    fw = torch.ones(13, device=prediction.device, dtype=prediction.dtype) if feature_weights is None \
        else torch.as_tensor(feature_weights, device=prediction.device, dtype=prediction.dtype)
    if fw.shape != (13,) or not torch.isfinite(fw).all() or (fw < 0).any():
        raise ValueError("Invalid target feature weights")
    valid = observed & artificial_mask[..., None] & torch.isfinite(target) \
        & torch.isfinite(quality_weight) & (quality_weight > 0) & (fw > 0)
    weights = torch.where(valid, quality_weight * fw, 0)
    denominator = weights.sum()
    if not bool(denominator > 0):
        return MaskedLoss(None, 0., 0)
    if not torch.isfinite(prediction[valid]).all():
        raise FloatingPointError("Nonfinite prediction at an eligible target")
    # Invalid NaN targets/predictions must be sanitized BEFORE arithmetic.
    safe_pred = torch.where(valid, prediction, 0)
    safe_target = torch.where(valid, target, 0)
    element = F.huber_loss(safe_pred, safe_target, reduction="none", delta=delta)
    loss = (element * weights).sum()/denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite masked Huber loss")
    return MaskedLoss(loss, float(denominator.detach().cpu()), int(valid.sum().detach().cpu()))


class ReconstructionMetrics:
    def __init__(self, delta=1.0):
        self.delta = delta
        self.weight = np.zeros(13)
        self.abs_error = np.zeros(13)
        self.huber = np.zeros(13)
        self.cells = np.zeros(13, dtype=np.int64)

    def update(self, prediction, target, observed, quality_weight, artificial_mask, feature_weights=None):
        fw = np.ones(13) if feature_weights is None else np.asarray(feature_weights)
        valid = observed & artificial_mask[..., None] & np.isfinite(target) \
            & np.isfinite(quality_weight) & (quality_weight > 0) & (fw > 0)
        if not np.isfinite(prediction[valid]).all():
            raise FloatingPointError("Nonfinite evaluation predictions")
        w = np.where(valid, quality_weight*fw, 0)
        error = np.abs(np.where(valid, prediction, 0)-np.where(valid, target, 0))
        huber = np.where(error <= self.delta, .5*error**2, self.delta*(error-.5*self.delta))
        axes = tuple(range(error.ndim-1))
        self.weight += w.sum(axis=axes)
        self.abs_error += (w*error).sum(axis=axes)
        self.huber += (w*huber).sum(axis=axes)
        self.cells += valid.sum(axis=axes)

    def report(self, names, schema_scale):
        total = float(self.weight.sum())
        per_feature = {}
        for k, name in enumerate(names):
            w = self.weight[k]
            mae = float(self.abs_error[k]/w) if w else None
            per_feature[name] = dict(mae_standardized=mae,
                                     mae_schema_units=mae*float(schema_scale[k]) if w else None,
                                     huber=float(self.huber[k]/w) if w else None,
                                     valid_cells=int(self.cells[k]), weight_sum=float(w))
        return dict(mae_standardized=float(self.abs_error.sum()/total) if total else None,
                    huber=float(self.huber.sum()/total) if total else None,
                    valid_cells=int(self.cells.sum()), weight_sum=total, per_feature=per_feature)
