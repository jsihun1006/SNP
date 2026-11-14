import hashlib
from pathlib import Path

BODY7_SHA256="d28b58e620a6b1da09427c1c0e21ec419d9a71602379eebcee48ca32bd6f0667"


def checkpoint_identity(path, kind):
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if kind=="body7" and digest!=BODY7_SHA256:
        raise ValueError("Checkpoint is not the verified official CSPNeXt-X Body-7 file; declare alternative initialization explicitly")
    return digest


def check_backbone_shapes(backbone, checkpoint):
    import torch
    obj=torch.load(str(checkpoint),map_location="cpu")
    state=obj.get("state_dict",obj)
    weights={k[len("backbone."):]:v for k,v in state.items() if k.startswith("backbone.")}
    expected=backbone.state_dict()
    missing=[k for k,v in expected.items() if k not in weights or weights[k].shape!=v.shape]
    if missing:
        raise ValueError(f"Backbone initialization is incompatible ({len(missing)} tensors): {missing[:10]}")
    return dict(matched_backbone_tensors=len(expected),source_backbone_tensors=len(weights))
