import argparse
import hashlib
import json
from pathlib import Path
from .probe_environment import require_cuda


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/rtmpose_x_two_points.py")
    p.add_argument("--checkpoint", type=Path, required=True, help="Body-7 CSPNeXt-X backbone weights")
    p.add_argument("--initialization-kind", choices=["body7", "alternative"], default="body7")
    p.add_argument("--initialization-note", default="")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    require_cuda(a.device)
    if min(a.batch_size, a.accumulate) < 1:
        p.error("batch-size and accumulate must be positive")
    if a.initialization_kind == "alternative" and not a.initialization_note:
        p.error("Alternative initialization requires --initialization-note")
    if not a.checkpoint.is_file():
        raise FileNotFoundError("Body-7 checkpoint missing; no silent substitute")
    from ..pose.checkpoints import checkpoint_identity, check_backbone_shapes
    digest=checkpoint_identity(a.checkpoint,a.initialization_kind)
    from .prepare_data import read_annotations
    for split in ("train", "val"):
        doc = read_annotations(a.data_root/"annotations"/f"{split}.json")
        if any(len(ann.get("keypoints", [])) != 6 for ann in doc["annotations"]):
            raise ValueError("Run prepare_data --task pose to flatten the two-point annotations")
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmpose.utils import register_all_modules
    register_all_modules()
    from ..pose import head  # register custom head
    cfg = Config.fromfile(str(a.config))
    from mmpose.registry import MODELS
    initialization_check=check_backbone_shapes(MODELS.build(cfg.model.backbone),a.checkpoint)
    cfg.work_dir = str(a.output.resolve())
    cfg.model.backbone.init_cfg = dict(type="Pretrained", prefix="backbone.", checkpoint=str(a.checkpoint.resolve()))
    cfg.optim_wrapper.accumulative_counts = a.accumulate
    cfg.train_dataloader.batch_size = a.batch_size
    for loader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        loader.dataset.data_root = str(a.data_root.resolve())
    # Select the requested GPU even on multi-GPU Windows hosts; do not launch distributed processes.
    import torch
    torch.cuda.set_device(a.device)
    if a.resume:
        if not a.resume.is_file():
            raise FileNotFoundError(a.resume)
        cfg.resume, cfg.load_from = True, str(a.resume.resolve())
    a.output.mkdir(parents=True, exist_ok=True)
    evidence = dict(initialization_kind=a.initialization_kind, note=a.initialization_note,
                    checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=digest, **initialization_check,
                    batch_size=a.batch_size, accumulation=a.accumulate,
                    effective_batch_size=a.batch_size*a.accumulate, paper_batch_size=128,
                    device=a.device, automatic_early_stopping=False)
    (a.output/"training_choices.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    cfg.dump(str(a.output/"resolved_config.py"))
    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
