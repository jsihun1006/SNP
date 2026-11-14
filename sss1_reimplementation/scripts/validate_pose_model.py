"""Real architecture/initialization smoke test; the two-point head is untrained."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    import torch
    from mmengine.config import Config
    from mmpose.registry import MODELS
    from mmpose.utils import register_all_modules
    from ..pose import head
    from ..pose.checkpoints import checkpoint_identity,check_backbone_shapes
    register_all_modules()
    cfg=Config.fromfile(str(Path(__file__).resolve().parents[1]/"configs/rtmpose_x_two_points.py"))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cfg.dump(str(a.output.with_name("validated_pose_config.py")))
    model=MODELS.build(cfg.model)
    digest=checkpoint_identity(a.checkpoint,"body7")
    shapes=check_backbone_shapes(model.backbone,a.checkpoint)
    weights=torch.load(str(a.checkpoint),map_location="cpu")["state_dict"]
    model.backbone.load_state_dict({k[len("backbone."):]:v for k,v in weights.items() if k.startswith("backbone.")},strict=True)
    model.eval()
    observed=[]
    with torch.no_grad():
        x,y=model.head(model.backbone(torch.zeros(1,3,384,288)))
    assert tuple(x.shape)==(1,2,576) and tuple(y.shape)==(1,2,768)
    # Check actual configured augmentations on a two-point COCO fixture, not just config text.
    from mmengine.dataset import Compose
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    from ..pose.metainfo import METAINFO
    from ..image_io import read_image, write_image
    import numpy as np
    import cv2
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/"fixture.jpg"
        write_image(path,np.full((384,288,3),128,np.uint8))
        sample=dict(img_path=str(path),bbox=np.array([[20.,20.,260.,370.]],np.float32),
                    bbox_score=np.ones(1,np.float32),keypoints=np.array([[[140.,180.],[140.,340.]]],np.float32),
                    keypoints_visible=np.ones((1,2),np.float32),**parse_pose_metainfo(METAINFO))
        batch=Compose(cfg.train_pipeline)(sample)
        assert tuple(batch["inputs"].shape)==(3,384,288)
        observed=list(batch["data_samples"].gt_instance_labels.keys())
        from mmpose.apis import inference_topdown
        model.cfg=cfg
        model.dataset_meta=parse_pose_metainfo(METAINFO)
        decoded=inference_topdown(model,read_image(path),np.array([[20.,20.,260.,370.]],np.float32))
        decoded_shape=list(decoded[0].pred_instances.keypoints.shape)
        assert decoded_shape==[1,2,2]
    report=dict(purpose="architecture and official Body-7 initialization only; two-point head UNTRAINED",
                device="cpu",checkpoint_sha256=digest,**shapes,simcc_x_shape=list(x.shape),simcc_y_shape=list(y.shape),
                pipeline_input_shape=list(batch["inputs"].shape),target_fields=observed,trained_inference_verified=False)
    report["untrained_head_topdown_decode_shape"]=decoded_shape
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
