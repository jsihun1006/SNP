from pathlib import Path
import numpy as np


class MMPoseEstimator:
    def __init__(self, config, checkpoint, device="cpu"):
        import torch
        from mmpose.apis import init_model
        from mmpose.utils import register_all_modules
        register_all_modules()
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trained TWO-keypoint checkpoint required: {checkpoint}")
        # init_model can otherwise silently ignore a 17-keypoint head mismatch.
        state = torch.load(str(checkpoint), map_location="cpu")
        weights = state.get("state_dict", state)
        candidates = [v for k, v in weights.items() if k.endswith("head.final_layer.weight")]
        if len(candidates) != 1 or candidates[0].shape[0] != 2:
            raise ValueError("Checkpoint must contain a trained 2-keypoint RTMCCHead")
        self.model = init_model(str(Path(config)), checkpoint=None, device=device)
        self.model.load_state_dict(weights, strict=True)
        if self.model.head.out_channels != 2:
            raise ValueError("Model must predict pelvis and ground, in that order")

    def predict(self, image, boxes):
        from mmpose.apis import inference_topdown
        if not boxes:
            return np.empty((0, 2, 2))
        samples = inference_topdown(self.model, image, np.array([b.bbox for b in boxes]), bbox_format="xyxy")
        result = np.array([s.pred_instances.keypoints[0, :, :2] for s in samples])
        if result.shape != (len(boxes), 2, 2):
            raise RuntimeError(f"Unexpected pose output: {result.shape}")
        return result
