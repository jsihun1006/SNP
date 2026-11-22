"""Use upstream RTMCCHead; optional, explicitly labelled paper 1/n experiment."""
from mmpose.registry import MODELS
from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead


@MODELS.register_module()
class TwoPointRTMCCHead(RTMCCHead):
    def __init__(self, paper_attention_normalization=False, **kwargs):
        if kwargs.get("out_channels") != 2:
            raise ValueError("TwoPointRTMCCHead requires exactly two keypoints")
        super().__init__(**kwargs)
        self.paper_attention_normalization = paper_attention_normalization
        if paper_attention_normalization:
            # Scaling the input of Wo scales the GAU branch only, preserving its residual.
            self.gau.o.register_forward_pre_hook(lambda module, args: (args[0] / 2.,))
