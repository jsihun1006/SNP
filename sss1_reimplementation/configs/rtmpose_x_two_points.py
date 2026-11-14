"""Self-contained MMPose 1.3.2 configuration; local Body-7 backbone required for training."""
_base_ = []
import pathlib
import sss1_reimplementation
from sss1_reimplementation.pose.metainfo import METAINFO

root = str(pathlib.Path(sss1_reimplementation.__file__).resolve().parent)
default_scope = "mmpose"
custom_imports = dict(imports=["sss1_reimplementation.pose.head"], allow_failed_imports=False)
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
               dist_cfg=dict(backend="gloo"))
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer")
randomness = dict(seed=21)
default_hooks = dict(timer=dict(type="IterTimerHook"), logger=dict(type="LoggerHook", interval=50),
                     param_scheduler=dict(type="ParamSchedulerHook"),
                     checkpoint=dict(type="CheckpointHook", interval=10, max_keep_ckpts=3),
                     sampler_seed=dict(type="DistSamplerSeedHook"))
train_cfg = dict(by_epoch=True, max_epochs=700, val_interval=10)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(type="OptimWrapper", accumulative_counts=1,
                     optimizer=dict(type="AdamW", lr=1e-4, weight_decay=.05),
                     clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [dict(type="CosineAnnealingLR", begin=0, end=700, T_max=700,
                        eta_min=5e-6, by_epoch=True, convert_to_iter_based=True)]
auto_scale_lr = dict(enable=False, base_batch_size=128)
codec = dict(type="SimCCLabel", input_size=(288, 384), sigma=(6., 6.93),
             simcc_split_ratio=2., normalize=False, use_dark=False)
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(type="PoseDataPreprocessor", mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    backbone=dict(type="CSPNeXt", arch="P5", expand_ratio=.5,
                  deepen_factor=1.33, widen_factor=1.25, out_indices=(4,), channel_attention=True,
                  norm_cfg=dict(type="BN"), act_cfg=dict(type="SiLU"), init_cfg=None),
    head=dict(type="TwoPointRTMCCHead", in_channels=1280, out_channels=2,
              input_size=(288, 384), in_featuremap_size=(9, 12), simcc_split_ratio=2.,
              final_layer_kernel_size=7, paper_attention_normalization=False,
              gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0.,
                           drop_path=0., act_fn="SiLU", use_rel_bias=False, pos_enc=False),
              loss=dict(type="KLDiscretLoss", use_target_weight=True, beta=10., label_softmax=True),
              decoder=codec),
    test_cfg=dict(flip_test=False))
train_pipeline = [dict(type="LoadImage"), dict(type="GetBBoxCenterScale"),
                  dict(type="RandomFlip", direction="horizontal"),
                  dict(type="RandomBBoxTransform", scale_factor=[.5, 1.5], rotate_factor=90),
                  dict(type="TopdownAffine", input_size=(288, 384)),
                  dict(type="Albumentation", transforms=[dict(type="CoarseDropout", max_holes=1,
                       min_holes=1, max_height=.4, max_width=.4, min_height=.2, min_width=.2, p=.5)]),
                  dict(type="GenerateTarget", encoder=codec), dict(type="PackPoseInputs")]
val_pipeline = [dict(type="LoadImage"), dict(type="GetBBoxCenterScale"),
                dict(type="TopdownAffine", input_size=(288, 384)), dict(type="PackPoseInputs")]
train_dataloader = dict(batch_size=128, num_workers=0, persistent_workers=False,
                        sampler=dict(type="DefaultSampler", shuffle=True),
                        dataset=dict(type="CocoDataset", data_mode="topdown", metainfo=METAINFO,
                                     data_root=str(pathlib.Path(root)/"data"), ann_file="annotations/train.json",
                                     data_prefix=dict(img="train/"), pipeline=train_pipeline))
val_dataloader = dict(batch_size=32, num_workers=0, persistent_workers=False, drop_last=False,
                      sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
                      dataset=dict(type="CocoDataset", data_mode="topdown", metainfo=METAINFO,
                                   data_root=str(pathlib.Path(root)/"data"), ann_file="annotations/val.json",
                                   data_prefix=dict(img="val/"), test_mode=True, pipeline=val_pipeline))
test_dataloader = val_dataloader
# Diagnostic pixel metric only; official world-space evaluation uses evaluation/official.py.
val_evaluator = dict(type="PCKAccuracy", thr=.05, norm_item="bbox")
test_evaluator = val_evaluator
