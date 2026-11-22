METAINFO = dict(
    dataset_name='synloc_two_points',
    joint_weights=[
        1.0,
        1.0,
    ],
    keypoint_info=dict({
        0:
        dict(
            color=[
                255,
                120,
                0,
            ], id=0, name='pelvis', swap='', type='lower'),
        1:
        dict(
            color=[
                0,
                200,
                255,
            ], id=1, name='ground', swap='', type='lower')
    }),
    paper_info=dict(
        author='',
        container='',
        title='SSS-1 independent reimplementation',
        year='2026'),
    sigmas=[
        0.089,
        0.089,
    ],
    skeleton_info=dict(
        {0: dict(color=[
            0,
            255,
            0,
        ], id=0, link=(
            'pelvis',
            'ground',
        ))}))
auto_scale_lr = dict(base_batch_size=128, enable=False)
codec = dict(
    input_size=(
        288,
        384,
    ),
    normalize=False,
    sigma=(
        6.0,
        6.93,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'sss1_reimplementation.pose.head',
    ])
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='gloo'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=1.33,
        expand_ratio=0.5,
        init_cfg=None,
        norm_cfg=dict(type='BN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=1.25),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                288,
                384,
            ),
            normalize=False,
            sigma=(
                6.0,
                6.93,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=1280,
        in_featuremap_size=(
            9,
            12,
        ),
        input_size=(
            288,
            384,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=2,
        paper_attention_normalization=False,
        simcc_split_ratio=2.0,
        type='TwoPointRTMCCHead'),
    test_cfg=dict(flip_test=False),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    accumulative_counts=1,
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=700,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=700,
        eta_min=5e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=21)
root = '/Users/kkj_1120/performance_analyst/projects/metrica_hack/more-than-a-hack-2027/sss1_reimplementation'
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/val.json',
        data_mode='topdown',
        data_prefix=dict(img='val/'),
        data_root=
        '/Users/kkj_1120/performance_analyst/projects/metrica_hack/more-than-a-hack-2027/sss1_reimplementation/data',
        metainfo=dict(
            dataset_name='synloc_two_points',
            joint_weights=[
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        255,
                        120,
                        0,
                    ],
                    id=0,
                    name='pelvis',
                    swap='',
                    type='lower'),
                1:
                dict(
                    color=[
                        0,
                        200,
                        255,
                    ],
                    id=1,
                    name='ground',
                    swap='',
                    type='lower')
            }),
            paper_info=dict(
                author='',
                container='',
                title='SSS-1 independent reimplementation',
                year='2026'),
            sigmas=[
                0.089,
                0.089,
            ],
            skeleton_info=dict({
                0:
                dict(color=[
                    0,
                    255,
                    0,
                ], id=0, link=(
                    'pelvis',
                    'ground',
                ))
            })),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                288,
                384,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(norm_item='bbox', thr=0.05, type='PCKAccuracy')
train_cfg = dict(by_epoch=True, max_epochs=700, val_interval=10)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='annotations/train.json',
        data_mode='topdown',
        data_prefix=dict(img='train/'),
        data_root=
        '/Users/kkj_1120/performance_analyst/projects/metrica_hack/more-than-a-hack-2027/sss1_reimplementation/data',
        metainfo=dict(
            dataset_name='synloc_two_points',
            joint_weights=[
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        255,
                        120,
                        0,
                    ],
                    id=0,
                    name='pelvis',
                    swap='',
                    type='lower'),
                1:
                dict(
                    color=[
                        0,
                        200,
                        255,
                    ],
                    id=1,
                    name='ground',
                    swap='',
                    type='lower')
            }),
            paper_info=dict(
                author='',
                container='',
                title='SSS-1 independent reimplementation',
                year='2026'),
            sigmas=[
                0.089,
                0.089,
            ],
            skeleton_info=dict({
                0:
                dict(color=[
                    0,
                    255,
                    0,
                ], id=0, link=(
                    'pelvis',
                    'ground',
                ))
            })),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(
                rotate_factor=90,
                scale_factor=[
                    0.5,
                    1.5,
                ],
                type='RandomBBoxTransform'),
            dict(input_size=(
                288,
                384,
            ), type='TopdownAffine'),
            dict(
                transforms=[
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=0.5,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        288,
                        384,
                    ),
                    normalize=False,
                    sigma=(
                        6.0,
                        6.93,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(
        rotate_factor=90,
        scale_factor=[
            0.5,
            1.5,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        288,
        384,
    ), type='TopdownAffine'),
    dict(
        transforms=[
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                288,
                384,
            ),
            normalize=False,
            sigma=(
                6.0,
                6.93,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/val.json',
        data_mode='topdown',
        data_prefix=dict(img='val/'),
        data_root=
        '/Users/kkj_1120/performance_analyst/projects/metrica_hack/more-than-a-hack-2027/sss1_reimplementation/data',
        metainfo=dict(
            dataset_name='synloc_two_points',
            joint_weights=[
                1.0,
                1.0,
            ],
            keypoint_info=dict({
                0:
                dict(
                    color=[
                        255,
                        120,
                        0,
                    ],
                    id=0,
                    name='pelvis',
                    swap='',
                    type='lower'),
                1:
                dict(
                    color=[
                        0,
                        200,
                        255,
                    ],
                    id=1,
                    name='ground',
                    swap='',
                    type='lower')
            }),
            paper_info=dict(
                author='',
                container='',
                title='SSS-1 independent reimplementation',
                year='2026'),
            sigmas=[
                0.089,
                0.089,
            ],
            skeleton_info=dict({
                0:
                dict(color=[
                    0,
                    255,
                    0,
                ], id=0, link=(
                    'pelvis',
                    'ground',
                ))
            })),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                288,
                384,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(norm_item='bbox', thr=0.05, type='PCKAccuracy')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        288,
        384,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
