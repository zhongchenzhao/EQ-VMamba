_base_ = [
    '../_base_/models/upernet_swin.py',
    # '../_base_/datasets/pascal_voc12.py',
    '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
num_classes = 21      # pascal_voc12
crop_size = (512, 512)      # pascal_voc12
data_preprocessor = dict(size=crop_size)
# norm_cfg = dict(type='SyncBN', requires_grad=True)          #############
norm_cfg = dict(type='EQSyncBatchNorm2d', requires_grad=True)          #############

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MM_EQVSSM',
        out_indices=(0, 1, 2, 3),
        eq_tranNum=4,
        pretrained="/data0/zzc/projects/VMamba/classification/exp/eqvssm1_small_0229s/20251124102409/ckpt_epoch_ema_best.pth",        # Fconv
        dims=96,
        # depths=(2, 2, 15, 2),
        depths=(2, 2, 20, 2),               # eqvmambav2v_small_224.yaml
        ssm_d_state=1,
        ssm_dt_rank="auto",
        # ssm_ratio=2.0,
        ssm_ratio=1.0,                      # eqvmambav2v_small_224.yaml
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.3,
        norm_layer="ln2d",
    ),
    decode_head=dict(
        type='EQUPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='EQFCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
)

# ======== ========
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader


# =========================================================
# Visualization
# =========================================================

# visualizer = dict(
#     type='SegLocalVisualizer',
#     name='visualizer',
#     vis_backends=[
#         dict(
#             type='LocalVisBackend',
#         )
#     ],
#     alpha=0.5,
# )
#
# default_hooks = dict(
#     visualization=dict(
#         type='SegVisualizationHook',
#         draw=True,
#         show=False,
#         interval=1,
#     )
# )