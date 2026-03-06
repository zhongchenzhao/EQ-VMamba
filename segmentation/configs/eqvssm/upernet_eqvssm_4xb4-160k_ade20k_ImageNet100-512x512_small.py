_base_ = [
    '../swin/swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
# norm_cfg = dict(type='SyncBN', requires_grad=True)          #############
norm_cfg = dict(type='EQSyncBatchNorm2d', requires_grad=True)          #############
num_classes = 150

model = dict(
    backbone=dict(
        type='MM_EQVSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="/data0/zzc/projects/VMamba/classification/exp/eqvssm1_small_0229s/20251124102409/ckpt_epoch_ema_best.pth",
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
# train_dataloader = dict(batch_size=4) # as gpus=4


# =========================================================
# Visualization
# =========================================================
# train_dataloader = dict(batch_size=1)
# val_dataloader = dict(batch_size=1)
# test_dataloader = val_dataloader
#
#
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