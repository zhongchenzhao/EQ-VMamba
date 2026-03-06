_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
# norm_cfg = dict(type='SyncBN', requires_grad=True)          #############
norm_cfg = dict(type='EQSyncBatchNorm2d', requires_grad=True)          #############
num_classes = 150

model = dict(
    backbone=dict(
        type='MM_EQVSSM',
        out_indices=(0, 1, 2, 3),
        eq_tranNum=4,
        pretrained="/data0/zzc/projects/VMamba/classification/exp/eqvssm1_tiny_0230s/20251024103824/ckpt_epoch_ema_best.pth",
        dims=96,
        # depths=(2, 2, 5, 2),
        depths=(2, 2, 8, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        # ssm_ratio=2.0,
        ssm_ratio=1.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
        norm_layer="ln2d",
    ),
    decode_head=dict(
        type='EQUPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=num_classes,         # pascal_voc12
        norm_cfg=norm_cfg,
        # norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        tranNum=4,
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
        num_classes=num_classes,         # pascal_voc12
        norm_cfg=norm_cfg,
        align_corners=False,
        tranNum=4,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
)
# train_dataloader = dict(batch_size=4) # as gpus=4

