# https://github.com/zhuqinfeng1999/Samba/blob/main/configs/samba/samba_upernet-15k_loveda-512x512_6e4.py
_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
num_classes = 7
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        # pretrained="/data0/zzc/projects/VMamba/classification/exp/vssm1_tiny_0230s/20251021220221/ckpt_epoch_ema_best.pth",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
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
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
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
        end=15000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU, use 2 GPUs
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
train_cfg = dict(type='IterBasedTrainLoop', max_iters=15000, val_interval=1000)
