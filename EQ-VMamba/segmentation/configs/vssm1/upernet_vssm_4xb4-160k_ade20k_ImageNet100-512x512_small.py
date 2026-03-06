_base_ = [
    '../swin/swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="/data0/zzc/projects/VMamba/classification/exp/vssm1_small_0229/20251110143203/ckpt_epoch_ema_best.pth",
        # copied from classification/configs/vssm/vssm_small_224.yaml
        dims=96,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.3,
        norm_layer="ln2d",
    ),)
# train_dataloader = dict(batch_size=4) # as gpus=4




# =========================================================
# Visualization
# =========================================================
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader


visualizer = dict(
    type='SegLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(
            type='LocalVisBackend',
        )
    ],
    alpha=0.5,
)

default_hooks = dict(
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        show=False,
        interval=1,
    )
)