#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : eq_uper_head.py
@Author: ZhongchenZhao
@Date  : 2025/11/23 19:03
@Desc  : 
"""

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import ConvModule
from mmseg.models.utils import resize
from mmengine.registry import MODELS
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG
# from .decode_head import BaseDecodeHead
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM

# from ..utils import resize
# from .decode_head import BaseDecodeHead
# from .psp_head import PPM
from eq_modules import Fconv_PCA, Fconv_1X1_out, Fconv_PCA_out, EQSyncBatchNorm2d, EQDropout, EQPixelShuffle_out




class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

# ================ ================

@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class EQUPerHead(BaseDecodeHead):
    """
    from EQUPerHead
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), tranNum=4, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,  # (1, 2, 3, 6)
            self.in_channels[-1] // tranNum,  # [96, 192, 384, 768]
            self.channels // tranNum,  # 512
            # conv_cfg=self.conv_cfg,         # dict(type='Conv2d')
            conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, inP=None, output_padding=0,
                          transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
            # norm_cfg=self.norm_cfg,         # dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
            act_cfg=self.act_cfg,  # {'type': 'ReLU'}
            align_corners=self.align_corners  # False
        )

        self.bottleneck = ConvModule(
            self.in_channels[-1] // tranNum + len(pool_scales) * self.channels // tranNum,
            self.channels // tranNum,
            3,
            padding=1,
            # conv_cfg=self.conv_cfg,         # dict(type='Conv2d')
            conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, inP=None, output_padding=0,
                          transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
            # norm_cfg=self.norm_cfg,         # dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
            act_cfg=self.act_cfg)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer, [96, 192, 384,]
            l_conv = ConvModule(
                in_channels // tranNum,
                self.channels // tranNum,
                1,
                # conv_cfg=self.conv_cfg,       # dict(type='Conv2d')
                conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, ifIni=0, Smooth=True, iniScale=1.0),
                # norm_cfg=self.norm_cfg,       # dict(type='SyncBN', requires_grad=True)
                norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels // tranNum,
                self.channels // tranNum,
                3,
                padding=1,
                # conv_cfg=self.conv_cfg,         # dict(type='Conv2d')
                conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, inP=None, output_padding=0,
                              transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                # norm_cfg=self.norm_cfg,         # dict(type='SyncBN', requires_grad=True)
                norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels // tranNum,
            self.channels,
            3,
            padding=1,
            # conv_cfg=self.conv_cfg,         # dict(type='Conv2d')
            conv_cfg=dict(type='Fconv_PCA_out', tranNum=tranNum, inP=None, output_padding=0,
                          transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
            # norm_cfg=self.norm_cfg,         # dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=self.act_cfg)

        # 20251123, copy from BaseDecodeHead
        self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)  # Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
        # self.conv_seg = Fconv_1X1_out(inNum=self.fpn_bottleneck_channels // tranNum, outNum=self.out_channels, tranNum=tranNum)

        if self.dropout_ratio > 0:
            # self.dropout = nn.Dropout2d(self.dropout_ratio)         # Dropout2d(p=0.1, inplace=False)
            self.dropout = EQDropout(self.dropout_ratio)  # Dropout2d(p=0.1, inplace=False)
        else:
            self.dropout = None

    def psp_forward(self, x):
        """Forward function of PSP module."""
        psp_outs = [x]  # [torch.Size([10, 768, 7, 7])]
        psp_outs.extend(self.psp_modules(x))
        # [torch.Size([10, 768, 7, 7]), torch.Size([10, 512, 7, 7]), torch.Size([10, 512, 7, 7]), torch.Size([10, 512, 7, 7]), torch.Size([10, 512, 7, 7])]
        # return output

        psp_outs = torch.cat(psp_outs, dim=1)  # torch.Size([10, 2816, 7, 7])
        output = self.bottleneck(psp_outs)  # torch.Size([10, 512, 7, 7])
        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)
        # [torch.Size([10, 96, 56, 56]), torch.Size([10, 192, 28, 28]),
        # torch.Size([10, 384, 14, 14]), torch.Size([10, 768, 7, 7])]

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # [torch.Size([10, 512, 56, 56]), torch.Size([10, 512, 28, 28]), torch.Size([10, 512, 14, 14])]

        laterals.append(self.psp_forward(inputs[-1]))
        # [torch.Size([10, 512, 56, 56]), torch.Size([10, 512, 28, 28]), torch.Size([10, 512, 14, 14]), torch.Size([10, 512, 7, 7])]
        # return self.psp_forward(inputs[-1])

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):  # [3, 2, 1]
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        # laterals: [torch.Size([10, 512, 56, 56]), torch.Size([10, 512, 28, 28]), torch.Size([10, 512, 14, 14]), torch.Size([10, 512, 7, 7])]

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature

        fpn_outs.append(laterals[-1])
        # [torch.Size([10, 512, 56, 56]), torch.Size([10, 512, 28, 28]), torch.Size([10, 512, 14, 14]), torch.Size([10, 512, 7, 7])]

        for i in range(used_backbone_levels - 1, 0, -1):  # [3, 2, 1]
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],  # torch.Size([56, 56])
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        # torch.Size([10, 2048, 56, 56])
        feats = self.fpn_bottleneck(fpn_outs)  # torch.Size([10, 512, 56, 56])
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)  # torch.Size([10, 512, 56, 56])
        output = self.cls_seg(output)  # torch.Size([10, 150, 56, 56])
        return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)  # Dropout2d(p=0.1, inplace=False)
        output = self.conv_seg(feat)
        return output

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':  # True
            inputs = [inputs[i] for i in self.in_index]  # in_index=[0, 1, 2, 3]
        else:
            inputs = inputs[self.in_index]

        return inputs





