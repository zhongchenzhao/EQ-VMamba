#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : eq_fcn_head.py
@Author: ZhongchenZhao
@Date  : 2025/11/23 19:02
@Desc  : 
"""


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

from eq_modules import Fconv_PCA_out, Fconv_1X1_out, EQSyncBatchNorm2d, EQDropout
# print("====print(MODELS.module_dict.keys())====:", MODELS.module_dict.keys())



class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        
        '''
        convs = self.convs Sequential(
                    (0): ConvModule(
                        (conv): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                        (bn): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (activate): ReLU(inplace=True)
                    ))
        '''

        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output



@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class EQFCNHeadV0(BaseDecodeHead):
    """ from FCNHead """
    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 tranNum=4,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs          # 1
        self.concat_input = concat_input    # False
        self.kernel_size = kernel_size      # 3
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        
        convs = []
        convs.append(
            ConvModule(
                self.in_channels // tranNum,    # 384
                self.channels // tranNum,       # 256
                kernel_size=kernel_size,        # 3
                padding=conv_padding,           # 1
                dilation=dilation,              # 1
                # conv_cfg=self.conv_cfg,         # None, if cfg is None: cfg = dict(type='Conv2d'), mmcv/.../build_conv_layer.py
                conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, inP=None, output_padding=0, 
                              transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                # norm_cfg=self.norm_cfg,         # dict(type='SyncBN', requires_grad=True)
                norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                act_cfg=self.act_cfg        # dict(type='ReLU')
                ))
        '''
        convs = self.convs Sequential(
                    (0): ConvModule(
                        (conv): Fconv_PCA()
                        (bn): EQSyncBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (activate): ReLU(inplace=True)
                    ))
        '''
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels // tranNum,
                    self.channels // tranNum,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, inP=None, output_padding=0,
                                  transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                    norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:           # num_convs=1
            self.convs = nn.Sequential(*convs)
        # print("************* self.convs", self.convs)

        if self.concat_input:       # False
            self.conv_cat = ConvModule(
                self.in_channels  // tranNum + self.channels // tranNum,
                self.channels // tranNum,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, inP=None, output_padding=0, 
                              transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                act_cfg=self.act_cfg)
            
        # 20251123, copy from BaseDecodeHead
        # self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)  # Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
        self.conv_seg = Fconv_1X1_out(inNum=self.channels // tranNum, outNum=self.out_channels, tranNum=tranNum)
        
        
        if self.dropout_ratio > 0:
            # self.dropout = nn.Dropout2d(self.dropout_ratio)         # Dropout2d(p=0.1, inplace=False)
            self.dropout = EQDropout(self.dropout_ratio)         # Dropout2d(p=0.1, inplace=False)
        else:
            self.dropout = None



    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)      # torch.Size([10, 384, 14, 14])
        feats = self.convs(x)                   # torch.Size([10, 256, 14, 14])

        if self.concat_input:       # False
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats


    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)          # torch.Size([10, 256, 14, 14])
        output = self.cls_seg(output)                   #  torch.Size([10, 150, 14, 14])
        return output


    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)           # EQDropout2d(p=0.1, inplace=False)
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
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:           # None
            inputs = inputs[self.in_index]      # in_index=2

        return inputs



@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class EQFCNHead(BaseDecodeHead):
    """ from EQFCNHead """
    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 tranNum=4,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs  # 1
        self.concat_input = concat_input  # False
        self.kernel_size = kernel_size  # 3
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation

        convs = []
        convs.append(
            ConvModule(
                self.in_channels // tranNum,  # 384
                self.channels,  # 256
                kernel_size=kernel_size,  # 3
                padding=conv_padding,  # 1
                dilation=dilation,  # 1
                # conv_cfg=self.conv_cfg,         # None, if cfg is None: cfg = dict(type='Conv2d'), mmcv/.../build_conv_layer.py
                conv_cfg=dict(type='Fconv_PCA_out', tranNum=tranNum, inP=None, output_padding=0,
                              transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                # norm_cfg=self.norm_cfg,         # dict(type='SyncBN', requires_grad=True)
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=self.act_cfg  # dict(type='ReLU')
            ))
        '''
        convs = self.convs Sequential(
                    (0): ConvModule(
                        (conv): Fconv_PCA()
                        (bn): EQSyncBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (activate): ReLU(inplace=True)
                    ))
        '''
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels // tranNum,
                    self.channels // tranNum,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, inP=None, output_padding=0,
                                  transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                    norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:  # num_convs=1
            self.convs = nn.Sequential(*convs)
        # print("************* self.convs", self.convs)

        if self.concat_input:  # False
            self.conv_cat = ConvModule(
                self.in_channels // tranNum + self.channels // tranNum,
                self.channels // tranNum,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=dict(type='Fconv_PCA', tranNum=tranNum, inP=None, output_padding=0,
                              transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                act_cfg=self.act_cfg)

        # 20251123, copy from BaseDecodeHead
        self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)  # Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
        # self.conv_seg = Fconv_1X1_out(inNum=self.channels // tranNum, outNum=self.out_channels, tranNum=tranNum)

        if self.dropout_ratio > 0:
            # self.dropout = nn.Dropout2d(self.dropout_ratio)         # Dropout2d(p=0.1, inplace=False)
            self.dropout = EQDropout(self.dropout_ratio)  # Dropout2d(p=0.1, inplace=False)
        else:
            self.dropout = None

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)  # torch.Size([10, 384, 14, 14])
        feats = self.convs(x)  # torch.Size([10, 256, 14, 14])

        if self.concat_input:  # False
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)  # torch.Size([10, 256, 14, 14])
        output = self.cls_seg(output)  # torch.Size([10, 150, 14, 14])
        return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)  # EQDropout2d(p=0.1, inplace=False)
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
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:  # None
            inputs = inputs[self.in_index]  # in_index=2

        return inputs



@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class EQBconvFCNHead(BaseDecodeHead):
    """ from EQFCNHead, change Fconv to Bconv """
    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 tranNum=4,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs  # 1
        self.concat_input = concat_input  # False
        self.kernel_size = kernel_size  # 3
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation

        convs = []
        convs.append(
            ConvModule(
                self.in_channels // tranNum,  # 384
                self.channels,  # 256
                kernel_size=kernel_size,  # 3
                padding=conv_padding,  # 1
                dilation=dilation,  # 1
                # conv_cfg=self.conv_cfg,         # None, if cfg is None: cfg = dict(type='Conv2d'), mmcv/.../build_conv_layer.py
                conv_cfg=dict(type='Bconv_PCA_out', tranNum=tranNum, inP=None, output_padding=0,
                              transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                # norm_cfg=self.norm_cfg,         # dict(type='SyncBN', requires_grad=True)
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=self.act_cfg  # dict(type='ReLU')
            ))
        '''
        convs = self.convs Sequential(
                    (0): ConvModule(
                        (conv): Bconv_PCA()
                        (bn): EQSyncBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (activate): ReLU(inplace=True)
                    ))
        '''
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels // tranNum,
                    self.channels // tranNum,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=dict(type='Bconv_PCA', tranNum=tranNum, inP=None, output_padding=0,
                                  transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                    norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:  # num_convs=1
            self.convs = nn.Sequential(*convs)
        # print("************* self.convs", self.convs)

        if self.concat_input:  # False
            self.conv_cat = ConvModule(
                self.in_channels // tranNum + self.channels // tranNum,
                self.channels // tranNum,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=dict(type='Bconv_PCA', tranNum=tranNum, inP=None, output_padding=0,
                              transposed=False, ifIni=0, Smooth=True, iniScale=1.0),
                norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
                act_cfg=self.act_cfg)

        # 20251123, copy from BaseDecodeHead
        self.conv_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)  # Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))

        if self.dropout_ratio > 0:
            # self.dropout = nn.Dropout2d(self.dropout_ratio)         # Dropout2d(p=0.1, inplace=False)
            self.dropout = EQDropout(self.dropout_ratio)  # Dropout2d(p=0.1, inplace=False)
        else:
            self.dropout = None

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)  # torch.Size([10, 384, 14, 14])
        feats = self.convs(x)  # torch.Size([10, 256, 14, 14])

        if self.concat_input:  # False
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)  # torch.Size([10, 256, 14, 14])
        output = self.cls_seg(output)  # torch.Size([10, 150, 14, 14])
        return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)  # EQDropout2d(p=0.1, inplace=False)
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
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:  # None
            inputs = inputs[self.in_index]  # in_index=2

        return inputs



# ########
def rotate_and_shift(x, rotate_times=0, rotate_dims=[-2, -1], shift_times=0, shift_dim=1):
    """
    对输入张量进行逆时针旋转90度和通道轮换

    参数:
        x: 输入张量, 形状为 [batch, 4, d_inner, height, width]

    返回:
        处理后的张量, 形状与输入相同
    """
    # 逆时针旋转90度 (对最后两个维度)
    x_rot = torch.rot90(x, k=rotate_times, dims=rotate_dims)

    # 在通道维度(维度1)进行轮换，最后一个放到第一个
    x_shifted = torch.roll(x_rot, shifts=shift_times, dims=shift_dim)

    return x_shifted



def print_error(x_triton, x_torch, epsilon=1e-5):
    diff_abs = torch.abs(x_triton - x_torch).mean()
    diff = torch.abs(x_triton - x_torch) / (torch.abs(x_torch) + epsilon)
    print(f"Error mean (abs): {diff_abs.mean()}, Error max (abs): {diff_abs.max()}, "
          f"Error mean (relative): {diff.mean()}, Error max (relative): {diff.max()}, ")




if __name__ == "__main__":
    head = EQFCNHead(
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        tranNum=4,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='EQSyncBatchNorm2d', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        )

    xs = [torch.randn([10, 96, 56, 56]), torch.randn([10, 192, 28, 28]), 
          torch.randn([10, 384, 14, 14]), torch.randn([10, 768, 7, 7]), ]
    output = head(xs)
    print("output: ", output.shape)
    
    
    print("========")
    xs = [torch.randn([10, 24, 4, 56, 56]), torch.randn([10, 48, 4, 28, 28]), 
          torch.randn([10, 96, 4, 14, 14]), torch.randn([10, 128, 4, 7, 7]), ]
    xs_rot = [rotate_and_shift(x, rotate_times=1, rotate_dims=[-2, -1], shift_times=1, shift_dim=-3) for x in xs]
    xs2 = [torch.cat([x, x_rot], dim=0) for (x, x_rot) in zip(xs, xs_rot)]
    xs2 = [rearrange(x, 'b c t h w -> b (c t) h w').contiguous() for x in xs2]

    print([x.shape for x in xs2])

    # head = Fconv_PCA(192//4, 256//4, 3, stride=1, padding=None, dilation=1, groups=1, bias=True,
    #              tranNum=4, inP=None, output_padding=0, transposed=False, ifIni=0, Smooth=True, iniScale=1.0)

    y = head(xs2)
    print("y: ", y.shape)
    # y = rearrange(y, 'b (c t) h w -> b c t h w', t=4).contiguous()

    y, y_reverse = torch.split(y, [10, 10], dim=0)
    y_reverse = rotate_and_shift(y_reverse, rotate_times=-1, rotate_dims=[-2, -1], shift_times=0, shift_dim=2)
    print_error(y.detach().cpu(), y_reverse.detach().cpu())

