#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import math
import copy
import numpy as np
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from einops import rearrange, repeat


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit




# =====================================================
def cross_scan_eq_split_torch(xs: torch.Tensor):
    batch, tranNum, dim, height, width = xs.shape
    ys = xs.new_zeros(batch, tranNum, dim, height*width)

    ys[:, 0] = xs[:, 0].view(batch, dim, height*width)
    ys[:, 1] = torch.rot90(xs[:, 1], k=-1, dims=[-2, -1]).contiguous().view(batch, dim, height*width)
    ys[:, 2] = xs[:, 2].view(batch, dim, height*width).flip(dims=[-1])
    ys[:, 3] = torch.rot90(xs[:, 3], k=1, dims=[-2, -1]).contiguous().view(batch, dim, height*width)
    return ys.contiguous()



def cross_merge_eq_concatenate_torch(xs: torch.Tensor, height, width):
    batch, tranNum, dim, seqlen = xs.shape
    assert seqlen == height * width, f"seqlen {seqlen} != height {height} x width {width}"
    ys = xs.new_zeros(batch, tranNum, dim, height, width)

    ys[:, 0] = xs[:, 0].contiguous().view(batch, dim, height, width)
    ys[:, 1] = torch.rot90(xs[:, 1].contiguous().view(batch, dim, width, height), k=1, dims=[-2, -1])
    ys[:, 2] = xs[:, 2].contiguous().flip(dims=[-1]).view(batch, dim, height, width)
    ys[:, 3] = torch.rot90(xs[:, 3].contiguous().view(batch, dim, width, height), k=-1, dims=[-2, -1])
    return ys.contiguous()



class EQ_linear_inter(nn.Module):
    def __init__(self, inNum, outNum, tranNum=4, bias=True, iniScale=1.0):
        super(EQ_linear_inter, self).__init__()
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.weights = nn.Parameter(torch.Tensor(outNum, 1, inNum, tranNum), requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.Tensor(outNum, 1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()

    def forward(self, input):
        """
        input: (B, L, C*T)
        """
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        tempW = self.weights.repeat([1, tranNum, 1, 1])

        tempWList = [torch.cat([tempW[:, i:i + 1, :, -i:], tempW[:, i:i + 1, :, :-i]], dim=3) for i in range(tranNum)]
        tempW = torch.cat(tempWList, dim=1)

        weight = tempW.reshape([outNum * tranNum, inNum * tranNum])
        if self.bias:
            bias = self.c.repeat([1, tranNum]).reshape([1, outNum * tranNum])  # .cuda()
        else:
            bias = self.c
        return F.linear(input, weight, bias=bias)


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



class EQ_linear_output(nn.Module):
    def __init__(self, inNum, outNum, tranNum=4, bias=True, iniScale=1.0):
        super(EQ_linear_output, self).__init__()
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, 1), requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.Tensor(outNum))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        tempW = self.weights.repeat([1, 1, tranNum])

        weight = tempW.reshape([outNum, inNum * tranNum])

        return F.linear(input, weight, bias=self.c)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



class Dropout(nn.Module):
    # 这个之后可以放到e_linear里面去
    def __init__(self, p=0., tranNum=4):
        # nn.Dropout2d
        super(Dropout, self).__init__()
        self.tranNum = tranNum
        self.Dropout = nn.Dropout2d(p)

    def forward(self, X):
        sizeX = X.shape
        X = self.Dropout(X.reshape([-1, sizeX[-1] // self.tranNum, self.tranNum]))
        return X.reshape(sizeX)



class EQDropout(nn.Module):
    """Dropout with shared mask across batch (mask shape = (1, C, H, W))."""
    def __init__(self, p=0.5, tranNum=4):
        super().__init__()
        self.p = p
        self.tranNum = tranNum

    def forward(self, x):
        """
        :param x: B, C*t, H, W
        :return:
        """
        if not self.training or self.p == 0.0:
            return x

        assert len(x.shape) == 4, f"x.shape: {x.shape}"
        keep_prob = 1 - self.p
        x = rearrange(x, 'b (c t) h w -> (b t) c h w', t=self.tranNum).contiguous()
        B, C, H, W = x.shape

        mask = (torch.rand(1, C, H, W, device=x.device) < keep_prob).float()

        mask[:, :, (H+1)//2:, :(W+1)//2] = torch.rot90(mask[:, :, :(H+1)//2, :W//2], k=1, dims=[-2, -1])
        mask[:, :, H//2:, (W+1)//2:] = torch.rot90(mask[:, :, :(H+1)//2, :W//2], k=2, dims=[-2, -1])
        mask[:, :, :H//2, W//2:] = torch.rot90(mask[:, :, :(H+1)//2, :W//2], k=-1, dims=[-2, -1])

        x = x * mask / keep_prob
        x = rearrange(x, '(b t) c h w -> b (c t) h w', t=self.tranNum).contiguous()
        return x



class EQ_linear_inter_dt(EQ_linear_inter):
    """
    change to reset_parameters for self.weights
    """
    def __init__(self, inNum, outNum, tranNum=4, bias=True, iniScale=1.0, dt_rank=6, dt_scale=1.0):
        super(EQ_linear_inter_dt, self).__init__(inNum, outNum, tranNum, bias)
        self.dt_scale = dt_scale
        self.dt_rank = dt_rank
        self.reset_parameters_uniform()

    def reset_parameters_uniform(self) -> None:
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        nn.init.uniform_(self.weights, dt_init_std, dt_init_std)



class Fconv_PCA(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(Fconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())
        # Basis.shape: torch.Size([3, 3, 4, 9])
        # print(Basis[:, :, 0, 0])
        # print(Basis[:, :, 1, 0])
        # print(Basis[:, :, 2, 0])
        # print(Basis[:, :, 3, 0])

        self.ifbias = bias
        if ifIni:  # first layer of F-Conv
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, Basis.size(3)),
                                    requires_grad=True)  # torch.Size([16, 5, 1, 3, 3])
        # self.weights: torch.Size([16, 5, 1, 9])

        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()

    def forward(self, input):
        if self.training:  # default: True
            tranNum = self.tranNum  # 4
            outNum = self.outNum  # 16
            inNum = self.inNum  # 5
            expand = self.expand  # 1

            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)  # torch.Size([16, 4, 5, 1, 3, 3])
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            # Basis.shape: torch.Size([3, 3, 4, 9])
            # weights.shape: torch.Size([16, 5, 1, 9])

            Num = tranNum // expand  # 4
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]

            tempW = torch.cat(tempWList, dim=1)  # torch.Size([16, 4, 5, 1, 3, 3])

            _filter = tempW.reshape(
                [outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])  # torch.Size([64, 5, 3, 3])

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
        else:
            _filter = self.filter  # torch.Size([64, 3, 3, 3])
            if self.ifbias:
                _bias = self.bias
        output = F.conv2d(input, _filter, padding=self.padding, dilation=1, groups=1)
        if self.ifbias:
            output = output + _bias
        return output

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)
            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)

        return super(Fconv_PCA, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



class StrideFconv_PCA(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, stride=1, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(StrideFconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.stride = stride
        self.ifIni = ifIni
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())
        # Basis.shape: torch.Size([3, 3, 4, 9])

        self.ifbias = bias
        if ifIni:  # first layer of F-Conv
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, Basis.size(3)), requires_grad=True)  # torch.Size([16, 5, 1, 3, 3])
        # self.weights: torch.Size([16, 5, 1, 9])

        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()


    def forward(self, input):
        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0

            Num = tranNum // expand         # Num=4 for first layer; Num=1 for middle layer;
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)

        else:
            Num = self.tranNum // self.expand         # Num=4 for first layer; Num=1 for middle layer;
            _filter = self.filter
            if self.ifbias:
                _bias = self.bias

        self.register_buffer("filter", _filter)
        # padding fod different stride
        p = self.padding[0] if isinstance(self.padding, (tuple, list)) else self.padding
        stride = self.stride[0] if isinstance(self.stride, (tuple, list)) else self.stride

        input = F.pad(input, (p, p, p, p), mode="constant", value=0)
        if stride >= 2:
            # only for first layer where Num=4
            batch, in_channel, height, width = input.shape
            input = input.view(batch, in_channel, 1, height, width).repeat(1, 1, Num, 1, 1).contiguous()
            input = input.view(batch, in_channel*Num, height, width).contiguous()

            input[:, 1::4, :, :] = torch.roll(input[:, 1::4, :, :], shifts=1 - stride, dims=-2)
            input[:, 2::4, :, :] = torch.roll(input[:, 2::4, :, :], shifts=1 - stride, dims=-2)
            input[:, 2::4, :, :] = torch.roll(input[:, 2::4, :, :], shifts=1 - stride, dims=-1)
            input[:, 3::4, :, :] = torch.roll(input[:, 3::4, :, :], shifts=1 - stride, dims=-1)

            input = rearrange(input, 'b (c n) h w -> b (n c) h w', n=Num).contiguous()
            _filter = rearrange(_filter, '(c n) ... -> (n c) ...', n=Num).contiguous()
            output = F.conv2d(input, _filter, stride=stride, padding=0, dilation=1, groups=Num)
            output = rearrange(output, 'b (n c) h w -> b (c n) h w', n=Num).contiguous()

        else:
            output = F.conv2d(input, _filter, stride=stride, padding=0, dilation=1, groups=1)


        if self.ifbias:
            self.register_buffer("bias", _bias)
            output = output + _bias
        return output


    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)
            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
            # self.filter = _filter.detach()

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)
                # self.bias = _bias.detach()

        return super(StrideFconv_PCA, self).train(mode)


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



class GroupFconv_PCA(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):
        super(GroupFconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        assert outNum % inNum == 0, f"outNum ({outNum}) is not a multiple of inNum ({inNum})"

        self.tranNum = tranNum
        self.inNum = inNum
        self.outNum = outNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())
        # Basis.shape: torch.Size([3, 3, 4, 9])

        self.ifbias = bias
        if ifIni:  # first layer of F-Conv
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand
        self.weights = nn.Parameter(torch.Tensor(1, outNum, expand, Basis.size(3)))  # torch.Size([16, 5, 1, 3, 3])
        # self.weights: torch.Size([16, 5, 1, 9])

        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        if bias:
            self.c = nn.Parameter(torch.Tensor(1, 1, outNum, 1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()

    def forward(self, input):
        if self.training:  # default: True
            tranNum = self.tranNum  # 4
            outNum = self.outNum  # 16
            inNum = self.inNum  # 5

            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)  # torch.Size([16, 4, 5, 1, 3, 3])

            _filter = tempW.view(tranNum, outNum, self.sizeP, self.sizeP)  # torch.Size([64, 5, 3, 3])
            _filter = _filter.contiguous().view(tranNum * outNum, 1, self.sizeP, self.sizeP)

            if self.ifbias:
                _bias = self.c.repeat([1, tranNum, 1, 1]).reshape([1, tranNum*outNum, 1, 1])
        else:
            tranNum = self.tranNum  # 4
            inNum = self.inNum  # 5
            _filter = self.filter  # torch.Size([64, 3, 3, 3])
            if self.ifbias:
                _bias = self.bias

        input = rearrange(input, 'b (c t) h w -> b (t c) h w', t=self.tranNum).contiguous()
        self.register_buffer("filter", _filter)

        output = F.conv2d(input, _filter, padding=self.padding, dilation=1, groups=inNum * tranNum)

        if self.ifbias:
            self.register_buffer("bias", _bias)
            output = output + _bias
        output = rearrange(output, 'b (t c) h w -> b (c t) h w', t=tranNum).contiguous()
        return output


    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

            _filter = tempW.view(tranNum, outNum, self.sizeP, self.sizeP)  # torch.Size([64, 5, 3, 3])
            _filter = _filter.contiguous().view(tranNum * outNum, 1, self.sizeP, self.sizeP)

            self.register_buffer("filter", _filter)
            # self.filter = _filter.detach()

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)
                # self.bias = _bias.detach()

        return super(GroupFconv_PCA, self).train(mode)


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



class EQUpsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, tranNum=4):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                # m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(Fconv_PCA(3, inNum=num_feat, outNum=4*num_feat, tranNum=tranNum, padding=1, bias=True))
                m.append(TranPermute(tranNum=tranNum))
                m.append(nn.PixelShuffle(2))
                m.append(TranPermute(tranNum=tranNum))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(TranPermute(tranNum=tranNum))
            m.append(nn.PixelShuffle(3))
            m.append(TranPermute(tranNum=tranNum))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(EQUpsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops



class TranPermute(nn.Module):
    def __init__(self, tranNum=4):
        super().__init__()
        self.tranNum = tranNum

    def forward(self, x: torch.Tensor):
        return rearrange(x, 'b (c t) h w -> b (t c) h w', t=self.tranNum).contiguous()



def GetBasis_PCA(sizeP, tranNum=4, inP=None, Smooth=True):
    if inP == None:
        inP = sizeP  # 3
    # sizeP=3
    # tranNum=4

    inX, inY, Mask = MaskC(sizeP, tranNum)
    '''
    inX [[-1.  0.  1.]
         [-1.  0.  1.]
         [-1.  0.  1.]]
    inY [[-1. -1. -1.]
         [ 0.  0.  0.]
         [ 1.  1.  1.]]
    Mask [[1. 1. 1.]
          [1. 1. 1.]
          [1. 1. 1.]]
    '''

    X0 = np.expand_dims(inX, 2)  # (3, 3, 1)
    Y0 = np.expand_dims(inY, 2)  # (3, 3, 1)

    Mask = np.expand_dims(Mask, 2)  # (3, 3, 1)

    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)  # (1, 1, 4)

    #    theta = torch.FloatTensor(theta)
    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0
    #    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)
    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)

    BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)

    BasisC = np.reshape(BasisC, [sizeP * sizeP * tranNum, inP * inP])
    BasisS = np.reshape(BasisS, [sizeP * sizeP * tranNum, inP * inP])

    BasisR = np.concatenate((BasisC, BasisS), axis=1)

    U, S, VT = np.linalg.svd(np.matmul(BasisR.T, BasisR))

    Rank = np.sum(S > 0.0001)
    BasisR = np.matmul(np.matmul(BasisR, U[:, :Rank]), np.diag(1 / np.sqrt(S[:Rank] + 0.0000000001)))
    BasisR = np.reshape(BasisR, [sizeP, sizeP, tranNum, Rank])

    temp = np.reshape(BasisR, [sizeP * sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis=0) ** 2, axis=0) + np.std(np.sum(temp ** 2 * sizeP * sizeP, axis=0),
                                                              axis=0)) / np.mean(
        np.sum(temp, axis=0) ** 2 + np.sum(temp ** 2 * sizeP * sizeP, axis=0), axis=0)
    Trod = 1
    Ind = var < Trod
    Rank = np.sum(Ind)
    Weight = 1 / np.maximum(var, 0.04) / 25
    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight, 0), 0), 0) * BasisR

    return torch.FloatTensor(BasisR), Rank, Weight



def MaskC(SizeP, tranNum):
    p = (SizeP - 1) / 2
    x = np.arange(-p, p + 1) / p
    X, Y = np.meshgrid(x, x)
    C = X ** 2 + Y ** 2
    if tranNum == 4:
        Mask = np.ones([SizeP, SizeP])
    else:
        if SizeP > 4:
            Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)
        else:
            Mask = np.exp(-np.maximum(C - 1, 0) / 2)
    return X, Y, Mask



class eq_mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj


    @staticmethod
    def eq_A_log_init(d_state, d_inner, copies=-1, device=None, merge=False):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True

        return A_log


    @staticmethod
    def eq_D_init(d_inner, copies=-1, device=None, merge=False):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True

        return D


    @classmethod
    def eq_init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, tranNum=4):
        # dt proj ============================
        dt_projs = cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)

        dt_projs_bias = nn.Parameter(dt_projs.bias)  # (K, inner), torch.Size([96, 4])
        del dt_projs

        # A, D =======================================
        A_logs = cls.eq_A_log_init(d_state, d_inner, copies=1, merge=False)     # (K * D, N)
        Ds = cls.eq_D_init(d_inner, copies=1, merge=False)                      # (K * D)
        return A_logs, Ds, dt_projs_bias




# =====================================================
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)



class EQLayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor, tranNum=4):
        """
        x.shape = (B, C, H, W)
        """
        x = x.permute(0, 2, 3, 1)       # (B, H, W, C)

        weight = self.weight.repeat_interleave(tranNum)
        bias = self.bias.repeat_interleave(tranNum)
        normalized_shape = weight.shape

        x = nn.functional.layer_norm(x, normalized_shape, weight, bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x



class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        """
        x.shape = (B, C, H, W)
        """
        x = x.permute(0, 2, 3, 1)       # (B, H, W, C)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x



class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)



class EQMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = EQ_linear_inter(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = EQ_linear_inter(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.fc1(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        x = self.act(x)
        x = self.drop(x)
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.fc2(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        x = self.drop(x)
        return x





class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError





class EQSS2D(nn.Module):
    """
    from SS2Dv2, Equivariant SS2D
    """
    def __init__(
            self,
            d_model=96,
            d_state=1,
            ssm_ratio=1.0,
            tranNum=4,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v05",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.tranNum = int(tranNum)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first          # True
        self.with_dconv = d_conv > 1

        # tags for forward_type ==============================
        self.disable_force32, forward_type = self.checkpostfix("_no32", forward_type)                # False
        self.oact, forward_type = self.checkpostfix("_oact", forward_type)                           # False
        self.disable_z, forward_type = self.checkpostfix("_noz", forward_type)                       # True
        self.disable_z_act, forward_type = self.checkpostfix("_nozact", forward_type)                # False
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner*tranNum, channel_first, self.tranNum)   # False


        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)             # self.d_inner
        self.eq_in_proj = EQ_linear_inter(self.d_model//tranNum, d_proj//tranNum, tranNum=tranNum, bias=bias)  # bias False
        self.act: nn.Module = act_layer()           # nn.SiLU()


        # conv =======================================
        if self.with_dconv:         # True
            self.eq_conv2d = GroupFconv_PCA(sizeP=d_conv, inNum=self.d_inner//tranNum, outNum=self.d_inner,
                                            tranNum=tranNum, padding=(d_conv - 1) // 2, ifIni=1, bias=conv_bias)


        # x proj ============================
        self.eq_linear_BCdt = EQ_linear_inter(inNum=self.d_inner, outNum=self.dt_rank + self.d_state + self.d_state,
                                              tranNum=tranNum, bias=False)
        self.eq_linear_dt_projs = EQ_linear_inter_dt(inNum=self.dt_rank, outNum=self.d_inner, tranNum=tranNum,
                                                     bias=False, dt_rank=self.dt_rank, dt_scale=dt_scale)

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()                    # Identity()
        self.eq_out_proj = EQ_linear_inter(inNum=self.d_inner, outNum=self.d_model//tranNum,
                                           tranNum=tranNum, bias=bias)               # bias=False
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()        # Identity()


        if initialize in ["v0"]:            # True
            self.A_logs, self.Ds, self.dt_projs_bias = eq_mamba_init.eq_init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale=dt_scale, dt_init="random", dt_min=0.001, dt_max=0.1,
                dt_init_floor=1e-4,
                tranNum=tranNum,
            )
            

    def forward_core(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            **kwargs,
    ):
        delta_softplus = True
        tranNum = self.tranNum          # 4
        out_norm = self.out_norm        # EQLayerNorm2d((96,), eps=1e-05, elementwise_affine=True)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        x = rearrange(x, 'b (c t) h w -> b c t h w', t=tranNum).contiguous()
        batch, d_inner, tranNum, height, width = x.shape        # torch.Size([10, 96, 4, 56, 56])
        assert tranNum == 4, f"tranNum ({tranNum}) must be 4"

        d_state = self.d_state                                                  # 1
        k_group, d_inner, dt_rank = self.k_group, self.d_inner, self.dt_rank    # k_group=4, d_inner=96, dt_rank=6

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=delta_softplus):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=None)

        x_dbl = self.eq_linear_BCdt(rearrange(x, 'b c t h w -> b (h w) (c t)').contiguous())

        x_dbl = rearrange(x_dbl, 'b (h w) (c t) -> b c t h w', h=height, w=width, t=tranNum).contiguous()
        dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=1)
        # x_dbl: torch.Size([10, 8, 4, 56, 56])
        # dts:   torch.Size([10, 6, 4, 56, 56])
        # Bs:    torch.Size([10, 1, 4, 56, 56])
        # Cs:    torch.Size([10, 1, 4, 56, 56])

        dts = self.eq_linear_dt_projs(rearrange(dts, 'b r t h w -> b (h w) (r t)'))
        dts = rearrange(dts, 'b (h w) (c t) -> b c t h w', h=height, w=width, c=d_inner, t=tranNum).contiguous()
        # dts:   torch.Size([10, 96, 4, 56, 56])

        x = x.permute(0, 2, 1, 3, 4).contiguous()           # torch.Size([10, 4, 96, 3136])
        Bs = Bs.permute(0, 2, 1, 3, 4).contiguous()         # torch.Size([10, 4, 1, 3136])
        Cs = Cs.permute(0, 2, 1, 3, 4).contiguous()         # torch.Size([10, 4, 1, 3136])
        dts = dts.permute(0, 2, 1, 3, 4).contiguous()       # torch.Size([10, 4, 96, 3136])

        x = cross_scan_eq_split_torch(x)                    # (batch, tranNum, dim, height*width)
        Bs = cross_scan_eq_split_torch(Bs)                  # (batch, tranNum, d_state, height*width)
        Cs = cross_scan_eq_split_torch(Cs)                  # (batch, tranNum, d_state, height*width)
        dts = cross_scan_eq_split_torch(dts)                # (batch, tranNum, dim, height*width)

        x = rearrange(x, 'b t c l -> b (t c) l').contiguous()           # torch.Size([10, 384, 3136])
        dts = rearrange(dts, 'b t c l -> b (t c) l').contiguous()       # torch.Size([10, 384, 3136])

        delta_bias = self.dt_projs_bias.view(d_inner, 1).repeat(1, tranNum)
        As = -self.A_logs.view(d_inner, 1, d_state).repeat(1, tranNum, 1).contiguous().to(torch.float).exp()
        Ds = self.Ds.view(d_inner, 1).repeat(1, tranNum).contiguous().to(torch.float)
        #     A_log = A_log
        #     D = D.view(d_inner, 1).repeat(1, copies).contiguous()

        delta_bias = rearrange(delta_bias, 'c t -> (t c)').contiguous().to(x.device)
        As = rearrange(As, 'c t d -> (t c) d').contiguous().to(x.device)                              # (k * c, d_state), torch.Size([384, 1])
        Ds = rearrange(Ds, 'c t -> (t c)').contiguous().to(x.device)                              # (k * c, d_state), torch.Size([384, 1])

        if force_fp32:
            x, dts, Bs, Cs = to_fp32(x, dts, Bs, Cs)

        # #### original Mamba1 1D scan ####
        y: torch.Tensor = selective_scan(
            x, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus=True
        )  # torch.Size([10, 4, 96, 56, 56])


        y = rearrange(y, 'b (t c) l -> b t c l', t=tranNum).contiguous()  # torch.Size([10, 384, 3136])
        y = cross_merge_eq_concatenate_torch(y, height, width)
        y = y.permute(0, 2, 1, 3, 4).contiguous()  # torch.Size([10, 96, 4, 56, 56])

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                u=x, dts=dts, delta_bias=delta_bias,
                y=y, height=height, W=width,
            ))

        y = rearrange(y, 'b c t h w -> b (c t) h w', t=tranNum).contiguous()  # torch.Size([20, 96*4, 56, 56])
        y = out_norm(y)     # torch.Size([10, 96, 56, 56])

        return y.to(x.dtype)


    def forward(self, x: torch.Tensor, **kwargs):
        x = self.eq_in_proj(rearrange(x, 'b c h w -> b h w c').contiguous())             # torch.Size([10, 96, 56, 56])
        x = rearrange(x, 'b h w c -> b c h w').contiguous()

        if not self.disable_z:          # self.disable_z==True
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)

        if self.with_dconv:     # True
            x = self.eq_conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        x = self.forward_core(x)

        x = self.out_act(x)                 # Identity()
        if not self.disable_z:              # self.disable_z==True
            x = x * z

        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.dropout(self.eq_out_proj(x))
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True, tranNum=4):
        # LayerNorm = EQLayerNorm2d if channel_first else nn.LayerNorm          # channel_first: True
        # out_norm = LayerNorm(d_inner//tranNum)
        out_norm = EQLayerNorm2d(d_inner//tranNum)

        return out_norm, forward_type


    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value




# =====================================================
class EQVSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = EQLayerNorm2d,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=1.0,
            ssm_dt_rank: Any = "auto",
            tranNum=4,
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v05_noz",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            _SS2D: type = EQSS2D,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0                 # True
        self.mlp_branch = mlp_ratio > 0                 # True
        self.use_checkpoint = use_checkpoint            # False
        self.post_norm = post_norm                      # False

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim//tranNum)
            self.op = _SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                tranNum=tranNum,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        # self.drop_path = DropPath(drop_path)
        self.drop_path = EQDropout(drop_path, tranNum=tranNum)          # debug， 20251012

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim//tranNum)             # EQLayerNorm2d
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = EQMlp(in_features=hidden_dim//tranNum, hidden_features=mlp_hidden_dim//tranNum,
                             act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)


    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)



class EQVSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            eq_tranNum=4,
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="ln2d",  # "BN", "LN2D"
            downsample_version: str = "v3",  # "v1", "v2", "v3"
            patchembed_version: str = "v2",  # "v1", "v2"
            use_checkpoint=False,
            # =========================
            posembed=False,
            imgsize=224,
            _SS2D=EQSS2D,
            # _SS2D=SS2D,
            # =========================
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])         # True
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        norm_layer: nn.Module = EQLayerNorm2d
        ssm_act_layer: nn.Module = nn.SiLU
        mlp_act_layer: nn.Module = nn.GELU

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        self.patch_embed = self._make_patch_embed_v2(in_chans, dims[0], patch_size, eq_tranNum, patch_norm, norm_layer,
                                                     channel_first=self.channel_first)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = self._make_downsample_v3(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                tranNum=eq_tranNum,
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                tranNum=eq_tranNum,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))



        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features//eq_tranNum),  # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            # head=nn.Linear(self.num_features, num_classes),
            head=EQ_linear_output(inNum=self.num_features//eq_tranNum, outNum=num_classes, tranNum=eq_tranNum),
        ))

        self.apply(self._init_weights)      # maybe todo


    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed


    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}


    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}


    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, tranNum=4, patch_norm=True, norm_layer=EQLayerNorm2d,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            # nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            StrideFconv_PCA(sizeP=kernel_size, inNum=in_chans, outNum=embed_dim // (2*tranNum),
                            tranNum=tranNum, stride=stride, padding=padding, ifIni=1),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // (2*tranNum)) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            # nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            StrideFconv_PCA(sizeP=kernel_size, inNum=embed_dim // (2 * tranNum), outNum=embed_dim // tranNum,
                            tranNum=tranNum, stride=stride, padding=padding, ifIni=0),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim//tranNum) if patch_norm else nn.Identity()),
        )


    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, tranNum=4, norm_layer=EQLayerNorm2d, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            # nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            StrideFconv_PCA(sizeP=3, inNum=dim//tranNum, outNum=out_dim//tranNum, tranNum=tranNum, stride=2, padding=1, ifIni=0),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim//tranNum),
        )


    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=EQLayerNorm2d,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            tranNum=4,
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # ===========================
            _SS2D=EQSS2D,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(EQVSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                tranNum=tranNum,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))


    def forward(self, x: torch.Tensor):
        # x.shape: torch.Size([10, 3, 224, 224])
        x = self.patch_embed(x)         # torch.Size([10, 96, 56, 56])

        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.classifier(x)
        return x


    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()


        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"


    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4],
                                                             align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)



# =====================================================
class Backbone_EQVSSM(EQVSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), eq_tranNum=4, pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])

        self.out_indices = out_indices
        for i in out_indices:
            layer = EQLayerNorm2d(self.dims[i]//eq_tranNum)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")
            import sys
            sys.exit()


    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x

        return outs



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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = 10
    d_inner = 24
    height = 56
    width = 56

    print("================")
    eqvmamba_tiny = EQVSSM(
        depths=[2, 2, 8, 2],
        dims=96,
        drop_path_rate=0.2,
        # drop_path_rate=0.0,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
    ).to(device)

    x = torch.randn([batch, 3, 224, 224], device=device)
    x_rot = rotate_and_shift(x, rotate_times=1, rotate_dims=[-2, -1], shift_times=0, shift_dim=2)
    xs = torch.cat([x, x_rot], dim=0)

    ys = eqvmamba_tiny(xs)

    y, y_reverse = torch.split(ys, [batch, batch], dim=0)
    print_error(y.detach().cpu(), y_reverse.detach().cpu())



