#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : eq_mambairBconv_arch.py
@Author: ZhongchenZhao
@Date  : 2025/11/3 15:57
@Desc  : 
"""

# Code Implementation of the MambaIR Model
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

from basicsr.utils.registry import ARCH_REGISTRY



NEG_INF = -1000000

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

        # iniw = Getini_reg(inNum, outNum, tranNum)*iniScale #(outNum,1,inNum,expand)
        # self.weights = nn.Parameter(iniw, requires_grad=True)

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
        # self.dt_init_std = dt_rank ** -0.5 * dt_scale
        # 保存初始化参数
        self.dt_scale = dt_scale
        self.dt_rank = dt_rank
        self.reset_parameters_uniform()

    def reset_parameters_uniform(self) -> None:
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        nn.init.uniform_(self.weights, dt_init_std, dt_init_std)



class Bconv_PCA(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(Bconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA_Bconv(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())
        # Basis.shape: torch.Size([3, 3, 4, 9])

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
            expand = self.expand  # 1  first layer of F-Conv

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
                self.register_buffer("bias", _bias)
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
            _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
            self.register_buffer("filter", _filter)
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)

        return super(Bconv_PCA, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class Bconv_PCA_out(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(Bconv_PCA_out, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA_Bconv(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())

        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, 1, Basis.size(3)), requires_grad=True)
        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')

        # iniw = Getini_reg(Basis.size(3), inNum, outNum, 1, weight)*iniScale
        # self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.ifbias = bias
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
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)
            _filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
        else:
            _filter = self.filter
        _bias = self.c
        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + _bias

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)

            _filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
        return super(Bconv_PCA_out, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class StrideBconv_PCA(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, stride=1, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(StrideBconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.stride = stride
        self.ifIni = ifIni
        Basis, Rank, weight = GetBasis_PCA_Bconv(sizeP, tranNum, inP, Smooth=Smooth)
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

        return super(StrideBconv_PCA, self).train(mode)


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



class GroupBconv_PCA(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, dilation=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):
        super(GroupBconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        assert outNum % inNum == 0, f"outNum ({outNum}) is not a multiple of inNum ({inNum})"

        self.tranNum = tranNum
        self.inNum = inNum
        self.outNum = outNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA_Bconv(sizeP, tranNum, inP, Smooth=Smooth)
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
        if dilation == None:
            self.dilation = 1
        else:
            self.dilation = dilation
        self.reset_parameters()

    def forward(self, input):
        if self.training:  # default: True
            tranNum = self.tranNum  # 4
            outNum = self.outNum  # 16
            inNum = self.inNum  # 5
            expand = self.expand  # 1

            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)  # torch.Size([16, 4, 5, 1, 3, 3])
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            # self.Basis torch.Size([3, 3, 4, 9])
            # self.weights torch.Size([1, 96, 1, 9])
            # tempW torch.Size([1, 4, 96, 1, 3, 3])

            # _filter = tempW.view(tranNum, outNum, self.sizeP, self.sizeP)  # torch.Size([64, 5, 3, 3])
            # _filter = _filter.permute(1, 0, 2, 3).contiguous().view(outNum * tranNum, 1, self.sizeP, self.sizeP)

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

        output = F.conv2d(input, _filter, padding=self.padding, dilation=self.dilation, groups=inNum * tranNum)

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
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0

            _filter = tempW.view(tranNum, outNum, self.sizeP, self.sizeP)  # torch.Size([64, 5, 3, 3])
            _filter = _filter.contiguous().view(tranNum * outNum, 1, self.sizeP, self.sizeP)

            self.register_buffer("filter", _filter)
            # self.filter = _filter.detach()

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)
                # self.bias = _bias.detach()

        return super(GroupBconv_PCA, self).train(mode)


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



class EQPixelShuffle(nn.Module):
    def __init__(self, scale=2, tranNum=4):
        super().__init__()
        self.scale = scale
        self.tranNum = tranNum
        self.pixelshuffle = nn.PixelShuffle(scale)


    def forward(self, x: torch.Tensor):
        """
        x.shape: (b, c*t, h, w), t=4
        """
        # assert self.tranNum==4, f"self.tranNum {self.tranNum} must to be 4!"
        x = rearrange(x, 'b (c t) h w -> b (t c) h w', t=self.tranNum)
        x = self.pixelshuffle(x)
        x = rearrange(x, 'b (t c) h w -> b (c t) h w', t=self.tranNum)
        x = rearrange(x, 'b (c t) (h r) (w s) -> b c t h w r s', t=self.tranNum, r=self.scale, s=self.scale).contiguous()
        for rotate_times in range(1, self.tranNum):
            x[:, :, rotate_times, :, :] = torch.rot90(x[:, :, rotate_times, :, :], k=rotate_times, dims=(-2, -1))
        x = rearrange(x, 'b c t h w r s -> b (c t) (h r) (w s)').contiguous()
        return x





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

        # if copies > 0:
        #     A_log = A_log.view(d_inner, 1, d_state).repeat(1, copies, 1).contiguous()
        #     if merge:
        #         A_log = A_log.view(d_inner*copies, d_state)
        return A_log


    @staticmethod
    def eq_D_init(d_inner, copies=-1, device=None, merge=False):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True

        # if copies > 0:
        #     D = D.view(d_inner, 1).repeat(1, copies).contiguous()
        #     if merge:
        #         D = D.view(-1)
        return D


    @classmethod
    def eq_init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, tranNum=4):
        # dt proj ============================
        dt_projs = cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)

        # dt_projs_weight = nn.Parameter(dt_projs.weight)  # (K, inner, rank)
        # dt_projs_weight = dt_projs_weight.view(d_inner, 1, dt_rank).repeat(1, k_group, 1)
        dt_projs_bias = nn.Parameter(dt_projs.bias)  # (K, inner), torch.Size([96, 4])
        del dt_projs

        # A, D =======================================
        A_logs = cls.eq_A_log_init(d_state, d_inner, copies=1, merge=False)     # (K * D, N)
        Ds = cls.eq_D_init(d_inner, copies=1, merge=False)                      # (K * D)
        return A_logs, Ds, dt_projs_bias



class TranPermute(nn.Module):
    def __init__(self, tranNum=4):
        super().__init__()
        self.tranNum = tranNum

    def forward(self, x: torch.Tensor):
        return rearrange(x, 'b (c t) h w -> b (t c) h w', t=self.tranNum).contiguous()


class TranAvgPool(nn.Module):
    def __init__(self, tranNum):
        super(TranAvgPool, self).__init__()
        self.tranNum = tranNum

    def forward(self, x):
        x = rearrange(x, 'b (c t) h w -> b c t h w', t=self.tranNum)
        return x.mean(dim=2)


class EQLayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor, tranNum=4):
        """
        x.shape = (B, H, W, C)
        """
        weight = self.weight.repeat_interleave(tranNum)
        bias = self.bias.repeat_interleave(tranNum)
        normalized_shape = weight.shape

        x = nn.functional.layer_norm(x, normalized_shape, weight, bias, self.eps)       # (B, H, W, C)
        return x



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


def GetBasis_PCA_Bconv(sizeP, tranNum=4, inP=None, Smooth=True):
    if inP == None:
        inP = sizeP
    inp = inP // 2
    inX, inY, Mask = MaskC(sizeP, tranNum)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    Mask = np.expand_dims(np.expand_dims(Mask, 2), 3)
    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)

    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0

    X = X * inp
    Y = Y * inp

    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)

    k = np.reshape(np.arange(-inp, inp + 1), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(-inp, inp + 1), [1, 1, 1, 1, inP])

    # print(X[:,:,0,0,0])
    Basis = BicubicIni(X - k) * BicubicIni(Y - l)
    # print(Basis[:,:,1,2,2])

    Rank = inP * inP
    Weight = 1
    Basis = Basis.reshape([sizeP, sizeP, tranNum, Rank]) * Mask

    return torch.FloatTensor(Basis), Rank, Weight



def BicubicIni(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    Ind1 = (absx<=1)
    Ind2 = (absx>1)*(absx<=2)
    temp = Ind1*(1.5*absx3-2.5*absx2+1)+Ind2*(-0.5*absx3+2.5*absx2-4*absx+2)
    return temp


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



# =====================================================


class PermuteLayer(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class EQChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=60.0/(2*4), tranNum=4):
        super(EQChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            PermuteLayer(0, 2, 3, 1),
            # nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            EQ_linear_inter(num_feat//tranNum, num_feat // int(tranNum*squeeze_factor), tranNum=4, bias=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            EQ_linear_inter(num_feat // int(tranNum*squeeze_factor), num_feat//tranNum, tranNum=4, bias=True),
            PermuteLayer(0, 3, 1, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y



class EQCAB(nn.Module):
    """
    change squeeze_factor from 30 to 60.0/(2*4)
    """
    def __init__(self, num_feat, is_light_sr=True, compress_ratio=3, tranNum=4, squeeze_factor=60.0/(2*4)):
        super(EQCAB, self).__init__()

        if is_light_sr: # we use dilated-conv & DWConv for lightSR for a large ERF, True
            compress_ratio = 2
            self.cab = nn.Sequential(
                PermuteLayer(0, 2, 3, 1),
                EQ_linear_inter(num_feat//tranNum, num_feat // (tranNum*compress_ratio), tranNum=tranNum, bias=True),
                PermuteLayer(0, 3, 1, 2),
                GroupBconv_PCA(sizeP=3, inNum=num_feat // (tranNum*compress_ratio), outNum=num_feat // (tranNum*compress_ratio),
                               tranNum=tranNum, padding=1, ifIni=1, bias=True),
                nn.GELU(),
                PermuteLayer(0, 2, 3, 1),
                EQ_linear_inter(num_feat // (tranNum * compress_ratio), num_feat // tranNum, tranNum=tranNum, bias=True),
                PermuteLayer(0, 3, 1, 2),
                GroupBconv_PCA(3, inNum=num_feat // tranNum, outNum=num_feat // tranNum,
                               tranNum=tranNum, padding=2, ifIni=1, dilation=2, bias=True),
                EQChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                # nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                Bconv_PCA(3, inNum=num_feat//tranNum, outNum=num_feat // (tranNum*compress_ratio),
                          tranNum=tranNum, padding=1, bias=True),
                nn.GELU(),
                # nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                Bconv_PCA(3, inNum=num_feat//(tranNum*compress_ratio), outNum=num_feat // tranNum,
                          tranNum=tranNum, padding=1, bias=True),
                EQChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            EQLayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            EQLayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            EQLayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops



class EQSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1.2,
            tranNum=4,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.k_group = 4
        self.d_model = d_model
        self.d_state = d_state
        self.tranNum = tranNum
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 1.2*60=72
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.eq_in_proj = EQ_linear_inter(self.d_model // tranNum, (self.d_inner * 2) // tranNum, tranNum=tranNum,
                                          bias=bias)  # bias False

        self.eq_conv2d = GroupBconv_PCA(sizeP=d_conv, inNum=self.d_inner // tranNum, outNum=self.d_inner,
                                        tranNum=tranNum, padding=(d_conv - 1) // 2, ifIni=1, bias=conv_bias)

        self.act = nn.SiLU()

        self.eq_linear_BCdt = EQ_linear_inter(inNum=self.d_inner, outNum=self.dt_rank + self.d_state + self.d_state,
                                              tranNum=tranNum, bias=False)
        self.eq_linear_dt_projs = EQ_linear_inter_dt(inNum=self.dt_rank, outNum=self.d_inner, tranNum=tranNum,
                                                     bias=False, dt_rank=self.dt_rank, dt_scale=dt_scale)

        self.selective_scan = selective_scan_fn

        # self.out_norm = nn.LayerNorm(self.d_inner)
        self.eq_out_norm = EQLayerNorm(self.d_inner)

        self.eq_out_proj_z = EQ_linear_inter(inNum=self.d_inner, outNum=self.d_inner // tranNum,
                                             tranNum=tranNum, bias=bias)  # bias=False
        self.eq_out_proj = EQ_linear_inter(inNum=self.d_inner // tranNum, outNum=self.d_model // tranNum,
                                           tranNum=tranNum, bias=bias)  # bias=False
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None  # dropout=0

        self.A_logs, self.Ds, self.dt_projs_bias = eq_mamba_init.eq_init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale=dt_scale, dt_init="random", dt_min=0.001, dt_max=0.1,
            dt_init_floor=1e-4,
            tranNum=tranNum,
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

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
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape  # torch.Size([1, 16, 16, 72])
        L = H * W
        K = 4  # four scanning routes
        tranNum = self.tranNum  # 4
        d_state = self.d_state  # 10
        k_group, d_inner, dt_rank = self.k_group, self.d_inner, self.dt_rank  # k_group=4, d_inner=72, dt_rank=4

        x = rearrange(x, 'b (c t) h w -> b c t h w', t=tranNum).contiguous()
        batch, d_inner, tranNum, height, width = x.shape  # torch.Size([10, 96, 4, 56, 56])
        assert tranNum == 4, f"tranNum ({tranNum}) must be 4"

        x_dbl = self.eq_linear_BCdt(rearrange(x, 'b c t h w -> b (h w) (c t)').contiguous())
        x_dbl = rearrange(x_dbl, 'b (h w) (c t) -> b c t h w', h=height, w=width, t=tranNum).contiguous()
        dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=1)

        dts = self.eq_linear_dt_projs(rearrange(dts, 'b r t h w -> b (h w) (r t)'))
        dts = rearrange(dts, 'b (h w) (c t) -> b c t h w', h=height, w=width, c=d_inner, t=tranNum).contiguous()

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # torch.Size([10, 4, 96, 3136])
        Bs = Bs.permute(0, 2, 1, 3, 4).contiguous()  # torch.Size([10, 4, 1, 3136])
        Cs = Cs.permute(0, 2, 1, 3, 4).contiguous()  # torch.Size([10, 4, 1, 3136])
        dts = dts.permute(0, 2, 1, 3, 4).contiguous()  # torch.Size([10, 4, 96, 3136])

        x = cross_scan_eq_split_torch(x)  # (batch, tranNum, dim, height*width)
        Bs = cross_scan_eq_split_torch(Bs)  # (batch, tranNum, d_state, height*width)
        Cs = cross_scan_eq_split_torch(Cs)  # (batch, tranNum, d_state, height*width)
        dts = cross_scan_eq_split_torch(dts)  # (batch, tranNum, dim, height*width)
        # xs: torch.Size([1, 288, 256])
        # dts: torch.Size([1, 288, 256])
        # Bs: torch.Size([1, 4, 10, 256])
        # Cs: torch.Size([1, 4, 10, 256])
        # Ds: torch.Size([288])
        # As: torch.Size([288, 10])
        # dt_projs_bias: torch.Size([288])

        x = rearrange(x, 'b t c l -> b (t c) l').contiguous()  # torch.Size([10, 384, 3136])
        dts = rearrange(dts, 'b t c l -> b (t c) l').contiguous()  # torch.Size([10, 384, 3136])
        delta_bias = self.dt_projs_bias.view(d_inner, 1).repeat(1, tranNum)
        As = -self.A_logs.view(d_inner, 1, d_state).repeat(1, tranNum, 1).contiguous().to(torch.float).exp()
        Ds = self.Ds.view(d_inner, 1).repeat(1, tranNum).contiguous().to(torch.float)
        #     A_log = A_log
        #     D = D.view(d_inner, 1).repeat(1, copies).contiguous()

        delta_bias = rearrange(delta_bias, 'c t -> (t c)').contiguous().to(x.device)
        As = rearrange(As, 'c t d -> (t c) d').contiguous().to(x.device)  # (k * c, d_state), torch.Size([384, 1])
        Ds = rearrange(Ds, 'c t -> (t c)').contiguous().to(x.device)  # (k * c, d_state), torch.Size([384, 1])

        y = self.selective_scan(
            x, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        assert y.dtype == torch.float
        # out_y: torch.Size([1, 4, 72, 256])

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
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape  #

        xz = self.eq_in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # torch.Size([1, 16, 16, 72])

        x = x.permute(0, 3, 1, 2).contiguous()  # B, (C 4), H, W
        x = self.act(self.eq_conv2d(x))  # torch.Size([1, 288, 16, 16])
        x = self.forward_core(x)  # # B, (C 4), H, W

        assert x.dtype == torch.float32
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, (C 4)
        x = self.eq_out_norm(x)

        x = self.eq_out_proj_z(x)  # torch.Size([1, 16, 16, 72])
        x = x * F.silu(z)
        x = self.eq_out_proj(x)  # torch.Size([1, 16, 16, 60])
        if self.dropout is not None:
            x = self.dropout(x)
        return x



class EQVSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 60,
            drop_path: float = 0.013,
            norm_layer: Callable[..., torch.nn.Module] = partial(EQLayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        tranNum = 4
        self.tranNum = tranNum
        self.ln_1 = norm_layer(hidden_dim//tranNum)
        self.self_attention = EQSS2D(d_model=hidden_dim, d_state=d_state, expand=expand,dropout=attn_drop_rate, **kwargs)
        # self.drop_path = DropPath(drop_path)
        self.drop_path = EQDropout(drop_path, tranNum=4)          # debug， 20251012

        self.skip_scale= nn.Parameter(torch.ones(hidden_dim//tranNum))
        self.conv_blk = EQCAB(hidden_dim, is_light_sr)
        self.ln_2 = EQLayerNorm(hidden_dim//tranNum)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim//tranNum))


    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        skip_scale = self.skip_scale.repeat_interleave(self.tranNum)
        skip_scale2 = self.skip_scale2.repeat_interleave(self.tranNum)

        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*skip_scale + self.drop_path(self.self_attention(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x*skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x



class EQBasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: EQLayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=EQLayerNorm,
                 downsample=None,
                 use_checkpoint=False,is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(EQVSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=EQLayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:      # downsample==None
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:     # False
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:         # downsample=None
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops



class EQResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: EQLayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=EQLayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr = False):
        super(EQResidualGroup, self).__init__()

        tranNum = 4
        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = EQBasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':      # True
            # self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
            self.eqconv = Bconv_PCA(3, inNum=dim//tranNum, outNum=dim //tranNum, tranNum=tranNum, padding=1, bias=True)
        elif resi_connection == '3conv':
            # to save parameters and memory
            # self.conv = nn.Sequential(
            #     nn.Conv2d(dim, dim // 4, 3, 1, 1),
            #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #     nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #     nn.Conv2d(dim // 4, dim, 3, 1, 1))
            self.eqconv = nn.Sequential(
                Bconv_PCA(3, inNum=dim//tranNum, outNum=dim // (tranNum*4), tranNum=tranNum, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                PermuteLayer(0, 2, 3, 1),
                EQ_linear_inter(dim // (tranNum*4), dim // (tranNum*4), tranNum=tranNum, bias=True),
                PermuteLayer(0, 3, 1, 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                Bconv_PCA(3, inNum=dim//(tranNum*4), outNum=dim//tranNum, tranNum=tranNum, padding=1, bias=True),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.eqconv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


# ==========
class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        tranNum = 4
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim//tranNum)
        else:
            self.norm = None

    def forward(self, x):
        """
        input:  (B, C, H, W)
        output: (B, H*W, C)
        """
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops



class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """
        input:  (B, H*W, C)
        output: (B, C, H, W)
        """
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



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
                m.append(Bconv_PCA(3, inNum=num_feat//tranNum, outNum=4*num_feat//tranNum, tranNum=tranNum, padding=1, bias=True))
                m.append(EQPixelShuffle(scale=2, tranNum=tranNum))
        elif scale == 3:
            m.append(Bconv_PCA(3, inNum=num_feat//tranNum, outNum=9 * num_feat//tranNum, tranNum=tranNum, padding=1, bias=True))
            m.append(EQPixelShuffle(scale=3, tranNum=tranNum))
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



class EQUpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, tranNum=4):
        self.num_feat = num_feat
        m = []
        # m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        # m.append(nn.PixelShuffle(scale))
        m.append(Bconv_PCA(3, inNum=num_feat // tranNum, outNum=(scale**2) * num_out_ch, tranNum=tranNum, padding=1, bias=True))
        m.append(EQPixelShuffle(scale, tranNum=tranNum))
        m.append(TranAvgPool(tranNum=tranNum))
        super(EQUpsampleOneStep, self).__init__(*m)




# ==========
@ARCH_REGISTRY.register()
class EQMambaIRBconv(nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: EQLayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=EQLayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(EQMambaIRBconv, self).__init__()
        tranNum = 4
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        # self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.eqconv_first = Bconv_PCA(3, inNum=num_in_ch, outNum=embed_dim // tranNum,
                                               tranNum=tranNum, padding=1, ifIni=1,  bias=True)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim


        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        tranNum = 4
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = EQResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features//tranNum)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            # self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.eqconv_after_body = Bconv_PCA(3, inNum=embed_dim // tranNum, outNum=embed_dim // tranNum,
                                               tranNum=tranNum, padding=1, bias=True)

        elif resi_connection == '3conv':
            # to save parameters and memory
            # self.conv_after_body = nn.Sequential(
            #     nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
            #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #     nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
            #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #     nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            # )
            self.eqconv_after_body = nn.Sequential(
                Bconv_PCA(3, inNum=embed_dim // tranNum, outNum=embed_dim // (tranNum * 4), tranNum=tranNum, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                PermuteLayer(0, 2, 3, 1),
                EQ_linear_inter(embed_dim // (tranNum * 4), embed_dim // (tranNum * 4), tranNum=tranNum, bias=True),
                PermuteLayer(0, 3, 1, 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                Bconv_PCA(3, inNum=embed_dim // (tranNum * 4), outNum=embed_dim // tranNum, tranNum=tranNum, padding=1, bias=True),
            )

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            # self.conv_before_upsample = nn.Sequential(
            #     nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
            #     nn.LeakyReLU(inplace=True))

            self.eqconv_before_upsample = nn.Sequential(
                Bconv_PCA(3, inNum=embed_dim // tranNum, outNum=num_feat // tranNum, tranNum=tranNum, padding=1, bias=True),
                nn.LeakyReLU(inplace=True))

            self.equpsample = EQUpsample(upscale, num_feat)
            # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.eqconv_last = Bconv_PCA_out(3, inNum=num_feat // tranNum, outNum=num_out_ch, tranNum=tranNum, padding=1, bias=True),
        elif self.upsampler == 'pixelshuffledirect':            # True
            # for lightweight SR (to save parameters)
            self.equpsample = EQUpsampleOneStep(upscale, embed_dim, num_out_ch, tranNum=tranNum)

        else:
            # for image denoising
            # self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
            self.eqconv_last = Bconv_PCA_out(3, inNum=embed_dim // tranNum, outNum=num_out_ch,
                                               tranNum=tranNum, padding=1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, EQLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.eqconv_first(x)
            x = self.eqconv_after_body(self.forward_features(x)) + x
            x = self.eqconv_before_upsample(x)
            x = self.eqconv_last(self.equpsample(x))

        elif self.upsampler == 'pixelshuffledirect':            # upsampler='pixelshuffledirect'
            # for lightweight SR
            x = self.eqconv_first(x)
            x = self.eqconv_after_body(self.forward_features(x)) + x
            x = self.equpsample(x)

        else:
            # for image denoising
            x_first = self.eqconv_first(x)
            res = self.eqconv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.eqconv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


# =====================================================

def rotate_and_shift(x, rotate_times=1, rotate_dims=[-2, -1], shift_times=1, shift_dim=1):
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
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("================")
    block = EQVSSBlock(
        hidden_dim=60,
        drop_path=0.0043,
        norm_layer=EQLayerNorm,
        attn_drop_rate=0,
        d_state=10,
        expand=1.2,
        input_resolution=(64, 64), is_light_sr=True).to(device)

    x = torch.randn([4, 15, 4, 14, 14]).to(device)
    x_size = (14, 14)
    x_rot = rotate_and_shift(x, rotate_times=1, rotate_dims=[-2, -1], shift_times=1, shift_dim=2)
    xs = torch.cat([x, x_rot], dim=0)
    xs = rearrange(xs, 'b c t h w -> b (c t) h w').contiguous()
    xs = rearrange(xs, 'b (c t) h w -> b (h w) (c t)', t=4).contiguous()

    ys = block(xs, x_size)
    print("**** ys ", ys.shape)

    ys = rearrange(ys, 'b (h w) (c t) ->b (c t) h w', h=x_size[0], t=4).contiguous()

    ys = rearrange(ys, 'b (c t) h w -> b  c t h w', t=4).contiguous()
    y, y_reverse = torch.split(ys, [4, 4], dim=0)
    y_reverse = rotate_and_shift(y_reverse, rotate_times=-1, rotate_dims=[-2, -1], shift_times=-1, shift_dim=2)
    print_error(y.detach().cpu(), y_reverse.detach().cpu())


    print("================")


    net_g = EQMambaIRBconv(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        drop_rate=0.,
        d_state=10,
        mlp_ratio=1.2,
        drop_path_rate=0.1,
        norm_layer=EQLayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        upscale=3,
        img_range=1.,
        upsampler='pixelshuffledirect',
        resi_connection='1conv',
    ).to(device)


    x = torch.randn([5, 3, 16, 16], device=device)
    x_rot = rotate_and_shift(x, rotate_times=1, rotate_dims=[-2, -1], shift_times=0, shift_dim=2)
    xs = torch.cat([x, x_rot], dim=0)

    ys = net_g(xs)

    print("x", x.shape, 'y', y.shape)
    y, y_reverse = ys.chunk(2, dim=0)
    y_reverse = rotate_and_shift(y_reverse, rotate_times=-1, rotate_dims=[-2, -1], shift_times=0, shift_dim=2)
    print_error(y.detach().cpu(), y_reverse.detach().cpu())

