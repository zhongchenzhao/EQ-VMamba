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
from PIL import Image


from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_MMSEG
from mmdet.registry import MODELS as MODELS_MMDET
# from mmengine.registry   import CONV_LAYERS
# from mmcv.cnn.bricks.conv import CONV_LAYERS as MODELS_CONV


class Fconv_PCAV0(nn.Module):
    def __init__(self, inNum, outNum, sizeP, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 tranNum=4, inP=None, output_padding=0, transposed=False, ifIni=0, Smooth=True, iniScale=1.0, **kwargs):

        super(Fconv_PCAV0, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.in_channels = inNum
        self.out_channels = outNum
        self.kernel_size = sizeP
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = output_padding
        self.groups = groups

        Basis = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())

        # print(Basis[:, :, 0, 0])
        # print(Basis[:, :, 1, 0])
        # print(Basis[:, :, 2, 0])
        # print(Basis[:, :, 3, 0])

        self.ifbias = bias
        # first layer of F-Conv
        expand = 1 if ifIni else tranNum
        self.expand = expand
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
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

            Num = tranNum // expand  # 4
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                for i in range(expand)]

            tempW = torch.cat(tempWList, dim=1)  # torch.Size([16, 4, 5, 1, 3, 3])

            _filter = tempW.reshape(
                [outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])  # torch.Size([64, 5, 3, 3])

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
            #     self.register_buffer("bias", _bias)
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
                print("\n\n del self.filter \n\n")
                del self.filter
                if self.ifbias:
                    print("\n\n del self.bias \n\n")
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
            # self.filter = _filter.detach()
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)
                # self.bias = _bias.detach()

        return super(Fconv_PCAV0, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)




@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class Fconv_PCA(nn.Module):
    def __init__(self, inNum, outNum, sizeP, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 tranNum=4, inP=None, output_padding=0, transposed=False, ifIni=0, Smooth=True, iniScale=1.0, **kwargs):
        super(Fconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.in_channels = inNum
        self.out_channels = outNum
        self.kernel_size = sizeP
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = output_padding
        self.groups = groups

        self.ifbias = bias
        if ifIni:  # first layer of F-Conv
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand

        if sizeP == 1:      # 1x1 Conv2D
            self.kernel_size = 1
            self.stride = 1
            self.padding = 0
            self.dilation = 1
            self.groups = 1
            self.transposed = False
            self.output_padding = 0
            iniw = Getini_reg(1, inNum, outNum, self.expand) * iniScale
            self.weights = nn.Parameter(iniw, requires_grad=True)
            if bias:
                self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1), requires_grad=True)
            else:
                self.register_parameter('c', None)

        else:       # 3x3 Conv2D
            Basis = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
            self.register_buffer("Basis", Basis)  # .cuda())

            self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, Basis.size(3)), requires_grad=True)
            # self.weights: torch.Size([16, 5, 1, 9])

            # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
            if padding == None:
                self.padding = 0
            else:
                self.padding = padding
            if bias:
                self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1), requires_grad=True)
            else:
                self.register_parameter('c', None)

            self.reset_parameters()

    def forward(self, input):
        tranNum = self.tranNum  # 4
        outNum = self.outNum  # 16
        inNum = self.inNum  # 5
        expand = self.expand  # 1

        if self.sizeP == 1:
            tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1, tranNum, 1, 1, 1, 1])
            Num = tranNum // expand
            tempWList = [
                torch.cat([tempW[:, i * Num:(i + 1) * Num, :, -i:, ...], tempW[:, i * Num:(i + 1) * Num, :, :-i, ...]],
                          dim=3) for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, 1, 1])

            output = F.conv2d(input, _filter,
                              padding=self.padding,
                              dilation=1,
                              groups=1)
            if self.ifbias:
                bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])  # .cuda()
                output = output + bias
            return output

        else:
            if self.training:  # default: True

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
                    # self.register_buffer("bias", _bias)           # debug, 20251125
            else:
                _filter = self.filter  # torch.Size([64, 3, 3, 3])
                if self.ifbias:
                    _bias = self.bias

            output = F.conv2d(input, _filter, padding=self.padding, dilation=1, groups=1)
            if self.ifbias:
                output = output + _bias
            return output



    def train(self, mode=True):
        if self.sizeP == 1:
            return super(Fconv_PCA, self).train(mode)

        else:
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
                # self.register_buffer("filter", _filter)
                self.filter = _filter.detach()
                if self.ifbias:
                    _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                    # self.register_buffer("bias", _bias)
                    self.bias = _bias.detach()

            return super(Fconv_PCA, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class Fconv_PCA_out(nn.Module):
    def __init__(self, inNum, outNum, sizeP, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 tranNum=4, inP=None, output_padding=0, transposed=False, ifIni=0, Smooth=True, iniScale=1.0, **kwargs):
        super(Fconv_PCA_out, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.in_channels = inNum
        self.out_channels = outNum
        self.kernel_size = sizeP
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = output_padding
        self.groups = groups

        Basis = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
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
        output = F.conv2d(input, _filter, padding=self.padding, dilation=1, groups=1)

        if self.ifbias:
            _bias = self.c
            output = output + _bias
        return output


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
        return super(Fconv_PCA_out, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)



@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class EQPixelShuffle_out(nn.Module):
    def __init__(self, scale=2, tranNum=4):
        super().__init__()
        self.scale = scale
        self.tranNum = tranNum
        self.pixelshuffle = nn.PixelShuffle(scale)


    def forward(self, x: torch.Tensor):
        """
        x.shape: (b, c*t, h, w), t=4
        """
        # assert self.tranNum==4, f"self.tranNum {self.tranNum} must be 4!"
        x = rearrange(x, 'b (c t) h w -> b c t h w', t=self.tranNum)
        x = x[:, :, [1, 0, 2, 3], :, :]
        x = rearrange(x, 'b c t h w -> b (c t) h w')
        x = self.pixelshuffle(x)
        return x



class Fconv_1X1(nn.Module):
    def __init__(self, inNum, outNum, sizeP=1, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 tranNum=4, ifIni=0, Smooth=True, iniScale=1.0, **kwargs):
        super(Fconv_1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.in_channels = inNum
        self.out_channels = outNum
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1
        self.transposed = False
        self.output_padding = 0
        if bias:
            self.bias = nn.Parameter(torch.zeros(outNum))  # ← Tensor
        else:
            self.register_parameter('bias', None)

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(1, inNum, outNum, self.expand) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0

        if bias:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=True)
        else:
            self.c = torch.zeros(1, outNum, 1, 1)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1, tranNum, 1, 1, 1, 1])

        Num = tranNum // expand
        tempWList = [
            torch.cat([tempW[:, i * Num:(i + 1) * Num, :, -i:, ...], tempW[:, i * Num:(i + 1) * Num, :, :-i, ...]],
                      dim=3) for i in range(expand)]
        tempW = torch.cat(tempWList, dim=1)

        _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, 1, 1])

        bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])  # .cuda()

        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + bias



@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class Fconv_1X1_out(nn.Module):
    def __init__(self, inNum, outNum, tranNum=4, ifIni=0, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 Smooth=True, iniScale=1.0, ):
        super(Fconv_1X1_out, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.in_channels = inNum
        self.out_channels = outNum
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1
        self.transposed = False
        self.output_padding = 0

        iniw = Getini_reg(1, inNum, outNum, expand=1) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0

        if bias:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=True)
        else:
            self.c = torch.zeros(1, outNum, 1, 1)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1, 1, 1, tranNum, 1, 1])
        _filter = tempW.reshape([outNum, inNum * tranNum, 1, 1])
        bias = self.c.reshape([1, outNum, 1, 1])  # .cuda()
        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + bias



@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class EQSyncBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, **kwargs):
        super(EQSyncBatchNorm2d, self).__init__() 
        self.norm = nn.SyncBatchNorm(num_features=num_features, **kwargs)
        
    def forward(self, x: torch.Tensor, tranNum=4) -> torch.Tensor:
        """
        x: [B, C, H, W], C = base_channels * tranNum
        """
        x = rearrange(x, 'b (c t) h w -> (b t) c h w', t=tranNum).contiguous()
        x = self.norm(x)
        return rearrange(x, '(b t) c h w -> b (c t) h w', t=tranNum).contiguous()



@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
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



class StrideFconv_PCA(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, stride=1, dilation=1, groups=1, padding=None, ifIni=0, bias=True, Smooth=True,
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
        Basis = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
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
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, stride=1, dilation=1, groups=1, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):
        super(GroupFconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        assert outNum % inNum == 0, f"outNum ({outNum}) is not a multiple of inNum ({inNum})"

        self.tranNum = tranNum
        self.inNum = inNum
        self.outNum = outNum
        self.sizeP = sizeP
        Basis = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
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

        return super(GroupFconv_PCA, self).train(mode)


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


def GetBasis_PCA(sizeP, tranNum=4, inP=None, Smooth=True):
    if inP == None:
        inP = sizeP  # 3
    # sizeP=3
    # tranNum=4
    # Smooth=True

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

    return torch.FloatTensor(BasisR)


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



def Getini(sizeP, inNum, outNum, expand):
    
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    X0 = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0,0),0),4),0)
    y  = Y0[:,1]
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y,0),0),3),0)

    orlW = np.zeros([outNum,inNum,expand,sizeP,sizeP,1,1])
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                temp = np.array(Image.fromarray(((np.random.randn(3,3))*2.4495/np.sqrt((inNum)*sizeP*sizeP))).resize((sizeP,sizeP)))
                orlW[i,j,k,:,:,0,0] = temp
             
    v = np.pi/sizeP*(sizeP-1)
    k = np.reshape((np.arange(sizeP)),[1,1,1,1,1,sizeP,1])
    l = np.reshape((np.arange(sizeP)),[1,1,1,1,1,sizeP])

    tempA =  np.sum(np.cos(k*v*X0)*orlW,4)/sizeP
    tempB = -np.sum(np.sin(k*v*X0)*orlW,4)/sizeP
    A     =  np.sum(np.cos(l*v*y)*tempA+np.sin(l*v*y)*tempB,3)/sizeP
    B     =  np.sum(np.cos(l*v*y)*tempB-np.sin(l*v*y)*tempA,3)/sizeP 
    A     = np.reshape(A, [outNum,inNum,expand,sizeP*sizeP])
    B     = np.reshape(B, [outNum,inNum,expand,sizeP*sizeP]) 
    iniW  = np.concatenate((A,B), axis = 3)
    return torch.FloatTensor(iniW)



def Getini_reg(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
    return torch.FloatTensor(A)




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



# ================================ Bconv ================================


@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class Bconv_PCA(nn.Module):
    def __init__(self, inNum, outNum, sizeP, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 tranNum=4, inP=None, output_padding=0, transposed=False, ifIni=0, Smooth=True, iniScale=1.0, **kwargs):
        super(Bconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.in_channels = inNum
        self.out_channels = outNum
        self.kernel_size = sizeP
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = output_padding
        self.groups = groups

        self.ifbias = bias
        if ifIni:  # first layer of F-Conv
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand

        if sizeP == 1:      # 1x1 Conv2D
            self.kernel_size = 1
            self.stride = 1
            self.padding = 0
            self.dilation = 1
            self.groups = 1
            self.transposed = False
            self.output_padding = 0
            iniw = Getini_reg(1, inNum, outNum, self.expand) * iniScale
            self.weights = nn.Parameter(iniw, requires_grad=True)
            if bias:
                self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1), requires_grad=True)
            else:
                self.register_parameter('c', None)

        else:       # 3x3 Conv2D
            Basis = GetBasis_PCA_Bconv(sizeP, tranNum, inP, Smooth=Smooth)
            self.register_buffer("Basis", Basis)  # .cuda())

            self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, Basis.size(3)), requires_grad=True)
            # self.weights: torch.Size([16, 5, 1, 9])

            # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
            if padding == None:
                self.padding = 0
            else:
                self.padding = padding
            if bias:
                self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1), requires_grad=True)
            else:
                self.register_parameter('c', None)

            self.reset_parameters()

    def forward(self, input):
        tranNum = self.tranNum  # 4
        outNum = self.outNum  # 16
        inNum = self.inNum  # 5
        expand = self.expand  # 1

        if self.sizeP == 1:
            tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1, tranNum, 1, 1, 1, 1])
            Num = tranNum // expand
            tempWList = [
                torch.cat([tempW[:, i * Num:(i + 1) * Num, :, -i:, ...], tempW[:, i * Num:(i + 1) * Num, :, :-i, ...]],
                          dim=3) for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, 1, 1])

            output = F.conv2d(input, _filter,
                              padding=self.padding,
                              dilation=1,
                              groups=1)
            if self.ifbias:
                bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])  # .cuda()
                output = output + bias
            return output

        else:
            if self.training:  # default: True

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
                    # self.register_buffer("bias", _bias)           # debug, 20251125
            else:
                _filter = self.filter  # torch.Size([64, 3, 3, 3])
                if self.ifbias:
                    _bias = self.bias

            output = F.conv2d(input, _filter, padding=self.padding, dilation=1, groups=1)
            if self.ifbias:
                output = output + _bias
            return output



    def train(self, mode=True):
        if self.sizeP == 1:
            return super(Bconv_PCA, self).train(mode)

        else:
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
                # self.register_buffer("filter", _filter)
                self.filter = _filter.detach()
                if self.ifbias:
                    _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                    # self.register_buffer("bias", _bias)
                    self.bias = _bias.detach()

            return super(Bconv_PCA, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)





@MODELS.register_module()
@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class Bconv_PCA_out(nn.Module):
    def __init__(self, inNum, outNum, sizeP, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 tranNum=4, inP=None, output_padding=0, transposed=False, ifIni=0, Smooth=True, iniScale=1.0, **kwargs):
        super(Bconv_PCA_out, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.in_channels = inNum
        self.out_channels = outNum
        self.kernel_size = sizeP
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = output_padding
        self.groups = groups

        Basis = GetBasis_PCA_Bconv(sizeP, tranNum, inP, Smooth=Smooth)
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
        output = F.conv2d(input, _filter, padding=self.padding, dilation=1, groups=1)

        if self.ifbias:
            _bias = self.c
            output = output + _bias
        return output


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




def BicubicIni(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    Ind1 = (absx<=1)
    Ind2 = (absx>1)*(absx<=2)
    temp = Ind1*(1.5*absx3-2.5*absx2+1)+Ind2*(-0.5*absx3+2.5*absx2-4*absx+2)
    return temp


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

    return torch.FloatTensor(Basis)

