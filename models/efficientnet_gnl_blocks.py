""" EfficientNet, MobileNetV3, etc Blocks

Hacked together by / Copyright 2020 Ross Wightman
"""

""" GroupNL: Low-Resource and Robust CNN Design over Cloud and Device
EfficientNet, MobileNetV3, etc Blocks with GroupNL Conv
Modified from timm
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: jianhang.xie@my.cityu.edu.hk
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models.layers import create_conv2d, drop_path, make_divisible, create_act_layer
from timm.models.layers.activations import sigmoid

import math

__all__ = [
    'SqueezeExcite', 'ConvBnAct', 'DepthwiseSeparableConv', 'InvertedResidual', 'CondConvResidual', 'EdgeResidual']


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """ Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W] """
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


# GroupNLModule
class GroupNLModule(nn.Module):

    def __init__(self, in_channels, out_channels, seed_channels, kernel_size, period_range, shift_range,
                 stride=1, padding=0, dilation=1, num_terms=3, shuffle=False, learn=False, groups=1, bias=False,
                 sparse=True):
        super(GroupNLModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sparse = sparse
        self.bias = bias

        self.num_terms = num_terms
        self.period_range = period_range
        self.shift_range = shift_range
        self.learn = learn

        self.reduction_ratio = math.ceil(out_channels / seed_channels)
        self.seed_channels = seed_channels
        self.generated_channels = self.seed_channels * (self.reduction_ratio - 1)

        self.splits1 = _split_channels(self.seed_channels, num_terms)
        self.splits2 = _split_channels(self.generated_channels, num_terms)

        self.expand = int(self.splits2[0] / math.gcd(self.splits1[0], self.splits2[0]))
        self.expand_sin = int(self.seed_channels / self.num_terms)

        self.periods = nn.Parameter(period_range[0] + (period_range[1] - period_range[0])
                                    * torch.rand(self.expand * num_terms))
        self.shifts = nn.Parameter(period_range[0] + (period_range[1] - period_range[0])
                                   * torch.rand(self.expand * num_terms))
        if not self.learn:
            self.periods.requires_grad = False
            self.shifts.requires_grad = False

        if self.sparse:
            self.groups = math.gcd(self.in_channels, self.seed_channels)
        else:
            self.groups = groups

        self.conv1 = nn.Conv2d(self.in_channels, self.seed_channels, kernel_size, stride=self.stride,
                               padding=self.padding, bias=self.bias, dilation=self.dilation, groups=self.groups)

        self.shuffle = shuffle
        self.channel_shuffle = ChannelShuffle(groups=num_terms)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'period=[{}, {}], shift=[{}, {}], num_terms={}, seed_ratio={:.2f}, ' \
               'period_grad={}, shift_grad={}, shuffle={}'.format(
            self.period_range[0], self.period_range[1], self.shift_range[0], self.shift_range[1],
            self.num_terms, self.seed_channels / self.out_channels,
            self.periods.requires_grad, self.shifts.requires_grad, self.shuffle
        )

    def forward(self, x):
        x1 = self.conv1(x)

        x1 = F.normalize(x1)

        x_split = torch.split(x1, self.splits1, 1)

        x2 = torch.cat([x_split[i].repeat([1, self.expand, 1, 1]) for i in range(self.num_terms)], 1)

        x_shifts = self.shifts.repeat_interleave(self.expand_sin).reshape([1, self.generated_channels, 1, 1])
        x_periods = self.periods.repeat_interleave(self.expand_sin).reshape([1, self.generated_channels, 1, 1])

        x2 = torch.sin((x2 - x_shifts) * x_periods)

        if self.shuffle:
            x2 = self.channel_shuffle(x2)

        x = torch.cat([x1, x2], 1)

        return x


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.ReLU,
            gate_layer=nn.Sigmoid, force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """
    def __init__(
            self, in_chs, out_chs, kernel_size, stride=1, dilation=1, pad_type='',
            skip=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_path_rate=0.):
        super(ConvBnAct, self).__init__()
        self.has_residual = skip and stride == 1 and in_chs == out_chs
        self.drop_path_rate = drop_path_rate
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs)
        self.act1 = act_layer(inplace=True)

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            se_layer=None, drop_path_rate=0.,
            period_range=(1, 2), shift_range=(1, 5), num_terms=4, shuffle=False, learn=False, sparse=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        # self.conv_dw = create_conv2d(
        #     in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        # Sparse GroupNL module replace the DW conv
        self.conv_dw = GroupNLModule(in_channels=in_chs, out_channels=in_chs, seed_channels=int(in_chs/2),
                                     kernel_size=dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2,
                                     dilation=dilation, period_range=period_range, shift_range=shift_range,
                                     num_terms=num_terms, shuffle=shuffle, learn=learn, bias=False, sparse=sparse)
        self.bn1 = norm_layer(in_chs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.dim)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0.,
            period_range=(1, 2), shift_range=(1, 5), num_terms=4, shuffle=False, learn=False, sparse=True):
        super(InvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs)
        try:
            self.act1 = act_layer(inplace=True)
        except:
            self.act1 = act_layer()
        # Depth-wise convolution
        # self.conv_dw = create_conv2d(
        #     mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
        #     padding=pad_type, depthwise=True, **conv_kwargs)
        # Sparse GroupNL module replace the DW conv
        self.conv_dw = GroupNLModule(in_channels=mid_chs, out_channels=mid_chs, seed_channels=int(mid_chs / 2),
                                     kernel_size=dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2,
                                     dilation=dilation, period_range=period_range, shift_range=shift_range,
                                     num_terms=num_terms, shuffle=shuffle, learn=learn, bias=False, sparse=sparse)
        self.bn2 = norm_layer(mid_chs)
        try:
            self.act2 = act_layer(inplace=True)
        except:
            self.act2 = act_layer()

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs)

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.dim)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x


class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, se_layer=None, num_experts=0, drop_path_rate=0.):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)

        super(CondConvResidual, self).__init__(
            in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, dilation=dilation, pad_type=pad_type,
            act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size, se_layer=se_layer, norm_layer=norm_layer, conv_kwargs=conv_kwargs,
            drop_path_rate=drop_path_rate)

        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        shortcut = x

        # CondConv routing
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))

        # Point-wise expansion
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(
            self, in_chs, out_chs, exp_kernel_size=3, stride=1, dilation=1, pad_type='',
            force_in_chs=0, noskip=False, exp_ratio=1.0, pw_kernel_size=1, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.):
        super(EdgeResidual, self).__init__()
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Expansion convolution
        self.conv_exp = create_conv2d(
            in_chs, mid_chs, exp_kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs)

    def feature_info(self, location):
        if location == 'expansion':  # after SE, before PWL
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.dim)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x
