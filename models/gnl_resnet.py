""" GroupNL: Low-Resource and Robust CNN Design over Cloud and Device
ResNet with GroupNL Conv
Modified from timm
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: jianhang.xie@my.cityu.edu.hk
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.resnet import create_attn, _create_resnet

def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return None
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)


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


class GNLBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None,
                 period_range=(1, 2), shift_range=(1, 5), num_terms=4, shuffle=False, learn=False, sparse=True):
        super(GNLBottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        if stride > 1:
            self.conv2 = nn.Conv2d(
                first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
                padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        else:
            self.conv2 = GroupNLModule(first_planes, width, int(first_planes // 2),
                                       kernel_size=3, stride=stride,
                                       period_range=period_range, shift_range=shift_range, padding=1,
                                       num_terms=num_terms, shuffle=shuffle, learn=learn, sparse=True)

        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


# Standard GNL Module (Sparse = False) && Reduction Rate = 1 // 2
class GNLSTDR2Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None,
                 period_range=(1, 2), shift_range=(1, 5), num_terms=4, shuffle=False, learn=False, sparse=True):
        super(GNLSTDR2Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        if stride > 1:
            self.conv2 = nn.Conv2d(
                first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
                padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        else:
            self.conv2 = GroupNLModule(first_planes, width, int(first_planes // 2),
                                       kernel_size=3, stride=stride,
                                       period_range=period_range, shift_range=shift_range, padding=1,
                                       num_terms=num_terms, shuffle=shuffle, learn=learn, sparse=False)

        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


# Standard GNL Module (Sparse = False) && Reduction Rate = 1 // 4
class GNLSTDR4Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None,
                 period_range=(1, 2), shift_range=(1, 5), num_terms=4, shuffle=False, learn=False, sparse=True):
        super(GNLSTDR4Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        if stride > 1:
            self.conv2 = nn.Conv2d(
                first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
                padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        else:
            self.conv2 = GroupNLModule(first_planes, width, int(first_planes // 4),
                                       kernel_size=3, stride=stride,
                                       period_range=period_range, shift_range=shift_range, padding=1,
                                       num_terms=num_terms, shuffle=shuffle, learn=learn, sparse=False)

        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


@register_model
def gnl_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=GNLBottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50', pretrained, **model_args)


@register_model
def gnl_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=GNLBottleneck, layers=[3, 4, 23, 3],  **kwargs)
    return _create_resnet('resnet101', pretrained, **model_args)


@register_model
def gnl_std_r2_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=GNLSTDR2Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50', pretrained, **model_args)


@register_model
def gnl_std_r2_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=GNLSTDR2Bottleneck, layers=[3, 4, 23, 3],  **kwargs)
    return _create_resnet('resnet101', pretrained, **model_args)


@register_model
def gnl_std_r4_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=GNLSTDR4Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50', pretrained, **model_args)


@register_model
def gnl_std_r4_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=GNLSTDR4Bottleneck, layers=[3, 4, 23, 3],  **kwargs)
    return _create_resnet('resnet101', pretrained, **model_args)


if __name__ == '__main__':
    model = gnl_resnet101()
    print(model)