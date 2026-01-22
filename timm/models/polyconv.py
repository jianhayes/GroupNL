import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _addindent

_EPS = 1e-5


class ShieldPolynomialConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, num_seeds, num_terms, exp_range, fanout_factor, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, poly_bias=False, onebyone=False):

        assert out_channels == in_channels, "out channels has to be equal to in channels"

        super(ShieldPolynomialConv2d, self).__init__(
            in_channels, num_seeds, kernel_size, stride, padding, dilation, groups, bias=False)

        self.num_terms = num_terms
        self.exp_range = exp_range
        self.fanout_factor = fanout_factor
        self.mid_chs = int(num_seeds * fanout_factor)

        if onebyone:
            self.gn = nn.GroupNorm(num_groups=16, num_channels=self.mid_chs)
            self.act = nn.ReLU(inplace=True)
            self.onebyone = nn.Conv2d(self.mid_chs, out_channels, kernel_size=1, bias=False)
        else:
            self.onebyone = None

        self.register_buffer(
            'poly_exponent', torch.tensor(
                exp_range[0] + (exp_range[1] - exp_range[0]) * torch.rand(
                    num_terms, self.mid_chs, in_channels // groups)).unsqueeze(dim=-1).unsqueeze(dim=-1))
        if poly_bias:
            self.register_buffer(
                'poly_bias', torch.tensor(
                    torch.rand(self.mid_chs, in_channels // groups)).unsqueeze(dim=-1).unsqueeze(dim=-1))

    # override the print func to print out rbf kernel as well
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        if hasattr(self, 'poly_bias'):
            main_str += ', poly_terms={}, range=({}, {}), factor={}, poly_bias=True'.format(
                self.num_terms, self.exp_range[0], self.exp_range[1], self.fanout_factor)
        else:
            main_str += ', poly_terms={}, range=({}, {}), factor={}, poly_bias=False'.format(
                self.num_terms, self.exp_range[0], self.exp_range[1], self.fanout_factor)

        main_str += ')'
        return main_str

    def forward(self, x):

        tmp = self.weight.repeat(self.num_terms, self.fanout_factor, 1, 1, 1)
        tmp2 = torch.sum(tmp.abs().pow(self.__getattr__('poly_exponent')), dim=0)

        if hasattr(self, 'poly_bias'):
            tmp2 += self.__getattr__('poly_bias')

        w = torch.mul(tmp2, self.weight.sign())

        # weight standardization from https://github.com/joe-siyuan-qiao/WeightStandardization
        weight_mean = w.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True).mean(dim=3, keepdim=True)
        w = w - weight_mean
        std = w.view(w.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + _EPS
        w = w / std.expand_as(w)

        features = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.onebyone:
            return self.onebyone(self.act(self.gn(features)))
        else:
            return features


class ShieldRBFFamilyConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, num_seeds, exp_range, fanout_factor, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, rbf='gaussian'):

        assert out_channels == in_channels, "out channels has to be equal to in channels"

        super(ShieldRBFFamilyConv2d, self).__init__(
            in_channels, num_seeds, kernel_size, stride, padding, dilation, groups, bias)

        self.exp_range = exp_range
        self.fanout_factor = fanout_factor
        self.mid_chs = int(num_seeds * fanout_factor)

        # self.onebyone = None
        self.gn = nn.GroupNorm(num_groups=16, num_channels=self.mid_chs)
        self.act = nn.ReLU(inplace=True)
        self.onebyone = nn.Conv2d(self.mid_chs, out_channels, kernel_size=1, bias=False)

        self.register_buffer(
            'exponent', torch.tensor(
                exp_range * torch.rand(
                    self.mid_chs, in_channels // groups)).unsqueeze(dim=-1).unsqueeze(dim=-1))

        self.rbf = rbf  # remember which radial basis function to use

    # override the print func to print out rbf kernel as well
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        main_str += 'rbf={}, '.format(self.rbf)
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    def forward(self, x):

        tmp = self.weight.repeat(self.fanout_factor, 1, 1, 1)

        if self.rbf == 'gaussian':
            w = torch.exp(-(torch.pow(tmp.mul(self.__getattr__('exponent')), 2)))
        elif self.rbf == 'multiquadric':
            w = torch.sqrt(1 + torch.pow(tmp.mul(self.__getattr__('exponent')), 2))
        elif self.rbf == 'inverse_quadratic':
            w = torch.reciprocal(1 + torch.pow(tmp.mul(self.__getattr__('exponent')), 2))
        elif self.rbf == 'inverse_multiquadric':
            w = torch.reciprocal(torch.sqrt(1 + torch.pow(tmp.mul(self.__getattr__('exponent')), 2)))
        else:
            raise NotImplementedError

        # weight standardization from https://github.com/joe-siyuan-qiao/WeightStandardization
        weight_mean = w.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True).mean(dim=3, keepdim=True)
        w = w - weight_mean
        std = w.view(w.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + _EPS
        w = w / std.expand_as(w)

        features = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.onebyone:
            return self.onebyone(self.act(self.gn(features)))
        else:
            return features