import copy
import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from torchsummary import summary

from utils.utils import load_pretrain


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        self.flatten = flatten
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        return 1

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'output_size=' + str(self.output_size) \
            + ', pool_type=' + self.pool_type + ')'


class MobileTrackBlock(nn.Module):
    # Depthwise conv
    def __init__(self, in_planes, out_planes, stride, kernel_size, use_act, num_conv_branches, inference_mode):
        super(MobileTrackBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.use_act = use_act
        self.num_conv_branches = num_conv_branches
        self.inference_mode = inference_mode

        self.block = nn.Sequential()
        self.block.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=self.kernel_size,
                                         stride=stride,
                                         padding=self.padding,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         num_conv_branches=self.num_conv_branches)),
        self.block.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.out_planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         num_conv_branches=self.num_conv_branches,
                                         use_act=self.use_act))

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        return x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 num_conv_branches: int = 1,
                 use_act: bool = True) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.use_act = use_act

        # self.activation = nn.ReLU()
        if self.use_act:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # TODO: concat alternative

            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1 and padding != 0:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(out)

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


class SubNet(nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.in_channels = 3
        self.kernel_size = [3, 5, 7, 0]
        self.choices = len(self.kernel_size)
        self.block_num = [1, 3, 12, 15, 1]  # [1, 2, 8, 10, 1]
        self.block_channel = [64, 96, 192, 384, 1280]
        # self.block_channel = [16, 64, 96, 160, 480]

        self.block_stride = [2, 2, 2, 2, 1]
        self.inference_mode = False
        self.in_planes = 3
        self.kernel_size = [3, 5, 7, 0]
        self.architecture = [[2], [0, 1, 0], [0, 2, 0, 2, 0, 0, 2, 0, 0, 3, 3, 3],
                             [0, 1, 2, 2, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3], [0]]
        # self.architecture = [[2], [2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #                      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2]]

        self.stage0 = self._make_stage(self.block_stride[0], self.block_num[0], self.block_channel[0],
                                       architecture=self.architecture[0])
        self.stage1 = self._make_stage(self.block_stride[1], self.block_num[1], self.block_channel[1],
                                       architecture=self.architecture[1])
        self.stage2 = self._make_stage(self.block_stride[2], self.block_num[2], self.block_channel[2],
                                       architecture=self.architecture[2])
        self.stage3 = self._make_stage(self.block_stride[3], self.block_num[3], self.block_channel[3],
                                       architecture=self.architecture[3])
        self.stage4 = self._make_stage(self.block_stride[4], self.block_num[4], self.block_channel[4],
                                       architecture=self.architecture[4],
                                       last_use_act=False)
        self.stage = [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]
        # self.global_pool = SelectAdaptivePool2d(pool_type='avg')
        # self.classifier = nn.Linear(1280 * self.global_pool.feat_mult(), 1000)
        self.act = nn.ReLU()
        # self._get_path_back()  # update self.architecture

    def _make_stage(self, stride, num_blocks, out_channels, use_act=True, last_use_act=True, architecture=None):

        strides = [stride] + [1] * (num_blocks - 1)
        self.use_act = use_act

        blocks = []
        for ix, stride in enumerate(strides):
            if last_use_act is False and ix == len(strides) - 1:
                self.use_act = False
            kernel_size = self.kernel_size[architecture[ix]]
            if kernel_size == 0:
                continue
            blocks.append(MobileTrackBlock(in_planes=self.in_planes, out_planes=out_channels,
                                           stride=stride, kernel_size=kernel_size, use_act=self.use_act,
                                           num_conv_branches=2,
                                           inference_mode=self.inference_mode))  # todo num_conv_branches
            self.in_planes = out_channels
        return nn.Sequential(*blocks)

    def forward_features(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x


from ptflops import get_model_complexity_info

#
# model = resnet50()


from torchstat import stat

if __name__ == '__main__':
    net = SubNet()
    # 加载模型
    net = load_pretrain(net, 'pretrain/mobileone_epoch50.pth')
    # model = torch.load('pretrain/mobileone_epoch50.pth')

    net.eval()




    x = torch.randn(1, 3, 255, 255)
    x1 = net(x)



    # print(x.shape)
    net = reparameterize_model(net)
    x2 = net(x)
    # summary(net, (3, 255, 255), device='cpu')
    print('========================== The diff is', ((x2 - x1) ** 2).sum())
    stat(net, (3, 255, 255))

    summary(net, (3, 255, 255), device='cpu')
    macs, params = get_model_complexity_info(net, (3, 255, 255), as_strings=True, print_per_layer_stat=False,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # input_names = ['input']
    # output_names = ['output']
    # dummy_input = torch.randn(1, 3, 255, 255)
    # onnx_export_file = "NewTrack.onnx"
    # torch.onnx.export(net, dummy_input, onnx_export_file, input_names=input_names, output_names=output_names)
