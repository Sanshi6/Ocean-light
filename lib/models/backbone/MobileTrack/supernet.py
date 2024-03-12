import torch.nn as nn
from timm.models import SelectAdaptivePool2d
import torch
from lib.models.backbone.MobileTrack.mobileblock import MobileTrackBlock, reparameterize_model
from lib.models.backbone.MobileTrack.mobileblock import MobileOneBlock
import numpy as np


def forward_hook_fn(
        module,  # 被注册钩子的对象
        input,  # module 前向计算的输入
        output,  # module 前向计算的输出
):
    print("enter the hook")


class SuperNet(nn.Module):
    def __init__(self):
        super(SuperNet, self).__init__()
        self.in_channels = 3
        self.kernel_size = [3, 5, 7, 0]
        self.choices = len(self.kernel_size)
        self.block_num = [1, 3, 12, 15, 1]  # [1, 2, 8, 10, 1]
        self.block_channel = [64, 96, 192, 384, 1280]
        self.block_stride = [True, True, True, True, False]
        self.inference_mode = False
        self.in_planes = 3
        self.stage0 = self._make_stage(self.block_stride[0], self.block_num[0], self.block_channel[0])
        self.stage1 = self._make_stage(self.block_stride[1], self.block_num[1], self.block_channel[1])
        self.stage2 = self._make_stage(self.block_stride[2], self.block_num[2], self.block_channel[2], use_act=False)
        self.stage3 = self._make_stage(self.block_stride[3], self.block_num[3], self.block_channel[3])
        self.stage4 = self._make_stage(self.block_stride[4], self.block_num[4], self.block_channel[4], last_use_act=False)
        self.stage = [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]
        self.global_pool = SelectAdaptivePool2d(pool_type='avg')
        self.classifier = nn.Linear(1280 * self.global_pool.feat_mult(), 1000)
        self.act = nn.ReLU()

        self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1000, bias=False))

        self._get_path_back()  # update self.architecture
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_stage(self, stride, num_blocks, out_channels, use_act=True, last_use_act=True):
        if stride:
            strides = [2] + [1] * (num_blocks - 1)
        else:
            strides = [1] + [1] * (num_blocks - 1)

        if use_act is False:
            self.use_act = False
        else:
            self.use_act = True

        blocks = []
        for ix, stride in enumerate(strides):
            if last_use_act is False and ix == len(strides) - 1:
                self.use_act = False
            block = []
            if ix == 0:
                self.kernel_size = [3, 5, 7]
            for kernel_size in self.kernel_size:
                if kernel_size == 0:
                    block.append(nn.Identity())
                    continue
                block.append(MobileTrackBlock(in_planes=self.in_planes, out_planes=out_channels,
                                              stride=stride, kernel_size=kernel_size, use_act=self.use_act,
                                              num_conv_branches=1,
                                              inference_mode=self.inference_mode))
            blocks.append(nn.Sequential(*block))
            self.kernel_size = [3, 5, 7, 0]
            self.in_planes = out_channels
        return nn.Sequential(*blocks)

    # def _get_path_back(self):
    #     """randomly sample one path from the backbone supernet"""
    #     sta_num = [1, 2, 8, 10, 1]
    #     path_back = []
    #     for item in sta_num:
    #         path = [np.random.choice(3)] + np.random.choice(4, (item - 1)).tolist()
    #         path_back.append(path)
    #     self.architecture = path_back

    # def _get_path_back(self):
    #     """randomly sample one path from the backbone supernet"""
    #     # path_back = [[item for item in sub_list if item != 3] + [3] * sub_list.count(3) for sub_list in path_back]
    #     sta_num = [1, 2, 8, 10, 1]
    # path_back = []
    # for item in sta_num:
    #     path = np.random.choice(3, item).tolist()
    #     path_back.append(path)
    #
    # new_path_back = []
    # for num, sub_path in zip(sta_num, path_back):
    #     identity_num = np.random.randint(0, num)
    #     sub_path = sub_path[::-1]
    #     for i in range(identity_num):
    #         sub_path[i] = 3
    #     sub_path = sub_path[::-1]
    #     new_path_back.append(sub_path)

    # path_back = [np.random.choice(3, item).tolist() for item in sta_num]
    # identity_nums = np.random.randint(0, sta_num, len(path_back))
    #
    # new_path_back = [sub_path[::-1] for sub_path in path_back]
    # new_path_back = [[3 if i < identity_num else val for i, val in enumerate(sub_path)][::-1] for
    #                  sub_path, identity_num in zip(new_path_back, identity_nums)]
    # self.architecture = new_path_back

    def _get_path_back(self):
        sta_num = [1, 3, 12, 15, 1]

        path_back = []
        identity_nums = []
        new_path_back = []

        for item in sta_num:
            path = np.random.choice(3, item).tolist()
            path_back.append(path)

        for item in sta_num:
            identity_num = np.random.randint(int(item * 2 / 3) + 1)
            identity_nums.append(identity_num)

        for sub_path, identity_num in zip(path_back, identity_nums):
            reversed_sub_path = sub_path[::-1]
            modified_sub_path = []
            for i, val in enumerate(reversed_sub_path):
                if i < identity_num:
                    modified_sub_path.append(3)
                else:
                    modified_sub_path.append(val)
            new_path_back.append(modified_sub_path[::-1])
        self.architecture = new_path_back

    def forward_features(self, x):
        # architecture = [[0], [], [], [], [], [0]]
        # assert (len(self.blocks == len(architecture)))  # avoid bugs
        for layer, layer_arch in zip(self.stage, self.architecture):
            # assert (len(layer) == len(layer_arch))  # avoid bugs
            # for blocks, arch in zip(layer, layer_arch):
            for i in range(len(layer)):
                blocks = layer[i]
                arch = layer_arch[i]

                x = blocks[arch](x)

                if len(layer_arch) is self.block_num[2] and arch != 3:
                    stride8_out = x

                if len(layer_arch) == self.block_num[2]:
                    x = self.act(x)
        return x

    def forward(self, x):
        # x = self.forward_features(x)
        # x = self.global_pool(x)
        # x = x.flatten(1)
        # return self.classifier(x)
        x = self.forward_features(x)

        # x = self.globalpool(x)
        #
        # x = self.dropout(x)
        # x = x.contiguous().view(-1, 1280)
        # x = self.classifier(x)
        return x


#
# def get_path_back():
#     """randomly sample one path from the backbone supernet"""
#     # path_back = [[item for item in sub_list if item != 3] + [3] * sub_list.count(3) for sub_list in path_back]
#     sta_num = [1, 2, 8, 10, 1]
#     # path_back = []
#     # for item in sta_num:
#     #     path = np.random.choice(3, item).tolist()
#     #     path_back.append(path)
#     #
#     # new_path_back = []
#     # for num, sub_path in zip(sta_num, path_back):
#     #     identity_num = np.random.randint(0, num)
#     #     sub_path = sub_path[::-1]
#     #     for i in range(identity_num):
#     #         sub_path[i] = 3
#     #     sub_path = sub_path[::-1]
#     #     new_path_back.append(sub_path)
#
#     path_back = [np.random.choice(3, item).tolist() for item in sta_num]
#     identity_nums = np.random.randint(0, sta_num, len(path_back))
#
#     new_path_back = [sub_path[::-1] for sub_path in path_back]
#     new_path_back = [[3 if i < identity_num else val for i, val in enumerate(sub_path)][::-1] for sub_path, identity_num
#                      in zip(new_path_back, identity_nums)]
#
#     return new_path_back

import math


def get_path_back():
    cls_total_path = []
    channel_choice_path = np.random.choice(3, 1).tolist()
    kernel_choice_path = np.random.choice(4, 8).tolist()
    kernel_choice_path = [x for x in kernel_choice_path if x != 3] + [3]*kernel_choice_path.count(3)
    cls_total_path.append(channel_choice_path)
    cls_total_path.append(kernel_choice_path)

    reg_total_path = []
    channel_choice_path = np.random.choice(3, 1).tolist()
    kernel_choice_path = np.random.choice(4, 8).tolist()
    kernel_choice_path = [x for x in kernel_choice_path if x != 3] + [3]*kernel_choice_path.count(3)
    reg_total_path.append(channel_choice_path)
    reg_total_path.append(kernel_choice_path)

    cand_h_dict = {'cls': cls_total_path, 'reg': reg_total_path}
    return cand_h_dict


if __name__ == '__main__':
    for i in range(1000):
        new_path_back = get_path_back()
        print(new_path_back)
    # net = SuperNet()
    # x = torch.randn(1, 3, 224, 224)
    # net(x)
    # print(net)
    # net.load_state_dict(torch.load(r"E:\SiamProject\LightTrack-VGG\workdirs\model.pth"))
    # print(net)
    # sum_path = [[0], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0]]
    # for i in range(1000):
    #     path = get_path_back()
    #     print(path)
    #     for i, sub_path in enumerate(path):
    #         for j, num in enumerate(sub_path):
    #             sum_path[i][j] = sum_path[i][j] + num

    # print(sum_path)
    # print(path)
    # new_path = []
    # for sub_list in path:
    #     count_3 = sub_list.count(3)  # 统计列表中 3 的个数
    #
    #     my_list = [item for item in sub_list if item != 3]  # 移除所有的 3
    #     my_list.extend([3] * count_3)  # 将所有的 3 添加到列表的最后
    #     new_path.append(my_list)
    # print(new_path)

    # new_path = [[item for item in sub_list if item != 3] + [3] * sub_list.count(3) for sub_list in path]
    # print(new_path)
