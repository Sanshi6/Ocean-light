from lib.models.backbone import build_subnet
from lib.models.backbone import build_supernet_DP
from lib.models.super_connect import head_supernet, MC_BN, Point_Neck_Mobile_simple_DP, Point_Neck_Mobile_simple
from lib.utils.transform import name2path
from lib.models.submodels import build_subnet_head, build_subnet_BN, build_subnet_feat_fusor
from lib.models.super_model import Super_model, Super_model_MACs, Super_model_retrain
from lib.models.backbone.MobileTrack.supernet import SuperNet
import numpy as np
import random


# 为什么要叫做 light track M
# search_size template_size stride adj_channel build_module
class LightTrackM_Supernet(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=64, build_module=True):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Supernet, self).__init__(search_size=search_size, template_size=template_size,
                                                   stride=stride)  # ATTENTION
        # config #
        # which parts to search, self.search_back / self.search_ops / self.search_head
        # self.search_back, self.search_ops, self.search_head = 1, 1, 1
        # backbone config
        # self.stage_idx = [1, 2, 3]  # which stages to use
        # self.max_flops_back = 470
        # head config
        # self.channel_head = [128, 192, 256]
        # kernel head?
        # self.kernel_head = [3, 5, 0]  # 0 means skip connection
        # ?
        # self.tower_num = 8  # max num of layers in the head

        # choice
        # self.num_choice_channel_head = len(self.channel_head)
        # self.num_choice_kernel_head = len(self.kernel_head)

        # Compute some values #
        # channel_back: List[int] = [24, 40, 80, 96, 192]
        # self.in_c = [self.channel_back[idx] for idx in self.stage_idx]          # todo
        # strides: List[int] = [4, 8, 16, 16, 32] TODO
        # strides_use = [self.strides[idx] for idx in self.stage_idx]
        # strides_use_new = []
        # for item in strides_use:
        #     if item not in strides_use_new:
        #         strides_use_new.append(item)  # remove repeated elements
        # self.strides_use_new = strides_use_new

        # self.num_kernel_corr? num kernel corr?
        # self.num_kernel_corr = [int(round(template_size / stride) ** 2) for stride in strides_use_new]
        self.channel_list = [112, 256, 512]
        self.kernel_list = [3, 5, 7, 0]
        self.tower_num = 8
        if build_module:
            # self.features, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)          # todo
            self.features = SuperNet()
            self.feature_fusor = Point_Neck_Mobile_simple(inchannels=1280)  # stride=8, stride=16
            self.supernet_head = head_supernet(channel_list=self.channel_list, kernel_list=self.kernel_list,
                                               linear_reg=False, inchannels=adj_channel, towernum=self.tower_num)
        else:
            _, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)


class LightTrackM_FLOPs(Super_model_MACs):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=128):
        '''subclass calls father class's __init__ func'''
        super(LightTrackM_FLOPs, self).__init__(search_size=search_size, template_size=template_size,
                                                stride=stride)  # ATTENTION
        self.model = LightTrackM_Supernet(search_size=search_size, template_size=template_size,
                                          stride=stride, adj_channel=adj_channel)


class LightTrackM_Subnet(Super_model_retrain):
    def __init__(self, path_name, search_size=256, template_size=128, stride=16, adj_channel=128):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Subnet, self).__init__(search_size=search_size, template_size=template_size,
                                                 stride=stride)  # ATTENTION
        model_cfg = LightTrackM_Supernet(search_size=search_size, template_size=template_size,
                                         stride=stride, adj_channel=adj_channel, build_module=False)

        # 从这儿开始看
        path_backbone, path_head, path_ops = name2path(path_name, sta_num=model_cfg.sta_num)
        # build the backbone
        self.features = build_subnet(path_backbone, ops=path_ops)  # sta_num is based on previous flops
        # build the neck layer
        self.neck = build_subnet_BN(path_ops, model_cfg)
        # build the Correlation layer and channel adjustment layer
        self.feature_fusor = build_subnet_feat_fusor(path_ops, model_cfg, matrix=True, adj_channel=adj_channel)
        # build the head
        self.head = build_subnet_head(path_head, channel_list=model_cfg.channel_head, kernel_list=model_cfg.kernel_head,
                                      inchannels=adj_channel, linear_reg=True, towernum=model_cfg.tower_num)


class LightTrackM_Speed(LightTrackM_Subnet):
    def __init__(self, path_name, search_size=256, template_size=128, stride=16, adj_channel=128):
        super(LightTrackM_Speed, self).__init__(path_name, search_size=search_size, template_size=template_size,
                                                stride=stride, adj_channel=adj_channel)

    def forward(self, x, zf):
        # backbone
        xf = self.features(x)
        # BN before Point-wise Corr
        zf, xf = self.neck(zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # head
        oup = self.head(feat_dict)
        return oup


class SuperNetToolbox(object):
    def __init__(self, model):
        self.model = model

    def get_path_back(self, prob=None):
        """randomly sample one path from the backbone supernet"""
        if prob is None:
            path_back = [np.random.choice(self.model.num_choice_back, item).tolist() for item in self.model.sta_num]
        else:
            path_back = [np.random.choice(self.model.num_choice_back, item, prob).tolist() for item in
                         self.model.sta_num]
        # add head and tail
        path_back.insert(0, [0])
        path_back.append([0])
        return path_back

    def get_path_head_single(self):
        num_choice_channel_head = self.model.num_choice_channel_head                    # 3
        num_choice_kernel_head = self.model.num_choice_kernel_head                      # 4
        tower_num = self.model.tower_num                                                # 8
        oup = [random.randint(0, num_choice_channel_head - 1)]                       # num of choices for head's channel
        arch = [random.randint(0, num_choice_kernel_head - 2)]
        arch += list(np.random.choice(num_choice_kernel_head, tower_num - 1))           # 3x3 conv, 5x5 conv, skip
        oup.append(arch)
        return oup

    def get_path_head(self):
        """randomly sample one path from the head supernet"""
        cand_h_dict = {'cls': self.get_path_head_single(), 'reg': self.get_path_head_single()}
        return cand_h_dict

    def get_path_ops(self):
        """randomly sample an output position"""
        stage_idx = random.choice(self.model.stage_idx)
        block_num = self.model.sta_num[stage_idx]
        block_idx = random.randint(0, block_num - 1)
        return [stage_idx, block_idx]

    def get_one_path(self):
        """randomly sample one complete path from the whole supernet"""
        cand_back, cand_OP, cand_h_dict = None, None, None
        tower_num = self.model.tower_num
        if self.model.search_back or self.model.search_ops:
            # backbone operations
            cand_back = self.get_path_back()
        if self.model.search_ops:
            # backbone output positions
            cand_OP = self.get_path_ops()
        if self.model.search_head:
            # head operations·
            cand_h_dict = self.get_path_head()
        else:
            cand_h_dict = {'cls': [0, [0] * tower_num], 'reg': [0, [0] * tower_num]}  # use fix head (only one choice)
        return {'back': cand_back, 'ops': cand_OP, 'head': cand_h_dict}


import torch

if __name__ == '__main__':
    # net = LightTrackM_Supernet().features
    net = LightTrackM_Supernet()
    x = torch.ones([1, 3, 255, 255])
    # net(x, [[0], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [0]])
    # print(net)
    net_box = SuperNetToolbox(net)
    sub_net_box = net_box.get_one_path()
    print(sub_net_box)
