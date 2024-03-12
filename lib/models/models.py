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
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=64, build_module=True, path=None):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Supernet, self).__init__(search_size=search_size, template_size=template_size, stride=stride)  # ATTENTION
        backbone_path= None
        if path is not None:
            backbone_path = [[path['bit00']],
                             [path['bit10'], path['bit11'], path['bit12']],
                             [path['bit20'], path['bit21'], path['bit22'], path['bit23'], path['bit24'], path['bit25'], path['bit26'], path['bit27'], path['bit28'], path['bit29'], path['bit210'], path['bit211']],
                             [path['bit30'], path['bit31'], path['bit32'], path['bit33'], path['bit34'], path['bit35'], path['bit36'], path['bit37'], path['bit38'], path['bit39'], path['bit310'], path['bit311'], path['bit312'], path['bit313'], path['bit314']],
                             [path['bit40']]]

            cls_reg_path = {'cls': [path['cls00'], [path['cls10'], path['cls11'], path['cls12'], path['cls13'], path['cls14'], path['cls15'], path['cls16'], path['cls17']]],
                            'reg': [path['reg00'], [path['reg10'], path['reg11'], path['reg12'], path['reg13'], path['reg14'], path['reg15'], path['reg16'], path['reg17']]]}

            for i in range(path["bit1"]):
                backbone_path[1][-(i+1)] = 3

            for i in range(path["bit2"]):
                backbone_path[2][-(i + 1)] = 3

            for i in range(path["bit3"]):
                backbone_path[3][-(i + 1)] = 3

            for i in range(path["cls"]):
                cls_reg_path['cls'][1][-(i+1)] = 3

            for i in range(path["reg"]):
                cls_reg_path['reg'][1][-(i + 1)] = 3

            print("backbone_path = ", backbone_path)
            print("cls_reg_path = ", cls_reg_path)

        self.channel_list = [112, 256, 512]
        self.kernel_list = [3, 5, 7, 0]
        self.tower_num = 8
        if build_module:
            # self.features, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)          # todo
            self.features = SuperNet(path=backbone_path)      #
            self.feature_fusor = Point_Neck_Mobile_simple(inchannels=1280)  # stride=8, stride=16
            self.supernet_head = head_supernet(channel_list=self.channel_list, kernel_list=self.kernel_list,
                                               linear_reg=False, inchannels=adj_channel, towernum=self.tower_num, path=cls_reg_path)  #
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
    # # net = LightTrackM_Supernet().features
    # net = LightTrackM_Supernet()
    # x = torch.ones([1, 3, 255, 255])
    # # net(x, [[0], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [0]])
    # # print(net)
    # net_box = SuperNetToolbox(net)
    # sub_net_box = net_box.get_one_path()
    # print(sub_net_box)
    path ={'bit00': 2, 'bit10': 0, 'bit11': 1, 'bit12': 0, 'bit20': 0, 'bit21': 2, 'bit22': 0, 'bit23': 2, 'bit24': 0, 'bit25': 0, 'bit26': 2, 'bit27': 0, 'bit28': 0, 'bit29': 0, 'bit210': 2, 'bit211': 2, 'bit30': 0, 'bit31': 1, 'bit32': 2, 'bit33': 2, 'bit34': 0, 'bit35': 2, 'bit36': 2, 'bit37': 1, 'bit38': 1, 'bit39': 0, 'bit310': 1, 'bit311': 0, 'bit312': 2, 'bit313': 1, 'bit314': 2, 'bit40': 0, 'bit1': 0, 'bit2': 3, 'bit3': 8, 'cls00': 1, 'cls10': 0, 'cls11': 1, 'cls12': 0, 'cls13': 1, 'cls14': 2, 'cls15': 2, 'cls16': 2, 'cls17': 0, 'cls': 3, 'reg00': 1, 'reg10': 1, 'reg11': 2, 'reg12': 0, 'reg13': 0, 'reg14': 1, 'reg15': 2, 'reg16': 1, 'reg17': 0, 'reg': 4}
    LightTrackM_Supernet(path=path)
