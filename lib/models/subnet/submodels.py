from lib.models.backbone import build_subnet
from lib.models.backbone import build_supernet_DP
from lib.models.super_connect import head_supernet, MC_BN, Point_Neck_Mobile_simple_DP, Point_Neck_Mobile_simple
from lib.utils.transform import name2path
from lib.models.submodels import build_subnet_head, build_subnet_BN, build_subnet_feat_fusor
from lib.models.super_model import Super_model, Super_model_MACs, Super_model_retrain
from lib.models.backbone.MobileTrack.supernet import SuperNet
from lib.models.subnet.subnet import SubNet
from lib.models.subnet.sub_connect import sub_connect
from lib.utils.utils import load_pretrain, init_weights
import numpy as np
import random
import torch


# 为什么要叫做 light track M
# search_size template_size stride adj_channel build_module
class LightTrackM_Subnet(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=64, build_module=True):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Subnet, self).__init__(search_size=search_size, template_size=template_size, stride=stride)  # ATTENTION

        if build_module:
            # self.features, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)          # todo
            self.features = SubNet()      #
            self.feature_fusor = Point_Neck_Mobile_simple(inchannels=1280)  # stride=8, stride=16
            self.supernet_head = sub_connect()
        self.apply(init_weights)



class LightTrackM_FLOPs(Super_model_MACs):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=128):
        '''subclass calls father class's __init__ func'''
        super(LightTrackM_FLOPs, self).__init__(search_size=search_size, template_size=template_size,
                                                stride=stride)  # ATTENTION
        self.model = LightTrackM_Subnet(search_size=search_size, template_size=template_size,
                                          stride=stride, adj_channel=adj_channel)


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




if __name__ == '__main__':
    net = LightTrackM_Subnet().cuda()
    net = load_pretrain(net, 'pretrain/mobileone_epoch50.pth')
    template = torch.randn(1, 3, 128, 128).cuda()
    net.template(template)
    search = torch.randn(1, 3, 256, 256).cuda()
    net.track(search)
    print(net)
