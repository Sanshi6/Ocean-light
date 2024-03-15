from __future__ import absolute_import

from ptflops import get_model_complexity_info
from ray.air import RunConfig
from ray.tune import Experiment, TuneConfig, TuneError
from ray.tune.search.bayesopt import BayesOptSearch

import tracking._init_paths
import os
import argparse
import numpy as np
import sys

from models.backbone.MobileTrack.mobileblock import reparameterize_model
from models.subnet.submodels import LightTrackM_Subnet
from tracker.lighttrack import Lighttrack

sys.path.append("..")
import models.models as models
from utils.utils import load_pretrain
from tracking.test_ocean import auc_otb, eao_vot
from tracker.ocean import Ocean
from easydict import EasyDict as edict
import torch

import ray
from ray import tune

from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.ax import AxSearch

from hyperopt import hp
from pprint import pprint

parser = argparse.ArgumentParser(description='parameters for Ocean tracker')
parser.add_argument('--dataset', default='OTB100', type=str, help='dataset')
args = parser.parse_args()

if __name__ == '__main__':
    tracker = Lighttrack()
    path ={'bit00': 2, 'bit10': 0, 'bit11': 1, 'bit12': 0, 'bit20': 0, 'bit21': 2, 'bit22': 0, 'bit23': 2, 'bit24': 0, 'bit25': 0, 'bit26': 2, 'bit27': 0, 'bit28': 0, 'bit29': 0, 'bit210': 2, 'bit211': 2, 'bit30': 0, 'bit31': 1, 'bit32': 2, 'bit33': 2, 'bit34': 0, 'bit35': 2, 'bit36': 2, 'bit37': 1, 'bit38': 1, 'bit39': 0, 'bit310': 1, 'bit311': 0, 'bit312': 2, 'bit313': 1, 'bit314': 2, 'bit40': 0, 'bit1': 0, 'bit2': 3, 'bit3': 8, 'cls00': 1, 'cls10': 0, 'cls11': 1, 'cls12': 0, 'cls13': 1, 'cls14': 2, 'cls15': 2, 'cls16': 2, 'cls17': 0, 'cls': 3, 'reg00': 1, 'reg10': 1, 'reg11': 2, 'reg12': 0, 'reg13': 0, 'reg14': 1, 'reg15': 2, 'reg16': 1, 'reg17': 0, 'reg': 4}
    model = models.LightTrackM_Supernet(path=path)
    model = load_pretrain(model, 'snapshot/checkpoint_e500.pth')
    model.eval()
    model.features = reparameterize_model(model.features)
    model = model.cuda()

    config = dict()
    config['benchmark'] = 'OTB100'

    if args.dataset.startswith('OTB'):
        auc = auc_otb(tracker, model, config, '002')
        print("auc = {}".format(auc))
