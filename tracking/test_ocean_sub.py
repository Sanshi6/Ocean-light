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
    model = LightTrackM_Subnet().cuda()
    model = load_pretrain(model, 'snapshot/checkpoint_e38.pth')
    model.eval()
    model.features = reparameterize_model(model.features)
    model = model.cuda()

    macs, params = get_model_complexity_info(model.features, (3, 255, 255), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)
    macs = float(macs.split()[0])

    config = dict()
    config['benchmark'] = 'OTB100'

    if args.dataset.startswith('OTB'):
        auc = auc_otb(tracker, model, config, '004')
        print("auc = {}".format(auc))






