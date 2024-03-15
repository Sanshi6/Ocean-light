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
parser.add_argument('--arch', dest='arch', default='Ocean',
                    help='architecture of model')
# parser.add_argument('--resume', default='/home/ubuntu/yl/SiamProject/OceanOps-Search/snapshot/checkpoint_e500.pth', type=str,
#                     help='resumed model')
parser.add_argument('--resume', default='E:\CapstoneProject\Ocean-light\snapshot\checkpoint_e500.pth', type=str,
                    help='resumed model')
parser.add_argument('--cache_dir', default='TPE_results', type=str, help='directory to store cache')
parser.add_argument('--gpu_nums', default=1, type=int, help='gpu numbers')
parser.add_argument('--trial_per_gpu', default=8, type=int, help='trail per gpu')
parser.add_argument('--dataset', default='OTB100', type=str, help='dataset')

args = parser.parse_args()

print('==> However TPE is slower than GENE')

# prepare tracker
info = edict()
info.dataset = args.dataset


# args.resume = os.path.abspath(args.resume)


# fitness function
def fitness(config):
    # create model
    """
    if 'Ocean' in args.arch:
        model = models.__dict__[args.arch](align=info.align)
        tracker = Ocean(info)
    else:
        raise ValueError('not supported other model now')
    """
    assert torch.cuda.is_available()
    tracker = Lighttrack(hp=config)
    path = {'bit00': 2, 'bit10': 0, 'bit11': 1, 'bit12': 0, 'bit20': 0, 'bit21': 2, 'bit22': 0, 'bit23': 2, 'bit24': 0,
            'bit25': 0, 'bit26': 2, 'bit27': 0, 'bit28': 0, 'bit29': 0, 'bit210': 2, 'bit211': 2, 'bit30': 0,
            'bit31': 1, 'bit32': 2, 'bit33': 2, 'bit34': 0, 'bit35': 2, 'bit36': 2, 'bit37': 1, 'bit38': 1, 'bit39': 0,
            'bit310': 1, 'bit311': 0, 'bit312': 2, 'bit313': 1, 'bit314': 2, 'bit40': 0, 'bit1': 0, 'bit2': 3,
            'bit3': 8, 'cls00': 1, 'cls10': 0, 'cls11': 1, 'cls12': 0, 'cls13': 1, 'cls14': 2, 'cls15': 2, 'cls16': 2,
            'cls17': 0, 'cls': 3, 'reg00': 1, 'reg10': 1, 'reg11': 2, 'reg12': 0, 'reg13': 0, 'reg14': 1, 'reg15': 2,
            'reg16': 1, 'reg17': 0, 'reg': 4}
    model = models.LightTrackM_Supernet(path=path)
    model = load_pretrain(model, args.resume)
    model.eval()
    model.features = reparameterize_model(model.features)
    model = model.cuda()

    # macs, params = get_model_complexity_info(model.features, (3, 255, 255), as_strings=True, print_per_layer_stat=True,
    #                                          verbose=True)
    # macs = float(macs.split()[0])
    # 检查FLOPs是否满足要求
    # print("the model flops: ", macs)
    # if macs > 600 or macs < 2:
    #     raise TuneError("FLOPs exceeded budget")

    print('pretrained model has been loaded')
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    # trial_id = ray.train.get_context().get_trial_id()
    # trial_id = Trainable().trial_id
    trial_id = tune.get_trial_id()
    print("trial_id = ", trial_id)
    model_config = dict()
    model_config['benchmark'] = args.dataset

    # VOT and Ocean
    if args.dataset.startswith('VOT'):
        eao = eao_vot(tracker, model, model_config)
        # reporter(EAO=eao)
        print("eao = {}".format(eao))
        return {"AUC": eao}

    # OTB and Ocean
    if args.dataset.startswith('OTB'):
        auc = auc_otb(tracker, model, model_config, trial_id)
        print("auc = {}".format(auc))
        # reporter(AUC=auc)
        return {"AUC": auc}


if __name__ == "__main__":
    ray.init()

    params = {
        "penalty_k": tune.quniform(0.001, 0.2, 0.001),
        "scale_lr": tune.quniform(0.001, 0.8, 0.001),
        "window_influence": tune.quniform(0.001, 0.65, 0.001),
        "small_sz": tune.choice([255]),
        "big_sz": tune.choice([255]),
        "ratio": tune.choice([1]),
    }

    tuner = tune.Tuner(
        tune.with_resources(fitness, {"gpu": 0.5, "cpu": 4}), param_space=params,
        # run_config=RunConfig(name="my_tune_run",
        #                      storage_path="/home/ubuntu/yl/SiamProject/OceanOps-Search/ray_result", ),
        run_config=RunConfig(name="my_tune_run",
                             storage_path=r"E:\CapstoneProject\Ocean-light\result", ),
        tune_config=TuneConfig(num_samples=800, mode='max', metric='AUC',
                               max_concurrent_trials=8, search_alg=OptunaSearch()))

    results = tuner.fit()
    print(results.get_best_result(metric="AUC", mode="max").config)
