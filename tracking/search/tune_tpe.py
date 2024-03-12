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
parser.add_argument('--resume', default='/home/ubuntu/yl/SiamProject/OceanOps-Search/snapshot/checkpoint_e500.pth', type=str,
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
    tracker = Lighttrack()
    model = models.LightTrackM_Supernet(path=config)
    model = load_pretrain(model, args.resume)
    model.eval()
    model.features = reparameterize_model(model.features)
    model = model.cuda()

    macs, params = get_model_complexity_info(model.features, (3, 255, 255), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)
    macs = float(macs.split()[0])
    # 检查FLOPs是否满足要求
    print("the model flops: ", macs)
    if macs > 600 or macs < 2:
        raise TuneError("FLOPs exceeded budget")

    print('pretrained model has been loaded')
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    trial_id = ray.train.get_context().get_trial_id()
    # trial_id = tune.get_trial_id()
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
        # backbone
        "bit00": tune.choice([0, 1, 2]),

        "bit10": tune.choice([0, 1, 2]),
        "bit11": tune.choice([0, 1, 2]),
        "bit12": tune.choice([0, 1, 2]),

        "bit20": tune.choice([0, 1, 2]),
        "bit21": tune.choice([0, 1, 2]),
        "bit22": tune.choice([0, 1, 2]),
        "bit23": tune.choice([0, 1, 2]),
        "bit24": tune.choice([0, 1, 2]),
        "bit25": tune.choice([0, 1, 2]),
        "bit26": tune.choice([0, 1, 2]),
        "bit27": tune.choice([0, 1, 2]),
        "bit28": tune.choice([0, 1, 2]),
        "bit29": tune.choice([0, 1, 2]),
        "bit210": tune.choice([0, 1, 2]),
        "bit211": tune.choice([0, 1, 2]),

        "bit30": tune.choice([0, 1, 2]),
        "bit31": tune.choice([0, 1, 2]),
        "bit32": tune.choice([0, 1, 2]),
        "bit33": tune.choice([0, 1, 2]),
        "bit34": tune.choice([0, 1, 2]),
        "bit35": tune.choice([0, 1, 2]),
        "bit36": tune.choice([0, 1, 2]),
        "bit37": tune.choice([0, 1, 2]),
        "bit38": tune.choice([0, 1, 2]),
        "bit39": tune.choice([0, 1, 2]),
        "bit310": tune.choice([0, 1, 2]),
        "bit311": tune.choice([0, 1, 2]),
        "bit312": tune.choice([0, 1, 2]),
        "bit313": tune.choice([0, 1, 2]),
        "bit314": tune.choice([0, 1, 2]),

        "bit40": tune.choice([0, 1, 2]),

        "bit1": tune.choice([0, 1, 2]),
        "bit2": tune.choice([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        "bit3": tune.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),

        # cls
        "cls00": tune.choice([0, 1, 2]),
        "cls10": tune.choice([0, 1, 2]),
        "cls11": tune.choice([0, 1, 2]),
        "cls12": tune.choice([0, 1, 2]),
        "cls13": tune.choice([0, 1, 2]),
        "cls14": tune.choice([0, 1, 2]),
        "cls15": tune.choice([0, 1, 2]),
        "cls16": tune.choice([0, 1, 2]),
        "cls17": tune.choice([0, 1, 2]),
        "cls": tune.choice([0, 1, 2, 3, 4, 5, 6]),

        # reg
        "reg00": tune.choice([0, 1, 2]),
        "reg10": tune.choice([0, 1, 2]),
        "reg11": tune.choice([0, 1, 2]),
        "reg12": tune.choice([0, 1, 2]),
        "reg13": tune.choice([0, 1, 2]),
        "reg14": tune.choice([0, 1, 2]),
        "reg15": tune.choice([0, 1, 2]),
        "reg16": tune.choice([0, 1, 2]),
        "reg17": tune.choice([0, 1, 2]),
        "reg": tune.choice([0, 1, 2, 3, 4, 5, 6]),

    }

    # bayes = BayesOptSearch(space=params, metric='AUC',mode='max', points_to_evaluate=current_best_params)

    tuner = tune.Tuner(
                       tune.with_resources(fitness, {"gpu": 0.125, "cpu": 2}), param_space=params,
                       run_config=RunConfig(name="my_tune_run",
                                            storage_path="/home/ubuntu/yl/SiamProject/OceanOps-Search/ray_result", ),
                       tune_config=TuneConfig(num_samples=50000, mode='max', metric='AUC',
                                              max_concurrent_trials=8, search_alg=OptunaSearch()))

    results = tuner.fit()
    print(results.get_best_result(metric="AUC", mode="max").config)

# Before
# ------
#
# from ray import tune
#
# def train_fn(config, reporter):
#     reporter(metric=1)
#
# tuner = tune.Tuner(train_fn)
# tuner.fit()
#
# After
# -----
#
# from ray import train, tune
#
# def train_fn(config):
#     train.report({"metric": 1})
#
# tuner = tune.Tuner(train_fn)
# tuner.fit()
