import os
import threading
import time
import argparse
from mpi4py import MPI
# from __future__ import absolute_import

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

# parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
# parser.add_argument('--arch', dest='arch', default='SiamFCIncep22',
#                     help='architecture of model')
# parser.add_argument('--start_epoch', default=30, type=int, required=True, help='test end epoch')
# parser.add_argument('--end_epoch', default=50, type=int, required=True,
#                     help='test end epoch')
# parser.add_argument('--gpu_nums', default=4, type=int, required=True, help='test start epoch')
# parser.add_argument('--anchor_nums', default=5, type=int, help='anchor numbers')
# parser.add_argument('--threads', default=16, type=int, required=True)
# parser.add_argument('--dataset', default='VOT0219', type=str, help='benchmark to test')
# parser.add_argument('--align', default='False', type=str, help='align')
# args = parser.parse_args()
#
# # init gpu and epochs
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
# GPU_ID = rank % args.gpu_nums
# node_name = MPI.Get_processor_name()  # get the name of the node
# os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
# print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
# time.sleep(rank * 5)
#
# # run test scripts -- two epoch for each thread
# for i in range(2):
#     arch = args.arch
#     dataset = args.dataset
#     try:
#         epoch_ID += args.threads  # for 16 queue
#     except:
#         epoch_ID = rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch
#
#     if epoch_ID > args.end_epoch:
#         continue
#
#     resume = 'snapshot/checkpoint_e{}.pth'.format(epoch_ID)
#     print('==> test {}th epoch'.format(epoch_ID))
#     os.system(
#         'python ./tracking/test_ocean.py --arch {0} --resume {1} --dataset {2} --align {3} --epoch_test True'.format(
#             arch, resume, dataset, args.align))
#
# for i in range(5, 51):
#     tracker = Lighttrack()
#     model = LightTrackM_Subnet().cuda()
#     model = load_pretrain(model, 'snapshot/submodel/checkpoint_e{i}.pth')
#     model.eval()
#     model.features = reparameterize_model(model.features)
#     model = model.cuda()
#
#     config = dict()
#     config['benchmark'] = 'OTB100'
#
#     if args.dataset.startswith('OTB'):
#         auc = auc_otb(tracker, model, config, '{i}')
#         print("auc = {}".format(auc))


# 定义一个函数来加载模型和计算AUC
# def evaluate_model(i):
#     tracker = Lighttrack()
#     model = LightTrackM_Subnet().cuda()
#     checkpoint_path = f'snapshot/submodel/checkpoint_e{i}.pth'  # 修改为f-string以插入i
#     model = load_pretrain(model, checkpoint_path)
#     model.eval()
#     model.features = reparameterize_model(model.features)
#     model = model.cuda()
#
#     config = dict()
#     config['benchmark'] = 'OTB100'
#
#     if config['benchmark'].startswith('OTB'):
#         auc = auc_otb(tracker, model, config, str(i))  # 确保传入的是字符串形式的i
#         print(f"Thread {i}: auc = {auc}")
#
#
# # 创建线程列表
# threads = []
#
# # 初始化并启动线程
# for i in range(5, 51):
#     # 创建线程
#     thread = threading.Thread(target=evaluate_model, args=(i,))
#     threads.append(thread)
#     thread.start()
#
# # 等待所有线程完成
# for thread in threads:
#     thread.join()
#
# print("所有模型评估完成。")

from multiprocessing import Pool
import torch


# 定义一个函数来加载模型和计算AUC
def evaluate_model(i):
    tracker = Lighttrack()
    model = LightTrackM_Subnet().cuda()
    checkpoint_path = f'snapshot/submodel/checkpoint_e{i}.pth'
    model = load_pretrain(model, checkpoint_path)
    model.eval()
    model.features = reparameterize_model(model.features)
    model = model.cuda()

    config = dict()
    config['benchmark'] = 'OTB100'
    auc = 0
    if config['benchmark'].startswith('OTB'):
        auc = auc_otb(tracker, model, config, str(i))
        print(f"Process {i}: auc = {auc}")
    return auc


def main():
    # The maximum number of concurrent processes
    max_processes = 2

    # Create a pool of processes
    with Pool(processes=max_processes) as pool:
        # The map function blocks until all results are finished
        auc_list = pool.map(evaluate_model, range(5, 51))

    print("All model evaluations are complete.")
    print(f"AUC List: {auc_list}")


# Protect the program's entry point
if __name__ == '__main__':
    # This is required for multiprocessing on Windows
    torch.multiprocessing.freeze_support()
    main()
