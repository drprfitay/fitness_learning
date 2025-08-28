#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:05:30 2025

@author: itayta
"""
import sys
import argparse
import datetime
import os
import time
import argparse
import builtins
import datetime
import os
import random
import shutil
import string
import subprocess

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as tdist

from torch.utils.tensorboard import SummaryWriter
from lightning.fabric import Fabric

# Logging - you can use either wandb or tensorboard
#import wandb

### ITAYFOLD IMPORTS
from constants import *
from utils import *
from dataset import *
from rosetta_former.embedding_mlp import * 

def restart_from_checkpoint_using_fabric(fabric, ckp_path, state_to_load, run_variables=None):
    """
    Re-start from checkpoint
    """
    try:
        reminder = fabric.load(ckp_path, state_to_load)
    except FileNotFoundError:
        print(f'checkoint {ckp_path} does not exist. skipping.')
        return
    print(f"Found checkpoint at {ckp_path}")

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in reminder:
                run_variables[var_name] = reminder[var_name]


def save_using_fabric(fabric, args, epoch, model, optimizer, **kwargs):
    to_save = {
        'model': model,
        'optimizer': optimizer,
        'grad_scaler': fabric.strategy.precision,  # does not really matter.
        'args': args,
        'epoch': epoch
    }
    to_save.update(kwargs)
    checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
    candidate = os.path.join(args.output_dir, f'checkpoint-{epoch:05d}.pth')
    if epoch >= args.epochs - 1:
        checkpoint_paths.append(candidate)
    elif epoch > 0 and (epoch % getattr(args, 'save_every', args.epochs) == 0):
        checkpoint_paths.append(candidate)
    for checkpoint_path in checkpoint_paths:
        fabric.save(checkpoint_path, to_save)


def lsf_kill_this_seq_array_jobs():
    # USE WIITH CARE!
    # run this _inside_ a job - it will kill it and all subsequent jobs in the same job array.
    jobid = os.environ.get('LSB_JOBID')
    cmd = f'bkill "{jobid}"'
    subprocess.call(cmd, shell=True)


def lsf_get_time_left_for_job():
    # run this _inside_ the job to know how long it has until termination
    # get jobid + array (https://www.ibm.com/support/pages/accessing-lsf-batch-job-id-and-array-id-within-job-environment)
    jobid = os.environ.get('LSB_BATCH_JID')
    cmd = f'bjobs -noheader -o "time_left" "{jobid}"'
    sec = None
    try:
        out = subprocess.check_output(cmd, shell=True)
        t = out.decode().strip().split(' ')[0]
        sec = 0
        cur = 60
        for i in reversed(t.split(':')):
            sec += cur * int(i)
            cur *= 60
        print(f'\t-lsf time remain- Got {out.decode().strip()} = {sec} [sec]')
    except Exception as e:
        print(f'Cannot get remaining time for job {jobid}: {e} ({type(e).__name__})')
    return sec


def lsf_get_num_devices_and_num_nodes_for_current_job():
    # run _inside_ job to know how many NODES are being utilized
    hosts = {}
    for h in os.environ['LSB_HOSTS'].split(' '):
        hosts[h] = hosts.get(h, 0) + 1
    cur_host = os.uname().nodename.split('.')[0]
    # print(f'{os.environ["LSB_HOSTS"]} -> {hosts}. current host = {cur_host}')
    num_nodes = len(hosts.keys())
    num_devices = hosts[cur_host]
    return num_devices, num_nodes


def lsf_get_all_queues_for_user():
    """
    get all the possible queues the user can run on
    :return: string
    """
    return "waic-long waic-short waic-risk long-gpu short-gpu risk-gpu"
    q = subprocess.run(f'bqueues -u {os.environ["USER"]}  -o "QUEUE_NAME" -noheader',
                       shell=True, capture_output=True, text=True)
    return q.stdout.replace('\n', ' ').strip()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)
    builtins.print = print

num_devices, num_nodes = lsf_get_num_devices_and_num_nodes_for_current_job()
fabric = Fabric(accelerator="cuda", strategy="ddp", precision="16-mixed", devices=num_devices, num_nodes=num_nodes)
fabric.launch()
setup_for_distributed(fabric.is_global_zero)

cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.cuda.set_device(fabric.device)

seed = fabric.global_rank  # set different seed for each GPU
fabric.seed_everything(seed, workers=True)

# # run wandb only on the main process
# if fabric.is_global_zero and wandb is not None:
#     try:
#         wandb.init(project=args.wandb_project, id=args.wandb_id, resume=args.wandb_id, dir=args.output_dir)
#         # you might need to add "entity=" argument to the init method.
#         wandb.config.update(args, allow_val_change=True)
#     except Exception as e:
#         print(f'Could not start wandb. got {e}')
#         sys.exit(1)

# limit number of threads for the main process per GPU -- ask Shai about this
torch.set_num_threads(2)
torch.set_num_interop_threads(2)  # this must be done only once!
    # ----------------------------------------------------------

# prepare the dataloader
# NO NEED to define DistributedSampler here!
base_dataset_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/datasets/random_100k_train"
    
dataset = EsmGfpDataset(sequences_path="%s/sequences.csv" % base_dataset_path,
                        embeddings_path="%s/embeddings" % base_dataset_path,
                        embedding_layer = -1,
                        tokens_path="%s/tokens" % base_dataset_path,
                        mode="embeddings")
    
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
train_loader = fabric.setup_dataloaders(train_loader)