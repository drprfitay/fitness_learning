import sys
import argparse
import datetime
import os
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as tdist
from lightning.fabric import Fabric

import run_with_waic as waic
import waic_utils as waic_utils

# Logging - you can use either wandb or tensorboard
import wandb

import torch
import torch.nn.functional as F

from constants import *
from utils import *
from dataset import *
from rosetta_former.embedding_mlp import * 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('Running your project on WAIC', add_help=False)
    # add your arguments here.
    # for example:
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--save_every', type=int, default=10000,
                        help='save persistent checkpoint every n epochs')

    # learning rate scheduler: warmup followed by cosine decay
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # ----------------------------------------------------------
    # You MUST keep these arguments to work with "run_with_waic":
    # run_with_waic will set this output
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')

    # do not use these
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_id', type=str, default=None)

    # FABRIC ARGS - set by run_with_waic
    parser.add_argument('--accelerator', default='cuda', type=str)
    parser.add_argument('--strategy', default='ddp', type=str)
    parser.add_argument('--precision', default='16-mixed', type=str)
    # ----------------------------------------------------------
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # ----------------------------------------------------------
    # statup code - right after parsing the arguments
    # start lightning fabric - this controls acceleration, precision and parallelism of code
    num_devices, num_nodes = waic.lsf_get_num_devices_and_num_nodes_for_current_job()
    assert num_devices == torch.cuda.device_count(), (f'GPU device count for this node is incorrect. '
                                                      f'Got {num_devices} from LSF '
                                                      f'and {torch.cuda.device_count()} from PyTorch.')
    fabric = Fabric(accelerator=args.accelerator, strategy=args.strategy, precision=args.precision,
                    devices=num_devices, num_nodes=num_nodes)
    fabric.launch()

    # print will only work on rank 0 process
    waic.setup_for_distributed(fabric.is_global_zero)

    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.cuda.set_device(fabric.device)

    seed = fabric.global_rank  # set different seed for each GPU
    fabric.seed_everything(seed, workers=True)

    # run wandb only on the main process
    if fabric.is_global_zero and wandb is not None:
        try:
            wandb.init(project=args.wandb_project, id=args.wandb_id, resume=args.wandb_id, dir=args.output_dir)
            # you might need to add "entity=" argument to the init method.
            wandb.config.update(args, allow_val_change=True)
        except Exception as e:
            print(f'Could not start wandb. got {e}')
            sys.exit(1)

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

    # define your model and optimizer
    with fabric.init_module():
        
        x,y = dataset[0]

        model = EmbeddingMLP(x.shape[1], y.shape[0])        
        
        # effective batch size
        eff_batch_size = args.batch_size * fabric.world_size
        
        # adjust the learning rate according to batch size
        args.lr = args.blr * eff_batch_size / 256    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # fabric the model and optimizer:
    model, optimizer = fabric.setup(model, optimizer)


    # Make your code run with checkpoints!
    to_restore = {"epoch": args.start_epoch}
    waic.restart_from_checkpoint_using_fabric(fabric,
                                              os.path.join(args.output_dir, "checkpoint.pth"),
                                              state_to_load={'model': model, 'optimizer': optimizer,
                                                             'grad_scaler': fabric.strategy.precision},
                                              run_variables=to_restore)
    start_epoch = to_restore["epoch"]

    fabric.print(f"Start training for {args.epochs} epochs start epoch = {start_epoch}")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # training loop
        model.train()

        metric_logger = waic_utils.MetricLogger(delimiter="  ", fabric=fabric)
        metric_logger.add_meter('lr', waic_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20

        for data_iter_step, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            waic_utils.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)
            input, target = batch

            optimizer.zero_grad()
            output = model(input)
            loss = torch.nn.functional.nll_loss(output, target)

            fabric.backward(loss)
            optimizer.step()

            if wandb is not None and fabric.is_global_zero:
                wandb.log({'loss': loss.item()})

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        # save checkpoint after each epoch
        waic.save_using_fabric(fabric, args, epoch + 1, model, optimizer)

        # check LSF time
        got_enough_time = fabric.to_device(torch.ones(1))
        if fabric.global_rank == 0:
            time_left = waic.lsf_get_time_left_for_job()
            print(f'Job remaining time={time_left} [sec]')
            # check remaining time
            time_per_epoch = (time.time() - start_time) / (epoch + 1 - start_epoch)
            print(f'Executing epoch at avg time of {time_per_epoch:.3f} [sec] remaining time {time_left} [sec].')
            if time_left is None or time_per_epoch * 1.2 >= time_left:
                got_enough_time[0] = 0
                print('Not enough time for another epoch. Quitting.')

        # Broadcast got_enough_time to all processes: using fabric causes CuOOM error here.
        tdist.broadcast(got_enough_time, src=0)
        if got_enough_time[0].item() == 0:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if wandb is not None and fabric.is_global_zero:
        wandb.finish()


if __name__ == '__main__':
    if wandb is not None:
        # see: https://github.com/wandb/client/blob/master/docs/dev/wandb-service-user.md
        wandb.require("service")
        wandb.require("core")
        wandb.setup()
    main()
