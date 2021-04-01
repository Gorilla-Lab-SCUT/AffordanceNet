# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from os.path import join as opj
import torch
import torch.nn as nn
import numpy as np
from gorilla.config import Config
import models
import loss
from utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--checkpoint", type=str,
                        help="the path to checkpoints")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Number of gpus to use"
    )
    parser.add_argument(
        "--with_loss", help="show the test loss", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir != None:
        cfg.work_dir = args.work_dir
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
    print(cfg)
    logger = IOStream(opj(cfg.work_dir, 'run.log'))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))
    logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg.gpu))
    if cfg.get('seed', None) != None:
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
    if cfg.get('with_loss', None) == None:
        cfg.update({"with_loss": args.with_loss})
    model = build_model(cfg).cuda()
    if num_gpu > 1:
        model = nn.DataParallel(model)
        logger.cprint('Use DataParallel!')
    if args.checkpoint != None:
        print("Loading checkpoint....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            model.load_state_dict(torch.load(args.checkpoint))
        elif exten == '.pth':
            check = torch.load(args.checkpoint)
            model.load_state_dict(check['model_state_dict'])
    else:
        raise ValueError("Must specify a checkpoint path!")
    dataset_dict = build_dataset(cfg, test=True)
    loader_dict = build_loader(cfg, dataset_dict)
    train_loss = build_loss(cfg)
    optim_dict = build_optimizer(cfg, model)
    training = dict(
        model=model,
        dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        loss=train_loss,
        optim_dict=optim_dict,
        logger=logger
    )

    task_trainer = Trainer(cfg, training)
    task_trainer.run()
