import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
from dataset import *
from models import *
import loss
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

model_pool = {
    'dgcnn': DGCNN_Estimation,
    'pn2': PointNet_Estimation
}

optim_pool = {
    'sgd': SGD,
    'adam': Adam
}

init_pool = {
    'pn2_init': weights_init_pn2
}

scheduler_pool = {
    'step': StepLR,
    'cos': CosineAnnealingLR,
    'lr_lambda': LambdaLR
}


def build_model(cfg):
    if hasattr(cfg, 'model'):
        model_info = cfg.model
        weights_init = model_info.get('weights_init', None)
        model_name = model_info.type
        model_cls = model_pool[model_name]
        num_category = len(cfg.data.category)
        model = model_cls(model_info, num_category)
        if weights_init != None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)
        return model
    else:
        raise ValueError("Configuration does not have model config!")


def build_dataset(cfg, test=False):
    if hasattr(cfg, 'data'):
        data_info = cfg.data
        data_root = data_info.data_root
        afford_cat = data_info.category
        if_partial = cfg.training_cfg.get('partial', False)
        if_rotate = cfg.training_cfg.get('rotate', 'None')
        if_semi = cfg.training_cfg.get('semi', False)
        if_transform = True if if_semi else False
        if test:
            test_set = AffordNetDataset(
                data_root, 'test', partial=if_partial, rotate=if_rotate, semi=False)
            dataset_dict = dict()
            dataset_dict.update({"test_set": test_set})
            return dataset_dict
        train_set = AffordNetDataset(
            data_root, 'train', partial=if_partial, rotate=if_rotate, semi=if_semi)
        val_set = AffordNetDataset(
            data_root, 'val', partial=if_partial, rotate=if_rotate, semi=False)
        dataset_dict = dict(
            train_set=train_set,
            val_set=val_set
        )
        if if_semi:
            train_unlabel_set = AffordNetDataset_Unlabel(data_root)
            dataset_dict.update({"train_unlabel_set": train_unlabel_set})
        return dataset_dict
    else:
        raise ValueError("Configuration does not have data config!")


def build_loader(cfg, dataset_dict):
    if "test_set" in dataset_dict:
        test_set = dataset_dict["test_set"]
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
        loader_dict = dict()
        loader_dict.update({"test_loader": test_loader})
        return loader_dict
    train_set = dataset_dict["train_set"]
    val_set = dataset_dict["val_set"]
    batch_size_factor = 1 if not cfg.training_cfg.get('semi', False) else 2
    train_loader = DataLoader(train_set, batch_size=cfg.training_cfg.batch_size // batch_size_factor,
                              shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=8, drop_last=False)
    loader_dict = dict(
        train_loader=train_loader,
        val_loader=val_loader
    )
    if "train_unlabel_set" in dataset_dict:
        train_unlabel_set = dataset_dict["train_unlabel_set"]
        train_unlabel_loader = DataLoader(train_unlabel_set, num_workers=4,
                                          batch_size=cfg.training_cfg.batch_size//batch_size_factor, shuffle=True, drop_last=True)
        loader_dict.update({"train_unlabel_loader": train_unlabel_loader})

    return loader_dict


def build_loss(cfg):
    if cfg.training_cfg.get("semi", False):
        loss_fn = loss.VATLoss(warmup_epoch=0)
        # loss_fn = loss.SemiLoss()
    else:
        loss_fn = loss.EstimationLoss()
    return loss_fn


def build_optimizer(cfg, model):
    optim_info = cfg.optimizer
    optim_type = optim_info.type
    optim_info.pop("type")
    optim_cls = optim_pool[optim_type]
    optimizer = optim_cls(model.parameters(), **optim_info)
    scheduler_info = cfg.scheduler
    scheduler_name = scheduler_info.type
    scheduler_info.pop('type')
    scheduler_cls = scheduler_pool[scheduler_name]
    scheduler = scheduler_cls(optimizer, **scheduler_info)
    optim_dict = dict(
        scheduler=scheduler,
        optimizer=optimizer
    )
    return optim_dict
