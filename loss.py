# Copyright (c) Gorilla-Lab. All rights reserved.
from torch import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import contextlib


class EstimationLoss(nn.Module):
    def __init__(self):
        super(EstimationLoss, self).__init__()
        self.gamma = 0
        self.alpha = 0

    def forward(self, pred, target):
        temp1 = -torch.mul(pred**self.gamma,
                           torch.mul(1-target, torch.log(1-pred+1e-6)))
        temp2 = -torch.mul((1-pred)**self.gamma,
                           torch.mul(target, torch.log(pred+1e-6)))
        temp = temp1+temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.5-dice_positive-dice_negative, 0)

        DICELoss = torch.sum(temp3)

        return CELoss+1.0*DICELoss


class SemiLoss(nn.Module):
    def __init__(self):
        super(SemiLoss, self).__init__()

    def MSELoss(self, input1, input2):
        return torch.sum(torch.mean((input1-input2)**2, (0, 1)))

    def forward(self, labeled_pred, target, unlabeled_pred, validate=False):
        if not validate:
            bs = int(labeled_pred.size(0)) // 2
            pred = labeled_pred[0:bs, :, :]
        else:
            pred = labeled_pred
        temp1 = -torch.mul(1-target, torch.log(1-pred+1e-6))
        temp2 = -torch.mul(target, torch.log(pred+1e-6))
        temp = temp1+temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.0-dice_positive-dice_negative, 0)

        DICELoss = torch.sum(temp3)

        if validate:
            return CELoss+DICELoss
        else:
            pred_noise = labeled_pred[bs:, :, :]
            ul_pred1 = unlabeled_pred[0:bs, :, :]
            ul_pred2 = unlabeled_pred[bs:, :, :]
            consistentLoss = self.MSELoss(input1=torch.cat([pred.detach(), ul_pred1.detach(
            )], dim=0), input2=torch.cat([pred_noise, ul_pred2], dim=0))

            return CELoss+DICELoss+consistentLoss


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, warmup_epoch=0):
        super(VATLoss, self).__init__()
        self.xi = 1e-6
        self.eps = 2.0
        self.ip = 1
        self.warmup_epoch = warmup_epoch

    def MSELoss(self, input1, input2):
        return torch.sum(torch.mean((input1-input2)**2, (0, 1)))

    def forward(self, model, labeled_data, unlabeled_data, labeled_pred, target, unlabeled_pred, epoch, validate=False):
        pred = labeled_pred
        temp1 = -torch.mul(1-target, torch.log(1-pred+1e-6))
        temp2 = -torch.mul(target, torch.log(pred+1e-6))
        temp = temp1+temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.5-dice_positive-dice_negative, 0)

        DICELoss = torch.sum(temp3)

        if validate:
            return CELoss+DICELoss
        elif epoch < self.warmup_epoch:
            return CELoss+DICELoss
        else:
            d_labeled = torch.rand(labeled_data.shape).sub(0.5).cuda()
            d_labeled = _l2_normalize(d_labeled)
            d_unlabeled = torch.rand(unlabeled_data.shape).sub(0.5).cuda()
            d_unlabeled = _l2_normalize(d_unlabeled)
            with _disable_tracking_bn_stats(model):
                for _ in range(self.ip):
                    d_labeled.requires_grad_()
                    d_unlabeled.requires_grad_()
                    train_labeled_data_view2 = labeled_data + self.xi*d_labeled
                    train_unlabeled_data_view2 = unlabeled_data + self.xi*d_unlabeled
                    pred_label_view2 = model(
                        train_labeled_data_view2).permute(0, 2, 1)
                    pred_unlabel_view2 = model(
                        train_unlabeled_data_view2).permute(0, 2, 1)
                    pred_label_view2 = torch.sigmoid(
                        pred_label_view2).contiguous()
                    pred_unlabel_view2 = torch.sigmoid(
                        pred_unlabel_view2).contiguous()
                    consist_loss = self.MSELoss(input1=torch.cat([labeled_pred.detach(), unlabeled_pred.detach()], dim=0),
                                                input2=torch.cat([pred_label_view2, pred_unlabel_view2], dim=0))

                    consist_loss.backward()
                    d_labeled = _l2_normalize(d_labeled.grad)
                    d_unlabeled = _l2_normalize(d_unlabeled.grad)
                    model.zero_grad()
                r_adv_label = d_labeled * self.eps
                r_adv_unlabel = d_unlabeled * self.eps
                train_labeled_data_view2 = labeled_data + r_adv_label
                train_unlabeled_data_view2 = unlabeled_data + r_adv_unlabel
                pred_label_view2 = model(
                    train_labeled_data_view2).permute(0, 2, 1)
                pred_unlabel_view2 = model(
                    train_unlabeled_data_view2).permute(0, 2, 1)
                pred_label_view2 = torch.sigmoid(
                    pred_label_view2).contiguous()
                pred_unlabel_view2 = torch.sigmoid(
                    pred_unlabel_view2).contiguous()
                consist_loss = self.MSELoss(input1=torch.cat([labeled_pred.detach(), unlabeled_pred.detach()], dim=0),
                                            input2=torch.cat([pred_label_view2, pred_unlabel_view2], dim=0))
            return CELoss+DICELoss+consist_loss
