import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


def cal_sam(img_true, img_pred):
    eps = 1e-6
    inner_prod = torch.sum(img_true * img_pred, 1, keepdim=True)
    norm_true = torch.norm(img_true, p=2, dim=1, keepdim=True)
    norm_pred = torch.norm(img_pred, p=2, dim=1, keepdim=True)
    divisor = norm_true * norm_pred
    mask = torch.eq(divisor, 0)
    divisor = divisor + mask.float() * eps
    cos_theta = torch.sum(inner_prod / divisor, 1).clamp(-1 + eps, 1 - eps)
    sam = torch.acos(cos_theta)
    sam = torch.mean(sam) / np.pi
    return sam


@LOSS_REGISTRY.register()
class SAMLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SAMLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * cal_sam(pred, target)


def cal_gradient_c(x):
    if x.size(1) > 1:
        return x[:, 1:, :-1, :-1] - x[:, :-1, :-1, :-1]
    return torch.zeros_like(x[:, :, :-1, :-1])


def cal_gradient_x(x):
    if x.size(2) > 1:
        return x[:, :-1, 1:, :-1] - x[:, :-1, :-1, :-1]
    return torch.zeros_like(x[:, :-1, :, :-1])


def cal_gradient_y(x):
    if x.size(3) > 1:
        return x[:, :-1, :-1, 1:] - x[:, :-1, :-1, :-1]
    return torch.zeros_like(x[:, :-1, :-1, :])


def cal_gradient(inp):
    c_grad = cal_gradient_c(inp)
    x_grad = cal_gradient_x(inp)
    y_grad = cal_gradient_y(inp)
    grad = torch.sqrt(torch.pow(x_grad, 2) + torch.pow(y_grad, 2) + torch.pow(c_grad, 2) + 1e-6)
    return grad


@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_grad = cal_gradient(pred)
        target_grad = cal_gradient(target)
        return self.loss_weight * self.criterion(pred_grad, target_grad)
