"""
Training utilities for semantic segmentation models.
This module contains loss functions, weight initialization, and learning rate schedulers.
"""

import math
from functools import partial
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage  import distance_transform_edt


def IoU_loss(inputs, target, smooth=1e-5):
    """
    Intersection over Union (IoU) loss for semantic segmentation.
    
    Args:
        inputs (torch.Tensor): Model predictions (logits)
        target (torch.Tensor): Ground truth labels
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: IoU loss value
    """
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    # Resize inputs if dimensions don't match
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Convert to probabilities and reshape [N, H, W, C]
    temp_inputs = torch.softmax(inputs.permute(0, 2, 3, 1).contiguous().view(n, -1, c), dim=-1)
    temp_target = target.view(n, -1, ct)  # [N, H*W, C]

    # Calculate intersection and union (considering foreground classes only)
    intersection = torch.sum(temp_target[..., :-1] * temp_inputs, dim=[0, 1])
    union = torch.sum(temp_target[..., :-1] + temp_inputs, dim=[0, 1]) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return 1 - torch.mean(iou)


class Boundary_Loss(nn.Module):
    """
    Boundary-aware loss function for semantic segmentation.
    This loss emphasizes boundary regions to improve edge quality.
    """
    def __init__(self, alpha=1.0, gamma=2.0, boundary_weight=5.0):
        """
        Initialize Boundary Loss.
        
        Args:
            alpha (float): Balancing factor for positive/negative samples (default 1.0)
            gamma (float): Focusing parameter for hard examples (default 2.0)
            boundary_weight (float): Weight factor for boundary pixels (default 5.0)
        """
        super(Boundary_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight

    def forward(self, pred, target, boundary_mask):
        """
        Compute Boundary Loss.
        
        Args:
            pred (torch.Tensor): Predicted probability maps (batch_size, num_classes, H, W)
            target (torch.Tensor): Ground truth labels (batch_size, H, W)
            boundary_mask (torch.Tensor): Boundary mask (batch_size, H, W), 1 for boundary, 0 for non-boundary
            
        Returns:
            torch.Tensor: Computed boundary loss value
        """
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')  # Per-pixel loss

        # Calculate Focal Loss weighting
        p_t = torch.exp(-ce_loss)  # Predicted class probability
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss  # Focal Loss formula

        # Apply boundary weighting - higher loss for boundary pixels
        weighted_loss = focal_loss * (1 + self.boundary_weight * boundary_mask)

        # Return mean of weighted loss
        return weighted_loss.mean()


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    """
    Cross-Entropy Loss for semantic segmentation.
    
    Args:
        inputs (torch.Tensor): Model predictions
        target (torch.Tensor): Ground truth labels
        cls_weights (torch.Tensor): Class weights for imbalanced datasets
        num_classes (int): Number of segmentation classes
        
    Returns:
        torch.Tensor: Cross-entropy loss value
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    
    # Resize inputs if dimensions don't match
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Flatten predictions and targets for loss calculation
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    """
    Focal Loss for handling class imbalance in semantic segmentation.
    
    Args:
        inputs (torch.Tensor): Model predictions
        target (torch.Tensor): Ground truth labels
        cls_weights (torch.Tensor): Class weights
        num_classes (int): Number of classes
        alpha (float): Balancing parameter
        gamma (float): Focusing parameter
        
    Returns:
        torch.Tensor: Focal loss value
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    
    # Resize inputs if dimensions don't match
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # Flatten predictions and targets
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # Calculate Focal Loss components
    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

'权重初始化策略'
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

'学习率调度策略'
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

'设置学习率'
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr