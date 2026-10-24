# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (lr_schedule_values is not None or wd_schedule_values is not None) and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.amp.autocast('cuda'):
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, header='Test:', return_preds=False):
    """
    Returns a dict of scalar metrics: loss, acc1 (%), and for the 2-class ADNI
    task (AD = 0, NC = 1) the per-class recall/precision, macro-F1 and confusion
    matrix counts. With return_preds=True also returns (probs, targets) tensors
    in data-loader order.
    """
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")

    all_probs = []
    all_targets = []

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.amp.autocast('cuda'):
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, = accuracy(output, target, topk=(1,))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        all_probs.append(output.float().softmax(dim=-1).cpu())
        all_targets.append(target.cpu())

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    if probs.shape[1] == 2:
        stats.update(binary_metrics(probs.argmax(dim=-1), targets))
        print('* Acc@1 {acc:.3f}  loss {loss:.3f}  AD recall {ad:.3f}  NC recall {nc:.3f}  macro-F1 {f1:.3f}  '
              '[TP {tp} FN {fn} FP {fp} TN {tn}]'
              .format(acc=stats['acc1'], loss=stats['loss'], ad=stats['ad_recall'], nc=stats['nc_recall'],
                      f1=stats['macro_f1'], tp=stats['tp'], fn=stats['fn'], fp=stats['fp'], tn=stats['tn']))
    else:
        print('* Acc@1 {acc:.3f}  loss {loss:.3f}'.format(acc=stats['acc1'], loss=stats['loss']))

    if return_preds:
        return stats, probs, targets
    return stats


def binary_metrics(preds, targets, positive=0):
    """Confusion-matrix metrics with AD (class 0) as the positive class."""
    pos_pred, pos_true = preds == positive, targets == positive
    tp = int((pos_pred & pos_true).sum())
    fn = int((~pos_pred & pos_true).sum())
    fp = int((pos_pred & ~pos_true).sum())
    tn = int((~pos_pred & ~pos_true).sum())

    def div(a, b):
        return a / b if b else 0.0

    ad_recall, nc_recall = div(tp, tp + fn), div(tn, tn + fp)      # sensitivity, specificity
    ad_precision, nc_precision = div(tp, tp + fp), div(tn, tn + fn)
    ad_f1 = div(2 * ad_precision * ad_recall, ad_precision + ad_recall)
    nc_f1 = div(2 * nc_precision * nc_recall, nc_precision + nc_recall)
    return {
        'ad_recall': ad_recall, 'nc_recall': nc_recall,
        'ad_precision': ad_precision, 'nc_precision': nc_precision,
        'macro_f1': (ad_f1 + nc_f1) / 2,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
    }