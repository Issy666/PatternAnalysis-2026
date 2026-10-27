# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Training (AdamW, one epoch), evaluation and the AD/NC classification metrics.

import math # math.isfinite: checks the loss isn't inf/NaN
import torch
from timm.utils import accuracy # % correct (used in evaluate)

import utils # MetricLogger (logging), NativeScalerWithGradNormCount (AMP)


def get_parameter_groups(model, weight_decay):
    """Split the weights into 2 groups so weight decay only applies to the ones it should."""
    decay, no_decay = [], []
    for name, param in model.named_parameters(): # every learnable tensor, e.g. "stages.0.0.dwconv.weight"
        # 1-D params (biases, norm weights, layer scale gamma) get no weight decay; only conv/linear weights are decayed.
        # decay pulls weights towards 0, which would undo norm scales (start at 1) and gamma, and biases don't overfit
        (no_decay if len(param.shape) == 1 or name.endswith(".bias") else decay).append(param)
    print(f"Param groups: {len(decay)} tensors with weight decay {weight_decay}, {len(no_decay)} without")
    return [{"weight_decay": weight_decay, "params": decay}, {"weight_decay": 0., "params": no_decay}]


def create_optimizer(args, model):
    # AdamW (the optimiser ConvNeXt was designed and tuned with): adaptive step size per weight + weight decay applied
    # separately from the gradient (the "W"). Weight decay is set per group above, so AdamW's own stays 0.
    # lr here is only the starting value; train_one_epoch sets it every step from the warmup + cosine schedule
    return torch.optim.AdamW(get_parameter_groups(model, args.weight_decay), lr=args.lr, weight_decay=0.)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    start_steps, lr_schedule_values, num_training_steps_per_epoch):
    # trains the model for one pass over the training set; called once per epoch from main.py
    # always uses AMP (mixed precision), so training needs a CUDA GPU
    model.train(True) # training mode: DropPath/dropout on, BatchNorm uses batch statistics (model.eval() in evaluate)
    # running averages of loss/acc etc., printed every 10 batches to train.log
    metric_logger = utils.MetricLogger(delimiter="  ")
    # window_size=1: show the current lr, not an average
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # one batch per loop: samples (64, 3, 256, 256), targets (64,) with 0 = AD, 1 = NC
    # log_every times each batch and prints a progress line every 10 batches
    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step >= num_training_steps_per_epoch: # safety: never step past the end of the lr schedule
            continue
        it = start_steps + step  # global training iteration, e.g. epoch 3 batch 10 -> 3 x 285 + 10 = 865
        for param_group in optimizer.param_groups: # decay and no_decay groups
            param_group["lr"] = lr_schedule_values[it] # this step's lr from the warmup + cosine schedule

        # copy the batch from CPU to GPU; non_blocking + pinned memory overlap the copy with GPU work
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # forward pass: output = logits (64, 2), raw scores for AD and NC (softmax turns them into probabilities)
        # autocast: float16 for convs/matmuls (fast), float32 for softmax/norms/loss; weights stay float32
        with torch.amp.autocast('cuda'):
            output = model(samples)
            loss = criterion(output, targets) # label-smoothing cross-entropy, averaged over the batch

        loss_value = loss.item() # plain Python number for logging; the tensor is kept for backward()

        # inf/NaN loss = training has diverged (lr too high or float16 overflow); stop instead of writing useless checkpoints
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        # backward pass + AdamW step through the AMP loss scaler (see utils.NativeScalerWithGradNormCount)
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters())
        optimizer.zero_grad() # PyTorch adds gradients to the existing ones, so clear them for the next batch

        # wait for the GPU to finish this batch, so log_every's timings are correct
        torch.cuda.synchronize()

        # training accuracy of this batch: argmax of the logits = predicted class (0 = AD, 1 = NC)
        # lower than eval accuracy on the same images: DropPath + augmentation are on, and weights change during the epoch
        class_acc = (output.max(-1)[-1] == targets).float().mean()
        metric_logger.update(loss=loss_value, class_acc=class_acc, lr=optimizer.param_groups[0]["lr"],
                             grad_norm=grad_norm)

    print("Averaged stats:", metric_logger)
    # epoch averages; main.py writes them to log.txt as train_loss, train_class_acc, train_lr, ... (used by plot_log.py)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad() # no gradient tracking: less memory and faster, as nothing is learned here
def evaluate(data_loader, model, device, use_amp=False, header='Test:', return_preds=False):
    """
    Returns a dict of scalar metrics: loss, acc1 (%), and for the 2-class ADNI
    task (AD = 0, NC = 1) the per-class recall/precision, macro-F1 and confusion
    matrix counts. With return_preds=True also returns (probs, targets) tensors
    in data-loader order.
    """
    # plain cross-entropy (no label smoothing, unlike training), so val/test loss can go below train loss's ~0.20 floor;
    # it scores the model's real probabilities, which is what calibration needs
    criterion = torch.nn.CrossEntropyLoss()
    # use_amp is kept here (unlike training) so predict.py can evaluate on a CPU with use_amp=False

    metric_logger = utils.MetricLogger(delimiter="  ")

    all_probs = []
    all_targets = []

    # switch to evaluation mode: DropPath off, every block runs
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header): # val/test loaders: not shuffled, last batch kept (smaller)
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

        acc1, = accuracy(output, target, topk=(1,)) # % of slices whose highest score is the true class

        # weight by batch size so the smaller last batch doesn't count as much as a full one
        # (the loss used to be added with weight 1, slightly skewing val/test loss)
        batch_size = images.shape[0]
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        # softmax -> probabilities summing to 1 (the confidence scores); float32 for precision near 0/1; in loader order,
        # which matches dataset.samples only because val/test loaders don't shuffle
        all_probs.append(output.float().softmax(dim=-1).cpu())
        all_targets.append(target.cpu())

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    if probs.shape[1] == 2:
        # argmax = predict whichever class has prob > 0.5 (the decision threshold that could later be moved to favour AD)
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
    # AD = 0 because ImageFolder numbers class folders alphabetically (AD, NC); slice-level, 0.5 threshold (argmax)
    pos_pred, pos_true = preds == positive, targets == positive # predicted AD / really AD
    tp = int((pos_pred & pos_true).sum())   # predicted AD, really AD (caught)
    fn = int((~pos_pred & pos_true).sum())  # predicted NC, really AD (missed Alzheimer's - the worst error here)
    fp = int((pos_pred & ~pos_true).sum())  # predicted AD, really NC (false alarm)
    tn = int((~pos_pred & ~pos_true).sum()) # predicted NC, really NC (correctly cleared)

    def div(a, b):
        return a / b if b else 0.0 # 0 instead of crashing if a class is never predicted/present

    ad_recall, nc_recall = div(tp, tp + fn), div(tn, tn + fp)      # sensitivity, specificity
    ad_precision, nc_precision = div(tp, tp + fp), div(tn, tn + fn) # when we say AD/NC, how often we're right
    # F1: harmonic mean of precision and recall (high only if both are); macro-F1: mean of the 2 classes' F1 (equal weight)
    ad_f1 = div(2 * ad_precision * ad_recall, ad_precision + ad_recall)
    nc_f1 = div(2 * nc_precision * nc_recall, nc_precision + nc_recall)
    return {
        'ad_recall': ad_recall, 'nc_recall': nc_recall,
        'ad_precision': ad_precision, 'nc_precision': nc_precision,
        'macro_f1': (ad_f1 + nc_f1) / 2,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
    }