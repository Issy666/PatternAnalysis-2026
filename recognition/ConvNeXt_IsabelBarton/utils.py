"""
Training helpers: console logging (SmoothedValue, MetricLogger), the AMP loss scaler, the warmup + cosine
learning-rate schedule, and checkpoint saving / resuming / loading.

Reference: everything except load_trained_model is adapted from the official ConvNeXt implementation [2]
(Copyright (c) Meta Platforms, Inc. and affiliates, MIT licence):
https://github.com/facebookresearch/ConvNeXt/blob/main/utils.py
Parts of the original not needed here (e.g. distributed training) were removed.
"""

import glob
import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np

from pathlib import Path

import torch

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Add a value; n = how many items it averages over (e.g. the batch size), for a weighted global average."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """A set of named SmoothedValues (loss, accuracy, lr, ...) that prints progress lines during an epoch."""
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Add one value per named metric, e.g. update(loss=0.4, class_acc=0.8); None values are skipped."""
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        # lets a meter be read as an attribute, e.g. metric_logger.loss
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Yield the items of iterable (e.g. a DataLoader) and print progress, ETA, timings and GPU memory every
        print_freq items and at the end."""
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class NativeScalerWithGradNormCount:
    """AMP (mixed precision) loss scaler: does the backward pass and optimiser step, and returns the gradient norm."""
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, parameters):
        # scaled backward pass, then unscale the gradients in place so their norm can be measured (logged as grad_norm,
        # e.g. to spot a too-high lr), then the optimiser step (skipped if any gradient is inf/NaN) and scale update
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(optimizer)
        norm = get_grad_norm_(parameters)
        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters) -> torch.Tensor:
    """L2 norm of all gradients together (logged as grad_norm to spot exploding gradients)."""
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.)
    return torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """One value per optimiser step (epochs x niter_per_ep): linear warmup start_warmup_value -> base_value, then
    cosine decay base_value -> final_value. Used for the learning rate."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_model(args, epoch, model, optimizer, loss_scaler):
    """Save checkpoint-<epoch>.pth (or checkpoint-best.pth) with everything needed to resume, plus the training
    arguments, so the model and its data can be rebuilt later (load_trained_model). Keeps the last --save_ckpt_num
    numbered checkpoints."""
    output_dir = Path(args.output_dir)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }, output_dir / ('checkpoint-%s.pth' % epoch))

    if isinstance(epoch, int):
        old_ckpt = output_dir / ('checkpoint-%s.pth' % (epoch - args.save_ckpt_num))
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model, optimizer, loss_scaler):
    """
    Resume an interrupted run: --resume <path>, or else the highest-numbered checkpoint-<epoch>.pth in output_dir
    (checkpoint-best.pth is skipped as "best" isn't a number). Restores the weights, AdamW state, AMP scale and
    args.start_epoch, so training and the lr schedule continue where they stopped.
    """
    if not args.resume:
        epochs = [int(t) for t in (c.split('-')[-1].split('.')[0] for c in
                                   glob.glob(os.path.join(args.output_dir, 'checkpoint-*.pth'))) if t.isdigit()]
        if epochs:
            args.resume = os.path.join(args.output_dir, 'checkpoint-%d.pth' % max(epochs))
        print("Auto resume checkpoint: %s" % args.resume)
    if not args.resume:
        return

    # checkpoints also store args/optimizer state, which PyTorch >= 2.6 refuses to unpickle by default
    # (only safe for checkpoints we saved ourselves: loading an untrusted .pth this way can run arbitrary code)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    print("Resume checkpoint %s" % args.resume)
    if not isinstance(checkpoint['epoch'], int):
        raise SystemExit(f"Can't resume training from {args.resume} (epoch '{checkpoint['epoch']}'); give a "
                         f"numbered checkpoint. To test a checkpoint use predict.py --resume.")
    optimizer.load_state_dict(checkpoint['optimizer']) # AdamW's running averages, so step sizes don't restart
    args.start_epoch = checkpoint['epoch'] + 1 # continue at the next epoch, so the lr schedule picks up where it stopped
    loss_scaler.load_state_dict(checkpoint['scaler']) # AMP scale factor, so it doesn't re-search for a safe value
    print("With optim & sched!")


def load_trained_model(checkpoint_path, device):
    """
    Rebuild a trained model from its checkpoint for evaluation (predict.py, calibrate.py, visualise.py).
    Written for this project (not in the original utils.py).
    The checkpoint stores the training arguments, so the model and its data (paths, input size, 2.5D input) are
    rebuilt exactly as trained. Returns (model in eval mode on `device`, training args).
    """
    from modules import build_model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    train_args = checkpoint['args']
    model = build_model(train_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    print(f"Loaded {train_args.model} from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return model, train_args
