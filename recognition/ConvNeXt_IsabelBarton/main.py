
import argparse
import csv
import datetime
import json
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

from dataset import build_dataset
from optim_factory import create_optimizer
from train import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import modules
import normalise
import utils


def str2bool(v):
    # argparse's type=bool treats any non-empty string (including "False") as True
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training + evaluation script for MRI AD or CN classification', add_help=False)
    
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=150, type=int)

    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str,
                        help='Name of model to train (a model function in modules.py)')

    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='Drop path (stochastic depth) rate (default: 0.1)')
                        
    parser.add_argument('--input_size', default=256, type=int,
                        help='image input size')

    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help='Initial value of the per-channel layer scale (gamma); 0 disables it (default: 1e-6)')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str,
                        help='Optimizer (default: "adamw"')

    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer Epsilon (default: 1e-8)')

    parser.add_argument('--opt_betas', default=None, type=float, nargs='+',
                        help='Optimizer Betas (default: None, use opt default)')
    
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3,
                        help='learning rate (default: 4e-3)')

    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--warmup_steps', type=int, default=-1,
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--aa', type=str, default='',
                        help='Use AutoAugment policy, e.g. "rand-m9-mstd0.5-inc1". (default: "" = off)')
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT',
                        help='Color jitter factor (default: 0 = off)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bilinear',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')

    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')

    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')

    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')

    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')

    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/isabel/Desktop/data/ADNI/AD_NC/train', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='/home/isabel/Desktop/data/ADNI/AD_NC/validation', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='/home/isabel/Desktop/data/ADNI/AD_NC/test', type=str,
                        help='dataset path for evaluation')

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--data_set', default='ADNI', choices=['CIFAR', 'ADNI'],
                        type=str, help='dataset path')
    parser.add_argument('--stats_dir', default=normalise.DEFAULT_CACHE_DIR, type=str,
                        help='where the training-set mean/std JSON is cached')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--output_dir', default='/home/isabel/Desktop/results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/home/isabel/Desktop/log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use_amp', type=str2bool, default=True,
                        help='Use PyTorch automatic mixed precision (much faster on an RTX GPU)')

    # Checkpointing / resuming
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int,
                        help='save a checkpoint every N epochs')
    parser.add_argument('--save_ckpt_num', default=3, type=int,
                        help='number of recent epoch checkpoints to keep (checkpoint-best.pth is always kept)')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True,
                        help='resume from the latest checkpoint in output_dir if one exists')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')

    return parser

def build_model(args):
    # models are built straight from modules.py and trained from scratch (random initial weights)
    if not hasattr(modules, args.model):
        raise ValueError(f"Unknown model '{args.model}'. Available in modules.py: convnext_tiny")
    return getattr(modules, args.model)(
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path, # reduce overfitting by dropping blocks in the model
        layer_scale_init_value=args.layer_scale_init_value, #add a scalar to each block per channel to deterine the influence of each blocks
    )


def run_test(model, data_loader_test, dataset_test, device, args, checkpoint_name, extra=None):
    """Evaluate on the held-out test split and save metrics + per-slice predictions to output_dir."""
    print(f"Evaluating {checkpoint_name} on the test set ({len(dataset_test)} images)")
    test_stats, probs, targets = evaluate(data_loader_test, model, device, use_amp=args.use_amp,
                                          header='Test:', return_preds=True)
    print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.2f}%")

    if args.output_dir:
        results = {'checkpoint': checkpoint_name, 'num_test_images': len(dataset_test),
                   **{f'test_{k}': v for k, v in test_stats.items()}, **(extra or {})}
        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # per-slice softmax outputs, for calibration / reliability analysis later
        idx_to_class = {v: k for k, v in dataset_test.class_to_idx.items()}
        with open(os.path.join(args.output_dir, 'test_predictions.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'label', 'pred'] + [f'prob_{idx_to_class[i]}' for i in range(probs.shape[1])])
            for (path, _), target, p in zip(dataset_test.samples, targets.tolist(), probs.tolist()):
                pred = max(range(len(p)), key=p.__getitem__)
                writer.writerow([path, idx_to_class[target], idx_to_class[pred]] + [f'{x:.6f}' for x in p])
        print(f"Saved test_results.json and test_predictions.csv to {args.output_dir}")
    return test_stats


def main(args):

    print(args)

    #device to run on
    device = torch.device(args.device)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #enables the NVIDIA cuDNN auto-tuner to test multiple convolution algorithms and select the fastest one for your GPU  Works for inputs that are the same size
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(split='train', args=args)
    dataset_val, _ = build_dataset(split='val', args=args)
    dataset_test, _ = build_dataset(split='test', args=args)

    # LOG DIRECTORY
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # shuffle=True reshuffles the training data every epoch
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, shuffle=False,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    # test is evaluated in order, so predictions line up with dataset_test.samples
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # removed for ADNI as the entire image is bnot relevant to AD or NC classification only certain section so mixing classes has opposite of desired effect
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    #build the CONVNEXT
    model = build_model(args)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # create the EMA copy after the model has been moved to the device
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    optimizer = create_optimizer(args, model, skip_list=None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        run_test(model, data_loader_test, dataset_test, device, args, checkpoint_name=args.resume or 'initial weights')
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, header='Val:')
            print(f"Accuracy of the model on the {len(dataset_val)} validation images: {val_stats['acc1']:.1f}%")
            if max_accuracy < val_stats["acc1"]:
                max_accuracy = val_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max validation accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(val_acc1=val_stats['acc1'], head="perf", step=epoch)
                log_writer.update(val_loss=val_stats['loss'], head="perf", step=epoch)
                if 'ad_recall' in val_stats:
                    log_writer.update(val_ad_recall=val_stats['ad_recall'], head="perf", step=epoch)
                    log_writer.update(val_nc_recall=val_stats['nc_recall'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                val_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp, header='Val (EMA):')
                print(f"Accuracy of the model EMA on {len(dataset_val)} validation images: {val_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < val_stats_ema["acc1"]:
                    max_accuracy_ema = val_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(val_acc1_ema=val_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'val_{k}_ema': v for k, v in val_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # final, one-off evaluation on the held-out test split using the best validation checkpoint
    checkpoint_name = 'final epoch weights'
    best_path = Path(args.output_dir) / 'checkpoint-best.pth' if args.output_dir else None
    if best_path is not None and best_path.exists():
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        checkpoint_name = str(best_path)
    extra = {
        'best_val_acc1': max_accuracy,
        'epochs_trained_this_run': args.epochs - args.start_epoch,
        'training_time_this_run': total_time_str,
        'max_gpu_mem_mb': torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else None,
        'n_parameters': n_parameters,
    }
    run_test(model, data_loader_test, dataset_test, device, args, checkpoint_name, extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training + evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)