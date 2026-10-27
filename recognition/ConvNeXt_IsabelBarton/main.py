
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

from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEma

from dataset import build_dataset
from optim_factory import create_optimizer
from train import train_one_epoch, evaluate
from predict import run_test, patient_predictions, classification_metrics
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
    parser.add_argument('--epochs', default=60, type=int) # 100-epoch run peaked on val at 55, so cosine decay now finishes around there

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

    # None = each model's own default: 0.4 for the BatchNorm (*_bn) models and simple_cnn, 0 for the LayerNorm ConvNeXt
    parser.add_argument('--head_drop', type=float, default=None,
                        help='Dropout before the classifier (default: model default, 0.4 for *_bn / simple_cnn, '
                             '0 for LayerNorm convnext_*)')

    # EMA related parameters (off by default)
    parser.add_argument('--model_ema', type=str2bool, default=False,
                        help='Keep an exponential moving average copy of the model weights')
    # averages over ~1/(1 - decay) steps. paper's 0.9999 = 10,000 steps = ~32 of 300 ImageNet epochs (~11% of training)
    # ADNI has ~285 steps/epoch, so 0.9999 would average over ~35 of 100 epochs and still hold ~6% of the random
    # init weights at the end; 0.9997 averages over ~3,333 steps (~12 of 100 epochs), the same proportion as the paper
    parser.add_argument('--model_ema_decay', type=float, default=0.9997,
                        help='EMA decay: ema = decay * ema + (1 - decay) * weights (default: 0.9997)')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False,
                        help='Store the EMA copy on the CPU to save GPU memory')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters (optimiser is always AdamW, see optim_factory.py)
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer Epsilon (default: 1e-8)')

    parser.add_argument('--opt_betas', default=None, type=float, nargs='+',
                        help='Optimizer Betas (default: None, use opt default)')
    
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    # paper's 4e-3 is for batch 4096; square-root scaled to batch 64: 4e-3 * sqrt(64/4096) = 5e-4 (not yet tuned on ADNI)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')

    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    
    parser.add_argument('--warmup_epochs', type=int, default=5, # ~8% of 60 epochs, close to the paper's 20 of 300
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--warmup_steps', type=int, default=-1,
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    # small random affine only (no RandomResizedCrop / flip / RandAugment): see normalise.build_transform
    parser.add_argument('--affine_degrees', type=float, default=10,
                        help='Random rotation range in degrees, +-value (default: 10)')
    parser.add_argument('--affine_translate', type=float, default=0.05,
                        help='Random shift as a fraction of image size, +-value (default: 0.05)')
    parser.add_argument('--affine_scale', type=float, default=0.05,
                        help='Random zoom range, 1 +- value (default: 0.05)')
    parser.add_argument('--brightness', type=float, default=0.0,
                        help='Random brightness factor range, 1 +- value (default: 0, off)')
    parser.add_argument('--contrast', type=float, default=0.0,
                        help='Random contrast factor range, 1 +- value (default: 0, off)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic', 'nearest'],
                        help='Interpolation for the training resize / affine (default: bilinear)')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/isabel/Desktop/data/ADNI/AD_NC/train', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='/home/isabel/Desktop/data/ADNI/AD_NC/validation', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='/home/isabel/Desktop/data/ADNI/AD_NC/test', type=str,
                        help='dataset path for evaluation')

    # 2.5D input: channels = (slice below, slice, slice above) from the same scan instead of 3 greyscale copies
    parser.add_argument('--neighbour_slices', type=str2bool, default=False,
                        help='Use neighbouring slices of the same scan as the 3 input channels (default: False)')
    parser.add_argument('--neighbour_step', type=int, default=1,
                        help='Slice offset of the neighbours, e.g. 2 = 2 slices either side (default: 1)')

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
    # which validation score picks checkpoint-best.pth. slice accuracy hides the AD/NC split of errors and jumps
    # around by luck; patient macro-F1 weights AD and NC equally, ties go to the lower validation loss
    parser.add_argument('--best_metric', default='acc1', choices=['acc1', 'patient_macro_f1'],
                        help='Validation metric used to choose checkpoint-best.pth (default: acc1)')

    return parser

def build_model(args):
    # models are built straight from modules.py and trained from scratch (random initial weights)
    available = sorted(n for n in dir(modules)
                       if (n.startswith('convnext_') or n == 'simple_cnn') and callable(getattr(modules, n)))
    if args.model not in available:
        raise ValueError(f"Unknown model '{args.model}'. Available in modules.py: {', '.join(available)}")
    kwargs = dict(
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path, # reduce overfitting by dropping blocks in the model
        layer_scale_init_value=args.layer_scale_init_value, #add a scalar to each block per channel to deterine the influence of each blocks
    )
    if getattr(args, 'head_drop', None) is not None:
        kwargs['head_drop_rate'] = args.head_drop # otherwise each model keeps its own default
    return getattr(modules, args.model)(**kwargs)


def val_patient_metrics(dataset, probs):
    # probs are in dataset.samples order (val loader isn't shuffled); column 0 = P(AD)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    slices = [(path, idx_to_class[target], p[0]) for (path, target), p in zip(dataset.samples, probs.tolist())]
    return classification_metrics(patient_predictions(slices))





def main(args):

    print(args)

    #device to run on
    device = torch.device(args.device)
    # training always uses AMP (autocast/GradScaler on 'cuda'), so it needs a CUDA GPU
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise SystemExit("main.py needs a CUDA GPU (training always uses mixed precision); "
                         "use predict.py --resume <checkpoint> to evaluate a trained model on a CPU")

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

    # mixup/cutmix from the paper removed for ADNI: they blend or paste two brains together, so the label no longer
    # matches the anatomy that defines AD vs NC (see README "Why mixup and CutMix are not used")

    #build the CONVNEXT
    model = build_model(args)
    model.to(device)

    # EMA: a second copy of the model whose weights slowly follow the trained weights,
    # updated after every optimizer step in train.py; smooths out step-to-step noise
    model_ema = None
    if args.model_ema:
        # create the EMA copy after the model has been moved to the device
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay, # ema = decay * ema + (1 - decay) * current weights
            device='cpu' if args.model_ema_force_cpu else '', # '' = same device as the model
            resume='') # don't load from a file; resumed runs restore EMA from the checkpoint instead
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    #No. OF TRAINABLE PARAMATERS IN THE ENTIRE MODEL
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # RUn details
    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    # effective batch size: gradients from update_freq batches are summed before one optimizer step, e.g. 32 x 2 = 64
    total_batch_size = args.batch_size * args.update_freq
    # optimizer steps per epoch (rounded down), e.g. 18,300 // 64 = 285; the LR/WD schedules are built per step from this
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    #choose optimiser
    optimizer = create_optimizer(args, model)

    # AMP loss scaler: float16 rounds tiny gradients to 0, so the loss is multiplied by a large factor before backward(),
    # gradients are divided back before the optimiser step, and steps with inf/NaN gradients are skipped (factor then halved)
    loss_scaler = NativeScaler()

    # one lr per optimizer step for the whole run 
    # linear warmup 0 -> lr (random init gives large, unreliable early gradients), then cosine decay lr -> min_lr
    # (big exploring steps early, small settling steps at the end)
    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    # weight decay schedule built the same way (no warmup); weight_decay_end None = constant at weight_decay (0.05)
    # each step AdamW shrinks decayed weights: w <- w - lr * wd * w, so the real shrinking still follows the lr schedule
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay # start = end, so the cosine curve is a flat line
    # train.py only applies this to groups with wd > 0, so biases/norms/gamma (no_decay group) stay at 0
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values))) # 0.05 / 0.05 when constant


    # cross-entropy with label smoothing: target [1, 0] becomes [0.95, 0.05] (0.9 x one-hot + 0.1 x uniform),
    # so the model is penalised for being over-confident; plain cross-entropy if --smoothing 0
    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # resume an interrupted run: loads the latest checkpoint-<epoch>.pth in output_dir (or --resume <path>) and restores
    # weights, AdamW state, EMA, AMP scale and start_epoch, so training and the lr schedule continue where they stopped.
    # use a new output_dir per run
    utils.auto_load_model(
        args=args, model=model,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # the script skips training completely, tests the model on the test set once, saves the results, and exits
    # --eval True: skip training, test the loaded checkpoint once on the test set, save test_results.json +
    # test_predictions.csv (per-slice probabilities, for calibration) to output_dir, then exit.
    # pass --resume .../checkpoint-best.pth: otherwise auto_resume loads the LATEST epoch, not the best one.
    # use a new output_dir to avoid overwriting the training run's test results
    if args.eval:
        print(f"Eval only mode")
        run_test(model, data_loader_test, dataset_test, device, args, checkpoint_name=args.resume or 'initial weights') # 'initial weights' = nothing loaded (random, ~50%)
        return

    max_accuracy = 0.0 #accuracy counter
    best_score, best_epoch = None, None # (best_metric, -val loss) of checkpoint-best.pth, compared as a tuple
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0 #ema accuracy counter

    print("Start training for %d epochs" % args.epochs)
    #start epock timer
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        )
        
        # save results of epoch
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        
        # run validation after each epoch
        if data_loader_val is not None:
            val_stats, val_probs, _ = evaluate(data_loader_val, model, device, use_amp=True, header='Val:',
                                               return_preds=True)
            print(f"Accuracy of the model on the {len(dataset_val)} validation images: {val_stats['acc1']:.1f}%")
            # patient level: average each patient's 20 slice probabilities, as predict.py does for test
            val_stats.update({f'patient_{k}': v for k, v in val_patient_metrics(dataset_val, val_probs).items()})
            print(f"Patient-level validation: accuracy {val_stats['patient_acc1']:.1f}%  "
                  f"AD recall {val_stats['patient_ad_recall']:.3f}  macro-F1 {val_stats['patient_macro_f1']:.3f}")
            max_accuracy = max(max_accuracy, val_stats["acc1"])
            score = (val_stats[args.best_metric], -val_stats['loss'])
            if best_score is None or score > best_score:
                best_score, best_epoch = score, epoch
                if args.output_dir and args.save_ckpt:
                    # save the latest updated weights
                    utils.save_model(
                        args=args, model=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max validation accuracy: {max_accuracy:.2f}%  '
                  f'best {args.best_metric}: {best_score[0]:.4f} (epoch {best_epoch})')

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
                val_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=True, header='Val (EMA):')
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

    total_time = time.time() - start_time # final epoch time 
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
        'best_metric': args.best_metric,
        'best_val_score': best_score[0] if best_score else None,
        'best_epoch': best_epoch,
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