"""
Training entry point: train a model on ADNI (AD vs NC), validating every epoch, then test the best validation
checkpoint once.

    python main.py --model convnext_tiny_bn --neighbour_slices True --output_dir ~/Desktop/results/<run> ...

Afterwards, predict.py --predictions <output_dir>/test_predictions.csv analyses the test predictions and
plot_log.py <output_dir> plots the training curves (see README, Section 7.2).
Writes to --output_dir: checkpoint-<epoch>.pth / checkpoint-best.pth, log.txt (one JSON line per epoch),
test_results.json and test_predictions.csv.

Reference: adapted from the official ConvNeXt training script [2] (Copyright (c) Meta Platforms, Inc. and
affiliates, MIT licence): https://github.com/facebookresearch/ConvNeXt/blob/main/main.py
Changes for ADNI: mixup/CutMix, EMA, fine-tuning and distributed options removed; validation and test splits,
patient-level validation metrics and the --best_metric checkpoint rule added.
"""
import argparse
import datetime
import json
import os
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path

from timm.loss import LabelSmoothingCrossEntropy

from dataset import build_dataset
from modules import build_model
from train import create_optimizer, train_one_epoch, evaluate
from predict import run_test, patient_predictions, classification_metrics
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def str2bool(v):
    """Parse a True/False command-line value."""
    # argparse's type=bool treats any non-empty string (including "False") as True
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    """All training options (model, optimiser, schedule, augmentation, data paths, checkpointing)."""
    parser = argparse.ArgumentParser('ConvNeXt training + evaluation script for MRI AD or CN classification', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=60, type=int) # 100-epoch run peaked on val at 55, so cosine decay now finishes around there

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str,
                        help='Model to train: convnext_tiny, convnext_tiny_bn, convnext_nano_bn or simple_cnn (modules.py)')

    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='Drop path (stochastic depth) rate (default: 0.1)')

    parser.add_argument('--input_size', default=256, type=int,
                        help='image input size')

    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help='Initial value of the per-channel layer scale (gamma); 0 disables it (default: 1e-6)')

    # None = each model's own default: 0.4 for the BatchNorm (*_bn) models and simple_cnn, 0 for the LayerNorm ConvNeXt
    parser.add_argument('--head_drop', type=float, default=None,
                        help='Dropout before the classifier (default: model default, 0.4 for *_bn / simple_cnn, '
                             '0 for convnext_tiny)')

    # Optimization parameters (optimiser is always AdamW, see train.create_optimizer)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # paper's 4e-3 is for batch 4096; square-root scaled to batch 64: 4e-3 * sqrt(64/4096) = 5e-4
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='peak learning rate (default: 5e-4)')

    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='learning rate at the end of the cosine decay (default: 1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, # ~8% of 60 epochs, close to the paper's 20 of 300
                        help='epochs of linear learning-rate warmup')

    # Augmentation parameters (no RandomResizedCrop / flip / RandAugment / mixup: see normalise.build_transform and README)
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
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase probability: one noise-filled rectangle per image (default: 0.25)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/isabel/Desktop/data/ADNI_patientleakagefix/AD_NC/train', type=str,
                        help='training images (folder containing AD/ and NC/)')
    parser.add_argument('--val_data_path', default='/home/isabel/Desktop/data/ADNI_patientleakagefix/AD_NC/validation', type=str,
                        help='validation images, used for checkpoint selection')
    parser.add_argument('--eval_data_path', default='/home/isabel/Desktop/data/ADNI_patientleakagefix/AD_NC/test', type=str,
                        help='test images, evaluated once at the end')

    # 2.5D input: channels = (slice below, slice, slice above) from the same scan instead of 3 greyscale copies
    parser.add_argument('--neighbour_slices', type=str2bool, default=False,
                        help='Use neighbouring slices of the same scan as the 3 input channels (default: False)')
    parser.add_argument('--neighbour_step', type=int, default=1,
                        help='Slice offset of the neighbours, e.g. 2 = 2 slices either side (default: 1)')

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output_dir', default='/home/isabel/Desktop/results',
                        help='where checkpoints, log.txt and test results are saved (use a new folder per run)')
    parser.add_argument('--seed', default=0, type=int)

    # Checkpointing / resuming
    parser.add_argument('--save_ckpt_num', default=3, type=int,
                        help='number of recent epoch checkpoints to keep (checkpoint-best.pth is always kept)')
    parser.add_argument('--resume', default='',
                        help='checkpoint to resume training from (default: the latest one in output_dir, if any)')

    # which validation score picks checkpoint-best.pth. slice accuracy hides the AD/NC split of errors and jumps
    # around by luck; patient macro-F1 weights AD and NC equally, ties go to the lower validation loss
    parser.add_argument('--best_metric', default='acc1', choices=['acc1', 'patient_macro_f1'],
                        help='Validation metric used to choose checkpoint-best.pth (default: acc1)')

    return parser


def val_patient_metrics(dataset, probs):
    """Patient-level validation metrics: mean P(AD) over each scan's slices, then accuracy / recall / macro-F1."""
    # probs are in dataset.samples order (val loader isn't shuffled); column 0 = P(AD)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    slices = [(path, idx_to_class[target], p[0]) for (path, target), p in zip(dataset.samples, probs.tolist())]
    return classification_metrics(patient_predictions(slices))


def main(args):
    """Build the data, model, optimiser and schedule; train with validation each epoch; then test the best checkpoint."""

    print(args)

    # training always uses AMP (autocast/GradScaler on 'cuda'), so it needs a CUDA GPU
    if not torch.cuda.is_available():
        raise SystemExit("main.py needs a CUDA GPU (training always uses mixed precision); "
                         "use predict.py --resume <checkpoint> to evaluate a trained model on a CPU")
    device = torch.device('cuda')

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # enables the NVIDIA cuDNN auto-tuner, which tests several convolution algorithms and picks the fastest for this GPU (works because every input is the same size)
    cudnn.benchmark = True

    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)
    dataset_test = build_dataset(split='test', args=args)

    # shuffle=True reshuffles the training data every epoch; pinned memory speeds up the copy to the GPU
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # validation and test are evaluated in order, so predictions line up with dataset.samples
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=False,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # mixup/cutmix from the paper removed for ADNI: they blend or paste two brains together, so the label no longer
    # matches the anatomy that defines AD vs NC (see README "Why mixup and CutMix are not used")

    #build the model (trained from scratch: random initial weights)
    model = build_model(args)
    model.to(device)

    # number of trainable parameters in the whole model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # run details
    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    # optimizer steps per epoch (rounded down), e.g. 18,260 // 64 = 285; the lr schedule is built per step from this
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % args.batch_size)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

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
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs,
    )

    # cross-entropy with label smoothing: target [1, 0] becomes [0.95, 0.05] (0.9 x one-hot + 0.1 x uniform),
    # so the model is penalised for being over-confident; plain cross-entropy if --smoothing 0
    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # resume an interrupted run: loads the latest checkpoint-<epoch>.pth in output_dir (or --resume <path>) and restores
    # weights, AdamW state, AMP scale and start_epoch, so training and the lr schedule continue where they stopped.
    # use a new output_dir per run
    args.start_epoch = 0
    utils.auto_load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    max_accuracy = 0.0 #accuracy counter
    best_score, best_epoch = None, None # (best_metric, -val loss) of checkpoint-best.pth, compared as a tuple

    print("Start training for %d epochs" % args.epochs)
    # start the training timer
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            start_steps=epoch * num_training_steps_per_epoch, lr_schedule_values=lr_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
        )

        # save results of epoch
        utils.save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        # run validation after each epoch
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
            # save the latest updated weights
            utils.save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch="best")
        print(f'Max validation accuracy: {max_accuracy:.2f}%  '
              f'best {args.best_metric}: {best_score[0]:.4f} (epoch {best_epoch})')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time # total training time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # final, one-off evaluation on the held-out test split using the best validation checkpoint
    checkpoint_name = 'final epoch weights'
    best_path = Path(args.output_dir) / 'checkpoint-best.pth'
    if best_path.exists():
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
        'max_gpu_mem_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'n_parameters': n_parameters,
    }
    run_test(model, data_loader_test, dataset_test, device, args, checkpoint_name, extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training + evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
