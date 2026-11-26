
"""
Plot training vs validation loss/accuracy (and validation AD/NC recall) from a run's log.txt.

    python plot_log.py ~/Desktop/results/convnext_t_pretrained

Saves training_curves.png and lr_schedule.png (learning rate per epoch: linear warmup, then cosine decay) in the
run directory. Works while training is still going.

AI assistance: the learning-rate plot was scaffolded with Claude (Anthropic), then reviewed by me (see README,
"Artificial Intelligence Usage Disclosure").
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(run_dir):
    """Read run_dir/log.txt and save the training-curve and learning-rate plots next to it."""
    run_dir = Path(run_dir).expanduser()
    rows = [json.loads(line) for line in open(run_dir / 'log.txt') if line.strip()]
    epochs = [r['epoch'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [r['train_loss'] for r in rows], label='Train')
    if 'val_loss' in rows[0]:
        axes[0].plot(epochs, [r['val_loss'] for r in rows], label='Validation')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # train accuracy is always logged now (mixup/cutmix removed); the check keeps older logs without it working
    if 'train_class_acc' in rows[0]:
        axes[1].plot(epochs, [100 * r['train_class_acc'] for r in rows], label='Train')
    if 'val_acc1' in rows[0]:
        axes[1].plot(epochs, [r['val_acc1'] for r in rows], label='Validation')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()

    if 'val_ad_recall' in rows[0]:
        axes[2].plot(epochs, [r['val_ad_recall'] for r in rows], label='AD recall (sensitivity)')
        axes[2].plot(epochs, [r['val_nc_recall'] for r in rows], label='NC recall (specificity)')
    axes[2].set_title('Validation Recall per Class')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    fig.suptitle(run_dir.name)
    fig.tight_layout()
    out = run_dir / 'training_curves.png'
    fig.savefig(out, dpi=120)

    plt.close(fig)

    # learning rate: train.py sets it every step from the warmup + cosine schedule, and log.txt stores each
    # epoch's mean, so this is the per-epoch average of the schedule
    if 'train_lr' in rows[0]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(epochs, [r['train_lr'] for r in rows], 'o-', color='#2a78d6', lw=2, ms=4)
        ax.set_title(f'Learning Rate Schedule ({run_dir.name})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning rate (mean over the epoch)')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)) # whole epochs only
        ax.grid(color='#e6e5e1', lw=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        lr_out = run_dir / 'lr_schedule.png'
        fig.savefig(lr_out, dpi=120)
        plt.close(fig)
        print(f"Saved {lr_out}")

    best = max(rows, key=lambda r: r.get('val_acc1', 0))
    print(f"{len(rows)} epochs logged. Best validation accuracy {best.get('val_acc1', float('nan')):.2f}% "
          f"at epoch {best['epoch']}. Saved {out}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
