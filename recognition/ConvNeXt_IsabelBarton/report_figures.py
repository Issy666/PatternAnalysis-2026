"""
README figures that no other script makes, from a trained run:

  - example_input_output.png : the 3 neighbouring slices that form one 2.5D input, the normalised input, and the
                               model's P(AD) for each of the scan's 20 slices (a typical correctly classified AD scan)
  - failure_cases.png        : chosen test scans with Grad-CAM on the middle slice and P(AD) per slice

    python report_figures.py --run ~/Desktop/results/bn_25d_aug_e18
    python report_figures.py --run ... --cases 1005735:"A. Missed AD" 959645:"B. False alarm"

Needs the run's checkpoint-best.pth and test_predictions.csv. Subject and year in the titles come from the split's
split_manifest.csv (written by subject_datasplit.py), if it exists.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# the model code lives in recognition/ConvNeXt_IsabelBarton
CODE_DIR = Path(__file__).resolve().parent.parent / 'recognition' / 'ConvNeXt_IsabelBarton'
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from PIL import Image

from dataset import build_dataset, patient_id
from utils import load_trained_model
import visualise

INK, MUTED, ORANGE, BLUE = '#0b0b0b', '#898781', '#eb6834', '#2a78d6'

# the 4 failure cases discussed in the README (section 6.4), from ConvNeXt-T seed 0 (bn_25d_aug_e18)
DEFAULT_CASES = ['1005735:A. Missed AD, confident', '959645:B. False alarm, confident',
                 '1070417:C. False alarm, repeated person', '301063:D. Borderline (referred)']


def slice_bars(ax, probs):
    """P(AD) per slice: orange = called AD, blue = called NC; dashed = 0.5, solid = the scan's mean."""
    xs = sorted(probs)
    ax.bar(xs, [probs[s] for s in xs], color=[ORANGE if probs[s] > 0.5 else BLUE for s in xs])
    ax.axhline(0.5, color=MUTED, ls='--', lw=1)
    ax.axhline(np.mean(list(probs.values())), color=INK, lw=1.5)
    ax.set(ylim=(0, 1), xlabel='slice', ylabel='P(AD)')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main(args):
    run = Path(args.run).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    model, train_args = load_trained_model(run / 'checkpoint-best.pth', args.device)
    ds = build_dataset('test', train_args)
    norm = next(t for t in ds.transform.transforms if t.__class__.__name__ == 'Normalize')
    mean, std = torch.tensor(norm.mean), torch.tensor(norm.std)
    manifest_path = Path(train_args.data_path).parent / 'split_manifest.csv'
    manifest = {r['image_id']: r for r in csv.DictReader(open(manifest_path))} if manifest_path.exists() else {}

    # P(AD) per slice and dataset index per slice, by scan
    slice_probs, labels, index = defaultdict(dict), {}, defaultdict(dict)
    for r in csv.DictReader(open(run / 'test_predictions.csv')):
        scan, sl = patient_id(r['path']), int(Path(r['path']).stem.split('_')[1])
        slice_probs[scan][sl] = float(r['prob_AD'])
        labels[scan] = r['label']
    for i, (p, _) in enumerate(ds.samples):
        index[patient_id(p)][int(Path(p).stem.split('_')[1])] = i

    def scan_mean(scan):
        return float(np.mean(list(slice_probs[scan].values())))

    def middle(scan):
        s = sorted(index[scan])
        return s[len(s) // 2]

    def describe(scan):
        m = manifest.get(scan)
        return f"scan {scan}, subject {m['subject']}, {m['year']}" if m else f"scan {scan}"

    # ---- 1. example input / output: a typical correct AD scan (scan-mean P(AD) closest to 0.85) ----
    scan = min((s for s in slice_probs if labels[s] == 'AD' and scan_mean(s) > 0.5),
               key=lambda s: abs(scan_mean(s) - 0.85))
    k = middle(scan)
    paths = {sl: ds.samples[i][0] for sl, i in index[scan].items()}
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.5]})
    for ax, (sl, title) in zip(axes[:3], [(k - 1, 'R: slice below'), (k, 'G: this slice'), (k + 1, 'B: slice above')]):
        ax.imshow(Image.open(paths[sl]).convert('L'), cmap='gray')
        ax.set_title(f"{title}\n{Path(paths[sl]).name}", fontsize=9, color=INK)
        ax.axis('off')
    x = ds[index[scan][k]][0]
    axes[3].imshow((x * std[:, None, None] + mean[:, None, None]).clamp(0, 1).permute(1, 2, 0).numpy())
    axes[3].set_title('2.5D input (3 x 256 x 256)\npadded + normalised', fontsize=9, color=INK)
    axes[3].axis('off')
    slice_bars(axes[4], slice_probs[scan])
    pred = 'AD' if scan_mean(scan) > 0.5 else 'NC'
    axes[4].set_title(f"Output: P(AD) per slice\nscan mean {scan_mean(scan):.2f} -> predicted {pred} "
                      f"(true {labels[scan]})", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / 'example_input_output.png', dpi=120)
    plt.close(fig)
    print(f"Saved {out / 'example_input_output.png'} (scan {scan})")

    # ---- 2. failure cases: Grad-CAM (stage 3, predicted class) on the middle slice + P(AD) per slice ----
    cases = [c.split(':', 1) for c in args.cases]
    layer = visualise.cam_layer(model, 3)
    fig, axes = plt.subplots(len(cases), 2, figsize=(9.5, 3.0 * len(cases)),
                             gridspec_kw={'width_ratios': [1, 1.6]}, squeeze=False)
    for row, (scan, title) in enumerate(cases):
        x = ds[index[scan][middle(scan)]][0][None].to(args.device)
        target = torch.tensor([0 if scan_mean(scan) > 0.5 else 1], device=args.device)  # AD = 0
        cam, _ = visualise.grad_cam(model, layer, x, target)
        ax = axes[row, 0]
        ax.imshow(visualise.slice_image(x[0].cpu(), mean, std).numpy(), cmap='gray')
        ax.imshow(cam[0].cpu().numpy(), cmap='jet', alpha=0.4, vmin=0, vmax=1)
        ax.set_title(f"{title}\n{describe(scan)}", fontsize=9, color=INK)
        ax.axis('off')
        slice_bars(axes[row, 1], slice_probs[scan])
        axes[row, 1].set_title(f"true {labels[scan]}, scan mean P(AD) {scan_mean(scan):.2f}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / 'failure_cases.png', dpi=120)
    plt.close(fig)
    print(f"Saved {out / 'failure_cases.png'}")


if __name__ == '__main__':
    p = argparse.ArgumentParser('README figures: example input/output and failure cases')
    p.add_argument('--run', default='~/Desktop/results/bn_25d_aug_e18', help='run folder with checkpoint-best.pth')
    p.add_argument('--out', default=str(CODE_DIR / 'images'), help='where to save the figures')
    p.add_argument('--cases', nargs='+', default=DEFAULT_CASES, help='failure cases as scanID:"title"')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    main(p.parse_args())
