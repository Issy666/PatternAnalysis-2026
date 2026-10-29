"""
Grad-CAM and UMAP for a trained model, on the test set.

Grad-CAM (Selvaraju et al., 2017): which parts of a slice drove the prediction. The gradient of the predicted
class's logit with respect to a stage's feature maps is averaged over space to weight each channel; the weighted
sum of the maps (ReLU'd, upsampled to the input) is the heat map. Saves:
  - gradcam_examples.png : correct AD, correct NC, the most confident missed AD and false alarms (middle slice of
                            each scan), with the heat map over the slice
  - gradcam_mean.png     : the average heat map over every test scan's middle slice, per outcome. Brains are only
                            roughly aligned between scans, so this shows broad regions, not exact structures
  - gradcam_stats.json   : fraction of heat-map mass inside the (filled) brain outline vs the brain's share of the
                            image (acceptance criterion 4: anatomy, not background or border)

UMAP (McInnes et al., 2018): the pooled features before the classifier (forward_features), averaged over each
scan's 20 slices, projected to 2D. Saves umap_test.png (one point per scan) coloured by true label and by outcome
(correct / wrong / referred), and umap_test_slices.png (one point per slice, the features before averaging) coloured
by true label and by the slice's own outcome (its P(AD) > 0.5 from test_predictions.csv; referral is per scan only).

The checkpoint stores its training arguments, so the model and test data (incl. 2.5D input) are rebuilt exactly.
Outcomes come from calibrate.py's test_patient_predictions_calibrated.csv if it exists (calibrated probability,
validation-chosen threshold, referral), otherwise the run's test_predictions.csv at threshold 0.5.

Usage:
    python visualise.py --resume ~/Desktop/results/<run>/checkpoint-best.pth
Needs umap-learn for the UMAP plot (Grad-CAM works without it).
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import build_dataset, patient_id
from utils import load_trained_model

BLUE, ORANGE, INK, MUTED, GREY = '#2a78d6', '#eb6834', '#0b0b0b', '#898781', '#c9c8c3'


def get_args_parser():
    p = argparse.ArgumentParser('Grad-CAM + UMAP on the test set')
    p.add_argument('--resume', required=True, help='checkpoint-best.pth of the run')
    p.add_argument('--output_dir', default='', help='default: <run>/visualise')
    p.add_argument('--cam_stage', default=3, type=int, choices=[2, 3, 4],
                   help='ConvNeXt stage whose output Grad-CAM uses: 3 = 16x16 maps (default), 4 = 8x8 (last stage)')
    p.add_argument('--per_row', default=4, type=int, help='example scans per outcome in gradcam_examples.png')
    p.add_argument('--brain_threshold', default=0.05, type=float,
                   help='pixel intensity (0-1) above which a pixel counts as brain for the in-brain statistic')
    p.add_argument('--batch_size', default=48, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p


# ---------------------------------------------------------------------------
# Model, data and scan-level outcomes
# ---------------------------------------------------------------------------

def scan_outcomes(run_dir):
    """{scan id: dict(label, prob_AD, pred, referred)} plus a description of where it came from."""
    cal = run_dir / 'calibrate' / 'test_patient_predictions_calibrated.csv'
    out = {}
    if cal.exists():
        with open(cal, newline='') as f:
            for r in csv.DictReader(f):
                out[r['patient']] = dict(label=r['label'], prob_AD=float(r['prob_AD_calibrated']), pred=r['pred'],
                                         referred=r['referred'] == '1')
        return out, f'calibrated predictions from {cal}'
    probs, labels = defaultdict(list), {}
    with open(run_dir / 'test_predictions.csv', newline='') as f:
        for r in csv.DictReader(f):
            pid = patient_id(r['path'])
            probs[pid].append(float(r['prob_AD']))
            labels[pid] = r['label']
    for pid, ps in probs.items():
        p = float(np.mean(ps))
        out[pid] = dict(label=labels[pid], prob_AD=p, pred='AD' if p > 0.5 else 'NC', referred=False)
    return out, 'test_predictions.csv at threshold 0.5 (run calibrate.py for calibrated outcomes and referrals)'


def outcome(o):
    if o['referred']:
        return 'Referred'
    if o['label'] == o['pred']:
        return f"Correct {o['label']}"
    return 'Missed AD' if o['label'] == 'AD' else 'False alarm'


def middle_slices(dataset):
    """{scan id: dataset index of its middle slice} (slice 10 of 20)."""
    by_scan = defaultdict(list)
    for i, (path, _) in enumerate(dataset.samples):
        by_scan[patient_id(path)].append((int(Path(path).stem.rsplit('_', 1)[1]), i))
    return {pid: sorted(s)[len(s) // 2][1] for pid, s in by_scan.items()}


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def cam_layer(model, stage):
    if hasattr(model, 'stages'):  # ConvNeXt / ConvNeXtBN
        return model.stages[stage - 1]
    return model.features[4 * stage - 1]  # SimpleCNN: output of block `stage` (after its max-pool)


def grad_cam(model, layer, x, target):
    """Heat maps (N, H, W) in [0, 1] for class `target` (N,), and the logits. Batched: in eval mode no layer mixes
    samples (BatchNorm uses running statistics), so each sample's gradient depends only on its own logit."""
    store = {}
    handle = layer.register_forward_hook(lambda m, i, o: store.update(act=o))
    logits = model(x)  # weights require grad, so the hooked activations are part of the graph
    handle.remove()
    act = store['act']
    grads = torch.autograd.grad(logits.gather(1, target[:, None]).sum(), act)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)                       # one weight per channel
    cam = F.relu((weights * act).sum(1, keepdim=True))                    # weighted sum of the feature maps
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]
    return cam.detach(), logits.detach()


def brain_masks(images, threshold):
    """
    Filled brain mask per image: everything except the dark background connected to the image border.
    A plain intensity threshold would leave out the ventricles and other CSF, which are dark on T1 but inside the
    brain (and enlarged in AD), so the background is flood-filled from the border through dark pixels instead.
    images: (N, H, W) tensor in [0, 1].
    """
    dark = images <= threshold
    outside = torch.zeros_like(dark)
    outside[:, 0, :], outside[:, -1, :], outside[:, :, 0], outside[:, :, -1] = (
        dark[:, 0, :], dark[:, -1, :], dark[:, :, 0], dark[:, :, -1])
    while True:  # grow the border region through dark pixels until it stops changing
        grown = F.max_pool2d(outside[:, None].float(), 3, stride=1, padding=1)[:, 0].bool() & dark
        if torch.equal(grown, outside):
            return ~outside
        outside = grown


def slice_image(x, mean, std):
    """Undo Normalize and return the middle channel (this slice; R/B are its neighbours in 2.5D) in [0, 1]."""
    return (x[1] * std[1] + mean[1]).clamp(0, 1)


def run_gradcam(model, dataset, outcomes, middles, args, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layer = cam_layer(model, args.cam_stage)
    norm = next(t for t in dataset.transform.transforms if t.__class__.__name__ == 'Normalize')
    mean, std = torch.tensor(norm.mean), torch.tensor(norm.std)

    # heat map for the middle slice of every test scan, w.r.t. the class the model predicted for the scan
    ids = sorted(middles)
    cams, images = {}, {}
    for start in range(0, len(ids), args.batch_size):
        chunk = ids[start:start + args.batch_size]
        x = torch.stack([dataset[middles[i]][0] for i in chunk]).to(args.device)
        target = torch.tensor([0 if outcomes[i]['pred'] == 'AD' else 1 for i in chunk], device=args.device)  # AD = 0
        cam, _ = grad_cam(model, layer, x, target)
        for i, c, xi in zip(chunk, cam.cpu(), x.cpu()):
            cams[i], images[i] = c.numpy(), slice_image(xi, mean, std).numpy()

    # criterion 4: share of each heat map inside the (filled) brain, compared with the brain's share of the image,
    # which is what a heat map spread evenly over the image would score
    masks = brain_masks(torch.from_numpy(np.stack([images[i] for i in ids])), args.brain_threshold).numpy()
    inside, area = {}, {}
    for i, m in zip(ids, masks):
        inside[i] = float((cams[i] * m).sum() / max(cams[i].sum(), 1e-8))
        area[i] = float(m.mean())
    groups = defaultdict(list)
    for i in ids:
        groups[outcome(outcomes[i])].append(i)

    def summary(sel):
        fi, fa = np.mean([inside[i] for i in sel]), np.mean([area[i] for i in sel])
        return {'n': len(sel), 'mean_fraction_inside_brain': float(fi), 'mean_brain_area_fraction': float(fa),
                'enrichment': float(fi / fa)}  # > 1 = the heat map favours the brain over the background

    stats = {'cam_stage': args.cam_stage, 'brain_threshold': args.brain_threshold, 'mask': 'filled brain outline',
             **summary(ids),
             'fraction_of_scans_with_>=75%_inside_brain': float(np.mean([inside[i] >= 0.75 for i in ids])),
             'by_outcome': {g: summary(v) for g, v in sorted(groups.items())}}
    with open(out / 'gradcam_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Grad-CAM (stage {args.cam_stage}): {stats['mean_fraction_inside_brain']:.1%} of the heat map is inside the "
          f"brain, which covers {stats['mean_brain_area_fraction']:.1%} of the image (enrichment "
          f"{stats['enrichment']:.2f}x); {stats['fraction_of_scans_with_>=75%_inside_brain']:.1%} of scans have "
          f">= 75% inside")
    for g, s in stats['by_outcome'].items():
        print(f"  {g:<12} n={s['n']:3d}  inside brain {s['mean_fraction_inside_brain']:.1%} "
              f"(brain area {s['mean_brain_area_fraction']:.1%}, {s['enrichment']:.2f}x)")

    # example grid: random correct cases, the most confident mistakes, the least decisive referrals
    rng = np.random.default_rng(args.seed)
    conf = {i: abs(outcomes[i]['prob_AD'] - 0.5) for i in ids}
    rows = {}
    for g in ['Correct AD', 'Correct NC']:
        if groups[g]:
            rows[g] = list(rng.choice(groups[g], min(args.per_row, len(groups[g])), replace=False))
    for g in ['Missed AD', 'False alarm']:
        rows[g] = sorted(groups[g], key=lambda i: -conf[i])[:args.per_row]
    if groups['Referred']:
        rows['Referred'] = sorted(groups['Referred'], key=lambda i: conf[i])[:args.per_row]
    rows = {g: v for g, v in rows.items() if v}

    fig, axes = plt.subplots(len(rows), args.per_row, figsize=(2.7 * args.per_row + 1.4, 2.9 * len(rows)),
                             squeeze=False)
    for r, (g, chosen) in enumerate(rows.items()):
        for c in range(args.per_row):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c >= len(chosen):
                continue
            i = chosen[c]
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
            ax.imshow(cams[i], cmap='jet', alpha=0.4, vmin=0, vmax=1)
            o = outcomes[i]
            ax.set_title(f"True {o['label']}, pred {o['pred']}\nP(AD) {o['prob_AD']:.2f}", fontsize=9, color=INK)
            ax.set_xlabel(f"{i} (in brain {inside[i]:.0%})", fontsize=8, color=MUTED)
        axes[r, 0].set_ylabel(g, fontsize=10, color=INK, labelpad=8)
    fig.suptitle(f"Grad-CAM, stage {args.cam_stage}, for the predicted class (middle slice of each test scan)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / 'gradcam_examples.png', dpi=120)
    plt.close(fig)
    print(f"Saved {out / 'gradcam_examples.png'}")

    # average heat map per outcome over the registered slices
    shown = [g for g in ['Correct AD', 'Correct NC', 'Missed AD', 'False alarm', 'Referred'] if groups[g]]
    fig, axes = plt.subplots(1, len(shown), figsize=(3.2 * len(shown), 3.6), squeeze=False)
    for ax, g in zip(axes[0], shown):
        ax.imshow(np.mean([images[i] for i in groups[g]], 0), cmap='gray')
        ax.imshow(np.mean([cams[i] for i in groups[g]], 0), cmap='jet', alpha=0.45)
        ax.set_title(f"{g} (n = {len(groups[g])})", fontsize=10, color=INK)
        ax.axis('off')
    fig.suptitle(f"Mean Grad-CAM per outcome, stage {args.cam_stage} (over mean slice)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'gradcam_mean.png', dpi=120)
    plt.close(fig)
    print(f"Saved {out / 'gradcam_mean.png'}")


# ---------------------------------------------------------------------------
# UMAP of the pooled features
# ---------------------------------------------------------------------------

@torch.no_grad()
def slice_features(model, dataset, args):
    """Pooled features before the classifier (model.forward_features), one row per slice in dataset.samples order."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    feats = []
    for x, _ in loader:
        x = x.to(args.device)
        feats.append(model.forward_features(x).float().cpu())
    return torch.cat(feats).numpy()


def slice_outcomes(run_dir):
    """{slice path: outcome} from the slice's own prediction in test_predictions.csv (P(AD) > 0.5)."""
    out = {}
    with open(run_dir / 'test_predictions.csv', newline='') as f:
        for r in csv.DictReader(f):
            o = {'label': r['label'], 'pred': 'AD' if float(r['prob_AD']) > 0.5 else 'NC', 'referred': False}
            out[r['path']] = outcome(o)
    return out


def plot_umap(emb, panels, title, path, point_size, alpha, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # draw all points in one random order, so no category is always painted on top of the others where they overlap
    order = np.random.default_rng(seed).permutation(len(emb))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8), squeeze=False)
    for ax, (panel_title, colours, values) in zip(axes[0], panels):
        values = np.array(values)
        ax.scatter(emb[order, 0], emb[order, 1], s=point_size, c=[colours[v] for v in values[order]], alpha=alpha,
                   linewidths=0)
        handles = [Line2D([], [], ls='', marker='o', ms=6, color=c, label=f"{n} ({(values == n).sum()})")
                   for n, c in colours.items() if (values == n).any()]
        ax.set_title(panel_title, fontsize=11, color=INK)
        ax.set_xlabel('UMAP-1', fontsize=9, color=INK)
        ax.set_ylabel('UMAP-2', fontsize=9, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.legend(handles=handles, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def run_umap(model, dataset, outcomes, run_dir, args, out):
    try:
        import umap
    except ImportError:
        print("umap-learn is not installed (pip install umap-learn); skipping the UMAP plots")
        return

    feats = slice_features(model, dataset, args)
    paths = [p for p, _ in dataset.samples]
    label_colours = {'AD': ORANGE, 'NC': BLUE}
    outcome_colours = {'Correct AD': ORANGE, 'Correct NC': BLUE, 'Missed AD': INK, 'False alarm': '#b8379a',
                       'Referred': GREY}

    # one point per scan: mean of its slices' features
    by_scan = defaultdict(list)
    for p, f in zip(paths, feats):
        by_scan[patient_id(p)].append(f)
    ids = sorted(by_scan)
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed).fit_transform(
        np.stack([np.mean(by_scan[i], 0) for i in ids]))
    np.save(out / 'umap_test_embedding.npy', emb)
    plot_umap(emb, [('True label', label_colours, [outcomes[i]['label'] for i in ids]),
                    ('Outcome', outcome_colours, [outcome(outcomes[i]) for i in ids])],
              f"UMAP of pooled test features, one point per scan ({len(ids)} scans)",
              out / 'umap_test.png', point_size=12, alpha=0.8)

    # one point per slice: the features before averaging, with each slice's own prediction
    per_slice = slice_outcomes(run_dir)
    emb_s = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed).fit_transform(feats)
    np.save(out / 'umap_test_slices_embedding.npy', emb_s)
    plot_umap(emb_s, [('True label', label_colours, [os.path.basename(os.path.dirname(p)) for p in paths]),
                      ('Slice outcome (P(AD) > 0.5)', outcome_colours, [per_slice[p] for p in paths])],
              f"UMAP of pooled test features, one point per slice ({len(paths)} slices)",
              out / 'umap_test_slices.png', point_size=3, alpha=0.5)


def main(args):
    torch.manual_seed(args.seed)
    run_dir = Path(args.resume).expanduser().resolve().parent
    out = Path(args.output_dir) if args.output_dir else run_dir / 'visualise'
    out.mkdir(parents=True, exist_ok=True)

    model, train_args = load_trained_model(args.resume, args.device) # rebuilt from the checkpoint's saved arguments
    dataset = build_dataset(split='test', args=train_args)
    outcomes, source = scan_outcomes(run_dir)
    print(f"Outcomes from {source}")
    middles = middle_slices(dataset)
    missing = set(middles) - set(outcomes)
    if missing:
        raise SystemExit(f"{len(missing)} test scans have no prediction (e.g. {sorted(missing)[:3]}); "
                         f"is this the right run / data?")

    run_gradcam(model, dataset, outcomes, middles, args, out)
    run_umap(model, dataset, outcomes, run_dir, args, out)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
