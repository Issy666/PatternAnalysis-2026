"""
Test a trained ConvNeXt and analyse its predictions.

Two ways to run it:
  1. From a checkpoint (runs the model on the test set, then analyses):
       python predict.py --resume ~/Desktop/results/<run>/checkpoint-best.pth
  2. From an existing test_predictions.csv (no model or GPU needed):
       python predict.py --predictions ~/Desktop/results/<run>/test_predictions.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import torch

from train import evaluate, binary_metrics

CLASSES = ['AD', 'NC'] # ImageFolder numbers folders alphabetically, so AD = 0 (positive class), NC = 1


def run_test(model, data_loader_test, dataset_test, device, args, checkpoint_name, extra=None):
    """Evaluate on the held-out test split and save metrics + per-slice predictions to output_dir."""
    print(f"Evaluating {checkpoint_name} on the test set ({len(dataset_test)} images)")
    # AMP is CUDA-only, so fall back to full precision on a CPU (e.g. someone running predict.py without a GPU)
    use_amp = torch.device(device).type == 'cuda'
    test_stats, probs, targets = evaluate(data_loader_test, model, device, use_amp=use_amp,
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


# ---------------------------------------------------------------------------
# Patient-level metrics
# ---------------------------------------------------------------------------

def load_predictions(csv_path):
    """Read test_predictions.csv into a list of (path, label, prob_AD) per slice."""
    with open(csv_path, newline='') as f:
        return [(row['path'], row['label'], float(row['prob_AD'])) for row in csv.DictReader(f)]


def patient_id(path):
    # filenames are <patientID>_<sliceID>.jpeg, e.g. 1003730_100.jpeg -> 1003730
    return os.path.basename(path).split('_')[0]


def patient_predictions(slices, threshold=0.5):
    """
    Combine each patient's slices into one prediction: average prob_AD over the
    patient's slices (20 in ADNI), then predict AD if the average > threshold.
    Returns a list of dicts sorted by patient ID.
    """
    by_patient = defaultdict(list)
    labels = {}
    for path, label, prob_ad in slices:
        pid = patient_id(path)
        by_patient[pid].append(prob_ad)
        # all slices of a patient share one diagnosis; a mismatch means the split mixed folders
        if labels.setdefault(pid, label) != label:
            raise ValueError(f"Patient {pid} has slices labelled both {labels[pid]} and {label}")

    patients = []
    for pid in sorted(by_patient):
        probs = by_patient[pid]
        mean_prob = sum(probs) / len(probs)
        patients.append({
            'patient': pid,
            'label': labels[pid],
            'pred': 'AD' if mean_prob > threshold else 'NC',
            'prob_AD': mean_prob,
            'n_slices': len(probs),
            # fraction of this patient's slices that the slice-level model (argmax) called AD
            'frac_slices_AD': sum(p > 0.5 for p in probs) / len(probs),
        })
    return patients


def classification_metrics(rows):
    """Accuracy, per-class recall/precision, macro-F1 and confusion matrix over rows with 'pred' and 'label'
    (patients, or slices for the slice-level comparison)."""
    # reuse the slice-level binary_metrics on class indices (AD = 0 is the positive class)
    preds = torch.tensor([CLASSES.index(r['pred']) for r in rows])
    targets = torch.tensor([CLASSES.index(r['label']) for r in rows])
    return {'acc1': (preds == targets).float().mean().item() * 100, 'num': len(rows),
            **binary_metrics(preds, targets)}


def save_patient_results(patients, metrics, threshold, output_dir):
    """Write patient_predictions.csv (one row per patient) and patient_results.json."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'patient_predictions.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(patients[0].keys()))
        writer.writeheader()
        for p in patients:
            writer.writerow({**p, 'prob_AD': f"{p['prob_AD']:.6f}", 'frac_slices_AD': f"{p['frac_slices_AD']:.3f}"})
    with open(os.path.join(output_dir, 'patient_results.json'), 'w') as f:
        json.dump({'threshold': threshold, **{f'patient_{k}': v for k, v in metrics.items()}}, f, indent=2)
    print(f"Saved patient_predictions.csv and patient_results.json to {output_dir}")


def print_metrics(title, m):
    print(f"\n{title}")
    print(f"  accuracy {m['acc1']:.2f}%   AD recall {m['ad_recall']:.3f}   NC recall {m['nc_recall']:.3f}   "
          f"macro-F1 {m['macro_f1']:.3f}")
    print(f"  TP {m['tp']}  FN {m['fn']}  FP {m['fp']}  TN {m['tn']}")


# ---------------------------------------------------------------------------
# Calibration: expected calibration error (ECE) and reliability diagram
# ---------------------------------------------------------------------------

def calibration(prob_ad, labels, n_bins=10):
    """
    Bin predictions by confidence and compare each bin's average confidence with its accuracy.

    prob_ad: list of P(AD) per item (slice or patient); labels: 'AD' / 'NC'.
    The prediction is the argmax (AD if prob_AD > 0.5), and confidence is the probability of
    the predicted class, max(prob_AD, 1 - prob_AD). With 2 classes it is always >= 0.5, so the
    bins split [0.5, 1] into n_bins equal widths (10 bins -> 0.05 wide).
    Calibration is a property of the probabilities, so it ignores --threshold.
    """
    width = 0.5 / n_bins
    bins = [{'lo': 0.5 + i * width, 'hi': 0.5 + (i + 1) * width, 'n': 0, 'conf_sum': 0.0, 'correct': 0}
            for i in range(n_bins)]
    conf_correct, conf_wrong = [], []
    for p, label in zip(prob_ad, labels):
        pred = 'AD' if p > 0.5 else 'NC'
        conf = max(p, 1 - p)
        correct = pred == label
        b = bins[min(int((conf - 0.5) / width), n_bins - 1)] # conf = 1.0 goes in the last bin
        b['n'] += 1
        b['conf_sum'] += conf
        b['correct'] += correct
        (conf_correct if correct else conf_wrong).append(conf)

    total = len(prob_ad)
    ece = mce = 0.0
    for b in bins:
        if b['n']:
            b['confidence'] = b.pop('conf_sum') / b['n']
            b['accuracy'] = b['correct'] / b['n']
            gap = abs(b['accuracy'] - b['confidence'])
            ece += b['n'] / total * gap # ECE: gap in each bin, weighted by the fraction of items in it
            mce = max(mce, gap)         # MCE: the worst bin's gap
        else:
            b.pop('conf_sum')
            b['confidence'] = b['accuracy'] = None

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        'n': total, 'n_bins': n_bins,
        'ece': ece, 'mce': mce,
        'accuracy': mean([1.0] * len(conf_correct) + [0.0] * len(conf_wrong)),
        'mean_confidence': mean(conf_correct + conf_wrong), # > accuracy = over-confident overall
        # the open research dilemma: confidence should be clearly lower on the predictions that are wrong
        'mean_confidence_correct': mean(conf_correct),
        'mean_confidence_wrong': mean(conf_wrong),
        'bins': bins,
    }


def print_calibration(title, c):
    print(f"\n{title}")
    print(f"  ECE {c['ece']:.4f}   MCE {c['mce']:.4f}   accuracy {c['accuracy']:.3f}   "
          f"mean confidence {c['mean_confidence']:.3f}")
    print(f"  mean confidence when correct {c['mean_confidence_correct']:.3f}, "
          f"when wrong {c['mean_confidence_wrong']:.3f}")


def plot_reliability(calibrations, path):
    """
    Reliability diagram, one column per level (e.g. slices, patients).
    Top: accuracy per confidence bin against the diagonal (perfect calibration); bars below the
    diagonal = over-confident, above = under-confident. Bottom: how many items fall in each bin.
    """
    import matplotlib
    matplotlib.use('Agg') # write to file, no display needed
    import matplotlib.pyplot as plt

    blue, ink, muted, grid = '#2a78d6', '#0b0b0b', '#898781', '#e6e5e1'
    fig, axes = plt.subplots(2, len(calibrations), figsize=(5 * len(calibrations), 6.5),
                             gridspec_kw={'height_ratios': [3, 1]}, squeeze=False)
    for col, (name, c) in enumerate(calibrations.items()):
        top, bottom = axes[0, col], axes[1, col]
        width = 0.5 / c['n_bins']
        filled = [b for b in c['bins'] if b['n']]
        mids = [(b['lo'] + b['hi']) / 2 for b in filled]

        top.plot([0.5, 1], [0.5, 1], '--', color=muted, lw=1.5, label='Perfect calibration')
        top.bar(mids, [b['accuracy'] for b in filled], width=width * 0.9, color=blue, label='Accuracy')
        # hatched gap between each bar and the diagonal = that bin's calibration error
        top.bar(mids, [b['confidence'] - b['accuracy'] for b in filled], bottom=[b['accuracy'] for b in filled],
                width=width * 0.9, fill=False, edgecolor=ink, hatch='//', lw=0, label='Gap (confidence - accuracy)')
        top.set(xlim=(0.5, 1), ylim=(0, 1), ylabel='Accuracy')
        top.set_title(f"{name}  (n = {c['n']:,})\nECE {c['ece']:.3f}, mean confidence {c['mean_confidence']:.2f}, "
                      f"accuracy {c['accuracy']:.2f}", fontsize=10, color=ink)

        bottom.bar(mids, [b['n'] for b in filled], width=width * 0.9, color=muted)
        bottom.set(xlim=(0.5, 1), xlabel='Confidence (probability of the predicted class)', ylabel='Count')

        for ax in (top, bottom):
            ax.grid(axis='y', color=grid, lw=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
    axes[0, 0].legend(loc='upper left', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Selective classification: reject curve (refer the least-confident cases to a specialist)
# ---------------------------------------------------------------------------

def reject_curve(prob_ad, labels, max_reject=0.5, step=0.01, target=0.2):
    """
    Refer the least-confident predictions to a specialist and score the rest.

    For each rejection rate r (0, 1%, ... max_reject), the r least-confident items are rejected and
    accuracy / AD recall / NC recall are measured on the items the model keeps. Compared with:
      - oracle: rejects the wrong predictions first (the best any confidence ranking could do)
      - random: rejecting at random leaves accuracy unchanged (flat line at the overall accuracy)
    The further the model's curve rises above random and towards the oracle, the better its
    confidence separates right from wrong. target = the acceptance criterion's 20% referral limit.
    """
    items = []
    for p, label in zip(prob_ad, labels):
        pred = 'AD' if p > 0.5 else 'NC'
        items.append({'conf': max(p, 1 - p), 'label': label, 'pred': pred, 'correct': pred == label})
    n = len(items)
    by_conf = sorted(items, key=lambda x: x['conf'])            # least confident first = rejected first
    by_oracle = sorted(items, key=lambda x: (x['correct'], x['conf'])) # wrong ones first

    def kept_metrics(ranked, n_reject):
        kept = ranked[n_reject:]
        m = classification_metrics(kept)
        return {'accuracy': m['acc1'] / 100, 'ad_recall': m['ad_recall'], 'nc_recall': m['nc_recall']}

    curve = []
    for i in range(int(round(max_reject / step)) + 1):
        rate = i * step
        n_reject = int(round(rate * n))
        rejected = by_conf[:n_reject]
        curve.append({
            'reject_rate': rate,
            'n_rejected': n_reject,
            # confidence below which a case is referred (the threshold a clinic would set)
            'conf_threshold': by_conf[n_reject]['conf'] if n_reject < n else None,
            **kept_metrics(by_conf, n_reject),
            'oracle_accuracy': kept_metrics(by_oracle, n_reject)['accuracy'],
            # fraction of all the model's errors that ended up with the specialist
            'errors_caught': sum(not x['correct'] for x in rejected) / max(1, sum(not x['correct'] for x in items)),
        })

    # area under the accuracy-vs-rejection curve, normalised to [0, 1] (higher = better ranking)
    area = sum((a['accuracy'] + b['accuracy']) / 2 * step for a, b in zip(curve, curve[1:])) / max_reject
    at_target = min(curve, key=lambda c: abs(c['reject_rate'] - target))
    return {'n': n, 'accuracy_no_reject': curve[0]['accuracy'], 'area_under_curve': area,
            'target_reject_rate': target, 'at_target': at_target, 'curve': curve}


def print_reject(title, r):
    t = r['at_target']
    print(f"\n{title}")
    print(f"  no rejection: accuracy {r['accuracy_no_reject']:.3f}")
    print(f"  reject {t['reject_rate']:.0%} ({t['n_rejected']} least confident, confidence < {t['conf_threshold']:.3f}): "
          f"accuracy {t['accuracy']:.3f} (oracle {t['oracle_accuracy']:.3f}), AD recall {t['ad_recall']:.3f}, "
          f"NC recall {t['nc_recall']:.3f}, {t['errors_caught']:.0%} of errors referred")


def plot_reject(rejects, path):
    """Accuracy of the kept predictions vs the fraction referred to a specialist, one panel per level."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    blue, orange, ink, muted, grid = '#2a78d6', '#eb6834', '#0b0b0b', '#898781', '#e6e5e1'
    fig, axes = plt.subplots(1, len(rejects), figsize=(5 * len(rejects), 4.2), squeeze=False)
    for ax, (name, r) in zip(axes[0], rejects.items()):
        rates = [c['reject_rate'] * 100 for c in r['curve']]
        ax.plot(rates, [c['oracle_accuracy'] for c in r['curve']], '--', color=orange, lw=2,
                label='Oracle (wrong ones first)')
        ax.plot(rates, [c['accuracy'] for c in r['curve']], '-', color=blue, lw=2, label='Model confidence')
        ax.axhline(r['accuracy_no_reject'], color=muted, lw=1.5, ls=':', label='Random rejection')
        t = r['at_target']
        ax.axvline(t['reject_rate'] * 100, color=muted, lw=1)
        ax.plot(t['reject_rate'] * 100, t['accuracy'], 'o', color=blue, ms=8, mec='#fcfcfb', mew=2)
        ax.annotate(f"{t['reject_rate']:.0%} referred: {t['accuracy']:.1%}", (t['reject_rate'] * 100, t['accuracy']),
                    xytext=(8, -14), textcoords='offset points', fontsize=9, color=ink)
        ax.set(xlim=(0, rates[-1]), ylim=(min(r['accuracy_no_reject'], 0.6) - 0.05, 1.01),
               xlabel='Referred to specialist (% least confident)', ylabel='Accuracy of the kept predictions')
        ax.set_title(f"{name}  (n = {r['n']:,})", fontsize=10, color=ink)
        ax.grid(axis='y', color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(loc='lower right', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Example predictions
# ---------------------------------------------------------------------------

def pick_examples(slices, per_row=4, reject_rate=0.2, seed=0):
    """
    Choose example test slices for each outcome, at most one slice per patient per row
    (so a row isn't 4 near-identical slices of the same brain).
      - Correct AD / Correct NC: random (seeded), i.e. typical correct predictions
      - Missed AD (FN) / False alarm (FP): the MOST confident mistakes, the dangerous ones
      - Referred: the least confident slices, which a reject threshold would send to a specialist
    Returns {row name: [slice dicts]}.
    """
    import random
    items = []
    for path, label, p in slices:
        pred = 'AD' if p > 0.5 else 'NC'
        items.append({'path': path, 'patient': patient_id(path), 'label': label, 'pred': pred,
                      'prob_AD': p, 'conf': max(p, 1 - p)})
    # confidence cut-off for referral at reject_rate (same ranking as reject_curve)
    cutoff = sorted(x['conf'] for x in items)[int(reject_rate * len(items))]

    def distinct(candidates):
        seen, out = set(), []
        for x in candidates:
            if x['patient'] not in seen:
                seen.add(x['patient'])
                out.append(x)
            if len(out) == per_row:
                break
        return out

    rng = random.Random(seed)
    def shuffled(xs):
        xs = list(xs)
        rng.shuffle(xs)
        return xs

    # correct and kept (above the referral cut-off), so they are cases the model would decide on its own
    tp = [x for x in items if x['label'] == x['pred'] == 'AD' and x['conf'] >= cutoff]
    tn = [x for x in items if x['label'] == x['pred'] == 'NC' and x['conf'] >= cutoff]
    fn = [x for x in items if x['label'] == 'AD' and x['pred'] == 'NC']
    fp = [x for x in items if x['label'] == 'NC' and x['pred'] == 'AD']
    return {
        'Correct AD': distinct(shuffled(tp)),
        'Correct NC': distinct(shuffled(tn)),
        'Missed AD (most confident)': distinct(sorted(fn, key=lambda x: -x['conf'])),
        'False alarm (most confident)': distinct(sorted(fp, key=lambda x: -x['conf'])),
        'Referred (least confident)': distinct(sorted(items, key=lambda x: x['conf'])),
    }, cutoff


def print_examples(examples, cutoff):
    print(f"\nExample predictions (referral cut-off: confidence < {cutoff:.3f})")
    for row, xs in examples.items():
        print(f"  {row}")
        for x in xs:
            print(f"    {os.path.basename(x['path']):18s} true {x['label']}  pred {x['pred']}  "
                  f"confidence {x['conf']:.3f}  prob_AD {x['prob_AD']:.3f}")


def plot_examples(examples, cutoff, path):
    """Grid of test slices: one row per outcome, each titled with true / predicted label and confidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    ink, secondary = '#0b0b0b', '#52514e'
    n_rows = len(examples)
    n_cols = max(len(xs) for xs in examples.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols + 1.6, 3.0 * n_rows), squeeze=False)
    for r, (row, xs) in enumerate(examples.items()):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c >= len(xs):
                continue
            x = xs[c]
            # the original greyscale slice as stored on disk (before padding / normalisation)
            ax.imshow(Image.open(x['path']).convert('L'), cmap='gray')
            referred = ' (referred)' if x['conf'] < cutoff else ''
            ax.set_title(f"True {x['label']}, predicted {x['pred']}\nconfidence {x['conf']:.2f}{referred}",
                         fontsize=9, color=ink)
            ax.set_xlabel(os.path.basename(x['path']), fontsize=8, color=secondary)
        axes[r, 0].set_ylabel(row, fontsize=10, color=ink, labelpad=10)
    fig.suptitle(f"Example test predictions (referred if confidence < {cutoff:.2f})", fontsize=11, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.98)) # leave room for the suptitle above the first row's titles
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def get_args_parser():
    # reuse main.py's arguments (model, data paths, input size, batch size...) so the model and test
    # transforms are built exactly as in training; imported here, not at the top, because main.py imports run_test
    from main import get_args_parser as main_args_parser
    parser = argparse.ArgumentParser('ConvNeXt test + analysis script', parents=[main_args_parser()])
    parser.add_argument('--predictions', default='', type=str,
                        help='existing test_predictions.csv to analyse (skips running the model)')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='predict AD when the patient-average prob_AD is above this (default: 0.5)')
    parser.add_argument('--n_bins', default=10, type=int,
                        help='confidence bins over [0.5, 1] for ECE / the reliability diagram (default: 10)')
    parser.add_argument('--reject_rate', default=0.2, type=float,
                        help='fraction of least-confident cases referred to a specialist, reported from the '
                             'reject curve (default: 0.2, the acceptance criterion\'s limit)')
    # output_dir defaults to the checkpoint's / CSV's folder (see main below) instead of main.py's results root;
    # device falls back to the CPU, which main.py does not allow
    parser.set_defaults(output_dir=None, device='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


def main(args):
    if args.predictions:
        # mode 2: analyse an existing CSV, no model needed
        csv_path = args.predictions
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(csv_path))
    else:
        # mode 1: load the checkpoint, run it on the test set and write test_predictions.csv
        if not args.resume:
            raise SystemExit("Give either --resume <checkpoint.pth> or --predictions <test_predictions.csv>")
        from main import build_model
        from dataset import build_dataset

        # separate subfolder so the training run's own test_results.json / test_predictions.csv aren't overwritten
        args.output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.resume)), 'predict')
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir

        dataset_test, args.nb_classes = build_dataset(split='test', args=args)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, shuffle=False, # in order, so predictions line up with dataset_test.samples
            batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem)

        model = build_model(args)
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.to(args.device)
        run_test(model, data_loader_test, dataset_test, args.device, args, checkpoint_name=args.resume)
        csv_path = os.path.join(output_dir, 'test_predictions.csv')

    slices = load_predictions(csv_path)
    # slice-level metrics again from the CSV (always argmax / 0.5), for side-by-side comparison
    slice_metrics = classification_metrics([{'pred': 'AD' if p > 0.5 else 'NC', 'label': label} for _, label, p in slices])
    print_metrics(f"Slice level ({len(slices)} slices, threshold 0.5)", slice_metrics)

    patients = patient_predictions(slices, threshold=args.threshold)
    metrics = classification_metrics(patients)
    print_metrics(f"Patient level ({len(patients)} patients, mean prob_AD over slices, threshold {args.threshold})",
                  metrics)
    save_patient_results(patients, metrics, args.threshold, output_dir)

    # calibration: does the model's confidence match how often it is right?
    calibrations = {
        'Slices': calibration([p for _, _, p in slices], [label for _, label, _ in slices], args.n_bins),
        'Patients': calibration([p['prob_AD'] for p in patients], [p['label'] for p in patients], args.n_bins),
    }
    for name, c in calibrations.items():
        print_calibration(f"Calibration, {name.lower()} ({c['n_bins']} bins)", c)
    with open(os.path.join(output_dir, 'calibration.json'), 'w') as f:
        json.dump(calibrations, f, indent=2)
    plot_reliability(calibrations, os.path.join(output_dir, 'reliability_diagram.png'))

    # reject curve: does referring the least-confident cases leave the kept ones more accurate?
    rejects = {
        'Slices': reject_curve([p for _, _, p in slices], [label for _, label, _ in slices], target=args.reject_rate),
        'Patients': reject_curve([p['prob_AD'] for p in patients], [p['label'] for p in patients],
                                 target=args.reject_rate),
    }
    for name, r in rejects.items():
        print_reject(f"Reject curve, {name.lower()}", r)
    with open(os.path.join(output_dir, 'reject_curve.json'), 'w') as f:
        json.dump(rejects, f, indent=2)
    plot_reject(rejects, os.path.join(output_dir, 'reject_curve.png'))

    # example slices for each outcome (needs the test images at the paths stored in the CSV)
    examples, cutoff = pick_examples(slices, reject_rate=args.reject_rate, seed=args.seed)
    print_examples(examples, cutoff)
    if all(os.path.exists(x['path']) for xs in examples.values() for x in xs):
        plot_examples(examples, cutoff, os.path.join(output_dir, 'example_predictions.png'))
    else:
        print("Test images not found at the paths in the CSV; skipping example_predictions.png")


if __name__ == '__main__':
    main(get_args_parser().parse_args())
