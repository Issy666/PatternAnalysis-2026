"""
Post-hoc calibration, decision threshold and reject (referral) rule, all chosen on VALIDATION and then applied
once to test. No retraining.

1. Temperature scaling (Guo et al., 2017): calibrated P(AD) = sigmoid(logit / T), where logit = log(P(AD) / P(NC))
   is the model's log-odds. T > 1 softens over-confident predictions. T is the value that minimises the
   validation negative log-likelihood (NLL). It never changes which class a single slice predicts.
2. Decision threshold: predict AD when the patient's mean calibrated P(AD) > t. t is chosen on validation with a
   rule fixed in advance (--threshold_rule), not by looking at test.
3. Reject rule: refer to a specialist every patient whose P(AD) is within +-b of t (the least decisive cases).
   b is set so that --reject_rate (default 20%) of VALIDATION patients are referred; test may refer more or fewer.

"Patient" = one scan: the 20 slices sharing a filename ID, as in predict.py.

The checkpoint stores its training arguments, so the model and the validation set (path, 2.5D input, input
size) are rebuilt exactly as trained. Test predictions are read from the training run's test_predictions.csv,
which main.py writes from the same checkpoint-best.pth.

Usage:
    python calibrate.py --resume ~/Desktop/results/<run>/checkpoint-best.pth
    python calibrate.py --resume ... --threshold_rule min_ad_recall --min_ad_recall 0.9
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

from predict import CLASSES, calibration, classification_metrics, patient_id, plot_reliability, print_calibration

EPS = 1e-6  # predictions are stored with 6 decimals, so P(AD) is clamped to [1e-6, 1 - 1e-6] before taking logits


def get_args_parser():
    p = argparse.ArgumentParser('Temperature scaling + validation-chosen threshold and reject rule')
    p.add_argument('--resume', required=True, help='checkpoint-best.pth of the run to calibrate')
    p.add_argument('--test_predictions', default='',
                   help="test_predictions.csv from the same checkpoint (default: the run folder's)")
    p.add_argument('--output_dir', default='', help='default: <run>/calibrate')
    p.add_argument('--rerun_val', action='store_true', help='recompute val_predictions.csv even if it exists')
    p.add_argument('--fit_level', default='patient', choices=['patient', 'slice'],
                   help='fit T on patient-level (mean of 20 slices) or slice-level NLL (default: patient, '
                        'the level decisions and referrals are made at)')
    p.add_argument('--threshold_rule', default='macro_f1', choices=['macro_f1', 'min_ad_recall'],
                   help='macro_f1: threshold with the best validation macro-F1 (middle of any tied range); '
                        'min_ad_recall: highest threshold whose validation AD recall >= --min_ad_recall')
    p.add_argument('--min_ad_recall', default=0.9, type=float)
    p.add_argument('--reject_rate', default=0.2, type=float,
                   help='fraction of validation patients to refer; sets the referral band (default: 0.2)')
    p.add_argument('--n_bins', default=10, type=int)
    p.add_argument('--batch_size', default=96, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def predict_validation(checkpoint_path, out_csv, args):
    """Run the checkpoint on its own validation set and write val_predictions.csv (same columns as test)."""
    from main import build_model
    from dataset import build_dataset
    from train import evaluate

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    train_args = checkpoint['args'] # the exact arguments the model was trained with
    train_args.nb_classes = 2
    dataset_val, _ = build_dataset(split='val', args=train_args)
    loader = torch.utils.data.DataLoader(dataset_val, shuffle=False, batch_size=args.batch_size,
                                         num_workers=args.num_workers, pin_memory=True)
    model = build_model(train_args)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    print(f"Running {checkpoint_path} (epoch {checkpoint['epoch']}) on {len(dataset_val)} validation slices "
          f"from {train_args.val_data_path}")
    _, probs, targets = evaluate(loader, model, args.device, use_amp=torch.device(args.device).type == 'cuda',
                                 header='Val:', return_preds=True)

    idx_to_class = {v: k for k, v in dataset_val.class_to_idx.items()}
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label', 'pred'] + [f'prob_{idx_to_class[i]}' for i in range(probs.shape[1])])
        for (path, _), t, p in zip(dataset_val.samples, targets.tolist(), probs.tolist()):
            w.writerow([path, idx_to_class[t], idx_to_class[int(np.argmax(p))]] + [f'{x:.6f}' for x in p])
    print(f"Saved {out_csv}")
    return train_args


def load_slices(csv_path):
    """-> (patient ids, labels 'AD'/'NC', slice log-odds of AD) as numpy arrays."""
    pids, labels, logits = [], [], []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            p = min(max(float(row['prob_AD']), EPS), 1 - EPS)
            pids.append(patient_id(row['path']))
            labels.append(row['label'])
            logits.append(math.log(p / (1 - p)))
    return np.array(pids), np.array(labels), np.array(logits)


def scaled_probs(logits, T):
    return 1 / (1 + np.exp(-logits / T))


def to_patients(pids, labels, slice_probs):
    """Mean P(AD) over each patient's slices -> (patient ids, labels, probs), sorted by patient ID."""
    groups = defaultdict(list)
    lab = {}
    for pid, l, p in zip(pids, labels, slice_probs):
        groups[pid].append(p)
        lab[pid] = l
    ids = sorted(groups)
    return np.array(ids), np.array([lab[i] for i in ids]), np.array([np.mean(groups[i]) for i in ids])


def nll(labels, prob_ad):
    p_true = np.where(labels == 'AD', prob_ad, 1 - prob_ad)
    return float(-np.mean(np.log(np.clip(p_true, EPS, 1))))


# ---------------------------------------------------------------------------
# Fitting on validation
# ---------------------------------------------------------------------------

def fit_temperature(pids, labels, logits, level):
    """Grid search over T (0.25 to 10, log-spaced): the T with the lowest validation NLL."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(10), 400))
    def loss(T):
        p = scaled_probs(logits, T)
        if level == 'patient':
            _, pl, pp = to_patients(pids, labels, p)
            return nll(pl, pp)
        return nll(labels, p)
    losses = [loss(T) for T in grid]
    best = int(np.argmin(losses))
    if best in (0, len(grid) - 1):
        print(f"WARNING: best T {grid[best]:.3f} is at the edge of the search range")
    return float(grid[best]), losses[best], loss(1.0)


def metrics_at(labels, prob_ad, t, keep=None):
    keep = np.ones(len(labels), bool) if keep is None else keep
    rows = [{'label': l, 'pred': 'AD' if p > t else 'NC'} for l, p in zip(labels[keep], prob_ad[keep])]
    return classification_metrics(rows)


def choose_threshold(labels, prob_ad, rule, min_ad_recall):
    """Threshold on validation patients with a rule fixed in advance."""
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [metrics_at(labels, prob_ad, t) for t in candidates]
    if rule == 'macro_f1':
        f1 = np.array([s['macro_f1'] for s in scores])
        tied = candidates[np.isclose(f1, f1.max())]
        # many thresholds tie on a small validation set; take the middle of the best range, not an edge
        return float(tied[len(tied) // 2]), f"best validation macro-F1 {f1.max():.3f} (tied over " \
                                            f"{tied.min():.2f}-{tied.max():.2f}, middle taken)"
    ok = [t for t, s in zip(candidates, scores) if s['ad_recall'] >= min_ad_recall]
    if not ok:
        raise SystemExit(f"No threshold reaches validation AD recall {min_ad_recall}")
    return float(max(ok)), f"highest threshold with validation AD recall >= {min_ad_recall}"


def choose_reject_band(prob_ad, t, rate):
    """Half-width b so that `rate` of validation patients have |P(AD) - t| < b (the least decisive ones)."""
    margins = np.sort(np.abs(prob_ad - t))
    k = int(round(rate * len(margins)))
    return float(margins[k]) if k < len(margins) else float('inf')


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(name, labels, prob_ad, t, slice_labels, slice_probs, n_bins, band=None):
    m = metrics_at(labels, prob_ad, t)
    out = {'setting': name, 'threshold': t, 'n_patients': int(len(labels)),
           'accuracy': m['acc1'] / 100, 'ad_recall': m['ad_recall'], 'nc_recall': m['nc_recall'],
           'macro_f1': m['macro_f1'], 'tp': m['tp'], 'fn': m['fn'], 'fp': m['fp'], 'tn': m['tn'],
           'nll_patient': nll(labels, prob_ad),
           'ece_patient': calibration(list(prob_ad), list(labels), n_bins)['ece'],
           'ece_slice': calibration(list(slice_probs), list(slice_labels), n_bins)['ece']}
    if band is not None:
        refer = np.abs(prob_ad - t) < band
        keep = ~refer
        wrong = np.where(labels == 'AD', prob_ad <= t, prob_ad > t)
        k = metrics_at(labels, prob_ad, t, keep)
        out.update({'reject_band': band, 'referred_rate': float(refer.mean()), 'n_referred': int(refer.sum()),
                    'kept_accuracy': k['acc1'] / 100, 'kept_ad_recall': k['ad_recall'],
                    'kept_nc_recall': k['nc_recall'],
                    'errors_referred': float((wrong & refer).sum() / max(wrong.sum(), 1))})
    return out


def print_row(r):
    s = (f"  {r['setting']:<34} t={r['threshold']:.2f}  acc {r['accuracy']:.3f}  AD {r['ad_recall']:.3f}  "
         f"NC {r['nc_recall']:.3f}  F1 {r['macro_f1']:.3f}  NLL {r['nll_patient']:.3f}  "
         f"ECE pat {r['ece_patient']:.3f} slice {r['ece_slice']:.3f}")
    print(s)
    if 'reject_band' in r:
        print(f"  {'':<34} refer |P(AD)-t| < {r['reject_band']:.3f}: {r['referred_rate']:.1%} referred "
              f"({r['n_referred']}), kept acc {r['kept_accuracy']:.3f}  AD {r['kept_ad_recall']:.3f}  "
              f"NC {r['kept_nc_recall']:.3f}, {r['errors_referred']:.0%} of errors referred")


def main(args):
    run_dir = os.path.dirname(os.path.abspath(args.resume))
    out = args.output_dir or os.path.join(run_dir, 'calibrate')
    os.makedirs(out, exist_ok=True)
    test_csv = args.test_predictions or os.path.join(run_dir, 'test_predictions.csv')
    val_csv = os.path.join(out, 'val_predictions.csv')

    if args.rerun_val or not os.path.exists(val_csv):
        predict_validation(args.resume, val_csv, args)
    else:
        print(f"Using existing {val_csv} (--rerun_val to recompute)")

    v_pid, v_lab, v_logit = load_slices(val_csv)
    t_pid, t_lab, t_logit = load_slices(test_csv)

    # ---- everything below is fitted on VALIDATION only ----
    T, nll_T, nll_1 = fit_temperature(v_pid, v_lab, v_logit, args.fit_level)
    print(f"\nTemperature (fitted on validation, {args.fit_level} NLL): T = {T:.3f}  "
          f"(validation NLL {nll_1:.4f} at T=1 -> {nll_T:.4f})")
    _, vp_lab, vp_prob = to_patients(v_pid, v_lab, scaled_probs(v_logit, T))
    t_star, why = choose_threshold(vp_lab, vp_prob, args.threshold_rule, args.min_ad_recall)
    band = choose_reject_band(vp_prob, t_star, args.reject_rate)
    print(f"Threshold: t = {t_star:.2f} ({why})")
    print(f"Reject rule: refer if |P(AD) - {t_star:.2f}| < {band:.3f} ({args.reject_rate:.0%} of validation patients)")

    # ---- results: validation (for reference) and test (applied once, nothing fitted on it) ----
    results = {'temperature': T, 'fit_level': args.fit_level, 'threshold': t_star, 'threshold_rule': why,
               'reject_band': band, 'reject_rate_target': args.reject_rate,
               'val_predictions': val_csv, 'test_predictions': test_csv}
    reliability = {}
    for split, pid, lab, logit in [('validation', v_pid, v_lab, v_logit), ('test', t_pid, t_lab, t_logit)]:
        raw_slices, cal_slices = scaled_probs(logit, 1.0), scaled_probs(logit, T)
        _, pl, raw_p = to_patients(pid, lab, raw_slices)
        _, _, cal_p = to_patients(pid, lab, cal_slices)
        rows = [summarise('raw (T=1, t=0.5)', pl, raw_p, 0.5, lab, raw_slices, args.n_bins),
                summarise(f'calibrated (T={T:.2f}, t=0.5)', pl, cal_p, 0.5, lab, cal_slices, args.n_bins),
                summarise(f'calibrated + threshold (t={t_star:.2f})', pl, cal_p, t_star, lab, cal_slices,
                          args.n_bins, band=band)]
        print(f"\n{split.capitalize()} ({len(pl)} patients, {len(lab)} slices)")
        for r in rows:
            print_row(r)
        results[split] = rows
        if split == 'test':
            reliability = {'Test patients, raw': calibration(list(raw_p), list(pl), args.n_bins),
                           f'Test patients, T = {T:.2f}': calibration(list(cal_p), list(pl), args.n_bins)}
            with open(os.path.join(out, 'test_patient_predictions_calibrated.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['patient', 'label', 'prob_AD_raw', 'prob_AD_calibrated', 'pred', 'referred'])
                ids, _, _ = to_patients(pid, lab, cal_slices)
                for i, l, rp, cp in zip(ids, pl, raw_p, cal_p):
                    w.writerow([i, l, f'{rp:.6f}', f'{cp:.6f}', 'AD' if cp > t_star else 'NC',
                                int(abs(cp - t_star) < band)])

    for name, c in reliability.items():
        print_calibration(f"Calibration, {name}", c)
    plot_reliability(reliability, os.path.join(out, 'reliability_before_after.png'))
    with open(os.path.join(out, 'calibration_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved calibration_results.json, test_patient_predictions_calibrated.csv to {out}")


if __name__ == '__main__':
    main(get_args_parser().parse_args())
