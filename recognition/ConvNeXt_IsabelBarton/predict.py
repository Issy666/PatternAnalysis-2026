import csv
import json
import os

import torch

from train import evaluate


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