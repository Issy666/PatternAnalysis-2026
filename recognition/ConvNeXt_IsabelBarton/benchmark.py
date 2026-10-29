"""
Resource profile of trained models (README section 6.3): parameters, weight size, peak training VRAM and training
time (from the run's test_results.json), and inference latency / throughput / memory measured here.

    python benchmark.py ~/Desktop/results/baseline_cnn_25d_aug_e18 ~/Desktop/results/bn_25d_aug_e18 ...

Latency is for one scan = a batch of its 20 slices (2.5D input, 256 x 256), after warm-up; on the GPU with mixed
precision (as in evaluation), on the CPU in float32. Throughput uses batches of 96 slices. Timings vary a little
between runs and with whatever else is using the GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

# the model code lives in recognition/ConvNeXt_IsabelBarton
CODE_DIR = Path(__file__).resolve().parent.parent / 'recognition' / 'ConvNeXt_IsabelBarton'
sys.path.insert(0, str(CODE_DIR))

import torch

from utils import load_trained_model


def time_forward(model, batch_size, device, amp, reps, size=256):
    """Mean milliseconds per forward pass of one batch, after 5 warm-up passes."""
    x = torch.randn(batch_size, 3, size, size, device=device)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp and device == 'cuda'):
        for _ in range(5):
            model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1000


def main(args):
    torch.backends.cudnn.benchmark = True
    rows = {}
    for run in args.runs:
        run = Path(run).expanduser()
        model, train_args = load_trained_model(run / 'checkpoint-best.pth', 'cuda')
        trained = json.load(open(run / 'test_results.json'))
        n_params = sum(p.numel() for p in model.parameters())

        torch.cuda.reset_peak_memory_stats()
        gpu_ms = time_forward(model, 20, 'cuda', True, 50)
        infer_vram = torch.cuda.max_memory_allocated() / 2 ** 20
        throughput = 96 / (time_forward(model, 96, 'cuda', True, 20) / 1000)
        model.cpu()
        cpu_ms = time_forward(model, 20, 'cpu', False, 3)

        rows[run.name] = {
            'model': train_args.model, 'params_M': n_params / 1e6, 'weights_MB': n_params * 4 / 2 ** 20,
            'train_peak_vram_MB': trained.get('max_gpu_mem_mb'), 'train_time': trained.get('training_time_this_run'),
            'gpu_ms_per_scan': gpu_ms, 'gpu_slices_per_s': throughput, 'infer_peak_vram_MB': infer_vram,
            'cpu_ms_per_scan': cpu_ms,
        }

    print(f"\nGPU: {torch.cuda.get_device_name(0)}\n")
    print("| Run | Model | Parameters | Weights | Peak training VRAM | Training time | GPU latency / scan | "
          "GPU throughput | Inference VRAM | CPU latency / scan |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['model']} | {r['params_M']:.2f} M | {r['weights_MB']:.1f} MB | "
              f"{r['train_peak_vram_MB'] / 1024:.1f} GB | {r['train_time']} | {r['gpu_ms_per_scan']:.1f} ms | "
              f"{r['gpu_slices_per_s']:,.0f} slices/s | {r['infer_peak_vram_MB']:.0f} MB | {r['cpu_ms_per_scan']:.0f} ms |")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved {args.out}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        sys.exit('benchmark.py needs a CUDA GPU')
    p = argparse.ArgumentParser('Resource profile of trained runs')
    p.add_argument('runs', nargs='+', help='run folders (each with checkpoint-best.pth and test_results.json)')
    p.add_argument('--out', default='', help='optional JSON file for the numbers')
    main(p.parse_args())
