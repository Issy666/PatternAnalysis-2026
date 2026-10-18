
"""
Sanity check for the ADNI data pipeline (dataset.py + normalise.py + main.py args).

Run from anywhere:
    python test_dataset.py
    python test_dataset.py --data_path /other/train --val_data_path /other/validation ...

The first run computes the training-set mean/std (a minute or two) and caches it in stats/;
later runs load the cached file.
"""

import argparse
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from main import get_args_parser
from dataset import build_dataset
from normalise import PadToSquare

# image counts on disk for /home/isabel/Desktop/data/ADNI/AD_NC
EXPECTED_SIZES = {'train': 18300, 'val': 3222, 'test': 9000}


def check_split(split, args):
    dataset, nb_classes = build_dataset(split=split, args=args)
    print(f"\n[{split}] root = {dataset.root}")
    print(f"[{split}] classes = {dataset.class_to_idx}  num images = {len(dataset)}")

    assert nb_classes == 2, f"expected 2 classes, got {nb_classes}"
    assert dataset.class_to_idx == {'AD': 0, 'NC': 1}, f"unexpected class mapping {dataset.class_to_idx}"
    if len(dataset) != EXPECTED_SIZES[split]:
        print(f"[{split}] WARNING: expected {EXPECTED_SIZES[split]} images (fine if you're using a different copy of the data)")

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    print(f"[{split}] batch images {tuple(images.shape)} {images.dtype}  labels {labels.tolist()}")
    print(f"[{split}] pixel mean {images.mean():.3f}  std {images.std():.3f}  "
          f"min {images.min():.3f}  max {images.max():.3f}")

    assert images.shape == (16, 3, args.input_size, args.input_size), f"bad batch shape {tuple(images.shape)}"
    assert torch.isfinite(images).all(), "batch contains NaN/inf"
    assert set(labels.tolist()) <= {0, 1}, "labels outside {0, 1}"

    first_step = dataset.transform.transforms[0]
    assert isinstance(first_step, PadToSquare), f"first transform is {first_step}, expected PadToSquare"
    return dataset


def check_padding(val_dataset, args):
    """Check images are padded (not stretched) to square, with the original pixels untouched."""
    path = val_dataset.samples[0][0]
    raw = Image.open(path).convert('RGB')
    w, h = raw.size
    side = max(w, h)
    top, left = (side - h) // 2, (side - w) // 2
    print(f"\n[padding] {path}: raw size {w}x{h} -> padded {side}x{side} "
          f"({top} rows top/bottom, {left} cols left/right)")

    padded = np.asarray(PadToSquare()(raw))
    raw_arr = np.asarray(raw)
    assert padded.shape == (side, side, 3), f"padded shape {padded.shape}"
    assert np.array_equal(padded[top:top + h, left:left + w], raw_arr), "original pixels changed by padding"
    border = np.ones((side, side), dtype=bool)
    border[top:top + h, left:left + w] = False
    assert (padded[border] == 0).all(), "padding is not black"

    # end-to-end: in the val/test pipeline the padded rows should come out as exactly (0 - mean) / std
    if side == args.input_size and top > 0:
        tensor = val_dataset.transform(raw)
        norm = val_dataset.transform.transforms[-1]
        expected = torch.tensor([-m / s for m, s in zip(norm.mean, norm.std)]).view(3, 1, 1)
        assert torch.allclose(tensor[:, :top, :], expected.expand(3, top, side), atol=1e-5), \
            "padded rows in the transformed tensor are not the normalised black value"
        assert torch.allclose(tensor[:, -top:, :], expected.expand(3, top, side), atol=1e-5), \
            "padded rows in the transformed tensor are not the normalised black value"
        print(f"[padding] transformed tensor {tuple(tensor.shape)}: padded rows = normalised black "
              f"({expected.flatten()[0]:.3f}), no resizing applied")
    else:
        print(f"[padding] input_size {args.input_size} != {side}, padded image is resized to fit; "
              "skipping exact pixel check")


def main():
    parser = argparse.ArgumentParser('ADNI dataset test', parents=[get_args_parser()])
    args = parser.parse_args()
    args.data_set = 'ADNI'

    failures = 0
    datasets = {}
    for split in ['train', 'val', 'test']:
        try:
            datasets[split] = check_split(split, args)
        except Exception as e:
            failures += 1
            print(f"[{split}] FAILED: {type(e).__name__}: {e}")

    if 'val' in datasets:
        try:
            check_padding(datasets['val'], args)
        except Exception as e:
            failures += 1
            print(f"[padding] FAILED: {type(e).__name__}: {e}")

    print("\nALL CHECKS PASSED" if failures == 0 else f"\n{failures} check(s) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
