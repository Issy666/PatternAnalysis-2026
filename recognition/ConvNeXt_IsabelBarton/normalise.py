
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from timm.data.random_erasing import RandomErasing
from torchvision import transforms

import numpy as np
from PIL import Image

# ImageNet defaults — used as fallback when stats file not found
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# stored next to this file so the cache is found no matter which directory the script is run from
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent / "stats")


class PadToSquare:
    """
    Pad a PIL image with black borders so it becomes square, keeping the
    original centred. ADNI slices are 256 wide x 240 tall with a pure black
    background, so this adds 8 rows top and bottom rather than stretching
    the brain vertically.
    """

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        if w == h:
            return img
        fill = self.fill if img.mode in ("L", "I", "F") else (self.fill,) * len(img.getbands())
        square = Image.new(img.mode, (side, side), fill)
        square.paste(img, ((side - w) // 2, (side - h) // 2))
        return square

    def __repr__(self):
        return f"{self.__class__.__name__}(fill={self.fill})"

def compute_stats(
    image_paths: List[str],
    disease: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    target_size: int = 224,
) -> Dict[str, List[float]]:
    """
    Compute per-channel mean and std over the training set.

    Parameters
    ----------
    image_paths : list of str
        Paths to training images (from registry).
    disease : str
        Disease tag — used as the cache file key.
    cache_dir : str
        Directory where JSON stats files are stored.
    target_size : int
        All images are padded to square then resized to
        (target_size, target_size) before computing stats, matching what
        the model sees.

    Returns
    -------
    dict with keys 'mean' and 'std', each a list of 3 floats.
    """
    # the cache records how images were preprocessed, so stats computed a different way are recomputed
    preprocess = f"pad_square_resize_{target_size}"

    #have already calculated the mean and std for the training set, so we can cache it to avoid recalculating
    cache_path = Path(cache_dir) / f"{disease}_stats.json"
    if cache_path.exists():
        cached = _load_json(cache_path)
        if cached.get("preprocess") == preprocess:
            print(f"[normalization] Loading cached stats for '{disease}': {cache_path}")
            return cached
        print(f"[normalization] Cached stats in {cache_path} were computed with different "
              f"preprocessing ({cached.get('preprocess', 'unknown')}) — recomputing")

    print(f"[normalization] Computing stats for '{disease}' ({len(image_paths)} images) …")

    # running per-channel sums of x and x^2, accumulated per image (vectorised, single pass)
    n      = 0
    sum_x  = np.zeros(3, dtype=np.float64)
    sum_x2 = np.zeros(3, dtype=np.float64)

    for path in image_paths:
        try:
            arr = _load_rgb_array(path, target_size)   # (H, W, 3) float [0,1]
        except Exception as e:
            print(f"  [WARN] Skipping {path}: {e}")
            continue

        n      += arr.shape[0] * arr.shape[1]
        sum_x  += arr.sum(axis=(0, 1))
        sum_x2 += (arr ** 2).sum(axis=(0, 1))

    if n < 2:
        print("[normalization] Not enough valid images — using ImageNet defaults")
        stats = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    else:
        mean     = sum_x / n
        variance = (sum_x2 - n * mean ** 2) / (n - 1)
        std      = np.sqrt(variance)
        stats = {
            "mean": mean.tolist(),
            "std":  std.tolist(),
            "preprocess": preprocess,
        }

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  mean = {[f'{v:.4f}' for v in stats['mean']]}")
    print(f"  std  = {[f'{v:.4f}' for v in stats['std']]}")
    print(f"  Saved → {cache_path}")
    return stats


def load_stats(
    disease: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    fallback_to_imagenet: bool = True,
) -> Dict[str, List[float]]:
    """
    Load precomputed stats from cache.

    Falls back to ImageNet defaults if no cached file is found
    and fallback_to_imagenet=True.
    """
    cache_path = Path(cache_dir) / f"{disease}_stats.json"
    if cache_path.exists():
        return _load_json(cache_path)

    if fallback_to_imagenet:
        print(
            f"[normalization] No stats found for '{disease}'. "
            "Using ImageNet defaults. Run compute_stats() first."
        )
        return {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}

    raise FileNotFoundError(
        f"Stats file not found: {cache_path}. "
        "Run compute_stats() first."
    )


def get_normalize_transform(stats: Dict[str, List[float]]):
    """
    Return a torchvision.transforms.Normalize object from a stats dict.
    """
    return transforms.Normalize(mean=stats["mean"], std=stats["std"])


def build_transform(is_train,
    stats: Dict[str, List[float]],
    img_size: int = 224,
    args=None,
):
    """
    Return a complete torchvision transform pipeline.

    Parameters
    ----------
    is_train : bool
        If True, returns the training augmentation pipeline (small random
        affine + random erasing). Requires args.
    img_size : int
        Output height and width.

    Both pipelines first pad images to square (PadToSquare) so the brain
    is never stretched.
    """
    if is_train:
        # ADNI-specific instead of the paper's ImageNet pipeline (RandomResizedCrop 8-100% + horizontal flip):
        # AD shows up as SIZE/SHAPE changes (bigger ventricles, smaller hippocampus), so augmentation only mimics
        # small differences in head position and never zooms, stretches or mirrors the brain (see README)
        interpolation = {'bilinear': transforms.InterpolationMode.BILINEAR,
                         'bicubic': transforms.InterpolationMode.BICUBIC,
                         'nearest': transforms.InterpolationMode.NEAREST}[args.train_interpolation]
        return transforms.Compose([
            PadToSquare(),
            transforms.Resize((img_size, img_size), interpolation=interpolation), # no-op at 256 (same as eval)
            transforms.RandomAffine(
                degrees=args.affine_degrees,                               # rotate +-10 deg (head tilt)
                translate=(args.affine_translate, args.affine_translate),  # shift up to +-5% (~13 px at 256)
                scale=(1 - args.affine_scale, 1 + args.affine_scale),      # 0.95-1.05 (head size / scanner FOV)
                interpolation=interpolation,
                fill=0,                                                    # new corners are black, like the background
            ),
            # no horizontal flip: every sagittal slice faces the same way, and a mirrored brain never appears at test
            transforms.ToTensor(),
            transforms.Normalize(mean=stats["mean"], std=stats["std"]),
            # erase one random rectangle (filled with random noise) in 25% of images, after normalisation as in timm
            RandomErasing(probability=args.reprob, mode=args.remode, max_count=args.recount, device='cpu'),
        ])

    t = []
    t.append(PadToSquare())
    # no-op for 256x240 ADNI slices at img_size=256 (padding already gives 256x256)
    t.append(transforms.Resize((img_size, img_size)))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=stats["mean"], std=stats["std"]))

    return transforms.Compose(t)


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _load_rgb_array(path: str, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = PadToSquare()(img.convert("RGB")).resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr   # (H, W, 3)


def _load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)