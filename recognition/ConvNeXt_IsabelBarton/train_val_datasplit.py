#!/usr/bin/env python3
"""
Move a percentage of images from a train folder into a validation folder,
splitting by PATIENT so no patient's slices end up in both sets.

Filenames are expected to look like:  <patientID>_<sliceID>.<ext>
    e.g. 1442100_100.jpeg  -> patient 1442100, slice 100

If the train folder has class subfolders (e.g. train/tumor, train/healthy),
the same structure is recreated in the validation folder, and a patient is
still kept entirely on one side across ALL classes.

Usage:
    python split_train_val.py --train data/train --val data/val --percent 20
    python split_train_val.py --train data/train --val data/val --percent 20 --dry-run
"""

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FILENAME_RE = re.compile(r"^(?P<patient>.+)_(?P<slice>\d+)$")


def parse_args():
    p = argparse.ArgumentParser(description="Patient-level train/val split.")
    p.add_argument("--train", required=True, type=Path, help="Path to train folder")
    p.add_argument("--val", required=True, type=Path, help="Path to validation folder (created if missing)")
    p.add_argument("--percent", required=True, type=float,
                   help="Target percentage of IMAGES to move to validation (e.g. 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--dry-run", action="store_true", help="Show what would happen without moving files")
    return p.parse_args()


def group_by_patient(train_dir: Path):
    """Return {patient_id: [file paths]} and a list of files that didn't match the pattern."""
    patients = defaultdict(list)
    skipped = []
    for f in sorted(train_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        m = FILENAME_RE.match(f.stem)
        if m:
            patients[m.group("patient")].append(f)
        else:
            skipped.append(f)
    return patients, skipped


def choose_val_patients(patients: dict, percent: float, seed: int):
    """
    Shuffle patients, then add them to validation one at a time until the
    image count gets as close as possible to the target. Whole patients only,
    so the final percentage is approximate.
    """
    total = sum(len(v) for v in patients.values())
    target = total * percent / 100.0

    ids = sorted(patients)            # sort first so the seed gives identical results every run
    random.Random(seed).shuffle(ids)

    val_ids, count = [], 0
    for pid in ids:
        n = len(patients[pid])
        # Add this patient only if it gets us closer to the target
        if abs((count + n) - target) <= abs(count - target):
            val_ids.append(pid)
            count += n
        if count >= target:
            break
    return set(val_ids), count, total


def main():
    args = parse_args()

    if not args.train.is_dir():
        sys.exit(f"Train folder not found: {args.train}")
    if not 0 < args.percent < 100:
        sys.exit("--percent must be between 0 and 100")

    if args.val.exists() and any(args.val.rglob("*")):
        sys.exit(f"Validation folder {args.val} is not empty. Empty it first to avoid "
                 f"mixing with a previous split.")

    patients, skipped = group_by_patient(args.train)
    if not patients:
        sys.exit("No matching image files found.")
    if skipped:
        print(f"WARNING: {len(skipped)} file(s) don't match <patient>_<slice> and will stay in train:")
        for f in skipped[:10]:
            print(f"   {f}")
        if len(skipped) > 10:
            print("   ...")

    val_ids, val_count, total = choose_val_patients(patients, args.percent, args.seed)
    if not val_ids:
        sys.exit("Percentage too small: not even one patient fits. Try a larger --percent.")

    print(f"\nPatients total:      {len(patients)}")
    print(f"Patients -> val:     {len(val_ids)}")
    print(f"Images total:        {total}")
    print(f"Images -> val:       {val_count}  ({100 * val_count / total:.2f}% vs requested {args.percent}%)")

    moved = 0
    for pid in sorted(val_ids):
        for src in patients[pid]:
            dest = args.val / src.relative_to(args.train)   # keeps class subfolders
            if args.dry_run:
                print(f"[dry-run] {src} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            moved += 1

    # Leakage check: no patient should now appear in both folders
    if not args.dry_run:
        remaining, _ = group_by_patient(args.train)
        moved_to_val, _ = group_by_patient(args.val)
        overlap = set(remaining) & set(moved_to_val)
        if overlap:
            sys.exit(f"ERROR: patients found in both sets: {sorted(overlap)[:10]}")
        print(f"\nMoved {moved} files. Leakage check passed: no patient is in both train and val.")
    else:
        print(f"\nDry run complete. {moved} files would be moved.")

    # Save the list of validation patients so the split is documented
    if not args.dry_run:
        list_path = args.val / "val_patients.txt"
        list_path.write_text("\n".join(sorted(val_ids)) + "\n")
        print(f"Validation patient IDs written to {list_path}")


if __name__ == "__main__":
    main()