#!/usr/bin/env python3
"""
Split ADNI slices into train / validation / test by ADNI SUBJECT (person), so no person is in more than one split.

Why not split by the filename ID: in <id>_<slice>.jpeg the <id> is the ADNI *image* ID (one scan session), not the
person. meta_data_with_label.json maps each image ID to its original ADNI file, whose name contains the subject ID,
e.g. "ADNI_068_S_0473_MR_..._I932015.nii" -> subject 068_S_0473. The 1,525 scans come from only 680 people, many
scanned at several visits, and the course split puts the same person in train and test.

Subjects are also stratified by scan era: every CN scan is from >= 2017, but AD scans are a mix, and in the course
split every >= 2017 AD scan was in test. Each split here gets the same share of AD people with >= 2017 scans, so
the model sees 2017+ AD scans in training and can't use "2017+ scan" as a shortcut for CN.

Expects every image in <root>/train/<class>/ and empty <root>/validation and <root>/test. Files are MOVED, keeping
class subfolders. A manifest (split_manifest.csv) records every scan's subject, era, sequence and split.

Usage:
    python subject_datasplit.py --root ~/Desktop/data/ADNI_patientleakagefix/AD_NC \
        --meta ~/Desktop/data/ADNI_patientleakagefix/meta_data_with_label.json --val 10 --test 30 --dry-run

AI assistance: the patient-leakage investigation behind this script, and the script itself, were developed with
Claude (Anthropic), then reviewed and run by me; the leakage checks below verify the result on disk (see README,
"Artificial Intelligence Usage Disclosure").
"""

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "validation", "test"]
META_LABEL = {"NC": 0, "AD": 2}  # label coding in meta_data_with_label.json (1 = not in this dataset)


def parse_args():
    """Command-line options: data root, metadata file, split percentages, seed and --dry-run."""
    p = argparse.ArgumentParser(description="Subject-level train/validation/test split for ADNI.")
    p.add_argument("--root", required=True, type=Path, help="Folder containing train/, validation/, test/")
    p.add_argument("--meta", required=True, type=Path, help="meta_data_with_label.json")
    p.add_argument("--val", type=float, default=10, help="Target percentage of SCANS for validation (default 10)")
    p.add_argument("--test", type=float, default=30, help="Target percentage of SCANS for test (default 30)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--dry-run", action="store_true", help="Show the split without moving files")
    return p.parse_args()


def scan_info(raw_path):
    """(subject ID, acquisition year, MRI sequence name) parsed from a scan's original ADNI file name."""
    # "ADNI_T1_3T/ADNI_068_S_0473_MR_Accelerated_Sagittal_MPRAGE__br_raw_20171116173455030_44_S632432_I932015.nii"
    name = raw_path.split("/")[-1]
    subject = re.search(r"ADNI_(\d+_S_\d+)", name).group(1) # site_S_participant, e.g. 068_S_0473 = one person
    # sequence name between "_MR_" and the 17-digit acquisition timestamp (YYYYMMDDhhmmss...), e.g. 20171116...
    m = re.search(r"_MR_(.*?)_?_(?:br_raw_)?(\d{17})", name)
    return subject, int(m.group(2)[:4]), m.group(1).strip("_") # first 4 digits of the timestamp = year


def collect_scans(root, meta):
    """Return {image_id: dict(cls, subject, year, sequence, files)} for everything in root/train."""
    scans = {}
    for cls_dir in sorted(p for p in (root / "train").iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            image_id = f.stem.split("_")[0] # <image ID>_<slice>.jpeg -> image ID (one scan session)
            if image_id not in scans: # first slice of this scan: look up its subject and check its label
                if image_id not in meta:
                    sys.exit(f"Image ID {image_id} ({f}) is not in the metadata file")
                subject, year, sequence = scan_info(meta[image_id]["raw"])
                if meta[image_id]["label"] != META_LABEL.get(cls_dir.name):
                    sys.exit(f"Folder label {cls_dir.name} disagrees with metadata label "
                             f"{meta[image_id]['label']} for image {image_id}")
                scans[image_id] = dict(cls=cls_dir.name, subject=subject, year=year, sequence=sequence, files=[])
            scans[image_id]["files"].append(f)
    return scans


def assign_subjects(scans, targets, seed):
    """
    Group scans by subject and stratify subjects by (class, has a >= 2017 scan). Within each stratum, subjects are
    shuffled and each is given to the split furthest below its target share of that stratum's scans.
    Whole subjects only, so the final percentages are approximate.
    """
    # subject -> all of that person's scans, so a person is always assigned as a whole
    by_subject = defaultdict(list)
    for image_id, s in scans.items():
        by_subject[s["subject"]].append(image_id)

    # (class, has a >= 2017 scan) -> subjects, so each split gets a similar mix of classes and scan eras
    strata = defaultdict(list)
    for subject, ids in by_subject.items():
        classes = {scans[i]["cls"] for i in ids}
        if len(classes) > 1:
            sys.exit(f"Subject {subject} has scans in more than one class: {classes}")
        new_era = any(scans[i]["year"] >= 2017 for i in ids)
        strata[(classes.pop(), new_era)].append(subject)

    rng = random.Random(seed)
    split_of = {}
    for key in sorted(strata):  # sorted so the seed gives identical results every run
        subjects = sorted(strata[key])
        rng.shuffle(subjects)
        total = sum(len(by_subject[s]) for s in subjects)
        count = dict.fromkeys(SPLITS, 0)
        for subject in subjects:
            # deficit = how far each split is below its target number of scans in this stratum
            split = max(SPLITS, key=lambda sp: targets[sp] * total - count[sp])
            split_of[subject] = split
            count[split] += len(by_subject[subject])
    return split_of


def print_summary(scans, split_of):
    """Print scans, slices, people and >= 2017 scans per split and class, and each split's share of the scans."""
    print(f"\n{'split':<11}{'class':<6}{'scans':>6}{'slices':>8}{'people':>8}{'scans >=2017':>14}")
    for split in SPLITS:
        for cls in sorted({s['cls'] for s in scans.values()}):
            ids = [i for i, s in scans.items() if s["cls"] == cls and split_of[s["subject"]] == split]
            print(f"{split:<11}{cls:<6}{len(ids):>6}{sum(len(scans[i]['files']) for i in ids):>8}"
                  f"{len({scans[i]['subject'] for i in ids}):>8}{sum(scans[i]['year'] >= 2017 for i in ids):>14}")
    n = Counter(split_of[s["subject"]] for s in scans.values())
    print("\nShare of scans: " + ", ".join(f"{sp} {100 * n[sp] / len(scans):.1f}%" for sp in SPLITS))


def main():
    """Plan the subject-level split, check it for leakage, move the files, re-check on disk and write the manifest."""
    args = parse_args()
    if args.val <= 0 or args.test <= 0 or args.val + args.test >= 100:
        sys.exit("--val and --test must be positive and add up to less than 100")
    for split in ["validation", "test"]:
        d = args.root / split
        if d.exists() and any(f.is_file() for f in d.rglob("*")):
            sys.exit(f"{d} already contains files. Move everything back into train/ first.")

    meta = json.load(open(args.meta))
    scans = collect_scans(args.root, meta)
    print(f"Found {len(scans)} scans ({sum(len(s['files']) for s in scans.values())} slices) "
          f"from {len({s['subject'] for s in scans.values()})} subjects")
    slices_per_scan = Counter(len(s["files"]) for s in scans.values())
    if set(slices_per_scan) != {20}:
        print(f"WARNING: slices per scan {dict(slices_per_scan)} (expected 20 for every scan)")

    targets = {"validation": args.val / 100, "test": args.test / 100}
    targets["train"] = 1 - targets["validation"] - targets["test"]
    split_of = assign_subjects(scans, targets, args.seed)
    print_summary(scans, split_of)

    # leakage check on the plan: every subject in exactly one split
    subject_splits = defaultdict(set)
    for s in scans.values():
        subject_splits[s["subject"]].add(split_of[s["subject"]])
    assert all(len(v) == 1 for v in subject_splits.values())

    if args.dry_run:
        print("\nDry run: no files moved.")
        return

    # move validation/test scans out of train/ (train scans stay where they are)
    moved = 0
    for s in scans.values():
        split = split_of[s["subject"]]
        if split == "train":
            continue
        for f in s["files"]:
            dest = args.root / split / s["cls"] / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            moved += 1

    # leakage check on the files actually on disk: read every image ID back from each split folder
    on_disk = defaultdict(set)
    for split in SPLITS:
        for f in (args.root / split).rglob("*"):
            if f.suffix.lower() in IMAGE_EXTS:
                on_disk[split].add(scans[f.stem.split("_")[0]]["subject"])
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if on_disk[a] & on_disk[b]:
            sys.exit(f"ERROR: subjects in both {a} and {b}: {sorted(on_disk[a] & on_disk[b])[:10]}")
    print(f"\nMoved {moved} files. Leakage check passed: no ADNI subject is in more than one split.")

    manifest = args.root / "split_manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "subject", "class", "year", "sequence", "n_slices", "split"])
        for image_id, s in sorted(scans.items(), key=lambda kv: (split_of[kv[1]["subject"]], kv[1]["subject"])):
            w.writerow([image_id, s["subject"], s["cls"], s["year"], s["sequence"], len(s["files"]),
                        split_of[s["subject"]]])
    print(f"Split manifest written to {manifest}")


if __name__ == "__main__":
    main()
