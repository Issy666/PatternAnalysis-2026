"""
ADNI datasets: train / validation / test folders of <scanID>_<slice>.jpeg slices in AD/ and NC/ class folders.

References:
https://github.com/facebookresearch/ConvNeXt/blob/main/main.py
https://github.com/MeetRVyas/adni_code/tree/main/docs/assets
https://github.com/hunghuynh1110/Alzheimers-MRI-ConvNext/blob/main/dataset.py
https://github.com/slavaheroes/convnext-for-ad/blob/master/dataloaders/datasets.py
https://github.com/HusnuBarisBaydargil/ConvNext-Medical-Imaging/blob/main/dataset.py
https://pmc.ncbi.nlm.nih.gov/articles/PMC12691013/#ref-list1
https://link.springer.com/chapter/10.1007/978-3-031-14903-0_48
https://www.biorxiv.org/content/10.1101/2025.07.10.664260v1.full
https://www.mdpi.com/2075-4418/15/23/2997#Data_Availability_Statement
https://zenodo.org/records/17084199
"""
import os
from collections import defaultdict
from PIL import Image
from torchvision import datasets
import normalise


def patient_id(path):
    # filenames are <scanID>_<sliceID>.jpeg, e.g. 1003730_100.jpeg -> 1003730. The ID is one scan session (ADNI image ID);
    # the code calls it a "patient", but one person scanned at several visits has several IDs (see subject_datasplit.py)
    return os.path.basename(path).split('_')[0]


class NeighbourSliceFolder(datasets.ImageFolder):
    """
    2.5D input: the 3 input channels are (slice - step, slice, slice + step) from the same scan, instead of one
    greyscale slice copied 3 times, so the model sees some 3D context (e.g. the hippocampus and ventricles across
    neighbouring slices) at no extra cost.

    Files are <scanID>_<slice>.jpeg with 20 consecutive slices per scan. Neighbours are looked up only within the
    same scan in the same split folder, so no data crosses splits. At the first/last slices of a scan the missing
    neighbour is replaced by the nearest slice that exists (edge replication).

    The 3 slices are merged into one RGB PIL image before the transform, so every augmentation (affine, jitter,
    erasing) is applied identically to all 3 channels. samples / targets keep ImageFolder's order, so predictions
    still line up with dataset.samples in predict.py.
    """
    def __init__(self, root, transform=None, step=1):
        super().__init__(root, transform=transform)
        self.step = step
        # (class dir, scan ID) -> {slice index: path}
        scans = defaultdict(dict)
        for path, _ in self.samples:
            scan, slice_idx = os.path.splitext(os.path.basename(path))[0].rsplit('_', 1)
            scans[(os.path.dirname(path), scan)][int(slice_idx)] = path
        self.neighbours = {}  # path -> (path of slice - step, path of slice + step)
        for slices in scans.values():
            order = sorted(slices)
            for k, idx in enumerate(order):
                below = order[max(k - step, 0)]  # clamp at the scan's first/last slice
                above = order[min(k + step, len(order) - 1)]
                self.neighbours[slices[idx]] = (slices[below], slices[above])

    def __getitem__(self, index):
        path, target = self.samples[index]
        below, above = self.neighbours[path]
        channels = [Image.open(p).convert('L') for p in (below, path, above)]
        img = Image.merge('RGB', channels)  # R = slice below, G = this slice, B = slice above
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def build_dataset(split, args):
    """split is 'train', 'val' or 'test'. Classes are numbered alphabetically by ImageFolder: AD = 0, NC = 1."""
    # mean/std always come from the training split only, so val/test never leak into normalisation
    train_paths = [path for path, _ in datasets.ImageFolder(args.data_path).samples]
    stats = normalise.compute_stats(train_paths, 'ADNI', target_size=args.input_size)
    transform = normalise.build_transform(split == 'train', stats, img_size=args.input_size, args=args)

    data_dir = {'train': args.data_path, 'val': args.val_data_path, 'test': args.eval_data_path}[split]
    # getattr: checkpoints saved before 2.5D input existed don't have --neighbour_slices
    if getattr(args, 'neighbour_slices', False):
        return NeighbourSliceFolder(data_dir, transform=transform, step=args.neighbour_step)
    return datasets.ImageFolder(data_dir, transform=transform)
