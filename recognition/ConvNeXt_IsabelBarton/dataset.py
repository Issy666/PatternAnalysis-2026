
import os
from torchvision import datasets, transforms
import normalise

"""
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

def get_stats(args):
    # mean/std always come from the training split only, so val/test never leak into normalisation
    if args.data_set == 'ADNI':
        train_paths = [path for path, _ in datasets.ImageFolder(args.data_path).samples]
        return normalise.compute_stats(train_paths, args.data_set, cache_dir=args.stats_dir,
                                       target_size=args.input_size)
    return normalise.load_stats(args.data_set, cache_dir=args.stats_dir)


def build_dataset(split, args):
    # split is one of 'train', 'val' or 'test'
    is_train = split == 'train'
    stats = get_stats(args)
    transform = normalise.build_transform(is_train, stats, img_size=args.input_size, args=args)

    if args.data_set == 'ADNI':
        data_dir = {'train': args.data_path,
                    'val': args.val_data_path,
                    'test': args.eval_data_path}[split]
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        nb_classes = 2
    elif args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes
