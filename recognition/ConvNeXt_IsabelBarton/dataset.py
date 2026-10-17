
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

def build_dataset(is_train, args):
    stats = normalise.load_stats(args.disease)
    transform = normalise.build_transform(is_train, stats,, args)

    if args.data_set == 'ADNI':
        data_dir = args.data_path if is_train else args.val_data_path
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        nb_classes = 2
    elif args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes

def build_transform(is_train, args):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform