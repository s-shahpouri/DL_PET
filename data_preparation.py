import os
import glob
import random
import json
import warnings
from collections import defaultdict
from monai.transforms import (EnsureChannelFirstd, Compose, LoadImaged, Spacingd, Resized)
from monai.data import CacheDataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def load_data(data_dir):
    print("Loading data from:", data_dir)
    train_images = sorted(glob.glob(os.path.join(data_dir, "NAC", "*.nii.gz")))
    target_images = sorted(glob.glob(os.path.join(data_dir, "MAC", "*.nii.gz")))
    data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]
    print(f"Total images loaded: {len(data_dicts)}")
    return data_dicts


def group_patients_by_center(data_dicts):
    '''Gathering patients from different centers'''

    center_dict = defaultdict(list)
    for item in data_dicts:
        filename = os.path.basename(item["image"])
        center = filename.split('_')[1]  #Center name is the 2end part of the filename
        center_dict[center].append(filename)
    return center_dict

def split_data_for_center(center_data, train_ratio, val_ratio, data_dir, seed=None):
    random.seed(seed)
    total_samples = len(center_data)
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)

    random.shuffle(center_data)
    train_set = [{"image": os.path.join(data_dir, "NAC", data), "target": os.path.join(data_dir, "MAC", data)} for data in center_data[:train_samples]]
    val_set = [{"image": os.path.join(data_dir, "NAC", data), "target": os.path.join(data_dir, "MAC", data)} for data in center_data[train_samples:train_samples + val_samples]]
    test_set = [{"image": os.path.join(data_dir, "NAC", data), "target": os.path.join(data_dir, "MAC", data)} for data in center_data[train_samples + val_samples:]]
    return train_set, val_set, test_set

def prepare_dataloaders(train_files, val_files, config):
    train_transforms = Compose(
        [   LoadImaged(keys=["image", "target"]),
            EnsureChannelFirstd(keys=["image", "target"]),
            Spacingd(keys=["image", "target"], pixdim=(1.5, 1.5, 2.0)),
            Resized(keys=["image", "target"], spatial_size=(96, 96, 96), mode='trilinear'),
        ])

    val_transforms = Compose(
        [   LoadImaged(keys=["image", "target"]),
            EnsureChannelFirstd(keys=["image", "target"]),
            Spacingd(keys=["image", "target"], pixdim=(1.5, 1.5, 2.0)),
            Resized(keys=["image", "target"], spatial_size=(96, 96, 96), mode='trilinear'),
        ])

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=config['num_workers'])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=config['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['val_batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader


def prepare_data(config):
    print("Preparing data using config:", config[0])
    data_dir = config["data_dir"]
    data_dicts = load_data(data_dir)
    center_dict = group_patients_by_center(data_dicts)

    train_files, val_files, test_files = [], [], []
    for _, patients in center_dict.items():
        center_train, center_val, center_test = split_data_for_center(patients, config['train_ratio'], config['val_ratio'], data_dir, seed=config['seed_random'])
        train_files.extend(center_train)
        val_files.extend(center_val)
        test_files.extend(center_test)

    train_loader, val_loader = prepare_dataloaders(train_files, val_files, config)

    return train_loader, val_loader

