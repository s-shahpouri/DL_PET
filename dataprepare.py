
import os
import glob
import random
import warnings
from collections import defaultdict
from monai.transforms import (EnsureChannelFirstd, Compose, LoadImaged, Spacingd)
from monai.data import CacheDataset, DataLoader
from monai.transforms import Resized
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# Path
data_dir = '/home/shahpouriz/Data/Practic/ASC-PET-001'
directory = '/home/shahpouriz/Data/Practic/LOG'


# file names
train_images = sorted(glob.glob(os.path.join(data_dir, "NAC", "*.nii.gz")))
target_images = sorted(glob.glob(os.path.join(data_dir, "MAC", "*.nii.gz")))


# Making dictionary
data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]
patient_list = [os.path.basename(f["image"]) for f in data_dicts]


# Spliting data
total_patients = len(patient_list)
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2


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


# Fixed seed for repeatability
seed_random = 42

center_dict = group_patients_by_center(data_dicts)
train_files, val_files, test_files = [], [], []

for center, patients in center_dict.items():
    center_train, center_val, center_test = split_data_for_center(patients, train_ratio, val_ratio, data_dir, seed=seed_random)
    train_files.extend(center_train)
    val_files.extend(center_val)
    test_files.extend(center_test)

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
        Resized(keys=["image", "target"], spatial_size=(96, 96, 96), mode=('trilinear')),
    ])

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=8)

