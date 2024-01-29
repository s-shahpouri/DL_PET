import os
import glob
import random
import json
import warnings
from collections import defaultdict
from monai.transforms import (EnsureChannelFirstd, Compose, LoadImaged, Spacingd, Resized)
from monai.data import CacheDataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

class DataHandling:
    def __init__(self, config):
        self.config = config
        self.data_dir = config["data_dir"]
        self.train_ratio = config['train_ratio']
        self.val_ratio = config['val_ratio']
        self.seed_random = config['seed_random']

    def load_data(self):
        print("Loading data from:", self.data_dir)
        train_images = sorted(glob.glob(os.path.join(self.data_dir, "NAC", "*.nii.gz")))
        target_images = sorted(glob.glob(os.path.join(self.data_dir, "MAC", "*.nii.gz")))
        data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]
        print(f"Total images loaded: {len(data_dicts)}")
        return data_dicts

        
    def group_patients_by_center(self, data_dicts):
        '''Gathering patients from different centers'''

        center_dict = defaultdict(list)
        for item in data_dicts:
            filename = os.path.basename(item["image"])
            center = filename.split('_')[1]  #Center name is the 2end part of the filename
            center_dict[center].append(filename)
        return center_dict

    def split_data_for_center(self, center_data, seed=None):
        random.seed(seed if seed is not None else self.seed_random)
        total_samples = len(center_data)
        train_samples = int(total_samples * self.train_ratio)
        val_samples = int(total_samples * self.val_ratio)

        random.shuffle(center_data)
        train_set = [{"image": os.path.join(self.data_dir, "NAC", data), "target": os.path.join(self.data_dir, "MAC", data)} for data in center_data[:train_samples]]
        val_set = [{"image": os.path.join(self.data_dir, "NAC", data), "target": os.path.join(self.data_dir, "MAC", data)} for data in center_data[train_samples:train_samples + val_samples]]
        test_set = [{"image": os.path.join(self.data_dir, "NAC", data), "target": os.path.join(self.data_dir, "MAC", data)} for data in center_data[train_samples + val_samples:]]
        return train_set, val_set, test_set

    def prepare_dataloaders(self, train_files, val_files, test_files, config, loaders_to_prepare):
        loaders = {}


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

        if "train" in loaders_to_prepare:
            train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=config['num_workers'])
            train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
            loaders["train"] = train_loader


        if "val" in loaders_to_prepare:

            val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=config['num_workers'])
            val_loader = DataLoader(val_ds, batch_size=config['val_batch_size'], shuffle=False, num_workers=config['num_workers'])
            loaders["val"] = val_loader


        if "test" in loaders_to_prepare:
            test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=config['num_workers'])
            test_loader = DataLoader(test_ds, batch_size=config['val_batch_size'], shuffle=False, num_workers=config['num_workers'])
            loaders["test"] = test_loader




        return loaders


    def prepare_data(self, loaders_to_prepare=["train", "val", "test"]):
        data_dicts = self.load_data()
        center_dict = self.group_patients_by_center(data_dicts)

        train_files, val_files, test_files = [], [], []
        for _, patients in center_dict.items():
            center_train, center_val, center_test = self.split_data_for_center(patients)
            train_files.extend(center_train)
            val_files.extend(center_val)
            test_files.extend(center_test)

        # Pass self.config to prepare_dataloaders
        loaders = self.prepare_dataloaders(train_files, val_files, test_files, self.config, loaders_to_prepare)
        
        return loaders, val_files, test_files

