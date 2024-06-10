import glob
import os
import random
import math
from collections import defaultdict
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, SpatialPadd, RandSpatialCropSamplesd, CenterSpatialCropd
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import NormalizeIntensityd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np

class DataHandling:
    def __init__(self, data_dir, train_mode="NAC", target_mode="MAC", external_center='C5'):
        self.data_dir = data_dir
        self.train_mode = train_mode
        self.target_mode = target_mode
        self.external_center = external_center
        self._load_data()
        
    def _load_data(self):
        train_images = sorted(glob.glob(os.path.join(self.data_dir, self.train_mode, "*.nii.gz")))
        target_images = sorted(glob.glob(os.path.join(self.data_dir, self.target_mode, "*.nii.gz")))
        data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]

        random.seed(42)
        data_by_center = defaultdict(list)
        for data in data_dicts:
            center = data["image"].split('/')[-1].split('_')[1]
            data_by_center[center].append(data)

        test_files = data_by_center.pop(self.external_center, [])
        
        for center, files in data_by_center.items():
            if len(files) > 2:
                selected_for_test = random.sample(files, 2)
                test_files.extend(selected_for_test)
                for selected in selected_for_test:
                    files.remove(selected)
            else:
                test_files.extend(files)
                data_by_center[center] = []

        remaining_files = [file for files in data_by_center.values() for file in files]
        random.shuffle(remaining_files)

        total_size = len(remaining_files)
        train_size = math.floor(total_size * 0.8)

        self.train_files = remaining_files[:train_size]
        self.val_files = remaining_files[train_size:]
        self.test_files = test_files

    def get_data_split(self, split_name):
        if split_name == 'train':
            return self.train_files
        elif split_name == 'val':
            return self.val_files
        elif split_name == 'test':
            return self.test_files
        else:
            raise ValueError("Invalid split name. Choose among 'train', 'val', or 'test'.")


class ExtrenalRadioSetSetHandling:
    def __init__(self, data_dir, train_mode="NAC", target_mode="MAC"):
        self.data_dir = data_dir
        self.train_mode = train_mode
        self.target_mode = target_mode
        self.data_dicts = []

        self._load_data()
        
    def _load_data(self):
        train_images = sorted(glob.glob(os.path.join(self.data_dir, self.train_mode, "*.nii.gz")))
        target_images = sorted(glob.glob(os.path.join(self.data_dir, self.target_mode, "*.nii.gz")))
        self.data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]

    def get_data(self):
            return self.data_dicts



class ExternalRadioSetHandling:
    def __init__(self, data_dir, train_mode="NAC", target_mode="MAC", test_ratio=None, random_seed=42):
        self.data_dir = data_dir
        self.train_mode = train_mode
        self.target_mode = target_mode
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.data_dicts = []

        self._load_data()
        
    def _load_data(self):
        train_images = sorted(glob.glob(os.path.join(self.data_dir, self.train_mode, "*.nii.gz")))
        target_images = sorted(glob.glob(os.path.join(self.data_dir, self.target_mode, "*.nii.gz")))
        self.data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]

    def _split_data(self, data, split_ratio):
        np.random.seed(self.random_seed)
        np.random.shuffle(data)
        split_index = int(len(data) * split_ratio)
        return data[:split_index], data[split_index:]

    def get_split_data(self):
        # Shuffle data with a fixed random seed for reproducibility
        np.random.seed(self.random_seed)
        np.random.shuffle(self.data_dicts)
        
        if self.test_ratio is None:  # Default to 70-15-15 split
            train_ratio = 0.70
            val_ratio = 0.15
        else:  # Custom split based on the specified test ratio
            test_ratio = self.test_ratio
            remaining = 1 - test_ratio
            train_ratio = remaining * 0.8
            val_ratio = remaining * 0.2

        # Calculate the number of samples for each set
        num_data = len(self.data_dicts)
        num_train = int(num_data * train_ratio)
        num_val = int(num_data * val_ratio)

        # Split the data
        train_data = self.data_dicts[:num_train]
        val_data = self.data_dicts[num_train:num_train + num_val]
        test_data = self.data_dicts[num_train + num_val:]

        return train_data, val_data, test_data




from monai.transforms import RandAffined, RandGaussianNoised

class LoaderFactory:
    def __init__(self, train_files=None, val_files=None, test_files=None,
                 patch_size=[168, 168, 16], spacing=[4.07, 4.07, 3.00],
                 spatial_size=(168, 168, 320), normalize=False):
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.patch_size = patch_size
        self.spacing = spacing
        self.spatial_size = spatial_size
        self.normalize = normalize

        # Define a list of common preprocessing steps
        common_transforms = [
            LoadImaged(keys=["image", "target"]),
            EnsureChannelFirstd(keys=["image", "target"]),
            Spacingd(keys=["image", "target"], pixdim=self.spacing, mode='trilinear'),
            SpatialPadd(keys=["image", "target"], spatial_size=self.spatial_size, mode='constant'),
        ]

        # Optionally add normalization to the transformation pipeline
        if self.normalize == "minmax":
            common_transforms.append(NormalizeIntensityd(keys=["image", "target"]))
        elif self.normalize == "suvscale":
            common_transforms.append(ScaleIntensity(keys=["target"]))

        self.train_transforms = Compose(common_transforms + [
            RandAffined(keys=["image", "target"], prob=0.5, rotate_range=(0, 0, np.pi/15)),

            RandSpatialCropSamplesd(keys=["image", "target"], roi_size=self.patch_size, num_samples=40),
        ])
        
        self.val_transforms = Compose(common_transforms + [
            CenterSpatialCropd(keys=["image", "target"], roi_size=self.spatial_size),
        ])

        self.test_transforms = Compose(common_transforms + [
            CenterSpatialCropd(keys=["image", "target"], roi_size=self.spatial_size),
        ])

    def get_test_transforms(self):
        return self.test_transforms
    
    def get_loader(self, dataset_type="train", batch_size=4, num_workers=2, shuffle=True):
        data_files = None
        transform = None
        
        if dataset_type == "train":
            data_files = self.train_files
            transform = self.train_transforms
        elif dataset_type == "val":
            data_files = self.val_files
            transform = self.val_transforms
        elif dataset_type == "test":
            data_files = self.test_files
            transform = self.test_transforms
        else:
            raise ValueError(f"No files provided or unknown dataset type: {dataset_type}")
        
        if data_files is not None:
            ds = Dataset(data=data_files, transform=transform) if dataset_type != "train" else CacheDataset(data=data_files, transform=transform, cache_rate=1.0, num_workers=num_workers)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class ScaleIntensity:
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key] / 50  # Divide by this for SUV scaling (choosed based on experiment)
        return data
    




from monai.transforms import MapTransform
class ClampNegative(MapTransform):
    """
    A MONAI transform that sets negative pixel values to zero within a dictionary format.
    This is useful for ensuring that the output predictions do not have negative values.
    Operates on all specified keys.
    """
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        for key in self.keys:
            d = data[key]
            # Find negative values
            negative_values = d[d < 0]
            if len(negative_values) > 0:  # Check if there are any negative values
                min_negative = np.min(negative_values)
                print(f"Minimum negative value in {key}: {min_negative}")
            else:
                print(f"No negative values in {key}")
            # Clamp negative values to 0
            d[d < 0] = 0
            data[key] = d
        return data

    