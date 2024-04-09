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
        if self.normalize:
            common_transforms.append(NormalizeIntensityd(keys=["image", "target"]))

        self.train_transforms = Compose(common_transforms + [
            RandSpatialCropSamplesd(keys=["image", "target"], roi_size=self.patch_size, num_samples=20),
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


def visualize_axial_slice(data_loader, slice_index):
    # Manually iterate through the DataLoader to fetch the first batch
    for data_batch in data_loader:
        break  # Only need the first batch
    
    # Extract the image and target tensors from the batch
    image, target = data_batch["image"][0][0], data_batch["target"][0][0]  # Assuming batch size of 1
    
    # Determine global min and max for a unified color scale
    vmin = min(image[:, :, slice_index].min(), target[:, :, slice_index].min())
    vmax = max(image[:, :, slice_index].max(), target[:, :, slice_index].max())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(f"Data Check/Axial slice {slice_index}")
    
    img_plot = axes[0].imshow(np.rot90(np.flip(image[:, :, slice_index], axis=1)), cmap="jet")
    axes[0].set_title("Image Slice")
    
    tgt_plot = axes[1].imshow(np.rot90(np.flip(target[:, :, slice_index], axis=1)), cmap="jet")
    axes[1].set_title("Target Slice")
   
    
    # Add a single colorbar 
    fig.colorbar(img_plot, ax=axes, fraction=0.021, pad=0.04)
    
    plt.show()


def visualize_axial_slice2(data_loader, slice_index):
    # Manually iterate through the DataLoader to fetch the first batch
    for data_batch in data_loader:
        break  # Only need the first batch
    
    # Extract the image and target tensors from the batch
    image, target = data_batch["image"][0][0], data_batch["target"][0][0]  # Assuming batch size of 1
    
    # # Determine global min and max for a unified color scale
    # vmin = min(image[:, :, slice_index].min(), target[:, :, slice_index].min())
    # vmax = max(image[:, :, slice_index].max(), target[:, :, slice_index].max())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(f"Data Check/Axial slice {slice_index}")
    
    img_plot = axes[0].imshow(np.rot90(np.flip(image[:, slice_index, :], axis=1), k=3), cmap="jet")
    axes[0].set_title("Image Slice")
    
    tgt_plot = axes[1].imshow(np.rot90(np.flip(target[:, slice_index, :], axis=1), k=3), cmap="jet")
    axes[1].set_title("Target Slice")
   
    
    # Add a single colorbar 
    fig.colorbar(img_plot, ax=axes, fraction=0.021, pad=0.04)
    
    plt.show()


def visualize_coronal_slice(data, predict, n, title, cm , Norm = False):


    fig, axes = plt.subplots(1, 3, figsize=(12, 6))  # Adjusted for three plots and one colorbar

    titles = ["Input", "Ground_truth", title]
    slices = [
        np.rot90(data["image"][0, 0, :, n, :]),
        np.rot90(data["target"][0, 0, :, n, :]),
        np.rot90(predict.detach().cpu()[0, 0, :, n, :])
    ]

    # Display the images
    images = []
    for ax, slice, title in zip(axes, slices, titles):  # Leave the last axes for the colorbar
        img = ax.imshow(slice, cmap=cm)
        images.append(img)
        ax.set_title(title)
        ax.axis('off')

    if Norm == True:
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

    fig.colorbar(images[0], ax=axes, orientation='vertical', fraction=0.025, pad=0.04)

    # Make sure the aspect ratio is equal to make the colorbar align well
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()



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

    