
from monai.utils import first, set_determinism
from monai.transforms import (EnsureChannelFirstd, Compose, CropForegroundd, LoadImaged, Orientationd, RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd)
from monai.networks.nets import DynUNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset
from monai.apps import download_and_extract
from monai.transforms import CenterSpatialCropd
from monai.transforms import Resized
import torch
import matplotlib.pyplot as plt
import os
import glob
import torch.nn as nn
import json
from datetime import datetime
from data_preparation2 import DataHandling 
from DL_PET.model_maker import create_unet
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    SpatialPadd, ScaleIntensityd, CenterSpatialCropd
)
from monai.transforms import NormalizeIntensityd, RandWeightedCropd, RandSpatialCropSamplesd

import random
import math
from collections import defaultdict

data_dir = '/homes/zshahpouri/DLP/ASC-PET-001'
directory = '/homes/zshahpouri/DLP/Practic/LOG'
output_dir = '/homes/zshahpouri/DLP/Practic/OUT'


train_images = sorted(glob.glob(os.path.join(data_dir, "NAC", "*.nii.gz")))
target_images = sorted(glob.glob(os.path.join(data_dir, "MAC", "*.nii.gz")))
data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]


random.seed(42)
# Separate data based on the center
data_by_center = defaultdict(list)
for data in data_dicts:
    center = data["image"].split('/')[-1].split('_')[1]  # Assuming the format is always like /path/Cx_...
    # print(center)
    data_by_center[center].append(data)
# print(len(data_by_center['C5']))
# Initialize test set with all data from C5
test_files = data_by_center.pop('C5', [])

# From each remaining center, randomly select 2 for the test set and ensure they're removed from the training set
for center, files in data_by_center.items():
    if len(files) > 2:  # Ensure there are more than 2 files to choose from
        selected_for_test = random.sample(files, 2)
        test_files.extend(selected_for_test)
        # Remove selected files from the original list
        for selected in selected_for_test:
            files.remove(selected)
    else:
        test_files.extend(files)
        data_by_center[center] = []  # Empty the list as all files have been moved to test

# Recombine the remaining files for training and validation
remaining_files = [file for files in data_by_center.values() for file in files]
# print(len(remaining_files))
random.shuffle(remaining_files)  # Shuffle to ensure random distribution

total_size = len(remaining_files)
train_size = math.floor(total_size * 0.8)

train_files = remaining_files[:train_size]
val_files = remaining_files[train_size:]


patch_size = [168, 168, 16]
spacing = [4.07, 4.07, 3.00]
spatial_size = (168, 168, 320)
train_transforms = Compose(

    [   LoadImaged(keys=["image", "target"]),
        EnsureChannelFirstd(keys=["image", "target"]),
        Spacingd(keys=["image", "target"], pixdim= spacing, mode= 'trilinear'),
        
        SpatialPadd(keys=["image", "target"], spatial_size=spatial_size, mode='constant'),  # Pad to ensure minimum size
        
        RandSpatialCropSamplesd(keys=["image", "target"], roi_size=patch_size, num_samples=20),       
        ])

val_transforms = Compose(
    [   LoadImaged(keys=["image", "target"]),
        EnsureChannelFirstd(keys=["image", "target"]),
        Spacingd(keys=["image", "target"], pixdim=spacing, mode= 'trilinear'),
        SpatialPadd(keys=["image", "target"], spatial_size=spatial_size, mode='constant'),  # Ensure minimum size
        CenterSpatialCropd(keys=["image", "target"], roi_size=spatial_size),  # Ensure uniform size
    ])



train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)



import os
import datetime
from datetime import datetime


class TrainingLogger:
    def __init__(self, directory):
        self.directory = directory
        self.ensure_directory_exists(self.directory)
        self.log_file = self.create_log_file()

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_log_file(self):
        filename = f"{self.directory}/log_{self.get_date()}.txt"
        return open(filename, "w")

    def get_date(self):

        s = datetime.now()
        date = f"{s.month}_{s.day}_{s.hour}_{s.minute}"
        return date

    def log(self, message):
        print(message)
        self.log_file.write(message + "\n")

    def close(self):
        self.log_file.close()



starting_epoch = 0
decay_epoch = 4
learning_rate = 0.001

import torch
from monai.networks.nets import DynUNet
from torch import nn

class DecayLR:
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (self.epochs - self.decay_epochs)
    

import os
import torch
from monai.networks.nets import DynUNet

def get_kernels_strides(patch_size, spacing):
    """
    Adjusted function to use the correct variable names.
    """
    sizes = patch_size  
    spacings = spacing  
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {patch_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_network(patch_size, spacing):
    """
    Initializes the DynUNet with dynamically determined kernels and strides.
    """
    kernels, strides = get_kernels_strides(patch_size, spacing)
    print(kernels)
    print(strides)
    print(len(strides))
    net = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=2,
    )
    return net

# Example usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_network(patch_size, spacing)
model = model.to(device)
model


loss_function = torch.nn.MSELoss()

print('Defining optimizer...')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

max_epochs = 500
val_interval = 2
best_metric = float('inf')
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
train_losses = []
val_losses = []

# Define scheduler
print('Defining scheduler...')
lr_lambda = DecayLR(epochs=max_epochs, offset=0, decay_epochs=decay_epoch).step
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def deep_loss(outputs, target, loss_function, device, weights=None):
    """
    Compute the deep supervision loss for each output feature map.

    Parameters:
    - outputs: Tensor containing all output feature maps, including the final prediction.
    - target: The ground truth tensor.
    - loss_function: The loss function to apply.
    - device: The device on which to perform the calculations.
    - weights: A list of weights for each output's loss. Defaults to equal weighting if None.

    Returns:
    - Weighted average of the computed losses.
    """
    # Unbind the outputs along the first dimension to handle each feature map individually
    output_maps = torch.unbind(outputs, dim=1)
    
    if weights is None:
        # If no weights specified, use equal weights
        weights = [1.0 / len(output_maps)] * len(output_maps)
    elif sum(weights) != 1:
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]

    total_loss = 0.0
    for output, weight in zip(output_maps, weights):
        # Resize target to match the output size if necessary
        resized_target = torch.nn.functional.interpolate(target, size=output.shape[2:], mode='nearest').to(device)
        # Compute loss for the current output
        loss = loss_function(output, resized_target)
        # Accumulate weighted loss
        total_loss += weight * loss

    return total_loss






class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_function, scheduler, max_epochs, val_interval, directory, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler  # Add scheduler to the class initialization
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.directory = directory
        self.device = device  # Assuming device is passed as a parameter
        self.logger = TrainingLogger(directory)
        self.best_metric = float('inf')
        self.best_metric_epoch = -1


    def log(self):
        self.logger.log(f"train set: {len(train_files)}" )
        self.logger.log(f"validation set: {len(val_files)}")
        self.logger.log(f"max_epochs: {max_epochs}")
        self.logger.log(f"val_interval: {val_interval}")
        self.logger.log(f"model.filters: {model.filters}")



    def train(self):
        for epoch in range(self.max_epochs):
            self.logger.log("-" * 10)
            self.logger.log(f"epoch {epoch + 1}/{self.max_epochs}")

            self.model.train()
            epoch_loss = 0
            step = 0

            for batch_data in self.train_loader:
                step += 1
                inputs, targets = batch_data["image"].to(self.device), batch_data["target"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Check if deep supervision is used
                if isinstance(outputs, tuple) or (outputs.dim() > targets.dim()):
                    # Outputs from deep supervision
                    loss = deep_loss(outputs, targets, loss_function, device)
                else:
                    # Standard output handling
                    outputs = torch.squeeze(outputs)
                    targets = torch.squeeze(targets, dim=1)  # Adjust for channel dimension if necessary
                    loss = loss_function(outputs, targets)
                
# # L1 Regularization
#         l1_reg = torch.tensor(0., requires_grad=True).to(device)
#         for name, param in model.named_parameters():
#             l1_reg = l1_reg + torch.norm(param, 1)
        
#         loss += lambda_reg * l1_reg


                # loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.logger.log(f"{step}/{len(self.train_loader.dataset) // self.train_loader.batch_size}, train_loss: {loss.item():.4f}")

            epoch_loss /= step
            self.logger.log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # Step the scheduler here, after the training phase and before the validation phase
            self.scheduler.step()
            self.logger.log(f"current lr: {self.scheduler.get_last_lr()[0]}")

            # Validation logic remains largely the same
            if (epoch + 1) % self.val_interval == 0:
                self.model.eval()
                val_loss = 0
                roi_size = (168, 168, 32)
                sw_batch_size = 16
                
                with torch.no_grad():
                    for val_data in self.val_loader:
                        val_inputs, val_targets = val_data["image"].to(self.device), val_data["target"].to(self.device)

                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                        val_loss += self.loss_function(val_outputs, val_targets).item()

                val_loss /= len(self.val_loader)
                self.logger.log(f"Validation loss: {val_loss:.4f}")

                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    self.best_metric_epoch = epoch + 1
                    self.save_model()

        self.logger.close()

    def save_model(self):
        model_filename = f"model_{self.logger.get_date()}.pth"
        torch.save(self.model.state_dict(), os.path.join(self.directory, model_filename))
        self.logger.log(f"Saved {model_filename} model, best_metric: {self.best_metric:.4f}, epoch: {self.best_metric_epoch}")


    
trainer = ModelTrainer(model, train_loader, val_loader, optimizer, loss_function, scheduler, max_epochs, val_interval,directory, device)
trainer.log()
trainer.train()


