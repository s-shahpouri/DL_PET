
# from monai.utils import first, set_determinism
# from monai.transforms import (EnsureChannelFirstd, Compose, CropForegroundd, LoadImaged, Orientationd, RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd)
# from monai.networks.nets import UNet
# from monai.networks.layers import Norm
# from monai.inferers import sliding_window_inference
# from monai.data import CacheDataset, DataLoader, Dataset
# from monai.apps import download_and_extract
# from monai.transforms import CenterSpatialCropd
# from monai.transforms import Resized
# import torch
# import matplotlib.pyplot as plt
# import os
# import glob
# import torch.nn as nn
# import json
# from datetime import datetime
# from data_preparation2 import DataHandling 
# from DL_PET.model_maker import create_unet
# import numpy as np
# import nibabel as nib
# from monai.transforms import (
#     Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
#     SpatialPadd, ScaleIntensityd, CenterSpatialCropd
# )

# import math



# data_dir = '/homes/zshahpouri/DLP/ASC-PET-001'
# directory = '/homes/zshahpouri/DLP/Practic/LOG'
# output_dir = '/homes/zshahpouri/DLP/Practic/OUT'


# train_images = sorted(glob.glob(os.path.join(data_dir, "NAC", "*.nii.gz")))
# target_images = sorted(glob.glob(os.path.join(data_dir, "MAC", "*.nii.gz")))

# # data_dicts = [{"image": img, "target": tar} for img in train_images]
# data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]


# # Calculate split sizes
# total_size = len(data_dicts)
# train_size = math.floor(total_size * 0.7)
# val_size = math.floor(total_size * 0.2)
# # The test set gets the remaining data points
# test_size = total_size - train_size - val_size

# # Split the dataset
# train_files = data_dicts[:train_size]
# val_files = data_dicts[train_size:(train_size + val_size)]
# test_files = data_dicts[(train_size + val_size):]

# roi_size = [168, 168, 320]

# train_transforms = Compose(
#     [   LoadImaged(keys=["image", "target"]),
#         EnsureChannelFirstd(keys=["image", "target"]),
#         Spacingd(keys=["image", "target"], pixdim=(4.07, 4.07, 3.00)),
#         SpatialPadd(keys=["image", "target"], spatial_size=(200, 200, 350), mode='constant'),  # Pad to ensure minimum size
#         CenterSpatialCropd(keys=["image", "target"], roi_size=roi_size),  # Crop to ensure exact size
#         ])

# val_transforms = Compose(
#     [   LoadImaged(keys=["image", "target"]),
#         EnsureChannelFirstd(keys=["image", "target"]),
#         Spacingd(keys=["image", "target"], pixdim=(4.07, 4.07, 3.00)),
#         SpatialPadd(keys=["image", "target"], spatial_size=(200, 200, 350), mode='constant'),  # Pad to ensure minimum size
#         CenterSpatialCropd(keys=["image", "target"], roi_size=roi_size),  # Crop to ensure exact size
#         ])

# train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

# val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
# val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

# test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
# test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)


# import os
# import datetime
# from datetime import datetime


# class TrainingLogger:
#     def __init__(self, directory):
#         self.directory = directory
#         self.ensure_directory_exists(self.directory)
#         self.log_file = self.create_log_file()

#     def ensure_directory_exists(self, directory):
#         if not os.path.exists(directory):
#             os.makedirs(directory)

#     def create_log_file(self):
#         filename = f"{self.directory}/log_{self.get_date()}.txt"
#         return open(filename, "w")

#     def get_date(self):

#         s = datetime.now()
#         date = f"{s.month}_{s.day}_{s.hour}_{s.minute}"
#         return date

#     def log(self, message):
#         print(message)
#         self.log_file.write(message + "\n")

#     def close(self):
#         self.log_file.close()


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(32, 64, 128, 256),

#     act=(nn.ReLU6, {"inplace": True}),
#     strides=(2, 2, 2, 2),
#   num_res_units=2,
#     norm=Norm.BATCH,
# ).to(device)

# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# max_epochs = 500
# val_interval = 2
# best_metric = float('inf')
# best_metric_epoch = -1
# epoch_loss_values = []
# metric_values = []
# train_losses = []
# val_losses = []


# class ModelTrainer:
#     def __init__(self, model, train_loader, val_loader, optimizer, loss_function, max_epochs, val_interval, directory):
#         self.model = model
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.optimizer = optimizer
#         self.loss_function = loss_function
#         self.max_epochs = max_epochs
#         self.val_interval = val_interval
#         self.directory = directory
#         self.logger = TrainingLogger(directory)
#         self.best_metric = float('inf')
#         self.best_metric_epoch = -1


#     def log(self):
#         self.logger.log(f"train set: {len(train_files)}" )
#         self.logger.log(f"validation set: {len(val_files)}")
#         self.logger.log(f"max_epochs: {max_epochs}")
#         self.logger.log(f"val_interval: {val_interval}")
#         self.logger.log(f"model.channels: {model.channels}")


#     def train(self):

#         for epoch in range(self.max_epochs):
#             self.logger.log("-" * 10)
#             self.logger.log(f"epoch {epoch + 1}/{self.max_epochs}")

#             self.model.train()
#             epoch_loss = 0
#             step = 0

#             for batch_data in self.train_loader:
#                 step += 1
#                 inputs, targets = batch_data["image"].to(device), batch_data["target"].to(device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.loss_function(outputs, targets)
#                 loss.backward()
#                 self.optimizer.step()

#                 epoch_loss += loss.item()
#                 train_loss_info = f"{step}/{len(self.train_loader.dataset) // self.train_loader.batch_size}, train_loss: {loss.item():.4f}"
#                 self.logger.log(train_loss_info)

#             # Calculate average loss for the epoch
#             epoch_loss /= step
#             self.logger.log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

#             # Validation
#             if (epoch + 1) % self.val_interval == 0:
#                 self.model.eval()
#                 val_loss = 0
#                 with torch.no_grad():
#                     for val_data in self.val_loader:
#                         val_inputs, val_targets = val_data["image"].to(device), val_data["target"].to(device)
#                         val_outputs = self.model(val_inputs)
#                         val_loss += self.loss_function(val_outputs, val_targets).item()

#                     val_loss /= len(self.val_loader)
#                     self.logger.log(f"Validation loss: {val_loss:.4f}")

#                     if val_loss < self.best_metric:
#                         self.best_metric = val_loss
#                         self.best_metric_epoch = epoch + 1
#                         self.save_model()


#         self.logger.close()

#     def save_model(self):
#         model_filename = f"model_{self.logger.get_date()}.pth"
#         torch.save(self.model.state_dict(), os.path.join(self.directory, model_filename))
#         self.logger.log(f"Saved {model_filename} model, best_metric: {self.best_metric:.4f}, epoch: {self.best_metric_epoch} ")


    
# trainer = ModelTrainer(model, train_loader, val_loader, optimizer, loss_function, max_epochs, val_interval,directory)
# trainer.log()
# trainer.train()

import os
import torch
from model_maker import get_network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_network(patch_size = [168, 168, 16], spacing = [4.07, 4.07, 3.00])
model = model.to(device)

import torch
from torchviz import make_dot


# Create a dummy input tensor based on the patch_size
dummy_input = torch.randn(1, 1, *[168, 168, 16]).to(device)

# Perform a forward pass to get the output
# The deep_supervision flag in get_network is set to True, the output will be a list
output = model(dummy_input)[0] if isinstance(model(dummy_input), list) else model(dummy_input)

# Visualize the graph
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the visualization to a file
dot.format = 'png'
dot.render('dynunet_architecture')

# Print the path to the saved visualization file
print('The visualization of DynUNet is saved to "dynunet_architecture.png"')
