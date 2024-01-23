from monai.utils import first, set_determinism
from monai.transforms import (EnsureChannelFirstd, Compose, CropForegroundd, LoadImaged, Orientationd, RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd)
from monai.networks.nets import UNet
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

from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


data_dir = '/home/shahpouriz/Data/Practic/ASC-PET-001'
directory = '/home/shahpouriz/Data/Practic/LOG'



train_images = sorted(glob.glob(os.path.join(data_dir, "NAC", "*.nii.gz")))
target_images = sorted(glob.glob(os.path.join(data_dir, "MAC", "*.nii.gz")))

# data_dicts = [{"image": img, "target": tar} for img in train_images]
data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]

# making  smaller input to save time for practing stage:
train_files, val_files =  data_dicts[:160], data_dicts[-24:]
# train_files, val_files =  data_dicts[:16], data_dicts[-4:]

print(len(train_files))
print(len(val_files))


set_determinism(seed=0)



# crop_size = (180, 180, 312)  # Adjusted based on my data

train_transforms = Compose(
    [   LoadImaged(keys=["image", "target"]),
        EnsureChannelFirstd(keys=["image", "target"]),
        Spacingd(keys=["image", "target"], pixdim=(1.5, 1.5, 2.0)),
        Resized(keys=["image", "target"], spatial_size=(96, 96, 96), mode='trilinear'),
        # CenterSpatialCropd(keys=["image", "target"], roi_size=crop_size, lazy=True),
    ])

val_transforms = Compose(
    [   LoadImaged(keys=["image", "target"]),
        EnsureChannelFirstd(keys=["image", "target"]),
        Spacingd(keys=["image", "target"], pixdim=(1.5, 1.5, 2.0)),
        Resized(keys=["image", "target"], spatial_size=(96, 96, 96), mode=('trilinear')),
        # CenterSpatialCropd(keys=["image", "target"], roi_size=crop_size, lazy=True),

    ])

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=8)



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


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_function, max_epochs, val_interval, directory):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.directory = directory
        self.logger = TrainingLogger(directory)
        self.best_metric = float('inf')
        self.best_metric_epoch = -1


    def log(self):
        self.logger.log(f"train set: {len(train_files)}" )
        self.logger.log(f"validation set: {len(val_files)}")
        self.logger.log(f"max_epochs: {max_epochs}")
        self.logger.log(f"val_interval: {val_interval}")
        self.logger.log(f"model.channels: {model.channels}")


    def train(self):

        for epoch in range(self.max_epochs):
            self.logger.log("-" * 10)
            self.logger.log(f"epoch {epoch + 1}/{self.max_epochs}")

            self.model.train()
            epoch_loss = 0
            step = 0

            for batch_data in self.train_loader:
                step += 1
                inputs, targets = batch_data["image"].to(device), batch_data["target"].to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                train_loss_info = f"{step}/{len(self.train_loader.dataset) // self.train_loader.batch_size}, train_loss: {loss.item():.4f}"
                self.logger.log(train_loss_info)

            # Calculate average loss for the epoch
            epoch_loss /= step
            self.logger.log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # Validation
            if (epoch + 1) % self.val_interval == 0:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_data in self.val_loader:
                        val_inputs, val_targets = val_data["image"].to(device), val_data["target"].to(device)
                        val_outputs = self.model(val_inputs)
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
        self.logger.log(f"Saved {model_filename} model, best_metric: {self.best_metric:.4f}, epoch: {self.best_metric_epoch} ")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64),
    # default=[32, 64, 128, 256, 512, 32] MRI Recon
    act=(nn.ReLU6, {"inplace": True}),
    strides=(2, 2),
  num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

max_epochs = 600
val_interval = 2
best_metric = float('inf')
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
train_losses = []
val_losses = []


torch.backends.cudnn.benchmark = True
trainer = ModelTrainer(model, train_loader, val_loader, optimizer, loss_function, max_epochs, val_interval,directory)
trainer.log()
trainer.train()


