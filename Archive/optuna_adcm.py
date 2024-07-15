
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
import numpy as np
import nibabel as nib

import math
import os
import glob
import torch
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from functools import partial
from monai.networks.nets import DynUNet
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, SpatialPadd, RandCropByPosNegLabeld, CenterSpatialCropd)
import random
import math
from collections import defaultdict
import os
import torch
from src.model_maker import get_network
from DL_PET.src.data_preparation import DataHandling 
from DL_PET.src.data_preparation import LoaderFactory


config_file = '/students/2023-2024/master/Shahpouri/DL_PET/config.json'

with open(config_file, 'r') as f:
    config = json.load(f)

log_dir = config["log_dir"]

fdg_data_dir = config["fdg_data_dir"]
fdg_output_dir = config['fdg_output_dir']

ga_output_dir = config["ga_output_dir"]
ga_data_dir = config["ga_data_dir"]

artifact_dir = config["artifacts"]
artifact_output_dir = config ["artifact_output_dir"]

artifact_repeated_dir = config["artifact_repeated_dir"]
artifacts_repeated_output_dir = config["artifacts_repeated_output_dir"]



# data_handler = DataHandling(ga_data_dir, train_mode="NAC", target_mode="ADCM")

# train_files = data_handler.get_data_split('train')
# val_files = data_handler.get_data_split('val')
# test_files = data_handler.get_data_split('test')
# print(len(train_files))
# print(len(val_files))
# print(len(test_files))





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_network(patch_size = [168, 168, 16], spacing = [4.07, 4.07, 3.00])

model = model.to(device)

loss_function = torch.nn.MSELoss()
best_metric = float('inf')
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
train_losses = []
val_losses = []

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

import torch
import json
import optuna
from model_maker import get_network
from DL_PET.src.data_preparation import DataHandling, LoaderFactory
def add_activation_before_output(model, activation_fn):
    """
    Adds an activation function just before the output of the network.
    """
    # Replace the last conv layer with a sequential layer that has conv followed by activation
    old_output_conv = model.output_block.conv.conv
    new_output_block = nn.Sequential(
        old_output_conv,
        activation_fn
    )
    model.output_block.conv.conv = new_output_block

def objective(trial, data_handler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_network(patch_size=[168, 168, 16], spacing=[4.07, 4.07, 3.00])
    add_activation_before_output(model, nn.ReLU(inplace=True))

    model = model.to(device)

    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 20, 30, 40])
    # sw_batch_size = trial.suggest_int('sw_batch_size', 4, 128, step=4)
    num_samples = trial.suggest_int('num_samples', 10, 40, step=4)

    loader_factory = LoaderFactory(
        train_files=data_handler.get_data_split('train'),
        val_files=data_handler.get_data_split('val'),
        test_files=data_handler.get_data_split('test'),
        num_samples = num_samples,
        patch_size=[168, 168, 16],
        spacing=[4.07, 4.07, 3.00],
        spatial_size=(168, 168, 320),
        normalize="suvscale"
    )
    
    train_loader = loader_factory.get_loader('train', batch_size=batch_size, num_workers=2, shuffle=True)
    val_loader = loader_factory.get_loader('val', batch_size=1, num_workers=2, shuffle=False)
    
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()
    max_epochs = 4

    # Training and validation logic here
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, targets = batch_data["image"].to(device), batch_data["target"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = deep_loss(outputs, targets, loss_function, device)
            print(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_data in val_loader:
                val_inputs, val_targets = val_data["image"].to(device), val_data["target"].to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_function(val_outputs, val_targets).item()
        val_loss /= len(val_loader)
        print(val_loss)

    return val_loss

# Loading config and setting up data handling
config_file = '/students/2023-2024/master/Shahpouri/DL_PET/config.json'
with open(config_file, 'r') as f:
    config = json.load(f)

data_handler = DataHandling(config["ga_data_dir"], train_mode="NAC", target_mode="ADCM")

# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, data_handler), n_trials=100)
print("Best trial:", study.best_trial.number)
print("Best value (validation loss):", study.best_value)
print("Best hyperparameters:", study.best_params)
# Saving study results to DataFrame and then to CSV
df = study.trials_dataframe()
df.to_csv('optuna_study_results.csv', index=False)
