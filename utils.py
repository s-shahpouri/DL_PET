import os
import random
from collections import defaultdict
import nibabel as nib
import torch


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


def find_last_best_model(log_filepath):
    last_saved_model = None
    best_metric = None
    epoch = None
    with open(log_filepath, 'r') as file:
        for line in file:
            if "Saved" in line and ".pth" in line:
                parts = line.split(',')
                last_saved_model = parts[0].split()[1]  # Extract model filename
                best_metric = float(parts[1].split(': ')[1])  # Extract best metric
                epoch = int(parts[2].split(': ')[1])  # Extract epoch number
    return last_saved_model, best_metric, epoch



def find_model_info(log_filepath, model_filename):
    metric = None
    epoch = None
    with open(log_filepath, 'r') as file:
        for line in file:
            if model_filename in line:
                parts = line.split(',')
                metric = float(parts[1].split(': ')[1])  # Extract best metric
                epoch = int(parts[2].split(': ')[1])  # Extract epoch number
                return model_filename, metric, epoch
    return model_filename, metric, epoch


# # Info about the data
# import nibabel as nib
# import os

# def get_nifti_info(nifti_file):
#     """Returns the shape and size of the NIfTI file."""
#     nifti_img = nib.load(nifti_file)
#     data = nifti_img.get_fdata()
#     shape = data.shape
#     size = data.size
#     return shape, size

# # Assuming data_dicts is defined as shown above
# val_files = data_dicts

# for file_dict in val_files:
#     image_path = file_dict["image"]
#     target_path = file_dict["target"]

#     image_shape, image_size = get_nifti_info(image_path)
#     target_shape, target_size = get_nifti_info(target_path)

#     print(f"Image: {os.path.basename(image_path)}, Shape: {image_shape}, Size: {image_size}")
#     print(f"Target: {os.path.basename(target_path)}, Shape: {target_shape}, Size: {target_size}")



# def get_nifti_header(file_path):
#     nifti_img = nib.load(file_path)
#     header_info = nifti_img.header
#     return header_info
# # Dictionary to hold header information of test files before transformations
# test_headers_before = {}

# for file_dict in test_files:
#     image_path = file_dict["image"]
#     target_path = file_dict["target"]
#     image_header = get_nifti_header(image_path)
#     target_header = get_nifti_header(target_path)
#     # Extracting filename without extension for key
#     file_key = os.path.splitext(os.path.basename(image_path))[0]
#     test_headers_before[file_key] = {"image_header": image_header, "target_header": target_header}


# def save_nifti(data, filename, affine):
#     """Save the data as a NIfTI file with the provided affine matrix."""
#     nifti_img = nib.Nifti1Image(data, affine)
#     nib.save(nifti_img, filename)

# def save_output_with_affine(test_data, model, output_dir, file_names, affine_matrices):
#     """Save model output for test data to NIfTI files, preserving affine matrices."""
#     model.eval()
#     with torch.no_grad():
#         test_outputs = model(test_data["image"].to(device))

#     # Loop over each item in the batch and save outputs with the corresponding affine matrix
#     for i in range(test_outputs.shape[0]):  # Adjust based on your model's output shape
#         output_data = test_outputs[i, 0, :, :, :].detach().cpu().numpy()  # Assuming single-channel output
#         output_file_path = os.path.join(output_dir, f"DL_{file_names[i]}.gz")
#         affine = affine_matrices[file_names[i]]
#         save_nifti(output_data, output_file_path, affine)


def find_last_saved_model(log_filepath):
    last_saved_model = None
    best_metric = None
    epoch = None
    with open(log_filepath, 'r') as file:
        for line in file:
            if "Saved" in line and ".pth" in line:
                parts = line.split(',')
                last_saved_model = parts[0].split()[1]  # Extract model filename
                best_metric = float(parts[1].split(': ')[1])  # Extract best metric
                epoch = int(parts[2].split(': ')[1])  # Extract epoch number
    return last_saved_model, best_metric, epoch


# Function to parse the loss values from the log file
def parse_loss_values(log_filepath):
    train_losses = []
    val_losses = []
    with open(log_filepath, 'r') as file:
        for line in file:
            if 'average loss:' in line:
                loss_value = float(line.split(': ')[-1])
                train_losses.append(loss_value)
            if 'Validation loss:' in line:
                val_loss_value = float(line.split(': ')[-1])
                val_losses.append(val_loss_value)
    return train_losses, val_losses


import glob
class PairFinder:
    def __init__(self, data_dir, output_dir, hint):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.hint = hint

    def extract_common_name(self, filename):
        # Extracts the common name from a filename by removing the hint and extension
        return os.path.basename(filename).replace(f'_{self.hint}', '').split('.')[0]

    def find_file_pairs(self):
        # Finds pairs of files based on the hint and directories specified
        dl_files = glob.glob(os.path.join(self.output_dir, f'**/*{self.hint}*.nii.gz'), recursive=True)
        test_dict_list = []
        for dl_path in dl_files:
            common_name = self.extract_common_name(dl_path)
            search_pattern = os.path.join(self.data_dir, f'{common_name}*.nii.gz')
            found_org_files = glob.glob(search_pattern)
            if found_org_files:
                pair_dict = {
                    'predicted': dl_path,
                    'reference': found_org_files[0]
                }
                test_dict_list.append(pair_dict)
        return test_dict_list
    

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

def mean_error(predicted, reference):
    return np.mean(predicted - reference)

def mean_absolute_error(predicted, reference):
    return np.mean(np.abs(predicted - reference))

def relative_error(predicted, reference, epsilon=0.3):
    return np.mean((predicted - reference) / (reference + epsilon)) * 100

def absolute_relative_error(predicted, reference, epsilon=0.3):
    return np.mean(np.abs(predicted - reference) / (reference + epsilon)) * 100

def rmse(predicted, reference):
    return sqrt(np.mean((predicted - reference) ** 2))

def psnr(predicted, reference, peak):
    mse = np.mean((predicted - reference) ** 2)
    return 20 * log10(peak / sqrt(mse))

def calculate_ssim(predicted, reference):
    return ssim(predicted, reference, data_range=reference.max() - reference.min())



import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

def load_nifti_image(path):
    """Load a NIfTI image and return its data as a NumPy array."""
    return nib.load(path).get_fdata()

def calculate_metrics_for_pair(predicted_path, reference_path, scaling_factor=7):
    """
    Calculate metrics for a single pair of images, applying a scaling factor to the images.
    """
    predicted_img = load_nifti_image(predicted_path) * scaling_factor
    reference_img = load_nifti_image(reference_path) * scaling_factor
    

    peak = np.max([predicted_img.max(), reference_img.max()])
    metrics = {
        "mean_error": mean_error(predicted_img, reference_img),
        "mean_absolute_error": mean_absolute_error(predicted_img, reference_img),
        "relative_error": relative_error(predicted_img, reference_img),
        "absolute_relative_error": absolute_relative_error(predicted_img, reference_img),
        "rmse": rmse(predicted_img, reference_img),
        "psnr": psnr(predicted_img, reference_img, peak),
        "ssim": calculate_ssim(predicted_img, reference_img)
    }
    return metrics

def aggregate_metrics(metrics_list):
    """Aggregate metrics across all pairs and calculate mean and standard deviation."""
    aggregated_metrics = {key: [] for key in metrics_list[0]}
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated_metrics[key].append(value)
    
    return {metric: (np.mean(values), np.std(values)) for metric, values in aggregated_metrics.items()}