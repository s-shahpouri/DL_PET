import os
import random
from collections import defaultdict
import nibabel as nib
import torch
import glob
import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

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


class PairFinder:
    def __init__(self, data_dir, output_dir, hint):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.hint = hint

    def extract_common_name(self, filename):
        # Extracts the common name from a filename by removing the hint and extension
        return os.path.basename(filename).replace(f'_{self.hint}', '').split('.')[0]

    def identify_center(self, filename):
        # Identifies the center from the filename
        parts = filename.split('_')
        if len(parts) > 1:
            return parts[1]  # Assuming the second part of the filename denotes the center
        return None

    def find_file_pairs(self):
        # Finds pairs of files based on the hint and directories specified
        dl_files = glob.glob(os.path.join(self.output_dir, f'**/*{self.hint}*.nii.gz'), recursive=True)
        all_pairs = []
        c5_pairs = []
        rest_pairs = []
        for dl_path in dl_files:
            common_name = self.extract_common_name(dl_path)
            center = self.identify_center(common_name)
            search_pattern = os.path.join(self.data_dir, f'{common_name}*.nii.gz')
            found_org_files = glob.glob(search_pattern)
            if found_org_files:
                pair_dict = {
                    'predicted': dl_path,
                    'reference': found_org_files[0],
                    'center': center
                }
                all_pairs.append(pair_dict)
                if center == 'C5':
                    c5_pairs.append(pair_dict)
                else:
                    rest_pairs.append(pair_dict)
        return all_pairs, c5_pairs, rest_pairs
    


# import numpy as np
# import nibabel as nib
# from math import sqrt, log10
# from skimage.metrics import structural_similarity as ssim

# def mean_error(predicted, reference):
#     return np.mean(predicted - reference)

# def mean_absolute_error(predicted, reference):
#     return np.mean(np.abs(predicted - reference))

# def relative_error(predicted, reference, epsilon=0.0):
#     return np.mean((predicted - reference) / (reference + epsilon)) * 100

# # def absolute_relative_error(predicted, reference, epsilon=0.0):
# #     return np.mean(np.abs(predicted - reference) / (reference + epsilon)) * 100
# def absolute_relative_error(predicted, reference, threshold=0.003):
#     """
#     Calculate the absolute relative error for pixels where the reference value
#     is greater than a specified threshold.
    
#     Args:
#     predicted (np.array): The predicted image values.
#     reference (np.array): The reference (ground truth) image values.
#     threshold (float): The threshold value above which pixels are considered for calculation.
    
#     Returns:
#     float: The absolute relative error (%) for the specified pixels.
#     """
#     # Create a mask for pixels in the reference image above the threshold
#     mask = reference > threshold
    
#     # Apply the mask to both predicted and reference arrays
#     masked_predicted = predicted[mask]
#     masked_reference = reference[mask]
   
#     # Calculate the absolute relative error using the masked pixels
#     are = np.mean(np.abs(masked_predicted - masked_reference) / masked_reference) * 100
    
#     return are

# def rmse(predicted, reference):
#     return sqrt(np.mean((predicted - reference) ** 2))

# def psnr(predicted, reference, peak):
#     mse = np.mean((predicted - reference) ** 2)
#     return 20 * log10(peak / sqrt(mse))

# def calculate_ssim(predicted, reference):
#     return ssim(predicted, reference, data_range=reference.max() - reference.min())

# def load_nifti_image(path):
#     """Load a NIfTI image and return its data as a NumPy array."""
#     return nib.load(path).get_fdata()

# def calculate_metrics_for_pair(predicted_path, reference_path, scaling_factor=5, mask_val = 0.03):
#     """
#     Calculate metrics for a single pair of images, applying a scaling factor to the images.
#     A mask is applied where the reference image values are bigger than 0.03.
#     """
#     predicted_img = load_nifti_image(predicted_path) * scaling_factor
#     reference_img = load_nifti_image(reference_path) * scaling_factor

#     # Create mask from reference image where values are greater than 0.03
#     mask = reference_img > mask_val
    
#     # Apply the mask to both images
#     masked_predicted_img = predicted_img[mask]
#     masked_reference_img = reference_img[mask]

#     peak = np.max([masked_predicted_img.max(), masked_reference_img.max()])
#     metrics = {
#         "mean_error": mean_error(masked_predicted_img, masked_reference_img),
#         "mean_absolute_error": mean_absolute_error(masked_predicted_img, masked_reference_img),
#         "relative_error": relative_error(masked_predicted_img, masked_reference_img),
#         "absolute_relative_error": absolute_relative_error(masked_predicted_img, masked_reference_img),
#         "rmse": rmse(masked_predicted_img, masked_reference_img),
#         "psnr": psnr(masked_predicted_img, masked_reference_img, peak),
#         "ssim": calculate_ssim(masked_predicted_img, masked_reference_img)
#     }
#     return metrics

# def aggregate_metrics(metrics_list):
#     """Aggregate metrics across all pairs and calculate mean and standard deviation."""
#     aggregated_metrics = {key: [] for key in metrics_list[0]}
#     for metrics in metrics_list:
#         for key, value in metrics.items():
#             aggregated_metrics[key].append(value)
    
#     return {metric: (np.mean(values), np.std(values)) for metric, values in aggregated_metrics.items()}

import re

def ids(s):
    # This will match consecutive digits at the beginning of the string
    match = re.match(r"(\d+)", s)
    if match:
        return match.group(0)  # Returns the matched group
    else:
        return None  # or an empty string if you prefer
    

def find_dl_image_path(artifact_output, patient_folder_name, hint):
    # Construct a glob pattern to search for DL images with the matching patient folder name
    search_pattern = os.path.join(artifact_output, "**", f"{patient_folder_name}*{hint}.nii.gz")
    found_paths = glob.glob(search_pattern, recursive=True)
    if found_paths:
        return found_paths[0]  # Return the first match
    else:
        return None  # No match found
    
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



import os
import glob
import nibabel as nib

class Pairs:
    def __init__(self, nac_data_dir, mac_data_dir):
        self.nac_data_dir = nac_data_dir
        self.mac_data_dir = mac_data_dir

    def extract_common_name(self, filename):
        # Generalize this method based on how the filenames can be normalized to form pairs
        return os.path.basename(filename).split('_')[0]

    def find_file_pairs(self):
        # Finds pairs of files from NAC and MAC directories based on common names
        nac_files = glob.glob(os.path.join(self.nac_data_dir, '*.nii.gz'))
        mac_files = glob.glob(os.path.join(self.mac_data_dir, '*.nii.gz'))
        
        nac_dict = {self.extract_common_name(path): path for path in nac_files}
        mac_dict = {self.extract_common_name(path): path for path in mac_files}

        all_pairs = []
        for common_name, nac_path in nac_dict.items():
            mac_path = mac_dict.get(common_name)
            if mac_path:
                # pair_dict = {
                #     'nac': nac_path,
                #     'mac': mac_path
                # }
                # all_pairs.append(pair_dict)
                all_pairs.append((nac_path, mac_path))
        return all_pairs



def calculate_adcm(nac_img, mac_img, epsilon):

    nac_img = nac_img.astype(np.float32) * 2
    mac_img = mac_img.astype(np.float32) * 5

    # Initialize ADCM with zeros
    adcm = np.zeros_like(mac_img)

    # Calculate ADCM where NASC-PET is greater than epsilon
    mask = nac_img > epsilon
    adcm[mask] = mac_img[mask] / nac_img[mask]

    # Assign MAC values directly where NASC-PET is less than or equal to epsilon
    adcm[~mask] = mac_img[~mask]
    
    return adcm



def calculate_adcm_stat(nac_img, mac_img, epsilon):

    adcm = calculate_adcm(nac_img, mac_img, epsilon)
    
    # nonzero_values = adcm_contour[adcm_contour>0]
    adcm_med = np.median(adcm)
    adcm_mean = np.mean(adcm)
    adcm_max = np.max(adcm)
    
    return adcm, adcm_med, adcm_mean, adcm_max


def calculate_nac_mac_stat(nac_img, mac_img):

    nac_img = nac_img.astype(np.float32) * 2
    mac_img = mac_img.astype(np.float32) * 5

    nac_med = np.median(nac_img)
    nac_mean = np.mean(nac_img)
    nac_max = np.max(nac_img)


    mac_med = np.median(mac_img)
    mac_mean = np.mean(mac_img)
    mac_max = np.max(mac_img)

    
    return nac_med, nac_mean, nac_max, mac_med, mac_mean, mac_max