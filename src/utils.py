import os
import random
from collections import defaultdict
import glob
import nibabel as nib
import numpy as np
import re

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


def extract_id(filepath):
    # Extracts an identifier from the file path, this would depend on your file naming convention
    return os.path.basename(filepath).split('_')[0]

def ids(s):
    # This will match consecutive digits at the beginning of the string
    match = re.match(r"(\d+)", s)
    if match:
        return match.group(0)  # Returns the matched group
    else:
        return None  # or an empty string if you prefer
    



    
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class PairFinder:
    def __init__(self, nac_data_dir, mac_data_dir, dl_data_dir, hint):
        self.nac_data_dir = nac_data_dir
        self.mac_data_dir = mac_data_dir
        self.dl_data_dir = dl_data_dir
        self.hint = hint

    def extract_common_name(self, filename):
        # Extracts the common name from a filename by removing the hint and extension
        return os.path.basename(filename).replace(f'_{self.hint}', '').split('.')[0]

    def identify_center(self, filename):
        # Identifies the center from the filename
        parts = filename.split('_')
        if len(parts) > 1:
            if parts[1].startswith("C") and parts[1][1:].isdigit():
                return parts[1]  # Directly return 'C1', 'C2', etc.
            elif parts[1] == "dataset" and len(parts) > 2:
                center_part = 'C' + parts[2][1]  # Convert '06', '07' to 'C6', 'C7'
                if center_part in ['C6', 'C7']:
                    return center_part
        return 'rest'


    def find_file_triples(self):
        # Finds triples of files: NAC, MAC, and DL based on the hint and directories specified
        dl_files = glob.glob(os.path.join(self.dl_data_dir, f'**/*{self.hint}*.nii.gz'), recursive=True)
        center_triples = { 'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'rest': [] }

      
        for dl_path in dl_files:
            common_name = self.extract_common_name(dl_path)
            center = self.identify_center(common_name)

            # Ensure the dictionary initialization covers this center
            if center not in center_triples:
                print(f"Unexpected center {center} found, adding to dictionary.")
                center_triples[center] = []

            mac_search_pattern = os.path.join(self.mac_data_dir, f'{common_name}*.nii.gz')
            nac_search_pattern = os.path.join(self.nac_data_dir, f'{common_name}*.nii.gz')
            found_mac_files = glob.glob(mac_search_pattern)
            found_nac_files = glob.glob(nac_search_pattern)
            
            if found_mac_files and found_nac_files:
                triple_dict = {
                    'predicted': dl_path,
                    'reference': found_mac_files[0],
                    'nac': found_nac_files[0],
                    'center': center,
                    'common_name': common_name
                }
                
                center_triples[center].append(triple_dict)

        return center_triples



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

def calculate_dl_mac(nac_img, dl_adcm_img, nac_factor, mac_factor, val):

    dl_adcm_img = dl_adcm_img * val
    nac_img = nac_img * nac_factor
    
    dl_final = np.copy(nac_img)
    
    # Only perform division where NASC-PET is greater than epsilon
    mask = nac_img > 0
    dl_final[mask] = (dl_adcm_img[mask] * nac_img[mask]) / mac_factor
    
    return dl_final



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


def export_final_adcm_image(nac_path, dl_final_img, output_path):
    """
    Saves an image with the original header from the NAC image.
    
    Parameters:
        nac_path (str): Path to the original NAC image.
        dl_final_img (np.array): The DL final image data to save.
        output_path (str): Path where the output image will be saved.
    """
    # Load the NAC image to use its header and affine
    nac_nii = nib.load(nac_path)
    nac_header = nac_nii.header
    nac_affine = nac_nii.affine
    
    # Create a new Nifti image with the dl_final_img data and the original NAC header/affine
    dl_final_nii = nib.Nifti1Image(dl_final_img, affine=nac_affine, header=nac_header)
    
    # Save the new image to the specified output path
    nib.save(dl_final_nii, output_path)



import json
import torch

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.__dict__.update(config)
        self.device = self.get_device()

    def get_device(self):
        configured_device = self.__dict__.get("device")
        print(f"Requested device: {configured_device}")

        if configured_device.startswith("cuda"):
            if torch.cuda.is_available():
                device_index = configured_device.split(":")
                if len(device_index) == 2 and device_index[1].isdigit():
                    device_index = int(device_index[1])
                    if device_index < torch.cuda.device_count():
                        print(f"CUDA device {device_index} is available. Using CUDA.")
                        return torch.device(configured_device)
                    else:
                        print(f"CUDA device {device_index} is not available. Switching to CPU.")
                else:
                    print("Invalid CUDA device format or device index. Switching to CPU.")
            else:
                print("CUDA is not available. Switching to CPU.")
            return torch.device("cpu")
        else:
            print("Using CPU as default device.")
            return torch.device("cpu")

# models.py
from torch import nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
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


class CustomDynUNet(DynUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add a ReLU activation after the final convolution layer
        self.output_block = nn.Sequential(
            self.output_block,
            nn.ReLU(inplace=True)
        )


from monai.networks.nets import SegResNetDS
class CustomSegResNetDS(SegResNetDS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Modify the final head to include a ReLU activation
        for i, layer in enumerate(self.up_layers):
            head_conv = layer['head']
            relu = nn.ReLU(inplace=True)
            # Create a sequential container with Conv3d followed by ReLU
            self.up_layers[i]['head'] = nn.Sequential(head_conv, relu)


import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd

import os
import glob

class ImageProcessor:
    def __init__(self, artifact_output_dir, nac_factor, mac_factor, hint):
        self.artifact_output_dir = artifact_output_dir
        self.nac_factor = nac_factor
        self.mac_factor = mac_factor
        self.hint = hint

    def find_dl_image_path(self, patient_folder_name_image):
        """Find the deep learning image path based on the patient folder name and hint."""
        search_pattern = os.path.join(self.artifact_output_dir, "**", f"{patient_folder_name_image}*{self.hint}.nii.gz")
        print(f"Searching for DL image with pattern: {search_pattern}")  # Debug print

        # Print all files in the directory for verification
        all_files = glob.glob(os.path.join(self.artifact_output_dir, "**"), recursive=True)
        print(f"All files in directory: {all_files}")

        found_paths = glob.glob(search_pattern, recursive=True)
        print(f"Found paths: {found_paths}")  # Debug print
        if found_paths:
            return found_paths[0]  # Return the first match
        else:
            print(f"No DL image found for {patient_folder_name_image} with hint {self.hint}")
            return None  # No match found

    def load_and_store_images_to_df(self, df, test_files):
        """Load images and add them to the existing DataFrame."""
        dl_image_paths = []
        image_matrices = []
        target_matrices = []
        dl_image_matrices = []
        difference_matrices = []

        for file_info, name in zip(test_files, df['name']):
            image_path = file_info['image']
            target_path = file_info['target']
            
            # Find the corresponding DL image path
            dl_image_path = self.find_dl_image_path(name)
            dl_image_paths.append(dl_image_path)
            
            if dl_image_path is None:
                image_matrices.append(None)
                target_matrices.append(None)
                dl_image_matrices.append(None)
                difference_matrices.append(None)
                continue
            
            try:
                # Load images and apply factors
                image = (nib.load(image_path).get_fdata()) * self.nac_factor
                target = (nib.load(target_path).get_fdata()) * self.mac_factor
                dl_image = (nib.load(dl_image_path).get_fdata()) * self.mac_factor
                difference = (target - dl_image) / self.mac_factor
                difference = np.clip(difference, -1, 1)

                # Append the matrices to the lists
                image_matrices.append(image)
                target_matrices.append(target)
                dl_image_matrices.append(dl_image)
                difference_matrices.append(difference)

            except Exception as e:
                print(f"Error loading or processing images for {name}: {e}")
                image_matrices.append(None)
                target_matrices.append(None)
                dl_image_matrices.append(None)
                difference_matrices.append(None)
                continue
        
        # Add the new data to the DataFrame
        df['dl_image_path'] = dl_image_paths
        df['image_matrix'] = image_matrices
        df['target_matrix'] = target_matrices
        df['dl_image_matrix'] = dl_image_matrices
        df['difference_matrices'] = difference_matrices
        return df

def load_df_from_pickle(filename='/students/2023-2024/master/Shahpouri/DATA/Artifact_data.pkl'):
    """Load the DataFrame from a Pickle file."""
    try:
        df = pd.read_pickle(filename)
        print(f"DataFrame loaded from {filename}")
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None