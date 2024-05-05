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
            center_part = parts[1]  # Assuming the second part of the filename denotes the center
            if center_part in ['C1', 'C2', 'C3', 'C4', 'C5']:
                return center_part
        return 'rest'

    def find_file_triples(self):
        # Finds triples of files: NAC, MAC, and DL based on the hint and directories specified
        dl_files = glob.glob(os.path.join(self.dl_data_dir, f'**/*{self.hint}*.nii.gz'), recursive=True)
        all_triples = []
        center_triples = { 'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'rest': [] }

        for dl_path in dl_files:
            common_name = self.extract_common_name(dl_path)
            center = self.identify_center(common_name)
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
                all_triples.append(triple_dict)
                center_triples[center].append(triple_dict)

        return all_triples, center_triples



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