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

