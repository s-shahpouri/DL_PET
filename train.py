import os
import random
from collections import defaultdict

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

def load_model(directory, model_filename):
    model_path = os.path.join(directory, model_filename)
    if os.path.exists(model_path):
        print(f"Model file {model_filename} is loading.")
        # Load the model onto the CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"Model file {model_filename} not found.")



def find_model_info(log_filepath, model_filename):
    best_metric = None
    epoch = None
    with open(log_filepath, 'r') as file:
        for line in file:
            if model_filename in line:
                parts = line.split(',')
                best_metric = float(parts[1].split(': ')[1])  # Extract best metric
                epoch = int(parts[2].split(': ')[1])  # Extract epoch number
                return model_filename, best_metric, epoch
    return model_filename, best_metric, epoch

