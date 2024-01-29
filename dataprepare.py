
import os
import glob
import random
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# Path
data_dir = '/home/shahpouriz/Data/Practic/ASC-PET-001'
directory = '/home/shahpouriz/Data/Practic/LOG'


# file names
train_images = sorted(glob.glob(os.path.join(data_dir, "NAC", "*.nii.gz")))
target_images = sorted(glob.glob(os.path.join(data_dir, "MAC", "*.nii.gz")))


# Making dictionary
data_dicts = [{"image": img, "target": tar} for img, tar in zip(train_images, target_images)]
patient_list = [os.path.basename(f["image"]) for f in data_dicts]


# Spliting data
total_patients = len(patient_list)
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2


def group_patients_by_center(data_dicts):
    '''Gathering patients from different centers'''

    center_dict = defaultdict(list)
    for item in data_dicts:
        filename = os.path.basename(item["image"])
        center = filename.split('_')[1]  #Center name is the 2th part of the filename
        center_dict[center].append(filename)
    return center_dict

def split_data_for_center(center_data, train_ratio, val_ratio, seed=None):
    '''Spliting data for each center based on ratios'''

    random.seed(seed)  # Using fixed seed for repeatability

    total_samples = len(center_data)
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    # test_samples = total_samples - train_samples - val_samples

    random.shuffle(center_data)
    train_set = center_data[:train_samples]
    val_set = center_data[train_samples:train_samples + val_samples]
    test_set = center_data[train_samples + val_samples:]
    return train_set, val_set, test_set

# Fixed seed for repeatability
seed_random = 42

center_dict = group_patients_by_center(data_dicts)
train_files, val_files, test_files = [], [], []

for center, patients in center_dict.items():
    center_train, center_val, center_test = split_data_for_center(patients, train_ratio, val_ratio, seed=seed_random)
    train_files.extend(center_train)
    val_files.extend(center_val)
    test_files.extend(center_test)


# center_dict = group_patients_by_center(data_dicts)

# train_files, val_files, test_files = [], [], []

# # Set the random seed for replayability
# random.seed(seed_random)

# # Split data for each center and combine
# for center, patients in center_dict.items():
#     center_train, center_val, center_test = split_data_for_center(patients, train_ratio, val_ratio, test_ratio, seed=seed_random)
#     train_files.extend(center_train)
#     val_files.extend(center_val)
#     test_files.extend(center_test)

# # Optionally, shuffle the combined lists if needed
# random.shuffle(train_files)
# random.shuffle(val_files)
# random.shuffle(test_files)

# print(train_files)
# print(val_files)
# print(test_files)

# print(len(train_files))
# print(len(val_files))
# print(len(test_files))