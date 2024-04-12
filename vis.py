import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def display_patient_coronal(patient_folder_name, image, target, dl_image, difference_image, n, cmp ="gist_yarg"):
    """
    Display medical images for a patient: input, target, deep learning output, and the difference.

    Parameters:
    - patient_folder_name: The folder name for the patient, used in the title of the input image.
    - image: The input image numpy array.
    - target: The target image numpy array.
    - dl_image: The deep learning output image numpy array.
    - difference_image: The difference between the target and the dl_image.
    - n: The slice number to be displayed.
    """
    # cmap = LinearSegmentedColormap.from_list('bwr', ['red','white','blue'])

    # Define the colors with more emphasis on white in the middle
    colors = [
        (0.00, "red"),  # Blue at the beginning
        (0.40, "white"),  # Start introducing white sooner
        (0.60, "white"),  # Keep white going until 60%
        (1.00, "blue")  # Red at the end
    ]

    # Create the colormap
    cmap_name = 'custom_seismic_more_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    fig, axs = plt.subplots(1, 4, figsize=(12, 6), gridspec_kw={'wspace':0, 'hspace':0})

    # Turn off axes
    for ax in axs:
        ax.axis('off')
    
    # Input Image
    axs[0].set_title(f"NAC: {patient_folder_name}")
    input_slice = np.rot90(image[:, n, :])
    axs[0].imshow(input_slice, cmap=cmp, vmin=0, vmax=0.5)
    
    # Target Image
    axs[1].set_title("MAC")
    target_slice = np.rot90(target[:, n, :])
    axs[1].imshow(target_slice, cmap=cmp, vmin=0, vmax=5)
    
    # DL Image
    axs[2].set_title("DL Image")
    dl_slice = np.rot90(dl_image[:, n, :])
    axs[2].imshow(dl_slice, cmap=cmp, vmin=0, vmax=5)
    
    # Difference Image
    axs[3].set_title("Difference")
    difference_slice = np.rot90(difference_image[:, n, :])
    difference_display = axs[3].imshow(difference_slice, cmap=cm, vmin=-1, vmax=1)

    # Remove the space between the images and the axis numbers
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Add color bar to the last subplot
    # plt.colorbar(difference_display, ax=axs[3], fraction=0.05, pad=0.04)

    plt.show()



def display_patient_transverse(patient_folder_name, image, target, dl_image, difference_image, n):
    """
    Display medical images for a patient: input, target, deep learning output, and the difference.

    Parameters:
    - patient_folder_name: The folder name for the patient, used in the title of the input image.
    - image: The input image numpy array.
    - target: The target image numpy array.
    - dl_image: The deep learning output image numpy array.
    - difference_image: The difference between the target and the dl_image.
    - n: The slice number to be displayed.
    """
    # cmap = LinearSegmentedColormap.from_list('bwr', ['red','white','blue'])

    # Define the colors with more emphasis on white in the middle
    colors = [
        (0.00, "red"),  # Blue at the beginning
        (0.40, "white"),  # Start introducing white sooner
        (0.60, "white"),  # Keep white going until 60%
        (1.00, "blue")  # Red at the end
    ]

    # Create the colormap
    cmap_name = 'custom_seismic_more_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    fig, axs = plt.subplots(1, 4, figsize=(12, 6), gridspec_kw={'wspace':0, 'hspace':0})

    # Turn off axes
    for ax in axs:
        ax.axis('off')
    
    # Input Image
    axs[0].set_title(f"NAC: {patient_folder_name}")
    input_slice = np.rot90(image[:, :, n])
    axs[0].imshow(input_slice, cmap="gist_yarg", vmin=0, vmax=0.7)
    
    # Target Image
    axs[1].set_title("MAC")
    target_slice = np.rot90(target[:, :, n])
    axs[1].imshow(target_slice, cmap="gist_yarg", vmin=0, vmax=7)
    
    # DL Image
    axs[2].set_title("DL Image")
    dl_slice = np.rot90(dl_image[:, :, n])
    axs[2].imshow(dl_slice, cmap="gist_yarg", vmin=0, vmax=7)
    
    # Difference Image
    axs[3].set_title("Difference")
    difference_slice = np.rot90(difference_image[:, :, n])
    difference_display = axs[3].imshow(difference_slice, cmap=cm, vmin=-1.5, vmax=1.5)

    # Remove the space between the images and the axis numbers
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Add color bar to the last subplot
    # plt.colorbar(difference_display, ax=axs[3], fraction=0.05, pad=0.04)

    plt.show()
