import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def display_patient_coronal(patient_folder_name, image, target, dl_image, difference_image, n, cmp="gray"):
    """
    Display medical images for a patient: input, target, deep learning output, and the difference.
    """
    colors = [(0.00, "orangered"), (0.40, "white"), (0.8, "white"), (1.00, "blue")]
    cmap_name = 'custom_seismic_more_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=12)

    fig, axs = plt.subplots(1, 4, figsize=(8, 4), gridspec_kw={'wspace':0, 'hspace':0})

    # # Turn off axes
    # for ax in axs:
    #     ax.axis('off')

    # Configure vmin and vmax for each image type
    nac_display_range = (np.percentile(image, 0), np.percentile(image, 99.7))
    mac_display_range = (np.percentile(target, 0), np.percentile(target, 99.7))
    dl_display_range = (np.percentile(dl_image, 0), np.percentile(dl_image, 99.9))

    # Input Image

    input_slice = np.rot90(image[:, n, :])
    axs[0].imshow(input_slice, cmap=cmp, vmin=nac_display_range[0], vmax=nac_display_range[1])

    # Target Image
    axs[1].set_title(patient_folder_name)
    target_slice = np.rot90(target[:, n, :])
    axs[1].imshow(target_slice, cmap=cmp, vmin=mac_display_range[0], vmax=mac_display_range[1])

    # DL Image
 
    dl_slice = np.rot90(dl_image[:, n, ])
    axs[2].imshow(dl_slice, cmap=cmp, vmin=dl_display_range[0], vmax=dl_display_range[1])

    # Difference Image

    difference_slice = np.rot90(difference_image[:, n, :])
    axs[3].imshow(difference_slice, cmap=cm, vmin=-1, vmax=1)

    plt.show()

    # Configure vmin and vmax for each image type
    nac_display_range = (np.percentile(image, 0), np.percentile(image, 99.7))
    mac_display_range = (np.percentile(target, 0), np.percentile(target, 99.7))
    dl_display_range = (np.percentile(dl_image, 0), np.percentile(dl_image, 99.7))


def display_patient_transverse(patient_folder_name, image, target, dl_image, difference_image, n, cmp="gray"):
    """
    Display medical images for a patient: input, target, deep learning output, and the difference.
    """
    colors = [(0.00, "red"), (0.40, "white"), (0.8, "white"), (1.00, "blue")]
    cmap_name = 'custom_seismic_more_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=12)

    fig, axs = plt.subplots(1, 4, figsize=(4, 2), gridspec_kw={'wspace':0, 'hspace':0})

    # Turn off axes
    for ax in axs:
        ax.axis('off')

    # Configure vmin and vmax for each image type
    nac_display_range = (np.percentile(image, 0), np.percentile(image, 99.7))
    mac_display_range = (np.percentile(target, 0), np.percentile(target, 99.7))
    dl_display_range = (np.percentile(dl_image, 0), np.percentile(dl_image, 99.7))

    # Input Image

    input_slice = np.rot90(image[:, :, n])
    axs[0].imshow(input_slice, cmap=cmp, vmin=nac_display_range[0], vmax=nac_display_range[1])

    # Target Image
    axs[1].set_title(patient_folder_name)
    target_slice = np.rot90(target[:, :, n])
    axs[1].imshow(target_slice, cmap=cmp, vmin=mac_display_range[0], vmax=mac_display_range[1])

    # DL Image
 
    dl_slice = np.rot90(dl_image[:, :, n])
    axs[2].imshow(dl_slice, cmap=cmp, vmin=dl_display_range[0], vmax=dl_display_range[1])

    # Difference Image

    difference_slice = np.rot90(difference_image[:, :, n])
    axs[3].imshow(difference_slice, cmap=cm, vmin=-1, vmax=1)

    plt.show()

    # Configure vmin and vmax for each image type
    nac_display_range = (np.percentile(image, 0), np.percentile(image, 99.7))
    mac_display_range = (np.percentile(target, 0), np.percentile(target, 99.7))
    dl_display_range = (np.percentile(dl_image, 0), np.percentile(dl_image, 99.7))


def plot_adcm_final_trans(nac_img, dl_adcm_im, dl_final, mac_images, title_prefix=''):
    
    slice_idx = nac_img.shape[2] // 3  # Middle slice for the Z-axis
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(nac_img[:, :, slice_idx], cmap='jet')
    axes[0].set_title(f'{title_prefix} NAC')
    
    axes[1].imshow(dl_adcm_im[:, :, slice_idx], cmap='jet')
    axes[1].set_title(f'{title_prefix} ADCM')
    
    axes[2].imshow(dl_final[:, :, slice_idx], cmap='jet')
    axes[2].set_title(f'{title_prefix} DL_Final')
    
    axes[3].imshow(mac_images[:, :, slice_idx], cmap='jet')
    axes[3].set_title(f'{title_prefix} MAC')
    plt.show()

def plot_adcm_final_coronal(nac_img, dl_adcm_im, dl_final, mac_images, n, title_prefix=''):
    
    
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(np.rot90(nac_img[:, n, :]), cmap='jet')
    axes[0].set_title(f'{title_prefix} NAC')
    
    axes[1].imshow(np.rot90(dl_adcm_im[:, n, :]), cmap='jet')
    axes[1].set_title(f'{title_prefix} ADCM')
    
    axes[2].imshow(np.rot90(dl_final[:, n, :]), cmap='jet')
    axes[2].set_title(f'{title_prefix} DL_Final')
    
    axes[3].imshow(np.rot90(mac_images[:, n, :]), cmap='jet')
    axes[3].set_title(f'{title_prefix} MAC')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def model_visualize_coronal(data, predict, n, title, cm, Norm=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))  # Adjusted for three plots

    titles = ["Input", "Ground_truth", title]
    slices = [
        np.rot90(data["image"][0, 0, :, n, :]),
        np.rot90(data["target"][0, 0, :, n, :]),
        np.rot90(predict.detach().cpu()[0, 0, :, n, :])
    ]

    vmin, vmax = None, None
    if Norm:
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(slice.min() for slice in slices)
        vmax = max(slice.max() for slice in slices)

    # Display the images with color bars
    for ax, slice, title in zip(axes, slices, titles):
        img = ax.imshow(slice, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()



def visualize_axial_slice(data_loader, slice_index):
    # Manually iterate through the DataLoader to fetch the first batch
    for data_batch in data_loader:
        break  # Only need the first batch
    
    # Extract the image and target tensors from the batch
    image, target = data_batch["image"][0][0], data_batch["target"][0][0]  # Assuming batch size of 1
    
    # Determine global min and max for a unified color scale
    vmin = min(image[:, :, slice_index].min(), target[:, :, slice_index].min())
    vmax = max(image[:, :, slice_index].max(), target[:, :, slice_index].max())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(f"Data Check/Axial slice {slice_index}")
    
    img_plot = axes[0].imshow(np.rot90(np.flip(image[:, :, slice_index], axis=1)), cmap="jet")
    axes[0].set_title("Image Slice")
    
    tgt_plot = axes[1].imshow(np.rot90(np.flip(target[:, :, slice_index], axis=1)), cmap="jet")
    axes[1].set_title("Target Slice")
   
    
    # Add a single colorbar 
    fig.colorbar(img_plot, ax=axes, fraction=0.021, pad=0.04)
    
    plt.show()


def visualize_coronal_slice(data_loader, slice_index):
    # Manually iterate through the DataLoader to fetch the first batch
    for data_batch in data_loader:
        break  # Only need the first batch
    
    # Extract the image and target tensors from the batch
    image, target = data_batch["image"][0][0], data_batch["target"][0][0]  # Assuming batch size of 1
    
    # # Determine global min and max for a unified color scale
    # vmin = min(image[:, :, slice_index].min(), target[:, :, slice_index].min())
    # vmax = max(image[:, :, slice_index].max(), target[:, :, slice_index].max())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(f"Data Check/Axial slice {slice_index}")
    
    img_plot = axes[0].imshow(np.rot90(np.flip(image[:, slice_index, :], axis=1), k=3), cmap="jet")
    axes[0].set_title("Image Slice")
    
    tgt_plot = axes[1].imshow(np.rot90(np.flip(target[:, slice_index, :], axis=1), k=3), cmap="jet")
    axes[1].set_title("Target Slice")
   
    
    # Add a single colorbar 
    fig.colorbar(img_plot, ax=axes, fraction=0.021, pad=0.04)
    
    plt.show()


def visualize_coronal_masked(masked_nac_img, masked_predicted_img, masked_reference_img, slice_number):
    # Assuming these images are already 3D arrays after masking and you want a specific slice
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    titles = ['Masked NAC', 'Masked Predicted', 'Masked Reference']
    images = [masked_nac_img, masked_predicted_img, masked_reference_img]

    for ax, img, title in zip(axes, images, titles):
        cax = ax.imshow(np.rot90(img[:, slice_number, :]), cmap='jet')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(cax, ax=ax)

    plt.tight_layout()
    plt.show() 