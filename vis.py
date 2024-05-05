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


def model_visualize_coronal(data, predict, n, title, cm , Norm = False):


    fig, axes = plt.subplots(1, 3, figsize=(12, 6))  # Adjusted for three plots and one colorbar

    titles = ["Input", "Ground_truth", title]
    slices = [
        np.rot90(data["image"][0, 0, :, n, :]),
        np.rot90(data["target"][0, 0, :, n, :]),
        np.rot90(predict.detach().cpu()[0, 0, :, n, :])
    ]

    # Display the images
    images = []
    for ax, slice, title in zip(axes, slices, titles):  # Leave the last axes for the colorbar
        img = ax.imshow(slice, cmap=cm)
        images.append(img)
        ax.set_title(title)
        ax.axis('off')

    if Norm == True:
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

    fig.colorbar(images[0], ax=axes, orientation='vertical', fraction=0.025, pad=0.04)

    # Make sure the aspect ratio is equal to make the colorbar align well
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
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