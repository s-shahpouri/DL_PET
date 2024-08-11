import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

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





import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

# def dash_plot_artifact(image, target, dl_image, difference_image, n, cmp="gist_yarg"):
#     """
#     Display medical images for a patient: input, target, deep learning output, and the difference.
#     """
#     colors = [(0.00, "red"), (0.40, "white"), (0.8, "white"), (1.00, "blue")]
#     cmap_name = 'custom_seismic_more_white'
#     cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=12)

#     fig, axs = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={'wspace':0.3, 'hspace':0})

#     # Turn off axes
#     for ax in axs:
#         ax.axis('off')

#     # Configure vmin and vmax for each image type
#     nac_display_range = (np.percentile(image, 0), np.percentile(image, 99.9))
#     mac_display_range = (np.percentile(target, 0), np.percentile(target, 99.9))
#     dl_display_range = (np.percentile(dl_image, 0), np.percentile(dl_image, 99.9))

#     # Input Image
#     input_slice = np.rot90(image[:, n, :])
#     im0 = axs[0].imshow(input_slice, cmap=cmp, vmin=nac_display_range[0], vmax=nac_display_range[1])
#     # fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

#     # Target Image
#     # axs[1].set_title(patient_folder_name)
#     target_slice = np.rot90(target[:, n, :])
#     im1 = axs[1].imshow(target_slice, cmap=cmp, vmin=mac_display_range[0], vmax=mac_display_range[1])
#     # fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

#     # DL Image
#     dl_slice = np.rot90(dl_image[:, n, :])
#     im2 = axs[2].imshow(dl_slice, cmap=cmp, vmin=dl_display_range[0], vmax=dl_display_range[1])
#     # fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

#     # Difference Image
#     difference_slice = np.rot90(difference_image[:, n, :])
#     im3 = axs[3].imshow(difference_slice, cmap=cm, vmin=-1, vmax=1)
#     # fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

#     # Instead of plt.show(), use st.pyplot() to display the figure in Streamlit
#     st.pyplot(fig)
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def normalize_image(image):
    """Normalize image data to the range [0, 255] for display."""
    image_min, image_max = np.min(image), np.max(image)
    if image_max > image_min:  # Avoid division by zero
        image = (image - image_min) / (image_max - image_min) * 255
    return image.astype(np.uint8)

# def dash_plot_artifact(image, target, dl_image, difference_image, n):
#     """
#     Display medical images for a patient: input, target, deep learning output, and the difference.
#     """
#     # Ensure the slice index n is within the bounds of the image dimensions
#     n = min(n, image.shape[1] - 1)

#     # Extract the nth slice and rotate it for correct orientation
#     input_slice = np.rot90(image[:, n, :])
#     target_slice = np.rot90(target[:, n, :])
#     dl_slice = np.rot90(dl_image[:, n, :])
#     difference_slice = np.rot90(difference_image[:, n, :])

#     # Create a subplot grid in Plotly
#     fig = make_subplots(rows=1, cols=4, subplot_titles=("Input", "Target", "DL Output", "Difference"))

#     # Input Image
#     fig.add_trace(go.Heatmap(z=input_slice, colorscale="Gray", showscale=False), row=1, col=1)

#     # Target Image
#     fig.add_trace(go.Heatmap(z=target_slice, colorscale="gray", showscale=False), row=1, col=2)

#     # DL Image
#     fig.add_trace(go.Heatmap(z=dl_slice, colorscale="gray", showscale=False), row=1, col=3)

#     # Difference Image
#     fig.add_trace(go.Heatmap(z=difference_slice, colorscale="RdBu", showscale=False), row=1, col=4)

#     fig.update_layout(height=600, width=800, title_text=f"Slice {n} Visualization", showlegend=False)
#     st.plotly_chart(fig)


custom_colorscale = [
    [0.0, "red"],
    [0.4, "white"],
    [0.8, "white"],
    [1.0, "blue"]
]
import plotly.graph_objects as go
from plotly.subplots import make_subplots


high_contrast_greys = [
    [0.0, 'rgb(255, 255, 255)'],  # Pure white background
    [0.3, 'rgb(40, 40, 40)'],  # Dark grey
    [1.0, 'rgb(0, 0, 0)']         # Pure black for hot spots
]

def rotate_and_flip_cor(image, target, dl_image, difference_image=None, n=None):
    """
    Rotate and flip the images for correct display orientation.
    """
    input_slice = np.rot90(image[:, n, :])
    target_slice = np.rot90(target[:, n, :])
    dl_slice = np.rot90(dl_image[:, n, :])

    # Flip the slices vertically to match Bokeh's default origin
    input_slice = np.flipud(input_slice)
    target_slice = np.flipud(target_slice)
    dl_slice = np.flipud(dl_slice)

    if difference_image is not None:
        difference_slice = np.rot90(difference_image[:, n, :])
        difference_slice = np.flipud(difference_slice)
    else:
        difference_slice = None

    return input_slice, target_slice, dl_slice, difference_slice

def rotate_and_flip_axial(image, target, dl_image, difference_image=None, n=None):
    """
    Rotate and flip the images for correct display orientation.
    """
    input_slice = np.rot90(image[:, :, n])
    target_slice = np.rot90(target[:, :, n])
    dl_slice = np.rot90(dl_image[:, :, n])

    # Flip the slices vertically to match Bokeh's default origin
    input_slice = np.flipud(input_slice)
    target_slice = np.flipud(target_slice)
    dl_slice = np.flipud(dl_slice)

    if difference_image is not None:
        difference_slice = np.rot90(difference_image[:, :, n])
        difference_slice = np.flipud(difference_slice)
    else:
        difference_slice = None

    return input_slice, target_slice, dl_slice, difference_slice


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def dash_plot_artifact(image, target, dl_image, difference_image, n):
    """
    Display medical images for a patient: input, target, deep learning output, and the difference.
    """
    # Ensure the slice index n is within the bounds of the image dimensions
    n = min(n, image.shape[1] - 1)

    # Extract the nth slice without any rotation or normalization
    input_slice, target_slice, dl_slice, difference_slice = rotate_and_flip_cor(image, target, dl_image, difference_image, n)

    # Create a subplot grid in Plotly
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Input", "Target", "DL Output", "Difference"))

    # Input Image
    fig.add_trace(go.Heatmap(z=input_slice, colorscale=high_contrast_greys, showscale=False), row=1, col=1)

    # Target Image
    fig.add_trace(go.Heatmap(z=target_slice, colorscale=high_contrast_greys, showscale=False), row=1, col=2)

    # DL Image
    fig.add_trace(go.Heatmap(z=dl_slice, colorscale=high_contrast_greys, showscale=False), row=1, col=3)

    # Difference Image
    fig.add_trace(go.Heatmap(z=difference_slice, colorscale=custom_colorscale, showscale=False), row=1, col=4)

    # Adjust axis numbers sizes
    fig.update_xaxes(tickfont=dict(size=10), ticklen=0)  # Smaller size, shorter ticks
    fig.update_yaxes(tickfont=dict(size=10), ticklen=0)

    # Display the plot in Streamlit
    st.plotly_chart(fig)



def auto_adjust_contrast(slice_data, lower_percentile=0.1, upper_percentile=99.9):
    """
    Automatically adjust contrast by setting intensity limits based on percentiles.
    """
    vmin = np.percentile(slice_data, lower_percentile)
    vmax = np.percentile(slice_data, upper_percentile)
    return vmin, vmax

# Updated `dash_plot_model` function to include the `auto_adjust` option
def dash_plot_model_cor(input_slice, target_slice, dl_slice, colormap='Jet', title="", auto_adjust=False, width=700, height=350):
    """
    Display medical images for a patient: input, target, deep learning output, and the difference.
    """
    if auto_adjust:
        # Auto-adjust contrast for each slice
        vmin_input, vmax_input = auto_adjust_contrast(input_slice)
        vmin_target, vmax_target = auto_adjust_contrast(target_slice)
        vmin_dl, vmax_dl = auto_adjust_contrast(dl_slice)
    else:
        # Use full range of the data
        vmin_input, vmax_input = input_slice.min(), input_slice.max()
        vmin_target, vmax_target = target_slice.min(), target_slice.max()
        vmin_dl, vmax_dl = dl_slice.min(), dl_slice.max()

    # Create a subplot grid in Plotly
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Non-ASC", "CT-ASC", "DL-ASC"))

    # Input Image
    fig.add_trace(go.Heatmap(z=input_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.28, thickness=10, len=1),
                             zmin=vmin_input, zmax=vmax_input), row=1, col=1)

    # Target Image
    fig.add_trace(go.Heatmap(z=target_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.635, thickness=10, len=1),
                             zmin=vmin_target, zmax=vmax_target), row=1, col=2)

    # DL Image
    fig.add_trace(go.Heatmap(z=dl_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.99, thickness=10, len=1),
                             zmin=vmin_dl, zmax=vmax_dl), row=1, col=3)

    # Adjust axis numbers sizes and remove them from the second and third plots
    fig.update_xaxes(tickfont=dict(size=8), ticklen=0, automargin=True)
    fig.update_yaxes(tickfont=dict(size=8), ticklen=0, automargin=True)

    # Turn off axis labels for the second and third plots
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)

    # Adjust the layout to ensure the colorbars don't overlap and set custom size
    fig.update_layout(
        title=title,
        width=width,      # Set the width of the plot
        height=height,    # Set the height of the plot
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    
def vis_model_dash_cor(single_test_file, slice_number=85, colormap='Jet', auto_adjust=False):
    base_name, subfolder_path, dl_image_path = get_image_paths(single_test_file)

    if not os.path.exists(dl_image_path):
        os.makedirs(subfolder_path, exist_ok=True)

        # Initialize Streamlit progress bar and text
        st_progress_bar = st.progress(0)
        st_progress_text = st.empty()

        dl_image = run_model_and_save(single_test_file, subfolder_path, dl_image_path, st_progress_bar, st_progress_text)

        # Clear progress bar and text after completion
        st_progress_bar.empty()
        st_progress_text.empty()

    else:
        input_image, target_image, dl_image = load_images(single_test_file, dl_image_path)

    if dl_image is not None:
        input_image, target_image, dl_image = load_images(single_test_file, dl_image_path)
        input_slice, target_slice, dl_slice, _ = rotate_and_flip_cor(
            input_image,
            target_image,
            dl_image,
            None,
            slice_number
        )
        dash_plot_model_cor(
            input_slice,
            target_slice,
            dl_slice,
            colormap=colormap,
            title="",
            auto_adjust=auto_adjust
        )
    else:
        st.write("Failed to load or generate DL image.")

def vis_model_dash_axial(single_test_file, slice_number=100, colormap='Jet', auto_adjust=False):
    base_name, subfolder_path, dl_image_path = get_image_paths(single_test_file)

    if not os.path.exists(dl_image_path):
        os.makedirs(subfolder_path, exist_ok=True)
        dl_image = run_model_and_save(single_test_file, subfolder_path, dl_image_path)
    else:
        input_image, target_image, dl_image = load_images(single_test_file, dl_image_path)

    if dl_image is not None:
        input_slice, target_slice, dl_slice, _ = rotate_and_flip_axial(
            input_image,
            target_image,
            dl_image,
            None,
            slice_number
        )
        dash_plot_model_axial(
            input_slice,
            target_slice,
            dl_slice,
            colormap=colormap,
            title="",
            auto_adjust=auto_adjust
        )
    else:
        st.write("Failed to load or generate DL image.")





def dash_plot_model_axial(input_slice, target_slice, dl_slice, colormap='Jet', title="", auto_adjust=False, width=700, height=200):
    """
    Display medical images for a patient: input, target, deep learning output, and the difference.
    """
    if auto_adjust:
        # Auto-adjust contrast for each slice
        vmin_input, vmax_input = auto_adjust_contrast(input_slice)
        vmin_target, vmax_target = auto_adjust_contrast(target_slice)
        vmin_dl, vmax_dl = auto_adjust_contrast(dl_slice)
    else:
        # Use full range of the data
        vmin_input, vmax_input = input_slice.min(), input_slice.max()
        vmin_target, vmax_target = target_slice.min(), target_slice.max()
        vmin_dl, vmax_dl = dl_slice.min(), dl_slice.max()

    # Create a subplot grid in Plotly
    fig = make_subplots(rows=1, cols=3)

    # Input Image
    fig.add_trace(go.Heatmap(z=input_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.28, thickness=10, len=1),
                             zmin=vmin_input, zmax=vmax_input), row=1, col=1)

    # Target Image
    fig.add_trace(go.Heatmap(z=target_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.635, thickness=10, len=1),
                             zmin=vmin_target, zmax=vmax_target), row=1, col=2)

    # DL Image
    fig.add_trace(go.Heatmap(z=dl_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.99, thickness=10, len=1),
                             zmin=vmin_dl, zmax=vmax_dl), row=1, col=3)
    # Adjust axis numbers sizes and remove them from the second and third plots
    fig.update_xaxes(tickfont=dict(size=8), ticklen=0, automargin=True)
    fig.update_yaxes(tickfont=dict(size=8), ticklen=0, automargin=True)

    # Turn off axis labels for the second and third plots
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)


    # Adjust the layout to ensure the colorbars don't overlap
    fig.update_layout(
        width=width,      # Set the width of the plot
        height=height,    # Set the height of the plot
        margin=dict(l=10, r=10, t=30, b=10)
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig)




from src.utils import get_image_paths, run_model_and_save, load_images



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



def plot_losses(df_losses, method):
    # Average the losses per epoch for each type (Training and Validation)
    df_avg_losses = df_losses.groupby(['Epoch', 'Type']).mean().reset_index()

    # Filter data based on selections
    traces = []
    
    train_losses = df_avg_losses[df_avg_losses['Type'] == 'Training']
    train_trace = go.Scatter(
        x=train_losses['Epoch'],
        y=train_losses['Loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='blue')
    )
    traces.append(train_trace)

    val_losses = df_avg_losses[df_avg_losses['Type'] == 'Validation']
    val_trace = go.Scatter(
        x=val_losses['Epoch'],
        y=val_losses['Loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='red')
    )
    traces.append(val_trace)

    layout = go.Layout(
        title=f'{method} Training and Validation Losses',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss'),
        hovermode='closest'
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig


def plot_metric(result_df, metric, subtitle, dataset_labels, colors, grouping_column, tickvals, ticktext):
    fig = go.Figure()
    for dataset in dataset_labels:
        for group in result_df[grouping_column].unique():
            filtered_data = result_df[(result_df['Dataset'] == dataset) & (result_df[grouping_column] == group)]
            color_idx = (dataset_labels.index(dataset) * len(result_df[grouping_column].unique()) + list(result_df[grouping_column].unique()).index(group)) % len(colors)
            fig.add_trace(go.Box(
                y=filtered_data[metric],
                name=f"{dataset} ({group})",
                boxmean=True,
                marker_color=colors[color_idx],
                boxpoints='suspectedoutliers'
            ))

    fig.update_layout(
        title=f"                    {metric}",
        yaxis_title=subtitle,
        boxmode='group',
        legend=dict(font=dict(size=6)),
        margin=dict(l=0, r=0, t=40, b=40),
        height=250,
        width=600,
        xaxis=dict(
            linecolor='black',
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0
        ),
        yaxis=dict(
            linecolor='black',
            tickangle=0
        )
    )
    return fig


def model_visualize_coronal_dash(data, predict, n, title, cm, Norm=False):
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
    # Instead of plt.show(), use st.pyplot to display the plot in Streamlit
    st.pyplot(fig)

    # Close the figure after displaying it to avoid memory leaks
    plt.close(fig)

