import nibabel as nib
import pandas as pd
from src.data_preparation import LoaderFactory
from monai.inferers import sliding_window_inference
from src.model_manager import ModelLoader
from monai.transforms import Compose, Invertd, SaveImaged
from monai.data import decollate_batch
import os
import torch
import json
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import time

custom_colorscale = [
    [0.0, "red"],
    [0.4, "white"],
    [0.8, "white"],
    [1.0, "blue"]
]


high_contrast_greys = [
    [0.0, 'rgb(255, 255, 255)'],  # Pure white background
    [0.3, 'rgb(40, 40, 40)'],  # Dark grey
    [1.0, 'rgb(0, 0, 0)']         # Pure black for hot spots
]


class Config:
    """
    A class to handle configuration settings from a JSON file and manage device selection.

    Attributes:
        config_file (str): Path to the JSON configuration file.
        device (torch.device): The device (CPU or CUDA) selected based on the configuration.

    Methods:
        get_device():
            Determines and returns the appropriate computation device (CPU or CUDA).
    """
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

# Load configuration
config_file = 'src/config.json'
config = Config(config_file)


def load_df_from_pickle(filename='/students/2023-2024/master/Shahpouri/DATA/Artifact_data.pkl'):
    """Load the DataFrame from a Pickle file."""
    try:
        df = pd.read_pickle(filename)
        print(f"DataFrame loaded from {filename}")
        return df
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    

def get_image_paths(single_test_file, selected_model):
    base_name = os.path.splitext(os.path.splitext(os.path.basename(single_test_file['image']))[0])[0]
    subfolder_path = os.path.join(config.dash_output_dir, base_name)
    dl_image_path = os.path.join(subfolder_path, base_name, f"{base_name}_{selected_model}.nii.gz")
    return base_name, subfolder_path, dl_image_path


def load_images(single_test_file, dl_image_path):
    input_image = nib.load(single_test_file['image']).get_fdata()
    target_image = nib.load(single_test_file['target']).get_fdata()
    dl_image = nib.load(dl_image_path).get_fdata() if os.path.exists(dl_image_path) else None
    return input_image, target_image, dl_image


def run_model_and_save(single_test_file, subfolder_path, dl_image_path, selected_model, st_progress_bar, st_progress_text):
    loader_factory = LoaderFactory(
        train_files=None,
        val_files=None,
        test_files=[single_test_file],
        patch_size=config.patch_size,
        spacing=config.spacing,
        spatial_size=config.spatial_size,
        normalize=config.normalize
    )

    single_test_loader = loader_factory.get_loader('test', batch_size=1, num_workers=config.num_workers['test'], shuffle=False)
    model_loader = ModelLoader(config)
    model = model_loader.call_model()

    if selected_model == "ADCM":
        model_path = 'Results/model_5_2_14_27.pth'
    else:
        model_path = 'Results/model_4_24_23_17.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        raise FileNotFoundError(f"Model in {model_path} not found.")

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=loader_factory.get_test_transforms(),
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=subfolder_path, output_postfix=selected_model, resample=False), 
        ]
    )

    with torch.no_grad():
        for data in single_test_loader:
            # Simulate progress bar updates
            for progress in np.linspace(0, 0.7, num=7):
                time.sleep(0.1)  # Sleep to simulate processing time
                st_progress_bar.progress(progress)
                st_progress_text.text(f"Processing: {int(progress * 100)}% complete")

            # Perform actual inference
            data["pred"] = sliding_window_inference(
                data["image"].to(config.device), 
                (168, 168, 16), 
                64, 
                model,
                overlap=0.70
            )

            # Apply post-processing (which will save the processed image)
            post_processed = [post_transforms(i) for i in decollate_batch(data)]

            if selected_model == 'ADCM':
                original_nii = nib.load(dl_image_path)  # Load the original NIfTI image to get the meta
                dl_adcm = original_nii.get_fdata()
                nac_img = nib.load(single_test_file['image']).get_fdata()
                # Apply your custom calculation
                dl_final = calculate_dl_mac(nac_img, dl_adcm, 2, 5, 50)

                # Create a new NIfTI image with the same header and affine as the original
                dl_final_nii = nib.Nifti1Image(dl_final, affine=original_nii.affine, header=original_nii.header)

                # Save the final NIfTI image to the same path
                nib.save(dl_final_nii, dl_image_path)



            # Load the final processed image
            dl_image = nib.load(dl_image_path).get_fdata()

    return dl_image



def normalize_image(image):
    """Normalize image data to the range [0, 255] for display."""
    image_min, image_max = np.min(image), np.max(image)
    if image_max > image_min:  # Avoid division by zero
        image = (image - image_min) / (image_max - image_min) * 255
    return image.astype(np.uint8)


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


def auto_adjust_contrast(slice_data, lower_percentile=0.15, upper_percentile=99.85):
    """
    Automatically adjust contrast by setting intensity limits based on percentiles.
    """
    vmin = np.percentile(slice_data, lower_percentile)
    vmax = np.percentile(slice_data, upper_percentile)
    return vmin, vmax


def vis_artifact_dash_cor(input_slice, target_slice, dl_slice, difference_slice, slice_number, colormap='Greys', title="", auto_adjust=False, width=800, height=300):
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


    input_slice, target_slice, dl_slice, difference_slice = rotate_and_flip_cor(input_slice, target_slice, dl_slice, difference_slice, slice_number)

    # Create a subplot grid in Plotly
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Non-ASC", "CT-ASC", "DL-ASC", "Difference"))

    # Input Image
    fig.add_trace(go.Heatmap(z=input_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.20, thickness=10, len=1),
                             zmin=vmin_input, zmax=vmax_input), row=1, col=1)

    # Target Image
    fig.add_trace(go.Heatmap(z=target_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.465, thickness=10, len=1),
                             zmin=vmin_target, zmax=vmax_target), row=1, col=2)

    # DL Image
    fig.add_trace(go.Heatmap(z=dl_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.725, thickness=10, len=1),
                             zmin=vmin_dl, zmax=vmax_dl), row=1, col=3)

    fig.add_trace(go.Heatmap(z=difference_slice, colorscale=custom_colorscale, showscale=True, colorbar=dict(x=0.99, thickness=10, len=1)),
                            row=1, col=4)

    # Adjust axis numbers sizes and remove them from the second and third plots
    fig.update_xaxes(tickfont=dict(size=8), ticklen=0, automargin=True)
    fig.update_yaxes(tickfont=dict(size=8), ticklen=0, automargin=True)

    # Turn off axis labels for the second and third plotcd
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=4)
    # Adjust the layout to ensure the colorbars don't overlap and set custom size
    fig.update_layout(
        title=title,
        width=width,      # Set the width of the plot
        height=height,    # Set the height of the plot
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


def vis_artifact_dash_axial(input_slice, target_slice, dl_slice, difference_slice, slice_number, colormap='Greys', title="", auto_adjust=False, width=800, height=200):
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


    input_slice, target_slice, dl_slice, difference_slice = rotate_and_flip_axial(input_slice, target_slice, dl_slice, difference_slice, slice_number)

    # Create a subplot grid in Plotly
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Non-ASC", "CT-ASC", "DL-ASC", "Difference"))

    # Input Image
    fig.add_trace(go.Heatmap(z=input_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.20, thickness=10, len=1),
                             zmin=vmin_input, zmax=vmax_input), row=1, col=1)

    # Target Image
    fig.add_trace(go.Heatmap(z=target_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.465, thickness=10, len=1),
                             zmin=vmin_target, zmax=vmax_target), row=1, col=2)

    # DL Image
    fig.add_trace(go.Heatmap(z=dl_slice, colorscale=colormap, showscale=True, colorbar=dict(x=0.725, thickness=10, len=1),
                             zmin=vmin_dl, zmax=vmax_dl), row=1, col=3)

    fig.add_trace(go.Heatmap(z=difference_slice, colorscale=custom_colorscale, showscale=True, colorbar=dict(x=0.99, thickness=10, len=1)),
                            row=1, col=4)

    # Adjust axis numbers sizes and remove them from the second and third plots
    fig.update_xaxes(tickfont=dict(size=8), ticklen=0, automargin=True)
    fig.update_yaxes(tickfont=dict(size=8), ticklen=0, automargin=True)

    # Turn off axis labels for the second and third plotcd
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=4)
    # Adjust the layout to ensure the colorbars don't overlap and set custom size
    fig.update_layout(
        title=title,
        width=width,      # Set the width of the plot
        height=height,    # Set the height of the plot
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


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

    
def vis_model_dash_cor(single_test_file, selected_model, slice_number=85, colormap='Jet', auto_adjust=False):
    base_name, subfolder_path, dl_image_path = get_image_paths(single_test_file, selected_model)

    if not os.path.exists(dl_image_path):
        os.makedirs(subfolder_path, exist_ok=True)

        # Initialize Streamlit progress bar and text
        st_progress_bar = st.progress(0)
        st_progress_text = st.empty()

        dl_image = run_model_and_save(single_test_file, subfolder_path, dl_image_path, selected_model, st_progress_bar, st_progress_text)

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


def vis_model_dash_axial(single_test_file, selected_model, slice_number=100, colormap='Jet', auto_adjust=False):
    base_name, subfolder_path, dl_image_path = get_image_paths(single_test_file, selected_model)

    if not os.path.exists(dl_image_path):
        os.makedirs(subfolder_path, exist_ok=True)
        dl_image = run_model_and_save(single_test_file, subfolder_path, dl_image_path, selected_model)
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

