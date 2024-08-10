import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import json
from streamlit_lottie import st_lottie
from src.utils import Config, get_image_paths, run_model_and_save, load_images
from src.vis import rotate_and_flip_cor, rotate_and_flip_axial, dash_plot_model, dash_plot_model_axial
import os
from src.data_preparation import LoaderFactory
import torch
from monai.inferers import sliding_window_inference
from src.model_manager import ModelLoader
from plotly.subplots import make_subplots
import numpy as np
from monai.transforms import Compose, Invertd, SaveImaged
from monai.data import decollate_batch
import nibabel as nib

st.set_page_config(
    page_title="DL ASC PET App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/s-shahpouri/DL_PET',
        'Report a bug': "mailto:z.shahpouri@gmail.com",
        'About': "This App used for Deep learning modeling attenuation scatter correction of 68Ga-PSMA PET/CT. This project at artificial intelligence in cardiac imaging laboratory, Inselspital, University of Bern, was done for fulfilment of the master of Data science for life science at Hanze University of Applied Science."
    }
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load configuration
config_file = 'src/config.json'
config = Config(config_file)

# Load test files
with open('Results/test_files.json', 'r') as f:
    test_files = json.load(f)

# Prepare the test names for display
test_name = [
    (
        os.path.splitext(os.path.splitext(os.path.basename(file_info['image']))[0])[0],
        os.path.splitext(os.path.splitext(os.path.basename(file_info['target']))[0])[0]
    )
    for file_info in test_files
]

# Main visualization functions
def vis_model_dash_cor(single_test_file, slice_number=85, colormap='Jet'):
    base_name, subfolder_path, dl_image_path = get_image_paths(single_test_file)

    if not os.path.exists(dl_image_path):
        os.makedirs(subfolder_path, exist_ok=True)
        dl_image = run_model_and_save(single_test_file, subfolder_path, dl_image_path)
    else:
        input_image, target_image, dl_image = load_images(single_test_file, dl_image_path)

    if dl_image is not None:
        input_slice, target_slice, dl_slice, _ = rotate_and_flip_cor(
            input_image,
            target_image,
            dl_image,
            None,
            slice_number
        )
        dash_plot_model(
            input_slice,
            target_slice,
            dl_slice,
            colormap=colormap,
            title="DL Post-Processed (Coronal View)"
        )
    else:
        st.write("Failed to load or generate DL image.")

def vis_model_dash_axial(single_test_file, slice_number=100, colormap='Jet'):
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
            title="DL Post-Processed (Axial View)"
        )
    else:
        st.write("Failed to load or generate DL image.")

# Layout setup and execution
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown("<div class='header'>Deep Learning-Based PET Image Correction Toward Quantitative Imaging</div>", unsafe_allow_html=True)

# Define tabs
tab0, tab1, tab2 = st.tabs(["Home 🏠", "Model 🤖 ", "Artifacts 🩻"])

# Sidebar
with st.sidebar:
    st.sidebar.markdown("---")
    selected_one = st.sidebar.selectbox("Select a Test File", [name[0] for name in test_name])

    # List of available color maps in Plotly
    available_colormaps = ['Jet', 'Viridis', 'Plasma', 'Magma', 'Cividis', 'Turbo', 'RdBu']

    # Sidebar or main interface for selecting the color map
    selected_colormap = st.selectbox("Select a Color Map", available_colormaps, index=0)

    st.sidebar.markdown("---")

with tab1:
    single_test_file = next(
        (
            file for file in test_files 
            if os.path.splitext(os.path.splitext(os.path.basename(file['image']))[0])[0] == selected_one
        ),
        None
    )

    if single_test_file:
        # Create layout for coronal and axial views

        col1, col2 = st.columns([1, 3])
        with col1:
            slice_number_coronal = st.slider("Select Coronal Slice Index", 0, nib.load(single_test_file['image']).get_fdata().shape[1] - 1, 85)
        with col2:
            vis_model_dash_cor(single_test_file, slice_number=slice_number_coronal, colormap=selected_colormap)


        col3, col4 = st.columns([1, 3])
        with col3:
            slice_number_axial = st.slider("Select Axial Slice Index", 0, nib.load(single_test_file['image']).get_fdata().shape[2] - 1, 85)
        with col4:
            vis_model_dash_axial(single_test_file, slice_number=slice_number_axial, colormap=selected_colormap)
    else:
        st.write("Test file not found.")
