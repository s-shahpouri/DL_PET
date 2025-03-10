"""
This Streamlit application serves as a dashboard for evaluating and visualizing deep learning-based
attenuation scatter correction (ASC) in 68Ga-PSMA PET/CT imaging. The application provides tools for
visualizing model outputs, artifact detection, and an overview of the project's objectives.


Tabs:
1. **Model Evaluation 🧠**:
    - Allows users to select a deep learning model and a patient, adjust the colormap, and visualize slices of the image.
    - Users can adjust the slice index for coronal and axial views to inspect different parts of the PET/CT images.

2. **Artifacts Detection 🩻**:
    - Focuses on detecting artifacts within the PET/CT images.
    - Provides similar controls for slice selection and colormap adjustment, with an additional focus on visualizing discrepancies between the predicted and actual images.

3. **About 📜**:
    - Offers background information on the project and how it contributes to advancing quantitative imaging in PET/CT.
    - Contact details and links to relevant profiles and resources are provided.

Usage:
- Run the application in a Python environment with Streamlit installed.
- The interface allows users to interactively explore deep learning models' outputs and artifact detection results.
- Suitable for researchers and practitioners working on medical imaging, particularly in the context of PET/CT.

Author: Sama Shahpouri
Last Edit: 25-08-2024

Example:
```bash
streamlit run dashboard.py"""


import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import json
from src.dash import  vis_model_dash_axial, vis_model_dash_cor
import os
import nibabel as nib
from src.dash import Config, load_df_from_pickle
from src.dash import vis_artifact_dash_cor, vis_artifact_dash_axial


st.set_page_config(
    page_title="DL ASC PET App",
    page_icon="/students/2023-2024/master/Shahpouri/DL_PET/info/PET_ICONE.png",
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
df = load_df_from_pickle('/students/2023-2024/master/Shahpouri/DATA/Artifact_data.pkl')

# Layout setup and execution
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown("<div class='header'>CT-free ASC of PET images Application</div>", unsafe_allow_html=True)

# Define tabs
tab1, tab2, tab0 = st.tabs([" Model Evaluation 🧠 |", " Artifacts Detection 🩻 |", " About 📜 |"])

# Sidebar
with st.sidebar:

    model_labels = ['IMCM', 'ADCM']
    selected_model = st.selectbox("Select a DL Model", model_labels, index=0)

    selected_one = st.sidebar.selectbox("Select a patient", [name[0] for name in test_name])

    # List of available color maps in Plotly
    # available_colormaps = ['Jet', 'Viridis', 'Plasma', 'Magma', 'Cividis', 'Turbo', 'RdBu']
    available_colormaps = [
    'Jet', 'Greys','Viridis','Plasma',
    'Magma', 'Turbo', 'Thermal', 'Blackbody']

    # Sidebar or main interface for selecting the color map
    selected_colormap = st.selectbox("Change Color-Map", available_colormaps, index=0)
    auto_adjust = st.checkbox("Auto Adjust Contrast", value=False)
    st.divider()


    selected_patient = st.sidebar.selectbox("Select an artifactual data", df['name'])

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.divider()
    st.caption(f"Data Contributors: [Inselgruppe](https://inselgruppe.ch/de/die-insel-gruppe)",unsafe_allow_html=True)


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

        col1, col2, col3 = st.columns([1,0.1, 5])
        with col1:
            st.markdown("")
            slice_number_coronal = st.slider("Select Coronal Slice", 0, nib.load(single_test_file['image']).get_fdata().shape[1] - 1, 85)

            st.markdown("")
            st.header("")
            st.header("")
            st.header("")
            st.markdown("")

            
            slice_number_axial = st.slider("Select Axial Slice", 0, nib.load(single_test_file['image']).get_fdata().shape[2] - 1, 85)

        with col2:
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")

                
        with col3:
            vis_model_dash_cor(single_test_file, selected_model, slice_number=slice_number_coronal, colormap=selected_colormap, auto_adjust=auto_adjust)
            vis_model_dash_axial(single_test_file, selected_model, slice_number=slice_number_axial, colormap=selected_colormap, auto_adjust=auto_adjust)

    else:
        st.write("Test file not found.")


with tab2:
    st.session_state.active_tab = " Artifacts Detection 🩻 "
    selected_colormap = 'Greys'
    # Function to get patient data
    @st.cache_data(show_spinner=False)
    def get_patient_data(patient_name):
        patient_data = df[df['name'] == patient_name]
        if patient_data.empty:
            return None, None, None, None
        image = patient_data['image_matrix'].values[0]
        target = patient_data['target_matrix'].values[0]
        dl_image = patient_data['dl_image_matrix'].values[0]
        difference_image = patient_data['difference_matrices'].values[0]
        return image, target, dl_image, difference_image
    
    with st.spinner(f"Loading data for {selected_patient}..."):
        image, target, dl_image, difference_image = get_patient_data(selected_patient)


        col1, col2, col3 = st.columns([1,0.1, 5])
        
        with col1:
            st.markdown("")
            num_slices_cor = image.shape[1]  # Assuming images are 3D arrays with shape (x, y, z)
            slice_index_cor = st.slider("Select Slice Index", 0, num_slices_cor - 1, 85)  # Default to 85 or any valid slice index
        
            st.markdown("")
            st.header("")
            st.header("")
            st.header("")
            st.markdown("")

            num_slices_axial = image.shape[2]
            slice_index_axial = st.slider("Select Axial Slice", 0, num_slices_axial - 1, 110)

        with col2:
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
            st.markdown(".")
            st.markdown("")
                
        with col3:
            # vis_model_dash_cor(single_test_file, selected_model, slice_number=slice_number_coronal, colormap=selected_colormap, auto_adjust=auto_adjust)
            # vis_model_dash_axial(single_test_file, selected_model, slice_number=slice_number_axial, colormap=selected_colormap, auto_adjust=auto_adjust)
            if image is not None:

            # Display the selected slice using the provided function
                vis_artifact_dash_cor(image, target, dl_image, difference_image, slice_index_cor, colormap=selected_colormap, auto_adjust=auto_adjust)
                vis_artifact_dash_axial(image, target, dl_image, difference_image, slice_index_axial, colormap=selected_colormap, auto_adjust=auto_adjust)

            else:
                st.write(f"No data found for patient {selected_patient}.")


with tab0:
    st.markdown("")
    st.markdown("")
    st.markdown("""
        <div style="text-align: center; color: #762EB1; font-weight: bold; font-size: 24px;">
        Deep Learning-Based PET Image Correction Toward Quantitative Imaging
        </div>
        """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("""This dashboard is part of a master's thesis project focused on developing deep learning models to enhance PET imaging by directly correcting attenuation and scatter without relying on anatomical information from CT scans.
                    The primary goal is to create a universal model that works across different scanners,
                    And ensuring accurate artifact detection and correction in 68Ga-PET imaging.""")

    
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("""        <p style="font-weight: bold"> Key Features:</p>
        <ul>
        <li>Direct Attenuation and Scatter Correction (ASC) without CT data</li>
        <li>Universal model applicable to different scanners</li>
        <li>Artifact detection and correction for multi-center PET datasets</li>
        </ul>
        <p>Explore the different tabs to evaluate model performance and view artifact corrections.</p>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.divider()
    st.markdown("""
        <div style="color: #762EB1;">
        Contact: <a href="mailto:z.shahpouri@gmail.com" style="color: #762EB1;">z.shahpouri@gmail.com</a>
        </div>
        """, unsafe_allow_html=True)
