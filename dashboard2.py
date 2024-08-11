import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import json
from src.vis import  vis_model_dash_axial, vis_model_dash_cor
import os
import nibabel as nib
from src.utils import Config, load_df_from_pickle
from src.vis import dash_plot_artifact


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
df = load_df_from_pickle('/students/2023-2024/master/Shahpouri/DATA/Artifact_data.pkl')

# Layout setup and execution
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown("<div class='header'>Deep Learning-Based PET Image Correction Toward Quantitative Imaging</div>", unsafe_allow_html=True)

# Define tabs
tab0, tab1, tab2 = st.tabs(["Home 🏠", "Model 🤖 ", "Artifacts 🩻"])

# Sidebar
with st.sidebar:
        # Model selection
    model_labels = ['IMCM', 'ADCM']
    selected_model = st.selectbox("Select a Model", model_labels, index=0)

    st.sidebar.markdown("---")
    selected_one = st.sidebar.selectbox("Select a patient", [name[0] for name in test_name])

    # List of available color maps in Plotly
    # available_colormaps = ['Jet', 'Viridis', 'Plasma', 'Magma', 'Cividis', 'Turbo', 'RdBu']
    available_colormaps = [
    'Jet', 'Viridis', 'Plasma', 'Magma', 'Cividis', 'Turbo', 'RdBu', 
    'Inferno', 'Blues', 'Greens', 'Greys', 'Oranges', 'Purples', 'Reds', 
    'Purpor', 'Sunset', 'Sunsetdark', 'Teal', 'Tealgrn', 'Temps', 
    'Thermal', 'Aggrnyl', 'Agsunset', 'Amp', 'Deep', 'Dense', 'Earth', 
    'Electric', 'Pink', 'Portland', 'Rainbow', 'Spectral', 'Tropic', 
    'Brwnyl', 'Picnic', 'Puor', 'RdGy', 'RdYlBu', 'RdYlGn', 'Geyser', 
    'Icefire', 'Phase', 'Twilight', 'TwilightShifted', 
    'Bluered', 'Blackbody']

    # Sidebar or main interface for selecting the color map
    selected_colormap = st.selectbox("Change Color Map", available_colormaps, index=0)
    auto_adjust = st.checkbox("Auto Adjust Contrast", value=False)
    st.sidebar.markdown("---")


    selected_patient = st.sidebar.selectbox("Select a artifactual data", df['name'])

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

                
        with col3:
            vis_model_dash_cor(single_test_file, selected_model, slice_number=slice_number_coronal, colormap=selected_colormap, auto_adjust=auto_adjust)
            vis_model_dash_axial(single_test_file, selected_model, slice_number=slice_number_axial, colormap=selected_colormap, auto_adjust=auto_adjust)

    else:
        st.write("Test file not found.")


with tab2:
    st.markdown(f"Artifacts for Patient: {selected_patient}")
    
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

    if image is not None:
        # Add a slider for slice index selection
        num_slices = image.shape[1]  # Assuming images are 3D arrays with shape (x, y, z)
        slice_index = st.slider("Select Slice Index", 0, num_slices - 1, 85)  # Default to 85 or any valid slice index
        
        # Display the selected slice using the provided function
        dash_plot_artifact(image, target, dl_image, difference_image, slice_index)
    else:
        st.write(f"No data found for patient {selected_patient}.")