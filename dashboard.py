import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Compose, Invertd, SaveImaged
from src.utils import Config, load_df_from_pickle
from src.data_preparation import DataHandling, LoaderFactory
from src.vis import visualize_axial_slice, visualize_coronal_slice, dash_plot_artifact, display_patient_coronal
from src.model_manager import ModelLoader
import pandas as pd
import os
import plotly.graph_objs as go

config_file = 'src/config.json'
config = Config(config_file)


# Example usage
df = load_df_from_pickle('/students/2023-2024/master/Shahpouri/DATA/Artifact_data.pkl')
# Define paths to loss data
df_losses_paths = {
    "ADCM": '/students/2023-2024/master/Shahpouri/DATA/adcm_loss.pkl',
    "IMCM": '/students/2023-2024/master/Shahpouri/DATA/imcm_loss.pkl'
}

# Define the function for plotting training and validation losses
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


# Define tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Home", "PET & ASC", "Objective", "Data Prepration", "Model", "Artifacts"])

# Home tab content
with tab1:
    st.title("Deep Learning-Based PET Image Correction Toward Quantitative Imaging")
    st.write("Sama Shahpouri")
    st.write("Data Science for life science")
    st.write("Hanze university of applied science")

# Tab 1 content
with tab2:
    st.write("Content for Tab 1 goes here.")
    value = st.slider("Choose a value", 0, 100)
    st.write("Selected value:", value)

# Tab 2 content
with tab3:
    st.write("Content for Tab 2 goes here.")
    text_input = st.text_input("Enter some text")
    st.write(f"You entered: {text_input}")

# Tab 3 content
with tab4:
    st.write("Content for Tab 3 goes here.")
    st.bar_chart([1, 2, 3, 4])

# Tab 4 content
with tab5:
    st.sidebar.title("Model Options")
    
    # Select method - Placeholder for future use
    method = st.sidebar.selectbox("Select Method", ["ADCM", "IMCM"], index=0)

    # Load the appropriate DataFrame based on the selected method
    df_losses = load_df_from_pickle(df_losses_paths[method])

    # Plot the losses using the function
    fig = plot_losses(df_losses, method)
    st.plotly_chart(fig)
# Add divider for clarity
st.sidebar.markdown("---")

# Artifacts tab content
with tab6:
    # Sidebar for selecting a patient
    st.sidebar.title("Select a Patient")
    selected_patient = st.sidebar.selectbox("Patient Name", df['name'])
    st.header(f"Artifacts for Patient: {selected_patient}")
    
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