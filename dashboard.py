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

config_file = 'src/config.json'
config = Config(config_file)


# Example usage
df = load_df_from_pickle('/students/2023-2024/master/Shahpouri/DATA/Artifact_data.pkl')

hint = 'dl4_23'  # Replace with the actual hint
n = 85  # You can choose any valid slice index

# # Function to plot a sample patient
# def plot_sample_patient(df, patient_name):
#     # Find the patient data in the DataFrame
#     patient_data = df[df['name'] == patient_name]
    
#     if patient_data.empty:
#         st.write(f"No data found for patient {patient_name}")
#         return
    
#     # Extract the image matrices
#     image = patient_data['image_matrix'].values[0]
#     target = patient_data['target_matrix'].values[0]
#     dl_image = patient_data['dl_image_matrix'].values[0]
#     difference_image = patient_data['difference_matrices'].values[0]
    


#     # Display the images using the provided function
#     dash_plot_artifact(patient_name, image, target, dl_image, difference_image, n, cmp="gist_yarg")



#################################################################
# Sidebar for selecting a patient
st.sidebar.title("Select a Patient")
selected_patient = st.sidebar.selectbox("Patient Name", df['name'])


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
    st.write("Content for Tab 4 goes here.")
    st.line_chart([10, 20, 30, 40])


with tab6:

# Function to plot a sample patient
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
        st.header(f"Artifacts for Patient: {selected_patient}")
        dash_plot_artifact(selected_patient, image, target, dl_image, difference_image, n, cmp="gist_yarg")
    else:
        st.write(f"No data found for patient {selected_patient}.")