import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import json
from streamlit_lottie import st_lottie
from src.utils import Config, load_df_from_pickle
from src.vis import dash_plot_artifact, plot_losses, plot_metric
import os 

config_file = 'src/config.json'
config = Config(config_file)

# Example usage
df = load_df_from_pickle('/students/2023-2024/master/Shahpouri/DATA/Artifact_data.pkl')
df_losses_paths = {
    "ADCM": '/students/2023-2024/master/Shahpouri/DATA/adcm_loss.pkl',
    "IMCM": '/students/2023-2024/master/Shahpouri/DATA/imcm_loss.pkl'
}

df_dataset_result = {
    "⁶⁸GA": '/students/2023-2024/master/Shahpouri/DATA/metric_data_ga.csv',
    "¹⁸F-FDG": '/students/2023-2024/master/Shahpouri/DATA/metric_data_fdg.csv'
}

metrics_info = {
    'Mean Error (SUV)': 'ME',
    'Mean Absolure Error (SUV)': 'MAE',
    'Relative Error (SUV%)': 'RE(%)',
    'Root Mean Squared Error': 'RMSE',
    'Peak Signal-to-Noise Ratio': 'PSNR',
    'Structual Similarity Index': 'SSIM'
}

dataset_labels = ['IMCM', 'ADCM']
center_groups = ['Internal', 'External']
colors = ['cornflowerblue', 'royalblue', 'sandybrown', 'coral', 'yellowgreen', 'limegreen', 'plum', 'darkviolet', 'peachpuff', 'darkorange']


# st.set_page_config(layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# st.markdown('<link rel="stylesheet" href="/students/2023-2024/master/Shahpouri/DL_PET/style.css">', unsafe_allow_html=True)


# Logos and Header
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown("<div class='header'>Deep Learning-Based PET Image Correction Toward Quantitative Imaging</div>", unsafe_allow_html=True)

# Load the CSS for styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Define tabs with the desired labels
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Home 🏠", 
    "PET & ASC ⚛️", 
    "Objective 🎯", 
    "Data Preparation 🗂️", 
    "Model 🤖 ", 
    "Results 📊", 
    "Artifacts 🧬",
    "Discussion 🗣️",
    "Future Work: 🔮"
])



# Display the images in the sidebar
with st.sidebar:

    # Select method - Placeholder for future use
    method = st.sidebar.selectbox("Select Method", ["ADCM", "IMCM"], index=0)
    st.sidebar.markdown("---")
    # Define Streamlit layout
    st.sidebar.title("Result Tab Options")
    dataset_choice = st.sidebar.selectbox("Select Dataset", ["⁶⁸GA", "¹⁸F-FDG"], index=0)
    st.sidebar.markdown("---")
        # Sidebar for selecting a patient
    st.sidebar.title("Select a Patient")
    selected_patient = st.sidebar.selectbox("Patient Name", df['name'])



with tab0:

    # Load the Lottie animation from the JSON file
    with open("/students/2023-2024/master/Shahpouri/DATA/Aniki_Hamster.json", "r") as f:
        lottie_data = json.load(f)
    
    # Use columns to center the Lottie animation
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st_lottie(lottie_data, speed=1, reverse=False, loop=True, height=240, width=240)

    st.markdown("<div class='subheader'>Author: <strong>Sama Shahpouri, 460145</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Study: <strong>Data Science for Life Science</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Institute: <strong>Hanze University of Applied Sciences, Institute of Life Science & Technology</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Supervisors: <strong>Dr. Isaac Shiri, Dr. Dave, Dr.</strong></div>", unsafe_allow_html=True)
    


with tab1:
    st.title('')
    base = '/students/2023-2024/master/Shahpouri/DL_PET/info/'
    image_paths = [
        base + 'true.png',
        base + 'scatter.png',
        base + 'random.png',
        base + 'multiple.png',
        base + 'CT-ASC.png'
    ]

    captions = [
        'True Coincidence',
        'Scatter Coincidence',
        'Random Coincidence',
        'Multiple Coincidence',
        'Before & After CT base ASC'
    ]

    # Initialize session state variable for the image index
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    # Function to change image index
    def change_image_index(change):
        st.session_state.image_index += change
        # Ensure the index stays within bounds
        st.session_state.image_index = max(0, min(len(image_paths) - 1, st.session_state.image_index))

    # Display previous and next buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col2:
        if st.button('Previous') and st.session_state.image_index > 0:
            change_image_index(-1)

    with col3:
        current_image_path = image_paths[st.session_state.image_index]
        st.markdown('')
        st.image(current_image_path, width=300, caption=f"{captions[st.session_state.image_index]}")

    with col4:
        if st.button('Next') and st.session_state.image_index < len(image_paths) - 1:
            change_image_index(1)

# Tab 2 content
with tab2:
    st.write("Content for Tab 2 goes here.")
    text_input = st.text_input("Enter some text")
    st.write(f"You entered: {text_input}")

# Tab 3 content
with tab3:

    # Data for the first table
    data1 = {
        "Center": ["Center 1", "Center 2", "Center 3", "Center 4", "External Center", "Total"],
        "No": [56, 31, 45, 40, 12, 184],
        "Train": [43, 25, 35, 28, "-", 131],
        "Validation": [11, 4, 8, 10, "-", 33],
        "Test": [2, 2, 2, 2, 12, 20],
        "Scanner": ["Siemens Biograph 6", "GE Discovery IQ", "Siemens mCT", "Siemens Biograph 6", "Siemens Horizon", "-"],
        "Reconstruction": ["3D-OSEM", "3D-OSEM", "3D-OSEM", "3D-OSEM", "PSF+TOF+3D-OSEM", "-"],
        "Matrix size × Z*": ["168 × 168", "192 × 192", "200 × 200", "168 × 168", "180 × 180", "-"]
    }

    # Convert to DataFrame
    df1 = pd.DataFrame(data1)

    # Display the first table
    st.subheader("Data Distribution Across Centers")
    st.table(df1)

    # Space between tables or content
    st.markdown("<hr>", unsafe_allow_html=True)

    # Add more content or another table below this line
    st.subheader("Additional Information or Table")
    st.write("Add any additional data or tables here.")
# Tab 4 content
with tab4:
    # Load the appropriate DataFrame based on the selected method
    df_losses = load_df_from_pickle(df_losses_paths[method])

    # Plot the losses using the function
    fig = plot_losses(df_losses, method)
    st.plotly_chart(fig)
# Add divider for clarity


with tab5:


    result_df = pd.read_csv(df_dataset_result[dataset_choice])
    st.header("Quantitative metrics")

    # Define a layout with columns for grouping option and metrics selection
    col1, col2 = st.columns([1, 5])  # Adjust the column width ratio as needed


    with col1:
        # Radio button for selecting grouping option
        grouping_option = st.radio(
            "Select Grouping Option",
            options=["Grouped", "Centers"],
            index=0
        )

    with col2:
        # Multiselect for selecting which metrics to display
        selected_metrics = st.multiselect(
            "Select Metrics to Display",
            list(metrics_info.keys()),
            default=list(metrics_info.keys())
        )

    # Determine the grouping column based on the radio button selection
    if grouping_option == "Grouped":
        grouping_column = 'Center_Group'
        if dataset_choice == "⁶⁸GA":
            ticktext = ["Internal", "External"]
            tickvals = [0.5, 2.5]
        else:
            ticktext = ["External"]
            tickvals = [0.5]
    else:
        grouping_column = 'Center'
        ticktext = []
        tickvals = []
        centers = result_df['Center'].unique()
        for idx, center in enumerate(centers):
            ticktext.extend([center, ''])
            tickvals.extend([idx * 2, idx * 2 + 1])

    # Loop through the selected metrics
    for i in range(0, len(selected_metrics), 2):
        col1, col2 = st.columns(2)  # Create two columns for side-by-side plots

        # Plot the first metric in the first column
        with col1:
            if i < len(selected_metrics):
                metric = selected_metrics[i]
                subtitle = metrics_info[metric]
                fig = plot_metric(result_df, metric, subtitle, dataset_labels, colors, grouping_column, tickvals, ticktext)
                st.plotly_chart(fig)

        # Plot the second metric in the second column
        with col2:
            if i + 1 < len(selected_metrics):
                metric = selected_metrics[i + 1]
                subtitle = metrics_info[metric]
                fig = plot_metric(result_df, metric, subtitle, dataset_labels, colors, grouping_column, tickvals, ticktext)
                st.plotly_chart(fig)


# Artifacts tab content
with tab6:

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