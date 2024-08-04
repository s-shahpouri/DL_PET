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


st.set_page_config(
    page_title="DL ASC PET App",
    page_icon="📊",  # Example using the X-ray emoji
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/s-shahpouri/DL_PET',
        'Report a bug': "mailto:z.shahpouri@gmail.com",
        'About': "This App used for Deep learning modeling attenuation scatter correction of 68Ga-PSMA PET/CT. This project at artificial intelligence in cardiac imaging laboratory, Inselspital, University of Bern, was done for fulfilment of the master of Data science for life science at Hanze University of Applied Science."
    }
)
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


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Logos and Header
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown("<div class='header'>Deep Learning-Based PET Image Correction Toward Quantitative Imaging</div>", unsafe_allow_html=True)


# Define tabs with the desired labels
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Home 🏠", 
    "PET & ASC ⚛️", 
    "Objective 🎯", 
    "Data Preparation 🗂️", 
    "Model 🤖 ", 
    "Results 📊", 
    "Artifacts 🩻",
    "Discussion 🗣️",
    "Future Work: 🔮"
])



# Display the images in the sidebar
with st.sidebar:

    # Select method - Placeholder for future use
    method = st.sidebar.selectbox("Select Method", ["ADCM", "IMCM"], index=0)
    st.sidebar.markdown("---")
    # Define Streamlit layout
    # st.sidebar.title("Result Tab Options")
    dataset_choice = st.sidebar.selectbox("Select Dataset", ["⁶⁸GA", "¹⁸F-FDG"], index=0)
    st.sidebar.markdown("---")
        # Sidebar for selecting a patient
    # st.sidebar.title("Select a Patient")
    selected_patient = st.sidebar.selectbox("Select a Patient", df['name'])



with tab0:
    st.title('')
    # Load the Lottie animation from the JSON file
    with open("/students/2023-2024/master/Shahpouri/DATA/Aniki_Hamster.json", "r") as f:
        lottie_data = json.load(f)
    
    # Use columns to center the Lottie animation
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st_lottie(lottie_data, speed=1, reverse=False, loop=True, height=200, width=200)

    st.markdown("<div class='subheader'>Author: <strong>Sama Shahpouri, 460145</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Study: <strong>Data Science for Life Science</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Institute: <strong>Hanze University of Applied Sciences, Institute of Life Science & Technology</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Supervisors: <strong>Dr. Isaac Shiri, Dr. Dave, Dr.</strong></div>", unsafe_allow_html=True)
    


with tab1:
    st.title('')
    base = '/students/2023-2024/master/Shahpouri/DL_PET/info/'
    image_paths = [
        # base + 'true.png',
        # base + 'scatter.png',
        # base + 'random.png',
        # base + 'multiple.png',
        base + 'pet_coincidence.png',
        base + 'CT-ASC.png'
    ]

    captions = [
        # 'True Coincidence',
        # 'Scatter Coincidence',
        # 'Random Coincidence',
        # 'Multiple Coincidence',
        '',
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
    col1, col2, col3, col4, col5 = st.columns([1, 1, 6, 1, 1])

    with col2:
        if st.button('Previous') and st.session_state.image_index > 0:
            change_image_index(-1)

    with col3:
        current_image_path = image_paths[st.session_state.image_index]
        st.image(current_image_path, width=600, caption=f"{captions[st.session_state.image_index]}")
        st.caption('https://www.radiologycafe.com/frcr-physics-notes/molecular-imaging/pet-imaging/')
    with col4:
        if st.button('Next') and st.session_state.image_index < len(image_paths) - 1:
            change_image_index(1)

# Tab 2 content
with tab2:
    st.markdown("")
    st.markdown("")
    col1, col2, col3, col4 = st.columns([1,3,2,1])
    with col2:
        st.image(base + "workflow.png", width =400)
    with col3:
        st.caption('Recon = reconstruction;')
        st.caption('AC = attenuation correction;')
        st.caption('SC = scatter correction ')
        st.caption('NAC = noncorrected PET;')
        st.caption('MAC = CT-Measured attenuation/scatter corrected;')
        st.caption('DL PET = deep learning–corrected PET;')


# Tab 3 content
with tab3:
    st.markdown("")
    st.markdown("")

    # Data for the first table
    data1 = {
        "Center": ["Center 1", "Center 2", "Center 3", "Center 4", "External Center", "Total"],
        "No": [56, 31, 45, 40, 12, 184],
        "Train": [43, 25, 35, 28, "-", 131],
        "Validation": [11, 4, 8, 10, "-", 33],
        "Test": [2, 2, 2, 2, 12, 20],
        # "Scanner": ["Siemens Biograph 6", "GE Discovery IQ", "Siemens mCT", "Siemens Biograph 6", "Siemens Horizon", "-"],
        # "Reconstruction": ["3D-OSEM", "3D-OSEM", "3D-OSEM", "3D-OSEM", "PSF+TOF+3D-OSEM", "-"],
        "Matrix size × Z*": ["168 × 168", "192 × 192", "200 × 200", "168 × 168", "180 × 180", "-"]
    }

    # Data for the second table
    data2 = {
        "Center": ["Center 6", "Center 7", "Total"],
        "No": [55, 43, 98],
        "Train": [39, 23, 62],
        "Validation": [6, 9, 15],
        "Test": [11, 10, 21],
        "Matrix size × Z*": ["272 × 200", "272 × 200", "-"]
    }

    col1, col2 = st.columns([1, 4])
    # Convert data to DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)


    with col1:
        # Radio button selection
        options = ["⁶⁸GA", "¹⁸F-FDG", "Artifactual Data"]
        selection = st.radio("Select Data-set:", options)
    with col2:
        # Initialize selection if not already set
        if 'selection' not in st.session_state:
            st.session_state.selection = "⁶⁸GA"
        
        # Update selection based on button press
        if selection:
            st.session_state.selection = selection

        # Display content based on radio button selection
        if st.session_state.selection == "⁶⁸GA":
            st.markdown(df1.style.set_table_styles(
                [{'selector': 'td', 'props': [('font-size', '12px')]}]
            ).to_html(), unsafe_allow_html=True)
            st.image(base + "donat_chart.png")
        elif st.session_state.selection == "¹⁸F-FDG":
            st.markdown(df2.style.set_table_styles(
                [{'selector': 'td', 'props': [('font-size', '12px')]}]
            ).to_html(), unsafe_allow_html=True)
        else:
            st.write("A group of 198 Artifactual Data")

# Tab 4 content
with tab4:
    st.markdown("")
    st.markdown("")
    # Load the appropriate DataFrame based on the selected method
    df_losses = load_df_from_pickle(df_losses_paths[method])

    # Plot the losses using the function
    fig = plot_losses(df_losses, method)
    st.plotly_chart(fig)
# Add divider for clarity


# with tab5:

#     st.markdown("")
#     st.markdown("")
#     result_df = pd.read_csv(df_dataset_result[dataset_choice])


#     # Define a layout with columns for grouping option and metrics selection
#     col1, col2 = st.columns([1, 5])  # Adjust the column width ratio as needed


#     with col1:
#         # Radio button for selecting grouping option
#         grouping_option = st.radio(
#             "",
#             options=["Grouped", "Centers"],
#             index=0
#         )

#     with col2:
#         # Multiselect for selecting which metrics to display
#         selected_metrics = st.multiselect(
#             "Select Metrics to Display",
#             list(metrics_info.keys()),
#             default=list(metrics_info.keys())
#         )

#     # Determine the grouping column based on the radio button selection
#     if grouping_option == "Grouped":
#         grouping_column = 'Center_Group'
#         if dataset_choice == "⁶⁸GA":
#             ticktext = ["Internal", "External"]
#             tickvals = [0.5, 2.5]
#         else:
#             ticktext = ["External"]
#             tickvals = [0.5]
#     else:
#         grouping_column = 'Center'
#         ticktext = []
#         tickvals = []
#         centers = result_df['Center'].unique()
#         for idx, center in enumerate(centers):
#             ticktext.extend([center, ''])
#             tickvals.extend([idx * 2, idx * 2 + 1])

#     # Loop through the selected metrics
#     for i in range(0, len(selected_metrics), 2):
#         col1, col2 = st.columns(2)  # Create two columns for side-by-side plots

#         # Plot the first metric in the first column
#         with col1:
#             if i < len(selected_metrics):
#                 metric = selected_metrics[i]
#                 subtitle = metrics_info[metric]
#                 fig = plot_metric(result_df, metric, subtitle, dataset_labels, colors, grouping_column, tickvals, ticktext)
#                 st.plotly_chart(fig)

#         # Plot the second metric in the second column
#         with col2:
#             if i + 1 < len(selected_metrics):
#                 metric = selected_metrics[i + 1]
#                 subtitle = metrics_info[metric]
#                 fig = plot_metric(result_df, metric, subtitle, dataset_labels, colors, grouping_column, tickvals, ticktext)
#                 st.plotly_chart(fig)

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# Tab 5 content

with tab5:
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
    st.markdown("")
    st.markdown("")
    result_df = pd.read_csv(df_dataset_result[dataset_choice])


    # Define a layout with columns for grouping option and metrics selection
    col1, col2 = st.columns([1, 5])  # Adjust the column width ratio as needed


    with col1:
        # Radio button for selecting grouping option
        grouping_option = st.radio(
            "",
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


with tab6:

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