import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config for wider layout
st.set_page_config(layout="wide")

st.title("Wind Energy Forecasting System Dashboard")

# --- Sidebar for Navigation and Upload ---
st.sidebar.header("Configuration")

# Upload Dataset Option
st.sidebar.subheader("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")
    st.session_state['data'] = data
else:
    if 'data' not in st.session_state:
        st.info("Please upload a dataset to get started.")
        st.stop()
    data = st.session_state['data']

# Ensure 'timestamp' is datetime for filtering
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    min_date = data['timestamp'].min().date()
    max_date = data['timestamp'].max().date()

    st.sidebar.subheader("Date and Time Filtering")
    selected_date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        data = data[(data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)]

# --- Main Content Area ---

st.header("Dataset Overview (Filtered)")
st.write(f"Showing {len(data)} rows out of {len(st.session_state['data'])} total.")
st.dataframe(data.head())

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["EDA Visualizations", "Processed Features", "Model Predictions & Metrics"])

with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("### Dataset Shape, dtypes, Missing Values")
    st.write(f"Shape: {data.shape}")
    st.write("Dtypes:")
    st.write(data.dtypes)
    st.write("Missing Values:")
    st.write(data.isnull().sum())

    st.markdown("### Summary Statistics")
    st.write(data.describe())

    st.markdown("### Histograms")
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    if 'timestamp' in numerical_cols: numerical_cols.remove('timestamp') # Remove if it got through as numeric

    if numerical_cols:
        fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(10, 5 * len(numerical_cols)))
        if len(numerical_cols) == 1: axes = [axes] # Handle single subplot case
        for i, col in enumerate(numerical_cols):
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No numerical columns for histograms.")

    st.markdown("### Boxplots")
    if numerical_cols:
        fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(10, 5 * len(numerical_cols)))
        if len(numerical_cols) == 1: axes = [axes] # Handle single subplot case
        for i, col in enumerate(numerical_cols):
            sns.boxplot(y=data[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No numerical columns for boxplots.")

    st.markdown("### Time Series Line Plots (Sampled for large datasets)")
    if 'timestamp' in data.columns and not data.empty:
        # Sample for performance if dataset is very large
        sample_data = data.sample(n=min(len(data), 1000), random_state=42) if len(data) > 1000 else data

        fig, axes = plt.subplots(3, 1, figsize=(18, 15))

        if 'actual_power_output_kw' in data.columns:
            sns.lineplot(x='timestamp', y='actual_power_output_kw', data=sample_data, ax=axes[0])
            axes[0].set_title('Actual Power Output over Time')
        else:
            axes[0].set_title('Actual Power Output (kW) column not found')

        if 'wind_speed_ms' in data.columns:
            sns.lineplot(x='timestamp', y='wind_speed_ms', data=sample_data, ax=axes[1])
            axes[1].set_title('Wind Speed over Time')
        else:
            axes[1].set_title('Wind Speed (m/s) column not found')

        if 'theoretical_power_curve_kwh' in data.columns:
            sns.lineplot(x='timestamp', y='theoretical_power_curve_kwh', data=sample_data, ax=axes[2])
            axes[2].set_title('Theoretical Power Curve over Time')
        else:
            axes[2].set_title('Theoretical Power Curve (KWh) column not found')

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Timestamp column not found or data is empty for time series plots.")

    st.markdown("### Correlation Heatmap")
    if numerical_cols:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numerical columns for correlation heatmap.")


with tab2:
    st.header("Processed Features")
    st.write("This section will display the engineered features after running `feature_engineering.py`.")
    st.dataframe(data.head())
    st.write(data.columns.tolist())

with tab3:
    st.header("Model Predictions and Metrics")

    # Display Model Prediction Plots
    st.markdown("### XGBoost Model Predictions")
    if os.path.exists('xgboost_actual_vs_predicted.png'):
        st.image('xgboost_actual_vs_predicted.png', caption='XGBoost: Actual vs Predicted Wind Power Output', use_column_width=True)
    else:
        st.warning("XGBoost prediction plot not found. Please run `model_training.py` first.")

    st.markdown("### LSTM Model Predictions")
    if os.path.exists('lstm_actual_vs_predicted.png'):
        st.image('lstm_actual_vs_predicted.png', caption='LSTM: Actual vs Predicted Wind Power Output', use_column_width=True)
    else:
        st.warning("LSTM prediction plot not found. Please run `model_training.py` first.")

    # Display Metric Comparison Table
    st.markdown("### Model Performance Comparison")
    # Assuming you'd save a metrics CSV or generate it here from a saved model
    st.info("Run `model_training.py` to generate model performance metrics and plots.")
    # Placeholder for a metrics table if loaded from a file or calculated here
    # st.dataframe(pd.DataFrame({'Model': ['XGBoost', 'LSTM'], 'MAE': [100, 150], 'RMSE': [150, 200], 'R2': [0.9, 0.85]}))

