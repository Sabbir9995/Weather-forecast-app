import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from sklearn.ensemble import RandomForestRegressor
import pickle
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

# Set Streamlit page configuration
st.set_page_config(
    page_title="Weather Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main App Title ---
st.title("Weather Forecasting App")

# --- Categorization Functions ---
def categorize_rainfall(mm):
    """Categorizes rainfall into 'Low', 'Medium', or 'High'."""
    if mm < 50:
        return 'Low'
    elif mm < 150:
        return 'Medium'
    else:
        return 'High'

def categorize_windspeed(ms):
    """Categorizes wind speed into 'Calm', 'Moderate', or 'Windy'."""
    if ms < 1.5:
        return 'Calm'
    elif ms < 3.5:
        return 'Moderate'
    else:
        return 'Windy'

def categorize_max_temp(temp):
    """Categorizes maximum temperature into 'Cool', 'Warm', or 'Hot'."""
    if temp < 28:
        return 'Cool'
    elif temp < 32:
        return 'Warm'
    else:
        return 'Hot'

def categorize_min_temp(temp):
    """Categorizes minimum temperature into 'Cold', 'Cool', or 'Warm'."""
    if temp < 15:
        return 'Cold'
    elif temp < 22:
        return 'Cool'
    else:
        return 'Warm'

def categorize_humidity(humidity):
    """Categorizes humidity into 'Low', 'Medium', or 'High'."""
    if humidity < 60:
        return 'Low'
    elif humidity < 80:
        return 'Medium'
    else:
        return 'High'

def categorize_sunshine(hours):
    """Categorizes sunshine hours into 'Low', 'Medium', or 'High'."""
    if hours < 4:
        return 'Low'
    elif hours < 7:
        return 'Medium'
    else:
        return 'High'

def categorize_cloud_coverage(coverage):
    """Categorizes cloud coverage into 'Clear', 'Partly Cloudy', or 'Cloudy'."""
    if coverage < 2:
        return 'Clear'
    elif coverage < 5:
        return 'Partly Cloudy'
    else:
        return 'Cloudy'

# --- Functions for data loading and merging ---
@st.cache_data
def load_weather_data(uploaded_file, value_column_name):
    """
    Loads an Excel file, skips the first 3 rows, renames columns, and
    returns a DataFrame with 'Year', 'Month', and the specified value column.
    """
    try:
        df = pd.read_excel(uploaded_file, skiprows=3)
        df.columns = ['SL', 'Station', 'Year', 'Month', value_column_name]
        return df[['Year', 'Month', value_column_name]]
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None

def merge_dataframes(dfs):
    """
    Merges a list of dataframes on 'Year' and 'Month' columns.
    """
    if not dfs:
        return pd.DataFrame()
    
    # Remove None values from the list of dataframes
    valid_dfs = [df for df in dfs if df is not None]
    
    if not valid_dfs:
        return pd.DataFrame()

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Year', 'Month']), valid_dfs)
    return merged_df

# --- Page Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1. Data Input", "2. Data Visualization", "3. Predict Weather", "4. Report Generation"])

# --- Page 1: Data Input ---
if page == "1. Data Input":
    st.header("1. Data Input")
    st.write("Upload the seven Excel files for weather data. Once all files are uploaded, the data will be processed and merged automatically.")

    # Initialize session state for files and processing status
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'weather_df' not in st.session_state:
        st.session_state.weather_df = pd.DataFrame()
    
    # Create file uploaders for each parameter with unique keys
    uploaded_files_current = {}
    uploaded_files_current['Humidity'] = st.file_uploader("Upload Humidity.xlsx", type="xlsx", key="hum_file")
    uploaded_files_current['MaxTemp'] = st.file_uploader("Upload Maximum Temperature.xlsx", type="xlsx", key="max_temp_file")
    uploaded_files_current['MinTemp'] = st.file_uploader("Upload Minimum Temperature.xlsx", type="xlsx", key="min_temp_file")
    uploaded_files_current['Rainfall'] = st.file_uploader("Upload Rainfall.xlsx", type="xlsx", key="rainfall_file")
    uploaded_files_current['Sunshine'] = st.file_uploader("Upload Sunshine.xlsx", type="xlsx", key="sunshine_file")
    uploaded_files_current['CloudCoverage'] = st.file_uploader("Upload Cloud Coverage.xlsx", type="xlsx", key="cloud_file")
    uploaded_files_current['WindSpeed'] = st.file_uploader("Upload Wind Speed.xlsx", type="xlsx", key="wind_file")

    # Check if all files are uploaded and if the file list has changed
    all_files_uploaded = all(file is not None for file in uploaded_files_current.values())
    current_file_names = {name: file.name if file else None for name, file in uploaded_files_current.items()}
    
    # --- Fix for AttributeError: Ensure DataFrame is reset if not all files are present ---
    if all_files_uploaded:
        # Check if the set of uploaded files has changed to avoid unnecessary reprocessing
        if current_file_names != st.session_state.uploaded_files:
            st.success("All files uploaded successfully! Processing data automatically...")
            
            dfs_to_merge = []
            for name, file in uploaded_files_current.items():
                df = load_weather_data(file, name)
                if df is not None:
                    dfs_to_merge.append(df)
            
            merged_df = merge_dataframes(dfs_to_merge)
            
            if not merged_df.empty:
                st.session_state.weather_df = merged_df[(merged_df['Year'] >= 1961) & (merged_df['Year'] <= 2023)].copy()
                
                # Apply categorization functions
                st.session_state.weather_df['Categorized_Rainfall'] = st.session_state.weather_df['Rainfall'].apply(categorize_rainfall)
                st.session_state.weather_df['Categorized_WindSpeed'] = st.session_state.weather_df['WindSpeed'].apply(categorize_windspeed)
                st.session_state.weather_df['Categorized_MaxTemp'] = st.session_state.weather_df['MaxTemp'].apply(categorize_max_temp)
                st.session_state.weather_df['Categorized_MinTemp'] = st.session_state.weather_df['MinTemp'].apply(categorize_min_temp)
                st.session_state.weather_df['Categorized_Humidity'] = st.session_state.weather_df['Humidity'].apply(categorize_humidity)
                st.session_state.weather_df['Categorized_Sunshine'] = st.session_state.weather_df['Sunshine'].apply(categorize_sunshine)
                st.session_state.weather_df['Categorized_CloudCoverage'] = st.session_state.weather_df['CloudCoverage'].apply(categorize_cloud_coverage)
                
                st.session_state.data_processed = True
                st.session_state.uploaded_files = current_file_names
                st.write("Merged, filtered, and categorized DataFrame:")
                st.dataframe(st.session_state.weather_df.head())
            else:
                st.error("Could not merge data. Please check the uploaded files.")
                st.session_state.data_processed = False
                st.session_state.weather_df = pd.DataFrame()
    else:
        # If not all files are uploaded, reset the data_processed flag and clear the DataFrame
        st.session_state.data_processed = False
        st.session_state.weather_df = pd.DataFrame()
        st.session_state.uploaded_files = current_file_names
        st.info("Please upload all seven files to proceed.")

    if st.session_state.data_processed:
        st.success("Data has been processed. You can now navigate to other pages.")

# --- Page 2: Data Visualization ---
elif page == "2. Data Visualization":
    st.header("2. Data Visualization")
    if st.session_state.weather_df.empty:
        st.warning("Please upload and process the data on the 'Data Input' page first.")
    else:
        st.write("Categorical distribution of weather parameters from 1961 to 2023.")
        
        visualization_type = st.radio(
            "Select visualization type:",
            ("Total Distribution", "Yearly Trend")
        )

        # Define parameters and their corresponding categories
        parameters_to_plot = {
            'Categorized_MaxTemp': ['Cool', 'Warm', 'Hot'],
            'Categorized_MinTemp': ['Cold', 'Cool', 'Warm'],
            'Categorized_Rainfall': ['Low', 'Medium', 'High'],
            'Categorized_CloudCoverage': ['Clear', 'Partly Cloudy', 'Cloudy'],
            'Categorized_Humidity': ['Low', 'Medium', 'High'],
            'Categorized_Sunshine': ['Low', 'Medium', 'High'],
            'Categorized_WindSpeed': ['Calm', 'Moderate', 'Windy']
        }

        if visualization_type == "Total Distribution":
            start_year, end_year = st.slider(
                "Select Year Range:",
                min_value=1961,
                max_value=2023,
                value=(1961, 2023)
            )

            filtered_df = st.session_state.weather_df[(st.session_state.weather_df['Year'] >= start_year) & (st.session_state.weather_df['Year'] <= end_year)]

            for param, categories in parameters_to_plot.items():
                st.subheader(f"Distribution of {param.replace('Categorized_', '')} ({start_year} - {end_year})")
                
                if not filtered_df.empty:
                    # Total count for each category in the filtered data
                    category_counts = filtered_df[param].value_counts().reindex(categories)
                    plot_df = category_counts.reset_index()
                    plot_df.columns = [param, 'Count']

                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    sns.barplot(data=plot_df, x=param, y='Count', palette='viridis', ax=ax, order=categories)
                    ax.set_title(f'Total Count by Category for {param.replace("Categorized_", "")}')
                    ax.set_xlabel(param.replace("Categorized_", ""))
                    ax.set_ylabel("Total Count of Months")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(f"No data available for {param.replace('Categorized_', '')} in the selected year range.")

        elif visualization_type == "Yearly Trend":
            for param, categories in parameters_to_plot.items():
                st.subheader(f"Distribution of {param.replace('Categorized_', '')}")
                
                # Count the occurrences of each category per year
                plot_df = st.session_state.weather_df.groupby(['Year', param]).size().reset_index(name='Count')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=plot_df, x='Year', y='Count', hue=param, palette='viridis', ax=ax, order=range(1961, 2023))
                ax.set_title(f'Category Distribution per Year for {param.replace("Categorized_", "")} (1961 - 2023)')
                ax.set_xlabel("Year")
                ax.set_ylabel("Count of Months")
                plt.xticks(rotation=90, ha='right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)


# --- Page 3: Predict Weather ---
elif page == "3. Predict Weather":
    st.header("3. Predict Weather")
    if st.session_state.weather_df.empty:
        st.warning("Please upload and process the data on the 'Data Input' page first.")
    else:
        st.write("Input a year and month to get predictions for all seven weather parameters.")

        # Train models if not already in session state
        if 'models' not in st.session_state:
            st.info("Training prediction models... This may take a moment.")
            
            # Define features and target parameters
            X = st.session_state.weather_df[['Year', 'Month']]
            parameters = ['MaxTemp', 'MinTemp', 'Rainfall', 'CloudCoverage', 'Humidity', 'Sunshine', 'WindSpeed']
            st.session_state.models = {}

            # Train a model for each parameter
            for param in parameters:
                y = st.session_state.weather_df[param]
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                st.session_state.models[param] = model
            st.success("Models trained successfully!")

        # User input for prediction
        year_input = st.number_input("Enter a year for prediction:", min_value=1961, max_value=2023, value=1961, step=1)
        month_input = st.selectbox("Select a month for prediction:", options=list(range(1, 13)))
        
        if st.button("Predict"):
            if 'models' in st.session_state:
                # Prepare input data for prediction
                input_data = pd.DataFrame([{'Year': year_input, 'Month': month_input}])
                
                # Make predictions for each parameter
                st.subheader("Prediction Results:")
                prediction_results = {}
                for param, model in st.session_state.models.items():
                    prediction = model.predict(input_data)[0]
                    prediction_results[param] = prediction
                    st.write(f"**{param}:** {prediction:.2f}")

                # Store prediction results in session state for report generation
                st.session_state.prediction_results = prediction_results
                st.session_state.prediction_input = {'Year': year_input, 'Month': month_input}
            else:
                st.error("Models are not yet trained. Please try again.")

# --- Page 4: Report Generation ---
elif page == "4. Report Generation":
    st.header("4. Report Generation")
    
    if 'prediction_results' not in st.session_state:
        st.warning("Please make a prediction on the 'Predict Weather' page first.")
    else:
        st.write("Generate a downloadable PDF report of the last prediction.")
        
        # Display the prediction results again
        st.subheader("Last Prediction:")
        st.write(f"Year: {st.session_state.prediction_input['Year']}")
        st.write(f"Month: {st.session_state.prediction_input['Month']}")
        
        for param, value in st.session_state.prediction_results.items():
            st.write(f"{param}: {value:.2f}")

        # PDF generation function
        def create_pdf(prediction_data):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='CenteredTitle', alignment=TA_CENTER, fontSize=30, spaceAfter=20))
            styles.add(ParagraphStyle(name='Heading', alignment=TA_CENTER, fontSize=30, spaceAfter=16))
            
            story = []
            
            story.append(Paragraph("Weather Prediction Report", styles['CenteredTitle']))
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph(f"**Year:** {prediction_data['input']['Year']}", styles['Normal']))
            story.append(Paragraph(f"**Month:** {prediction_data['input']['Month']}", styles['Normal']))
            story.append(Spacer(2, 1 * inch))

            story.append(Paragraph("--- Predicted Weather Data ---", styles['Heading']))
            story.append(Spacer(2, 1 * inch))

            for param, value in prediction_data['results'].items():
                story.append(Paragraph(f"**{param}:** {value:.2f}", styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        # Generate and download PDF button
        if st.button("Generate and Download PDF"):
            pdf_data = {
                'input': st.session_state.prediction_input,
                'results': st.session_state.prediction_results
            }
            pdf_file = create_pdf(pdf_data)
            st.download_button(
                label="Download Report",
                data=pdf_file,
                file_name=f"weather_report_{st.session_state.prediction_input['Year']}-{st.session_state.prediction_input['Month']}.pdf",
                mime="application/pdf"
            )
