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
    st.title("1. Data Input")
    st.write("Upload the seven Excel files for weather data. Once all files are uploaded, the data will be processed and merged.")

    if 'weather_df' not in st.session_state:
        st.session_state.weather_df = pd.DataFrame()
    
    # Create file uploaders for each parameter
    uploaded_files = {}
    uploaded_files['Humidity'] = st.file_uploader("Upload Humidity.xlsx", type="xlsx")
    uploaded_files['MaxTemp'] = st.file_uploader("Upload Maximum Temperature.xlsx", type="xlsx")
    uploaded_files['MinTemp'] = st.file_uploader("Upload Minimum Temperature.xlsx", type="xlsx")
    uploaded_files['Rainfall'] = st.file_uploader("Upload Rainfall.xlsx", type="xlsx")
    uploaded_files['Sunshine'] = st.file_uploader("Upload Sunshine.xlsx", type="xlsx")
    uploaded_files['CloudCoverage'] = st.file_uploader("Upload Cloud Coverage.xlsx", type="xlsx")
    uploaded_files['WindSpeed'] = st.file_uploader("Upload Wind Speed.xlsx", type="xlsx")

    # Check if all files are uploaded
    all_files_uploaded = all(file is not None for file in uploaded_files.values())

    if all_files_uploaded:
        st.success("All files uploaded successfully! Click the button below to process the data.")
        if st.button("Process and Merge Data"):
            # Load and clean each dataframe
            dfs_to_merge = []
            for name, file in uploaded_files.items():
                df = load_weather_data(file, name)
                if df is not None:
                    dfs_to_merge.append(df)
            
            # Merge all dataframes
            merged_df = merge_dataframes(dfs_to_merge)
            
            # Filter for the year range 1961-2023
            if not merged_df.empty:
                st.session_state.weather_df = merged_df[(merged_df['Year'] >= 1961) & (merged_df['Year'] <= 2023)].copy()
                
                # Apply categorization functions to create new categorical columns
                st.session_state.weather_df['Categorized_Rainfall'] = st.session_state.weather_df['Rainfall'].apply(categorize_rainfall)
                st.session_state.weather_df['Categorized_WindSpeed'] = st.session_state.weather_df['WindSpeed'].apply(categorize_windspeed)
                st.session_state.weather_df['Categorized_MaxTemp'] = st.session_state.weather_df['MaxTemp'].apply(categorize_max_temp)
                st.session_state.weather_df['Categorized_MinTemp'] = st.session_state.weather_df['MinTemp'].apply(categorize_min_temp)
                st.session_state.weather_df['Categorized_Humidity'] = st.session_state.weather_df['Humidity'].apply(categorize_humidity)
                st.session_state.weather_df['Categorized_Sunshine'] = st.session_state.weather_df['Sunshine'].apply(categorize_sunshine)
                st.session_state.weather_df['Categorized_CloudCoverage'] = st.session_state.weather_df['CloudCoverage'].apply(categorize_cloud_coverage)
                
                st.write("Merged, filtered, and categorized DataFrame:")
                st.dataframe(st.session_state.weather_df.head())
            else:
                st.error("Could not merge data. Please check the uploaded files.")

# --- Page 2: Data Visualization ---
elif page == "2. Data Visualization":
    st.title("2. Data Visualization")
    if st.session_state.weather_df.empty:
        st.warning("Please upload and process the data on the 'Data Input' page first.")
    else:
        st.write("Visualize the categorical weather parameters over a selected year range.")
        
        # Get min and max years from the dataframe
        min_year = int(st.session_state.weather_df['Year'].min())
        max_year = int(st.session_state.weather_df['Year'].max())

        # Year slider for visualization
        year_range = st.slider(
            "Select a year range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        filtered_df = st.session_state.weather_df[(st.session_state.weather_df['Year'] >= year_range[0]) & (st.session_state.weather_df['Year'] <= year_range[1])]

        parameters = {
            'Categorized_MaxTemp': ['Cool', 'Warm', 'Hot'],
            'Categorized_MinTemp': ['Cold', 'Cool', 'Warm'],
            'Categorized_Rainfall': ['Low', 'Medium', 'High'],
            'Categorized_CloudCoverage': ['Clear', 'Partly Cloudy', 'Cloudy'],
            'Categorized_Humidity': ['Low', 'Medium', 'High'],
            'Categorized_Sunshine': ['Low', 'Medium', 'High'],
            'Categorized_WindSpeed': ['Calm', 'Moderate', 'Windy']
        }

        for param, categories in parameters.items():
            st.subheader(f"Categorical Trends for {param.replace('Categorized_', '')}")
            
            # Count the occurrences of each category per year
            plot_df = filtered_df.groupby(['Year', param]).size().reset_index(name='Count')

            fig, ax = plt.subplots(figsize=(10, 5))
            # Use a bar plot to show category counts per year
            sns.barplot(data=plot_df, x='Year', y='Count', hue=param, palette='viridis', ax=ax, order=range(year_range[0], year_range[1] + 1))
            ax.set_title(f'Category Distribution for {param.replace("Categorized_", "")} ({year_range[0]} - {year_range[1]})')
            ax.set_xlabel("Year")
            ax.set_ylabel("Count of Months")
            plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for readability
            plt.tight_layout() # Adjust plot to ensure everything fits
            st.pyplot(fig)

# --- Page 3: Predict Weather ---
elif page == "3. Predict Weather":
    st.title("3. Predict Weather")
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
        month_input = st.selectbox("Select a month for prediction:", options=list(range(1, 12)))
        
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
    st.title("4. Report Generation")
    
    if 'prediction_results' not in st.session_state:
        st.warning("Please make a prediction on the 'Predict Weather' page first.")
    else:
        st.write("Generate a downloadable PDF report of the last prediction.")
        
        # Display the prediction results again
        st.subheader("Last Prediction:")
        st.write(f"**Year:** {st.session_state.prediction_input['Year']}")
        st.write(f"**Month:** {st.session_state.prediction_input['Month']}")
        
        for param, value in st.session_state.prediction_results.items():
            st.write(f"**{param}:** {value:.2f}")

        # PDF generation function
        def create_pdf(prediction_data):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='CenteredTitle', alignment=TA_CENTER, fontSize=24, spaceAfter=20))
            styles.add(ParagraphStyle(name='Heading', alignment=TA_CENTER, fontSize=30, spaceAfter=20))
            
            story = []
            
            story.append(Paragraph("Weather Prediction Report", styles['CenteredTitle']))
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph(f"**Year:** {prediction_data['input']['Year']}", styles['Normal']))
            story.append(Paragraph(f"**Month:** {prediction_data['input']['Month']}", styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

            story.append(Paragraph("--- Predicted Weather Data ---", styles['Heading']))
            story.append(Spacer(1, 0.1 * inch))

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
