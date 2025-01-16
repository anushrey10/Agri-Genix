import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and encoder
model = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Feature names from the model
required_features = model.feature_names_in_

# Set page configuration
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Sidebar for additional information
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
Welcome to the **Crop Yield Prediction App**!  
This tool predicts the crop yield based on:
- Country
- Year
- Max and Min Temperature
- Avg Humidity
- Crop Type  

🌍 **Ideal for farmers, researchers, and policymakers.**  
🌾 Developed using **Machine Learning** and **Streamlit**.
""")

st.sidebar.title("💡 Instructions")
st.sidebar.markdown("""
1. Fill in the details on the main page.
2. Click **Predict Yield** to get results.
3. Use the sliders for temperature and humidity adjustments.
""")

st.sidebar.title("📊 Additional Features")
st.sidebar.markdown("""
🔹 Supports common crops: **Wheat, Rice, Maize, etc.**  
🔹 Interactive sliders for temperature and humidity.  
🔹 Custom theme for a modern look.  
""")

# Main app title and description
st.title("🌾 Crop Yield Prediction")
st.markdown("""
Predict the yield of crops based on various factors.  
Simply input the details below and click **Predict Yield** to get results.
""")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    country = st.text_input("🌍 Country (e.g., India, USA)", value="India")
    year = st.number_input("📅 Year", min_value=1900, max_value=2100, value=2020)

with col2:
    max_temp = st.slider("🌡️ Max Temperature (°C)", -10.0, 50.0, 25.0, help="Select the maximum daily temperature.")
    min_temp = st.slider("🌡️ Min Temperature (°C)", -10.0, 40.0, 15.0, help="Select the minimum daily temperature.")
    avg_humidity = st.slider("💧 Avg Humidity (%)", 0.0, 100.0, 50.0, help="Select the average daily humidity.")

# Dropdown for selecting the crop
crop_list = ['Maize', 'Rice, paddy', 'Wheat', 'Potatoes', 'Sorghum', 'Soybeans', 'Sweet potatoes']
crop = st.selectbox("🌱 Select Crop", crop_list)

# Predict button
if st.button("📊 Predict Yield"):
    try:
        # Encode country and crop
        country_label = label_encoder.transform([country])[0]
        crop_feature = f"Crop_{crop}"

        # Construct the input DataFrame
        input_data = pd.DataFrame([{
            'Max_Temperature': max_temp,
            'Min_Temperature': min_temp,
            'Avg_Humidity': avg_humidity,
            crop_feature: 1,
            'Country': country_label
        }])

        # Add missing features with default value 0
        for feature in required_features:
            if feature not in input_data.columns:
                input_data[feature] = 0

        # Align column order with the model
        input_data = input_data[required_features]

        # Make the prediction
        prediction = model.predict(input_data)

        # Display the result
        st.success(f"🌾 Predicted Yield: **{prediction[0]:.2f} hg/ha**")
    except Exception as e:
        st.error(f"An error occurred: {e}")