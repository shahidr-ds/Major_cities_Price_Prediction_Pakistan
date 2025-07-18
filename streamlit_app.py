import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature names
model = joblib.load("xgb_model_tuned.pkl")
model_features = joblib.load("model_features_tuned.pkl")

# Title
st.title("🏡 Pakistan Real Estate Price Predictor")
st.markdown("Enter property details to predict the price (in Million PKR)")

# Sidebar Inputs
city = st.selectbox("City", ['Lahore', 'Karachi', 'Islamabad', 'Rawalpindi', 'Peshawar'])
province = st.selectbox("Province", ['Punjab', 'Sindh', 'Islamabad Capital', 'Khyber Pakhtunkhwa'])
property_type = st.selectbox("Property Type", ['House', 'Flat', 'Shop', 'Residential Plot', 'Commercial Plot', 'Building'])
bedroom = st.number_input("Bedrooms", min_value=0, step=1)
bath = st.number_input("Bathrooms", min_value=0, step=1)
area_sqft = st.number_input("Area (in square feet)", min_value=0.0, step=1.0)

# Feature Engineering
log_area = np.log1p(area_sqft)

# Prepare input features dictionary
input_data = {
    'location_city_te': hash(city) % 1_000_000,       # same encoding logic used in training
    'location_te': hash(f"{city}_{province}") % 1_000_000,
    'type_Building': int(property_type == 'Building'),
    'type_Commercial Plot': int(property_type == 'Commercial Plot'),
    'type_Flat': int(property_type == 'Flat'),
    'type_House': int(property_type == 'House'),
    'type_Residential Plot': int(property_type == 'Residential Plot'),
    'type_Shop': int(property_type == 'Shop'),
    'location_province_ Islamabad Capital': int(province == 'Islamabad Capital'),
    'location_province_ Khyber Pakhtunkhwa': int(province == 'Khyber Pakhtunkhwa'),
    'location_province_ Punjab': int(province == 'Punjab'),
    'location_province_ Sindh': int(province == 'Sindh'),
    'bedroom_imputed': bedroom,
    'bath': bath,
    'area_sqft': area_sqft,
    'log_area': log_area
}

# Create DataFrame and reorder columns to match model
input_df = pd.DataFrame([input_data])
input_df = input_df[model_features]  # Ensure column order and names match

# Prediction
if st.button("Predict Price"):
    log_price_pred = model.predict(input_df)[0]
    price_pred = np.expm1(log_price_pred)
    st.success(f"🏷️ Estimated Price: **{price_pred:.2f} Million PKR**")
