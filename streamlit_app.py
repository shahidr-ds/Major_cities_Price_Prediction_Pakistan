import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and features
model = joblib.load("xgb_model_tuned.pkl")
selected_features = joblib.load("model_features_tuned.pkl")

# --- UI Dropdown Friendly Names ---
city_map = {
    'Karachi': 'location_city_te', 
    'Lahore': 'location_city_te',
    'Islamabad': 'location_city_te',
    'Rawalpindi': 'location_city_te',
    'Peshawar': 'location_city_te',
    'Multan': 'location_city_te'
}

province_map = {
    'Punjab': 'location_province_ Punjab',
    'Sindh': 'location_province_ Sindh',
    'Khyber Pakhtunkhwa': 'location_province_ Khyber Pakhtunkhwa',
    'Islamabad Capital': 'location_province_ Islamabad Capital'
}

type_map = {
    'House': 'type_House',
    'Flat': 'type_Flat',
    'Shop': 'type_Shop',
    'Commercial Plot': 'type_Commercial Plot',
    'Residential Plot': 'type_Residential Plot',
    'Building': 'type_Building'
}

st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
st.title("🏠 Major Cities Real Estate Price Prediction (Pakistan)")

# User Inputs
city = st.selectbox("Select City", list(city_map.keys()))
province = st.selectbox("Select Province", list(province_map.keys()))
property_type = st.selectbox("Select Property Type", list(type_map.keys()))
bedrooms = st.number_input("Bedrooms", min_value=0, step=1, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0, step=1, value=2)
area_sqft = st.number_input("Area (in Square Feet)", min_value=0.0, step=50.0, value=1000.0)

# Preprocessing single input row
def create_input_df():
    input_dict = {feature: 0 for feature in selected_features}
    
    # Encoded features
    input_dict[type_map[property_type]] = 1
    input_dict[province_map[province]] = 1
    
    # Numerical features
    input_dict['bedroom_imputed'] = bedrooms
    input_dict['bath'] = bathrooms
    input_dict['area_sqft'] = area_sqft
    input_dict['log_area'] = np.log1p(area_sqft)

    # Handle TE features if present
    if 'location_city_te' in selected_features:
        # Assume dummy average encoding for city and full location
        city_te = {
            'Karachi': 0.72,
            'Lahore': 0.69,
            'Islamabad': 0.85,
            'Rawalpindi': 0.66,
            'Peshawar': 0.55,
            'Multan': 0.52
        }
        input_dict['location_city_te'] = city_te.get(city, 0.60)
    
    if 'location_te' in selected_features:
        # If location_te was based on city + area, fallback to city only
        input_dict['location_te'] = city_te.get(city, 0.60)

    return pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Price"):
    input_df = create_input_df()
    log_price_pred = model.predict(input_df)[0]
    price_million = np.expm1(log_price_pred)

    st.success(f"💰 Estimated Price: **{price_million:.2f} Million PKR**")

    with st.expander("🔍 Model Input Summary"):
        st.dataframe(input_df)
