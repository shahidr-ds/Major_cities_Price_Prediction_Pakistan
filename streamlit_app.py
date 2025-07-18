import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

# Load model and encoder
model = joblib.load("xgb_model_tuned.pkl")

# Define dropdown mapping for UI
city_map = {
    "Karachi": 0, "Lahore": 1, "Islamabad": 2, "Rawalpindi": 3, "Faisalabad": 4,
    "Peshawar": 5, "Multan": 6, "Hyderabad": 7, "Quetta": 8, "Sialkot": 9
}
property_types = ["House", "Flat", "Shop", "Commercial Plot", "Residential Plot", "Building"]

province_columns = {
    "Islamabad Capital": "location_province_ Islamabad Capital",
    "Punjab": "location_province_ Punjab",
    "Khyber Pakhtunkhwa": "location_province_ Khyber Pakhtunkhwa"
}

st.title("🏠 Real Estate Price Prediction App")

# Input UI
city = st.selectbox("Select City", list(city_map.keys()))
province = st.selectbox("Select Province", list(province_columns.keys()))
property_type = st.selectbox("Select Property Type", property_types)
bedroom = st.number_input("Bedrooms", min_value=0, step=1)
bath = st.number_input("Bathrooms", min_value=0, step=1)
area_sqft = st.number_input("Area (sqft)", min_value=1)

# Prediction
if st.button("Predict Price"):
    input_data = {
        "location_city_te": city_map[city],
        "location_te": city_map[city],  # Same encoding if you used same TargetEncoder
        "type_Building": 1 if property_type == "Building" else 0,
        "type_Commercial Plot": 1 if property_type == "Commercial Plot" else 0,
        "type_Flat": 1 if property_type == "Flat" else 0,
        "type_House": 1 if property_type == "House" else 0,
        "type_Shop": 1 if property_type == "Shop" else 0,
        "location_province_ Islamabad Capital": 1 if province == "Islamabad Capital" else 0,
        "location_province_ Khyber Pakhtunkhwa": 1 if province == "Khyber Pakhtunkhwa" else 0,
        "location_province_ Punjab": 1 if province == "Punjab" else 0,
        "bedroom_imputed": bedroom,
        "bath": bath,
        "area_sqft": area_sqft,
        "log_area": np.log1p(area_sqft)
    }

    input_df = pd.DataFrame([input_data])

    # Predict log price
    log_price_pred = model.predict(input_df)[0]
    price_pred = np.expm1(log_price_pred)

    st.success(f"🏷️ Estimated Price: Rs. {price_pred:,.0f}")
