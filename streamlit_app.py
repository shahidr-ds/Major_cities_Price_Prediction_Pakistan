import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and feature list
model = joblib.load("xgb_model.pkl")
model_features = joblib.load("model_features.pkl")  # This should be the feature list used during training

# Title
st.title("🏡 Major Cities House Price Prediction - Pakistan")

# User Inputs
st.header("Enter Property Details")

# Dropdowns and inputs
city = st.selectbox("City", ['Lahore', 'Karachi', 'Islamabad', 'Peshawar'])
province = st.selectbox("Province", ['Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Islamabad Capital'])
property_type = st.selectbox("Property Type", ['House', 'Flat', 'Shop', 'Commercial Plot', 'Building'])
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=2)
area = st.number_input("Area (sqft)", min_value=100, max_value=20000, value=1200)

# Feature engineering (manual encoding)
input_dict = {
    'bedroom_imputed': bedrooms,
    'bath': bathrooms,
    'area_sqft': area,
    'log_area': np.log1p(area),  # log(area) for model
    'location_city_te': hash(city) % 1000,  # Placeholder for city target encoding
    'location_te': hash(f"{city}_{province}") % 1000,  # Placeholder for combined encoding
}

# One-hot encoding
for pt in ['type_Building', 'type_Commercial Plot', 'type_Flat', 'type_House', 'type_Shop']:
    input_dict[pt] = 1 if pt.split('_')[1] == property_type else 0

for prov in ['location_province_ Punjab', 'location_province_ Sindh', 'location_province_ Khyber Pakhtunkhwa', 'location_province_ Islamabad Capital']:
    input_dict[prov] = 1 if prov.split('_ ')[1] == province else 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Reindex to match training feature order
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Predict
if st.button("Predict Price"):
    log_price = model.predict(input_df)[0]
    price = np.expm1(log_price)
    st.success(f"💰 Estimated Price: PKR {price:,.0f}")
