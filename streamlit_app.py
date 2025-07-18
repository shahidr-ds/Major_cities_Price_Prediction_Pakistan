import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load trained model and assets
# -------------------------------
model = joblib.load("xgb_model_tuned.pkl")
model_features = joblib.load("model_features_tuned.pkl")
label_encoders = joblib.load("label_encoders_tuned.pkl")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏠 Pakistan Real Estate Price Prediction App")

st.markdown("Enter the property details below to predict the **price (in millions)**.")

# City dropdown
city = st.selectbox("City", ['Lahore', 'Karachi', 'Islamabad', 'Rawalpindi', 'Peshawar', 'Multan'])

# Location dropdown
location = st.text_input("Exact Location (e.g., DHA Phase 6)", "")

# Property type
property_type = st.selectbox("Property Type", ['House', 'Flat', 'Shop', 'Building', 'Commercial Plot', 'Residential Plot'])

# Province
province = st.selectbox("Province", ['Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Islamabad Capital'])

# Other numerical inputs
bedroom = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bath = st.number_input("Bathrooms", min_value=0, max_value=20, value=3)
area_sqft = st.number_input("Area (in square feet)", min_value=0.0, value=1200.0)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Price"):

    # Raw input data
    input_data = {
        'location_city': city,
        'location': location,
        'type': property_type,
        'location_province': province,
        'bedroom_imputed': bedroom,
        'bath': bath,
        'area_sqft': area_sqft
    }

    df = pd.DataFrame([input_data])

    # Apply label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Feature engineering
    df['log_area'] = np.log1p(df['area_sqft'])

    # One-hot encode property types and provinces
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    province_dummies = pd.get_dummies(df['location_province'], prefix='location_province')

    df = pd.concat([df, type_dummies, province_dummies], axis=1)

    # Drop original string columns
    df.drop(columns=['type', 'location_province'], inplace=True)

    # Ensure all features used during training are present
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Final feature order
    df_final = df[model_features]

    # Predict
    log_price_pred = model.predict(df_final)[0]
    price_million = np.expm1(log_price_pred)

    st.success(f"💰 Predicted Price: **{price_million:.2f} million PKR**")
