import streamlit as st
import numpy as np
import pandas as pd
import joblib
import math

# Load model and related files
model = joblib.load("xgb_model_tuned.pkl")
feature_list = joblib.load("model_features_tuned.pkl")
label_maps = joblib.load("label_maps.pkl")  # for readable dropdowns

# -----------------------------
# Helper: preprocess user input
# -----------------------------
def preprocess_input(data):
    # Create a dataframe from user inputs
    df = pd.DataFrame([data])

    # Add one-hot columns for type and province
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # Log transform area
    df["log_area"] = np.log1p(df["area_sqft"])
    df = df[feature_list]
    return df

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Pakistan Real Estate Price Predictor", layout="centered")

st.title("🏠 Pakistan Real Estate Price Prediction")
st.markdown("Predict the property price (in PKR millions) using a machine learning model.")

# -----------------------------
# User Inputs
# -----------------------------
city = st.selectbox("City", options=label_maps["location_city_te"].keys())
location = st.selectbox("Location (Society)", options=label_maps["location_te"][city].keys())
ptype = st.selectbox("Property Type", options=label_maps["type"].keys())
province = st.selectbox("Province", options=label_maps["province"].keys())
bedroom = st.number_input("Bedrooms", min_value=0, max_value=20, step=1, value=3)
bath = st.number_input("Bathrooms", min_value=1, max_value=20, step=1, value=2)
area = st.number_input("Area (sqft)", min_value=100, max_value=100000, step=50, value=1200)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Price"):
    try:
        # Encode inputs
        input_data = {
            'location_city_te': label_maps["location_city_te"][city],
            'location_te': label_maps["location_te"][city][location],
            'bedroom_imputed': bedroom,
            'bath': bath,
            'area_sqft': area,
        }

        # One-hot encode property type
        for prop_type in ['type_Building', 'type_Commercial Plot', 'type_Flat', 'type_House', 'type_Shop']:
            input_data[prop_type] = 1 if prop_type == label_maps["type"][ptype] else 0

        # One-hot encode province
        for prov in ['location_province_ Islamabad Capital', 'location_province_ Khyber Pakhtunkhwa', 'location_province_ Punjab']:
            input_data[prov] = 1 if prov == label_maps["province"][province] else 0

        # Preprocess and predict
        df_input = preprocess_input(input_data)
        log_pred = model.predict(df_input)[0]
        price_million = np.expm1(log_pred)

        # Display prediction
        st.success(f"💰 Estimated Price: **PKR {price_million:,.2f} million**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
