import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="🏡 House Price Prediction (PKR)", layout="centered")
st.title("🏡 Major Cities House Price Predictor - Pakistan")
st.markdown("Estimate **log-transformed house prices** using the XGBoost model trained on real data.")

# 🔧 Sidebar Input
st.sidebar.header("📥 Enter Property Details")

# Province input (move to top)
province = st.sidebar.selectbox("Province", ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Islamabad Capital"])

# Target Encoded Features (numeric sliders)
location_city_te = st.sidebar.slider("City Target Encoding", 0.0, 30.0, 15.0)
location_te = st.sidebar.slider("Location Target Encoding", 0.0, 100.0, 50.0)

# Property type
property_type = st.sidebar.selectbox("Property Type", ["House", "Flat", "Commercial Plot", "Residential Plot", "Shop", "Building"])

# Numeric Inputs
bedroom = st.sidebar.slider("Bedrooms", 0, 10, 3)
bath = st.sidebar.slider("Bathrooms", 0, 10, 2)
area_sqft = st.sidebar.number_input("Area (sqft)", min_value=50.0, max_value=20000.0, value=1200.0)

# ✨ Add predict button
if st.button("🔍 Predict House Price"):

    # ----------------------------
    # Step 1: One-Hot Encode Inputs
    # ----------------------------
    types = ["Building", "Commercial Plot", "Flat", "House", "Residential Plot", "Shop"]
    provinces = ["Islamabad Capital", "Khyber Pakhtunkhwa", "Punjab", "Sindh"]

    type_encoded = [1.0 if property_type == t else 0.0 for t in types]
    province_encoded = [1.0 if province == p else 0.0 for p in provinces]

    # ----------------------------
    # Step 2: Feature Engineering
    # ----------------------------
    log_area = np.log1p(area_sqft)

    # Final input order (✅ must match training: 15 features)
    features = [
        location_city_te,
        location_te,
        *type_encoded,         # 6
        *province_encoded,     # 4
        bedroom,
        bath,
        area_sqft,
        log_area
    ]

    # ----------------------------
    # Step 3: Predict
    # ----------------------------
    st.markdown(f"✅ **Feature count being passed: {len(features)}**")
    X_input = scaler.transform([features])
    pred_log_price = model.predict(X_input)[0]
    pred_price = np.expm1(pred_log_price)

    # ----------------------------
    # Step 4: Show Prediction
    # ----------------------------
    st.markdown("---")
    st.subheader("📊 Predicted Price")
    st.metric("Estimated House Price (PKR in Millions)", f"{pred_price:.2f} M")
    st.caption("Predicted from log-transformed XGBoost model")
