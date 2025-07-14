import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="🏡 House Price Prediction (PKR)", layout="centered")
st.title("🏡 Major Cities House Price Predictor - Pakistan")
st.markdown("Estimate log-transformed house prices using the XGBoost model trained on real data.")

# Sidebar - User input
st.sidebar.header("📥 Enter Property Features")

location_city_te = st.sidebar.slider("City TE", min_value=0.0, max_value=30.0, value=15.0)
location_te = st.sidebar.slider("Location TE", min_value=0.0, max_value=100.0, value=50.0)
type_house = st.sidebar.selectbox("Property Type", ["House", "Flat", "Commercial Plot", "Residential Plot", "Shop", "Building"])
province = st.sidebar.selectbox("Province", ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Islamabad Capital"])
bedroom = st.sidebar.slider("Bedrooms", 0, 10, 3)
bath = st.sidebar.slider("Bathrooms", 0, 10, 2)
area_sqft = st.sidebar.number_input("Area (sqft)", min_value=50.0, max_value=20000.0, value=1200.0)
days_since_posted = st.sidebar.slider("Days Since Posted", 0, 365, 30)

# One-hot encodings for property type
types = ["Building", "Commercial Plot", "Flat", "House", "Residential Plot", "Shop"]
type_encoded = [1.0 if type_house == t else 0.0 for t in types]

# One-hot encodings for province
provinces = ["Islamabad Capital", "Khyber Pakhtunkhwa", "Punjab", "Sindh"]
province_encoded = [1.0 if province == p else 0.0 for p in provinces]

# Compute additional features
log_area = np.log1p(area_sqft)
log_area_price_ratio = 0  # Optional: let model learn from actual features
log_price_per_sqft = 0    # Placeholder; the model already learned from final engineered features

# Combine features
features = [
    location_city_te,
    location_te,
    *type_encoded,
    *province_encoded,
    bedroom,
    bath,
    area_sqft,
    days_since_posted,
    log_area,
    log_price_per_sqft,
    log_area_price_ratio
]

# Scale and predict
X_input = scaler.transform([features])
pred_log_price = model.predict(X_input)[0]
pred_price = np.expm1(pred_log_price)

# Output
st.markdown("---")
st.subheader("📊 Predicted Price")
st.metric("Estimated House Price (PKR in Millions)", f"{pred_price:.2f} M")
st.caption("Log-transformed target, trained with XGBoost")