import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Load model and scaler
# ----------------------------
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ----------------------------
# Set up Streamlit UI
# ----------------------------
st.set_page_config(page_title="🏡 House Price Prediction (PKR)", layout="centered")
st.title("🏡 Major Cities House Price Predictor - Pakistan")
st.markdown("Estimate log-transformed house prices using the XGBoost model trained on real data.")

# ----------------------------
# Sidebar - User Inputs
# ----------------------------
st.sidebar.header("📥 Enter Property Features")

# Grouped cities by province
province_city_map = {
    "Punjab": ["Lahore", "Rawalpindi", "Faisalabad", "Multan", "Sialkot"],
    "Sindh": ["Karachi", "Hyderabad"],
    "Khyber Pakhtunkhwa": ["Peshawar"],
    "Islamabad Capital": ["Islamabad"]
}

# Province and dynamic city dropdown
province = st.sidebar.selectbox("Province", list(province_city_map.keys()))
city = st.sidebar.selectbox("City", province_city_map[province])

# Property Type
property_type = st.sidebar.selectbox("Property Type", [
    "House", "Flat", "Commercial Plot", "Residential Plot", "Shop", "Building"
])

# Numeric Inputs
bedroom = st.sidebar.slider("Bedrooms", 0, 10, 3)
bath = st.sidebar.slider("Bathrooms", 0, 10, 2)
area_sqft = st.sidebar.number_input("Area (sqft)", min_value=50.0, max_value=20000.0, value=1200.0)

# ----------------------------
# Feature Engineering
# ----------------------------
# Dummy encoding: Property Type
type_list = ["Building", "Commercial Plot", "Flat", "House", "Residential Plot", "Shop"]
type_encoded = [1.0 if property_type == t else 0.0 for t in type_list]

# Dummy encoding: Province
province_list = ["Islamabad Capital", "Khyber Pakhtunkhwa", "Punjab", "Sindh"]
province_encoded = [1.0 if province == p else 0.0 for p in province_list]

# Target encoding for city (mock values)
city_te_map = {
    "Lahore": 25.1, "Karachi": 24.5, "Islamabad": 26.0, "Rawalpindi": 23.7,
    "Peshawar": 22.3, "Faisalabad": 21.9, "Multan": 21.4, "Hyderabad": 20.2,
    "Quetta": 19.5, "Sialkot": 20.0
}
location_city_te = city_te_map.get(city, 21.0)
location_te = 50.0  # constant or average encoding

# Log features
log_area = np.log1p(area_sqft)

# Final feature vector
features = [
    location_city_te,
    location_te,
    *type_encoded,         # 6 features
    *province_encoded,     # 4 features
    bedroom,
    bath,
    area_sqft,
    log_area
]

# Debug info
st.write("✅ Feature count:", len(features))
st.code(f"🧪 Features passed to model:\n{features}")

# ----------------------------
# Predict
# ----------------------------
if st.button("🔍 Predict Price"):
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    # Transform and predict
    X_input = scaler.transform([features])
    pred_log_price = model.predict(X_input)[0]
    pred_price = np.expm1(pred_log_price)

    st.metric("Estimated House Price (PKR in Millions)", f"{pred_price:.2f} M")
    st.caption("🔎 Model trained on log-transformed target using XGBoost")
    st.write(f"Log price predicted: {pred_log_price:.4f}")
