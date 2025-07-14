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

# Location (Society) Dropdown and Encoding
location_map = {
    "Lahore": {"DHA Lahore": 52.4, "Bahria Town": 48.9, "Johar Town": 46.5},
    "Karachi": {"DHA Karachi": 51.2, "Gulshan-e-Iqbal": 46.3},
    "Islamabad": {"DHA Islamabad": 53.1, "G-13": 49.5},
    "Rawalpindi": {"Bahria Town": 47.8},
    "Peshawar": {"Hayatabad": 42.0},
    "Faisalabad": {"Satiana Road": 40.5},
    "Multan": {"Model Town": 39.0},
    "Hyderabad": {"Qasimabad": 38.0},
    "Sialkot": {"Cantt": 37.2}
}
location_options = list(location_map.get(city, {"Unknown": 40.0}).keys())
selected_location = st.sidebar.selectbox("Location / Society", location_options)
location_te = location_map.get(city, {}).get(selected_location, 40.0)

# Property Type
property_type = st.sidebar.selectbox("Property Type", [
    "House", "Flat", "Residential Plot", "Shop"
])

# Numeric Inputs
bedroom = st.sidebar.slider("Bedrooms", 0, 10, 3)
bath = st.sidebar.slider("Bathrooms", 0, 10, 2)
area_sqft = st.sidebar.number_input("Area (sqft)", min_value=50.0, max_value=20000.0, value=1200.0)

# ----------------------------
# Feature Engineering (15 total)
# ----------------------------
# Property type dummies (4 used)
property_types = ["House", "Flat", "Shop", "Residential Plot"]
type_encoded = [1.0 if property_type == t else 0.0 for t in property_types]

# Province dummies (Punjab, Sindh, KP)
province_dummies = ["Punjab", "Sindh", "Khyber Pakhtunkhwa"]
province_encoded = [1.0 if province == p else 0.0 for p in province_dummies]

# Target Encodings
city_te_map = {
    "Lahore": 25.1, "Karachi": 24.5, "Islamabad": 26.0, "Rawalpindi": 23.7,
    "Peshawar": 22.3, "Faisalabad": 21.9, "Multan": 21.4, "Hyderabad": 20.2,
    "Sialkot": 20.0
}
city_te = city_te_map.get(city, 21.0)

# Log price per sqft and ratio
price_per_sqft = area_sqft / (bedroom + bath + 1)
log_price_per_sqft = np.log1p(price_per_sqft)
log_area_price_ratio = np.log1p(area_sqft / (bedroom + bath + 1))

# Mock days since posted
days_since_posted = 15.0

# Final feature vector (exact training order)
features = [
    *type_encoded,
    bath,
    bedroom,
    area_sqft,
    days_since_posted,
    city_te,
    location_te,
    *province_encoded,
    log_price_per_sqft,
    log_area_price_ratio
]

# Debug info
st.write("✅ Feature count:", len(features))
st.write("📐 Expected by scaler:", scaler.n_features_in_)

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
