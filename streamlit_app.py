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
# Dummy encoding: Property Type (must match training order)
type_list = ["Building", "Commercial Plot", "Flat", "House", "Residential Plot", "Shop"]
type_encoded = [1.0 if property_type == t else 0.0 for t in type_list]

# Dummy encoding: Province (must match training order)
province_list = ["Islamabad Capital", "Khyber Pakhtunkhwa", "Punjab", "Sindh"]
province_encoded = [1.0 if province == p else 0.0 for p in province_list]

# Target encoding for city (from training)
city_te_map = {
    "Lahore": 25.1, "Karachi": 24.5, "Islamabad": 26.0, "Rawalpindi": 23.7,
    "Peshawar": 22.3, "Faisalabad": 21.9, "Multan": 21.4, "Hyderabad": 20.2,
    "Sialkot": 20.0
}
location_city_te = city_te_map.get(city, 21.0)

# Log features
log_area = np.log1p(area_sqft)
log_area_price_ratio = log_area / area_sqft

# ✅ Final feature vector — must be exactly 15 features
features = [
    location_city_te,         # 1
    *type_encoded,            # 6
    *province_encoded,        # 4
    bedroom,                  # 1
    bath,                     # 1
    area_sqft,                # 1
    log_area,                 # 1
    log_area_price_ratio      # 1
]

# Show for debugging
st.write("✅ Feature count being passed:", len(features))
st.code(f"{features}")

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

    st.metric("🏷 Estimated House Price (PKR Millions)", f"{pred_price:.2f} M")
    st.caption("🔎 Predicted using XGBoost model trained on log-transformed prices")
    st.write(f"📉 Log Price Estimate: `{pred_log_price:.4f}`")
