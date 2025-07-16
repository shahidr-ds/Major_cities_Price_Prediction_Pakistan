# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model and Data
# -----------------------------
@st.cache_data
def load_model():
    return joblib.load("xgb_model_tuned.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Major_cities_data.csv")

model = load_model()
df = load_data()

# -----------------------------
# Prepare Dropdown Options
# -----------------------------
city_options = sorted(df['location_city'].dropna().unique())
society_map = df.groupby("location_city")["location"].unique().apply(sorted).to_dict()

# ✅ Allowed property types
allowed_types = ["House", "Flat", "Shop", "Residential Plot"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏠 Pakistan Real Estate Price Predictor")

selected_city = st.selectbox("📍 Select City", city_options)
selected_society = st.selectbox("🏘️ Select Society", society_map.get(selected_city, []))
selected_type = st.selectbox("🏗️ Property Type", allowed_types)

bedroom = st.slider("🛏️ Bedrooms", 0, 10, 3)
bathroom = st.slider("🛁 Bathrooms", 0, 10, 2)
area_sqft = st.number_input("📐 Area (sqft)", min_value=50, value=1000)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Price"):

    # Step 1: Base input
    input_data = {
        'location_city': selected_city,
        'location': selected_society,
        'type': selected_type,
        'bedroom_imputed': bedroom,
        'bath': bathroom,
        'area_sqft': area_sqft,
    }

    df_input = pd.DataFrame([input_data])

    # Step 2: Feature engineering
    df_input['log_area'] = np.log1p(df_input['area_sqft'])

    # Step 3: One-hot encoding for type
    for t in allowed_types:
        df_input[f"type_{t}"] = int(selected_type == t)

    # Step 4: One-hot for province
    province = df[df["location_city"] == selected_city]["location_province"].mode().iloc[0]
    for p in ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Islamabad Capital"]:
        df_input[f"location_province_ {p}"] = int(province == p)

    # Step 5: Target encoding for city and society
    city_te_map = df.drop_duplicates("location_city")[["location_city", "location_city_te"]].set_index("location_city").to_dict()["location_city_te"]
    society_te_map = df.drop_duplicates("location")[["location", "location_te"]].set_index("location").to_dict()["location_te"]

    df_input["location_city_te"] = city_te_map.get(selected_city, 0)
    df_input["location_te"] = society_te_map.get(selected_society, 0)

    # Step 6: Fill any missing model columns
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model_features]  # ensure column order

    # Step 7: Predict
    pred_log_price = model.predict(df_input)[0]
    pred_price = round(np.expm1(pred_log_price), 2)

    st.success(f"💰 Estimated Price: {pred_price:,} Million PKR")
