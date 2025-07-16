import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load raw data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Major_cities_data.csv")

df_raw = load_data()

# -----------------------------
# Fixed province/type structure
# -----------------------------
provinces = ["Punjab", "Sindh", "Islamabad Capital", "Khyber Pakhtunkhwa"]
property_types = {
    "Building": "type_Building",
    "Commercial Plot": "type_Commercial Plot",
    "Flat": "type_Flat",
    "House": "type_House",
    "Shop": "type_Shop"
}

# -----------------------------
# Fixed mappings
# -----------------------------
city_te_map = df_raw.drop_duplicates("location_city")[["location_city", "location_city_te"]].set_index("location_city")["location_city_te"].to_dict()
location_te_map = df_raw.drop_duplicates("location")[["location", "location_te"]].set_index("location")["location_te"].to_dict()

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏠 Real Estate Price Prediction")

selected_city = st.selectbox("City", sorted(city_te_map.keys()))
selected_location = st.selectbox("Location", sorted(location_te_map.keys()))
selected_province = st.selectbox("Province", provinces)
selected_type = st.selectbox("Property Type", list(property_types.keys()))

bedroom = st.number_input("Bedrooms", min_value=0, max_value=20, step=1)
bath = st.number_input("Bathrooms", min_value=0, max_value=20, step=1)
area_sqft = st.number_input("Area (sqft)", min_value=1)

# -----------------------------
# Prepare model input
# -----------------------------
def prepare_input():
    log_area = np.log1p(area_sqft)

    input_dict = {
        "location_city_te": city_te_map[selected_city],
        "location_te": location_te_map[selected_location],
        "bedroom_imputed": bedroom,
        "bath": bath,
        "area_sqft": area_sqft,
        "log_area": log_area,
    }

    # Property type one-hot
    for col in property_types.values():
        input_dict[col] = 1 if col == property_types[selected_type] else 0

    # Province one-hot
    for prov in provinces[:-1]:  # Skip last one for baseline if needed
        col = f"location_province_ {prov}"
        input_dict[col] = 1 if prov == selected_province else 0

    return pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    try:
        input_df = prepare_input()
        log_price = model.predict(input_df)[0]
        price = np.expm1(log_price)

        st.success(f"🏷️ **Estimated Price:** Rs {price:,.0f}")
    except Exception as e:
        st.error("⚠️ Prediction failed.")
        st.exception(e)
