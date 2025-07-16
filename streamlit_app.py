import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Load only the necessary raw data
# --------------------------------------------------
@st.cache_data
def load_data():
    df_raw = pd.read_csv("Major_cities_data.csv")
    return df_raw

df_raw = load_data()

# --------------------------------------------------
# Define province and type columns used in the model
# --------------------------------------------------
provinces = ["Punjab", "Sindh", "Islamabad Capital", "Khyber Pakhtunkhwa"]
property_types = {
    "House": "type_House",
    "Flat": "type_Flat",
    "Shop": "type_Shop",
    "Residential Plot": "type_Residential Plot"
}

# --------------------------------------------------
# Generate mappings from raw data
# --------------------------------------------------
city_te_map = df_raw.drop_duplicates("location_city")[["location_city", "location_city_te"]].set_index("location_city")["location_city_te"].to_dict()
location_te_map = df_raw.drop_duplicates("location")[["location", "location_te"]].set_index("location")["location_te"].to_dict()

# --------------------------------------------------
# Load the trained model
# --------------------------------------------------
model = joblib.load("best_model.pkl")

# --------------------------------------------------
# Streamlit App UI
# --------------------------------------------------
st.title("🏠 Real Estate Price Prediction App (Pakistan)")

# --- User Inputs ---
selected_city = st.selectbox("Select City", sorted(city_te_map.keys()))
selected_location = st.selectbox("Select Location", sorted(location_te_map.keys()))
selected_province = st.selectbox("Select Province", sorted(provinces))
selected_type = st.selectbox("Select Property Type", list(property_types.keys()))

bedroom = st.number_input("Bedrooms", min_value=0, max_value=20, step=1)
bath = st.number_input("Bathrooms", min_value=0, max_value=20, step=1)
area_sqft = st.number_input("Area (sqft)", min_value=1)

# --------------------------------------------------
# Prepare Input for Model
# --------------------------------------------------
def create_input_df():
    input_dict = {
        "location_city_te": city_te_map[selected_city],
        "location_te": location_te_map[selected_location],
        "bedroom_imputed": bedroom,
        "bath": bath,
        "area_sqft": area_sqft
    }

    # One-hot encode property type
    for col in property_types.values():
        input_dict[col] = 1 if col == property_types[selected_type] else 0

    # One-hot encode province
    for prov in provinces:
        col = f"location_province_ {prov}"
        input_dict[col] = 1 if prov == selected_province else 0

    return pd.DataFrame([input_dict])

# --------------------------------------------------
# Predict
# --------------------------------------------------
if st.button("Predict Price"):
    input_df = create_input_df()

    try:
        log_price_pred = model.predict(input_df)[0]
        price_pred = np.expm1(log_price_pred)  # Inverse of log1p

        st.success(f"🏷️ **Estimated Price:** Rs {price_pred:,.0f}")
    except Exception as e:
        st.error("⚠️ Prediction failed. Check model or input.")
        st.exception(e)
