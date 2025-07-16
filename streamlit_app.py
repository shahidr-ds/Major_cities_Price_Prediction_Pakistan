import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------
# 1. Load Model and Data
# --------------------------------------------
model = joblib.load("xgb_model_cleaned.pkl")
df = pd.read_csv("df_fe.csv")

# --------------------------------------------
# 2. Setup Mappings
# --------------------------------------------
city_te_values = df["location_city_te"].drop_duplicates().sort_values()
city_te_map = {v: v for v in city_te_values}

location_te_values = df["location_te"].drop_duplicates().sort_values()
location_te_map = {v: v for v in location_te_values}

province_cols = [col for col in df.columns if col.startswith("location_province_")]
city_to_province = (
    df.drop_duplicates("location_city_te")
      .set_index("location_city_te")[province_cols]
      .idxmax(axis=1)
      .str.replace("location_province_ ", "", regex=False)
      .str.strip()
      .to_dict()
)

# --------------------------------------------
# 3. UI Inputs
# --------------------------------------------
st.title("🏠 Pakistan Real Estate Price Predictor")

city_te = st.selectbox("Select City (Encoded)", sorted(city_te_map.keys()))
related_societies = df[df["location_city_te"] == city_te]["location_te"].drop_duplicates().sort_values()
loc_te = st.selectbox("Select Society (Encoded)", related_societies)

prop_type = st.selectbox("Property Type", ["House", "Flat", "Shop", "Residential Plot"])

bed = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
area = st.number_input("Area (sqft)", min_value=100, max_value=100000, value=1200, step=50)

# --------------------------------------------
# 4. Feature Builder
# --------------------------------------------
def build_features(bath, bed, area, loc_te, city_te, province, prop_type):
    return {
        "type_House": int(prop_type == "House"),
        "type_Flat": int(prop_type == "Flat"),
        "type_Shop": int(prop_type == "Shop"),
        "type_Residential Plot": int(prop_type == "Residential Plot"),
        "bath": bath,
        "bedroom_imputed": bed,
        "area_sqft": area,
        "days_since_posted": 30,  # assumed
        "location_city_te": city_te,
        "location_te": loc_te,
        "location_province_ Punjab": int(province == "Punjab"),
        "location_province_ Sindh": int(province == "Sindh"),
        "location_province_ Khyber Pakhtunkhwa": int(province == "Khyber Pakhtunkhwa"),
        "location_province_ Islamabad Capital": int(province == "Islamabad Capital"),
        "log_price_per_sqft": np.log(1.0),
        "log_area_price_ratio": np.log(1.0)
    }

# --------------------------------------------
# 5. Predict Button
# --------------------------------------------
if st.button("Predict Price"):
    try:
        province = city_to_province.get(city_te, "Punjab")
        features = build_features(bath, bed, area, loc_te, city_te, province, prop_type)
        df_input = pd.DataFrame([features])

        # Align with model input
        expected = model.get_booster().feature_names
        df_input = df_input[expected]

        pred_log = model.predict(df_input)[0]
        pred_price = round(np.exp(pred_log), 2)

        st.success(f"💰 Predicted Price: {pred_price} Million PKR")
        st.info(f"🏠 {prop_type} in society (encoded): {loc_te}, city (encoded): {city_te}, province: {province}")
    except Exception as e:
        st.error(f"Error: {e}")
