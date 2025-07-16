import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------
# 1. Load Model and Data
# --------------------------------------------
try:
    model = joblib.load("xgb_model_cleaned.pkl")
    df = pd.read_csv("df_fe.csv")
except Exception as e:
    st.error(f"❌ Failed to load model or data: {e}")
    st.stop()

# --------------------------------------------
# 2. Check for Raw Names (Required for Display)
# --------------------------------------------
required_cols = {"location_city", "location", "location_city_te", "location_te"}
if not required_cols.issubset(df.columns):
    st.error("❌ df_fe.csv must include: location_city, location, location_city_te, and location_te.")
    st.stop()

# --------------------------------------------
# 3. Build Mappings
# --------------------------------------------
# City: readable → encoded
city_map_df = df[["location_city", "location_city_te"]].drop_duplicates()
city_te_map = dict(zip(city_map_df["location_city"], city_map_df["location_city_te"]))

# Province from encoded city
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
# 4. Streamlit UI
# --------------------------------------------
st.set_page_config(page_title="Pakistan Real Estate Price Predictor", layout="centered")
st.title("🏠 Pakistan Real Estate Price Predictor")
st.write("Predict property prices in major cities of Pakistan using a trained XGBoost model.")

# City dropdown
selected_city = st.selectbox("📍 Select City", sorted(city_te_map.keys()))
city_te = city_te_map[selected_city]

# Filter societies for selected city
location_map_df = df[df["location_city_te"] == city_te][["location", "location_te"]].drop_duplicates()
society_te_map = dict(zip(location_map_df["location"], location_map_df["location_te"]))

# Society dropdown
selected_society = st.selectbox("🏘️ Select Society", sorted(society_te_map.keys()))
loc_te = society_te_map[selected_society]

# Property details
prop_type = st.selectbox("🏗️ Property Type", ["House", "Flat", "Shop", "Residential Plot"])
bed = st.number_input("🛏️ Bedrooms", min_value=0, max_value=10, value=3)
bath = st.number_input("🛁 Bathrooms", min_value=0, max_value=10, value=2)
area = st.number_input("📐 Area (sqft)", min_value=100, max_value=100000, value=1200, step=50)

# --------------------------------------------
# 5. Feature Builder
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
        "days_since_posted": 30,
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
# 6. Predict Button
# --------------------------------------------
if st.button("💰 Predict Price"):
    try:
        province = city_to_province.get(city_te, "Punjab")
        input_features = build_features(bath, bed, area, loc_te, city_te, province, prop_type)
        df_input = pd.DataFrame([input_features])

        # Ensure columns match model input
        expected_cols = model.get_booster().feature_names
        df_input = df_input[expected_cols]

        pred_log_price = model.predict(df_input)[0]
        pred_price = round(np.exp(pred_log_price), 2)

        st.success(f"💰 Predicted Price: {pred_price} Million PKR")
        st.caption(f"📊 Based on a {prop_type} in {selected_society}, {selected_city} — {area} sqft, {bed} bed, {bath} bath")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
