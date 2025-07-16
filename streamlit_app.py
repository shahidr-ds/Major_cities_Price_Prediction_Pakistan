import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------
# 1. Load Model and Data
# --------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_model_cleaned.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("df_fe.csv")

model = load_model()
df = load_data()

# --------------------------------------------
# 2. Validate Columns
# --------------------------------------------
required_cols = {"location_city", "location", "location_city_te", "location_te"}
if not required_cols.issubset(df.columns):
    st.error("❌ `df_fe.csv` must include: location_city, location, location_city_te, and location_te.")
    st.stop()

# --------------------------------------------
# 3. Create Mappings
# --------------------------------------------
# City → encoded
city_map_df = df[["location_city", "location_city_te"]].drop_duplicates()
city_te_map = dict(zip(city_map_df["location_city"], city_map_df["location_city_te"]))

# Province mapping from encoded city
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
# 4. Streamlit App UI
# --------------------------------------------
st.set_page_config("🏠 Pakistan Price Predictor", layout="centered")
st.title("🏠 Pakistan Real Estate Price Predictor")
st.markdown("Predict house/plot/shop prices in major Pakistani cities using machine learning.")

# --- City Dropdown ---
selected_city = st.selectbox("📍 Select City", sorted(city_te_map.keys()))
city_te = city_te_map[selected_city]

# --- Society Dropdown ---
societies_df = df[df["location_city_te"] == city_te][["location", "location_te"]].drop_duplicates()
society_te_map = dict(zip(societies_df["location"], societies_df["location_te"]))
selected_society = st.selectbox("🏘️ Select Society", sorted(society_te_map.keys()))
loc_te = society_te_map[selected_society]

# --- Property Details ---
property_type = st.selectbox("🏗️ Property Type", ["House", "Flat", "Shop", "Residential Plot"])
bedrooms = st.number_input("🛏️ Bedrooms", min_value=0, max_value=10, value=3)
bathrooms = st.number_input("🛁 Bathrooms", min_value=0, max_value=10, value=2)
area_sqft = st.number_input("📐 Area (sqft)", min_value=100, max_value=100000, value=1200, step=50)

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
        "days_since_posted": 30,  # assumed average
        "location_city_te": city_te,
        "location_te": loc_te,
        "location_province_ Punjab": int(province == "Punjab"),
        "location_province_ Sindh": int(province == "Sindh"),
        "location_province_ Khyber Pakhtunkhwa": int(province == "Khyber Pakhtunkhwa"),
        "location_province_ Islamabad Capital": int(province == "Islamabad Capital"),
        "log_price_per_sqft": np.log(1.0),  # placeholder
        "log_area_price_ratio": np.log(1.0)  # placeholder
    }

# --------------------------------------------
# 6. Predict Button
# --------------------------------------------
if st.button("💰 Predict Price"):
    try:
        province = city_to_province.get(city_te, "Punjab")
        features = build_features(
            bath=bathrooms,
            bed=bedrooms,
            area=area_sqft,
            loc_te=loc_te,
            city_te=city_te,
            province=province,
            prop_type=property_type
        )

        df_input = pd.DataFrame([features])
        expected_features = model.get_booster().feature_names
        df_input = df_input[expected_features]

        pred_log = model.predict(df_input)[0]
        predicted_price = round(np.exp(pred_log), 2)

        st.success(f"💰 Predicted Price: {predicted_price} Million PKR")
        st.caption(f"{property_type} in {selected_society}, {selected_city} | {area_sqft} sqft | {bedrooms} bed, {bathrooms} bath")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
