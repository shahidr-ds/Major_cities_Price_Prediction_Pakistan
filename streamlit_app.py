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
# 2. Recover Readable Mappings (Assumes Raw Names Exist)
# --------------------------------------------

# If you have raw names stored in 'location_city' and 'location' columns:
if 'location_city' in df.columns and 'location' in df.columns:
    city_map_df = df[['location_city', 'location_city_te']].drop_duplicates()
    city_te_map = dict(zip(city_map_df['location_city'], city_map_df['location_city_te']))

    location_map_df = df[['location', 'location_te', 'location_city_te']].drop_duplicates()

else:
    st.error("❌ Raw city and society names ('location_city' and 'location') not found in df_fe.csv.")
    st.stop()

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
# 3. Streamlit Interface
# --------------------------------------------
st.title("🏠 Pakistan Real Estate Price Predictor")

# Select readable city name
selected_city_name = st.selectbox("Select City", sorted(city_te_map.keys()))
city_te = city_te_map[selected_city_name]

# Filter related societies
related_societies = location_map_df[location_map_df["location_city_te"] == city_te]
society_te_map = dict(zip(related_societies["location"], related_societies["location_te"]))
selected_society_name = st.selectbox("Select Society", sorted(society_te_map.keys()))
loc_te = society_te_map[selected_society_name]

# Property details
prop_type = st.selectbox("Property Type", ["House", "Flat", "Shop", "Residential Plot"])
bed = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
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
# 5. Prediction
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
        st.info(f"🏙️ {selected_city_name} → {selected_society_name} | {prop_type} | 🛏️ {bed}, 🛁 {bath}, 📐 {area} sqft")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
