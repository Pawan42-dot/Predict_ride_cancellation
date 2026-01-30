import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("ride_cancellation_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title(" Ride Cancellation Predictor")

st.write("Enter ride details to predict cancellation risk")

# User inputs
booking_hour = st.slider("Booking Hour", 0, 23, 12)
ride_distance_km = st.slider("Ride Distance (km)", 1, 50, 10)
driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.0)
customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0)
surge_pricing = st.slider("Surge Pricing", 1.0, 3.0, 1.5)

pickup_location = st.selectbox("Pickup Location",
                                ["Downtown", "Mall", "Railway Station", "Residential Area"])

drop_location = st.selectbox("Drop Location",
                              ["Downtown", "Mall", "Railway Station", "Residential Area"])

weather = st.selectbox("Weather",
                        ["Foggy", "Rainy", "Stormy"])

# Build input row
input_dict = {
    "booking_hour": booking_hour,
    "ride_distance_km": ride_distance_km,
    "driver_rating": driver_rating,
    "customer_rating": customer_rating,
    "surge_pricing": surge_pricing
}

# Create empty dataframe
input_df = pd.DataFrame(columns=columns)
input_df.loc[0] = 0

# Fill numeric
for key in input_dict:
    input_df[key] = input_dict[key]

# Fill encoded categories
input_df[f"pickup_location_{pickup_location}"] = 1
input_df[f"drop_location_{drop_location}"] = 1
input_df[f"weather_{weather}"] = 1

# Predict
if st.button("Predict Cancellation"):
    prob = model.predict_proba(input_df)[0][1]

    if prob > 0.6:
        st.error(f" High Risk of Cancellation: {prob:.2f}")
    else:
        st.success(f" Low Risk of Cancellation: {prob:.2f}")
    st.balloons()
