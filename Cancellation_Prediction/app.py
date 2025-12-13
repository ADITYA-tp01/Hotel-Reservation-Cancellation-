import streamlit as st
import pandas as pd
import joblib

# Load Artifacts
model = joblib.load("artifacts/best_model_rf.joblib")
scaler = joblib.load("artifacts/scaler.joblib")
training_columns = joblib.load("artifacts/training_columns.pkl")

# Streamlit UI
st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="centered")

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Enter booking details to predict whether the reservation will be **Canceled** or **Not Canceled**.")

# Example Input Fields (Add all important ones you trained on)
lead_time = st.number_input("Lead Time (days before arrival booking was made):", min_value=0, max_value=500, value=50)
avg_price = st.number_input("Average Price per Room:", min_value=0.0, max_value=10000.0, value=150.0)
special_requests = st.number_input("Number of Special Requests:", min_value=0, max_value=5, value=0)

room_type = st.selectbox("Room Type:", ["Room_Type 1", "Room_Type 2", "Room_Type 3"])
meal_plan = st.selectbox("Meal Plan:", ["Meal Plan 1", "Meal Plan 2", "Not Selected"])
market_segment = st.selectbox("Market Segment:", ["Online", "Offline", "Corporate", "Aviation"])
repeated_guest = st.selectbox("Repeated Guest:", [0, 1])
car_parking = st.selectbox("Required Car Parking Spaces:", [0, 1, 2])
arrival_month = st.slider("Arrival Month:", 1, 12, 6)

# Prepare Input Data
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "lead_time": lead_time,
        "avg_price_per_room": avg_price,
        "no_of_special_requests": special_requests,
        "room_type": room_type,
        "meal_type": meal_plan,
        "market_segment": market_segment,
        "repeated_guest": repeated_guest,
        "required_car_parking_spaces": car_parking,
        "arrival_month": arrival_month
    }])

    # One-hot encode categorical variables
    input_data_enc = pd.get_dummies(input_data)

    # Reindex to match training columns
    input_data_enc = input_data_enc.reindex(columns=training_columns, fill_value=0)

    # Scale
    input_data_scaled = scaler.transform(input_data_enc)

    # Predict
    prediction = model.predict(input_data_scaled)[0]
    prob = model.predict_proba(input_data_scaled)[0][1]  # Probability of cancellation

    # Show Result
    if prediction == 1:
        st.error(f"❌ Booking is likely to be **Canceled** (Risk Score: {prob:.2f})")
    else:
        st.success(f"✅ Booking is likely to be **Not Canceled** (Risk Score: {prob:.2f})")
