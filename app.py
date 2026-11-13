import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('hotel_cancellation_model.joblib')

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="centered")

st.title("Hotel Revenue Protection: Cancellation Predictor")
st.write("Enter the booking details below to check the probability of cancellation and the recommended policy to protect hotel revenue.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input("Lead Time (Days)", min_value=0, max_value=737, value=30)
        adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, value=100.0)
        market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"])
        deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
        
    with col2:
        adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        babies = st.number_input("Babies", min_value=0, max_value=5, value=0)
        stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=50, value=2)
        stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=20, value=1)
        total_of_special_requests = st.slider("Special Requests", min_value=0, max_value=5, value=0)
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=20, value=0)

    submit_button = st.form_submit_button(label="Predict Booking Status")

if submit_button:
    input_data = pd.DataFrame({
        'hotel': [hotel],
        'lead_time': [lead_time],
        'arrival_date_year': [2016],
        'arrival_date_month': ['July'],
        'arrival_date_week_number': [27],
        'arrival_date_day_of_month': [15],
        'stays_in_weekend_nights': [stays_in_weekend_nights],
        'stays_in_week_nights': [stays_in_week_nights],
        'adults': [adults],
        'children': [children],
        'babies': [babies],
        'meal': ['BB'],
        'country': ['PRT'],
        'market_segment': [market_segment],
        'distribution_channel': ['TA/TO' if 'TA' in market_segment else 'Direct'],
        'is_repeated_guest': [0],
        'previous_cancellations': [previous_cancellations],
        'previous_bookings_not_canceled': [0],
        'reserved_room_type': ['A'],
        'assigned_room_type': ['A'],
        'booking_changes': [0],
        'deposit_type': [deposit_type],
        'agent': [9.0 if 'TA' in market_segment else 0.0],
        'days_in_waiting_list': [0],
        'customer_type': ['Transient'],
        'adr': [adr],
        'required_car_parking_spaces': [0],
        'total_of_special_requests': [total_of_special_requests],
        # Feature Engineering columns added during training
        'total_stay_nights': [stays_in_weekend_nights + stays_in_week_nights],
        'total_guests': [adults + children + babies],
        'has_special_requests': [1 if total_of_special_requests > 0 else 0],
        'has_car_parking': [0],
        'has_previous_cancellations': [1 if previous_cancellations > 0 else 0],
        'changes_per_lead_time': [0.0]
    })
    
    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    st.subheader("Analysis and Prediction Result:")
    
    if prediction == 1:
        st.error(f"Warning: High probability of cancellation! (Probability: {probability:.1f}%)")
        st.info("Recommendation for the Hotel: Require full deposit or a non-refundable policy to protect revenue.")
    else:
        st.success(f"Booking is stable and likely to be confirmed (Expected cancellation probability: {probability:.1f}%)")
        st.info("Recommendation for the Hotel: Allow booking without advance deposit.")
