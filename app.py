import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("flight_price_model.pkl")

st.title("Flight Price Prediction")

# User inputs
airline = st.selectbox("Airline", ['Air_India', 'IndiGo', 'SpiceJet', 'Vistara', 'GO_FIRST'])
source_city = st.selectbox("Source City", ['Delhi', 'Chennai', 'Mumbai', 'Kolkata', 'Hyderabad'])
departure_time = st.selectbox("Departure Time", ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night'])
arrival_time = st.selectbox("Arrival Time", ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night'])
destination_city = st.selectbox("Destination City", ['Delhi', 'Chennai', 'Mumbai', 'Kolkata', 'Hyderabad'])
stops = st.selectbox("Stops", [0, 1, 2, 3])
days_left = st.number_input("Days Left", min_value=1, max_value=365)
duration = st.number_input("Duration (hours)", min_value=0.5)
flight_class = st.selectbox("Class", ["Economy", "Business"])


# Create input dataframe
input_df = pd.DataFrame({
    "airline": [airline],
    "source_city": [source_city],
    "departure_time": [departure_time],
    "arrival_time": [arrival_time],
    "destination_city": [destination_city],
    "stops": [stops],
    "days_left": [days_left],
    "duration": [duration],
    "class": [flight_class], 
})

# Apply get_dummies
input_df = pd.get_dummies(input_df)

# Align columns with training data
model_features = model.feature_names_in_

for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure same column order
input_df = input_df[model_features]

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Flight Price: ₹{int(prediction)}")
