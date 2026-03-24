# app.py
from preprocessing import preprocess
import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipe = joblib.load("pipeline.pkl")
# pipe = joblib.load("../pipeline.pkl")
preprocess = pipe["preprocessor"]
models = pipe["models"]

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# --- Create DataFrame ---
data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "DeviceProtection": DeviceProtection,
    "OnlineBackup": OnlineBackup,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

# --- Preprocess data ---
data_processed = preprocess(data)
if "Churn" in data_processed.columns:
    data_processed = data_processed.drop("Churn", axis=1)

# --- Predict ---
if st.button("Predict Churn"):
    st.write("Predictions:")
    for name, model in models.items():
        pred = int(model.predict(data_processed)[0])
        prob = float(model.predict_proba(data_processed)[0][1])
        st.write(f"**{name}** → Prediction: {pred} (Churn Probability: {prob:.2f})")
