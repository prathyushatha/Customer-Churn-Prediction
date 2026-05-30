import streamlit as st
import pickle
import pandas as pd

# Load saved files
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))

st.title("Customer Churn Prediction")

st.header("Customer Information")

gender = st.selectbox("Gender", ["Male", "Female"])

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    value=500.0
)

if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=columns
    )

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    if "gender_Male" in input_data.columns:
        input_data["gender_Male"] = 1 if gender == "Male" else 0

    if "SeniorCitizen" in input_data.columns:
        input_data["SeniorCitizen"] = senior_citizen

    if "Partner_Yes" in input_data.columns:
        input_data["Partner_Yes"] = 1 if partner == "Yes" else 0

    if "Dependents_Yes" in input_data.columns:
        input_data["Dependents_Yes"] = 1 if dependents == "Yes" else 0

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)

    churn_probability = probability[0][1] * 100

    st.write(
        f"Churn Probability: {churn_probability:.2f}%"
    )

    if prediction[0] == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")
