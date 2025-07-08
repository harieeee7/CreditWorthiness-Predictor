# streamlit_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model artifacts
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
default_values = joblib.load("default_values.pkl")

st.set_page_config(page_title="Creditworthiness Predictor", layout="centered")

# Sidebar project info
with st.sidebar:
    st.title("📘 Project Info")

    st.markdown("""
    **🔍 Creditworthiness Prediction App**

    Predicts whether a user's loan will be approved based on key financial indicators.

    **🧠 Trained Features**:
    - Age
    - Annual Income
    - Monthly Debt Payments
    - Debt to Income Ratio
    - Payment History
    - Utility Bills Payment History
    - Previous Loan Defaults
    - Credit Score
    - Bankruptcy History
    - Interest Rate
    - Loan Amount

    **🛠️ Tools Used**:
    - Python, Pandas, NumPy
    - Scikit-learn, SMOTE
    - Streamlit, Joblib

    **📁 Dataset**:
    - Source: Provided `Loan.csv` file
    - Type: Binary classification (Loan Approved or Not)

    **👨‍💻 Developer**:
    [Haricharan Pentamalla](https://www.linkedin.com/in/haricharanpentamalla/)
    """)
    st.markdown("---")    


st.title("🔍 Creditworthiness Predictor")
st.write("Enter your financial details below:")

# Currency Selector
currency = st.radio("Select Currency:", ["INR (₹)", "USD ($)"])
symbol = "₹" if currency == "INR (₹)" else "$"
conversion = 1 if currency == "INR (₹)" else 83

# Input form
with st.form("predict_form"):
    age = st.number_input("🎂 Age", min_value=18, max_value=100, value=int(default_values.get("Age", 30)))
    income = st.number_input(f"💰 Annual Income ({symbol})", min_value=0.0, value=round(default_values["AnnualIncome"]/conversion, 2))
    debt = st.number_input(f"💳 Total Monthly Debts ({symbol})", min_value=0.0, value=round(default_values["MonthlyDebtPayments"]/conversion, 2))
    interest = st.number_input("📈 Interest Rate (%)", min_value=0.0, max_value=100.0, value=float(default_values.get("InterestRate", 12.0)))
    loan_amt = st.number_input(f"🏦 Loan Amount Requested ({symbol})", min_value=0.0, value=round(default_values["LoanAmount"]/conversion, 2))

    submit = st.form_submit_button("🔍 Predict")

if submit:
    # Prepare input
    user_input = {
        "Age": age,
        "AnnualIncome": income * conversion,
        "MonthlyDebtPayments": debt * conversion,
        "DebtToIncomeRatio": (debt * conversion) / (income * conversion + 1),
        "PaymentHistory": default_values["PaymentHistory"],
        "UtilityBillsPaymentHistory": default_values["UtilityBillsPaymentHistory"],
        "PreviousLoanDefaults": default_values["PreviousLoanDefaults"],
        "CreditScore": default_values["CreditScore"],
        "BankruptcyHistory": default_values["BankruptcyHistory"],
        "InterestRate": interest,
        "LoanAmount": loan_amt * conversion
    }

    df_input = pd.DataFrame([user_input])[feature_names]
    scaled_input = scaler.transform(df_input)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.success(f"✅ Loan is likely to be approved!\n\n**Confidence:** {proba * 100:.2f}%")
    else:
        st.error(f"❌ Loan is likely to be rejected.\n\n**Confidence:** {(1 - proba) * 100:.2f}%")
