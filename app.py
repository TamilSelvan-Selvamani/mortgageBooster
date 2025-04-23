import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and reference dataframe
clf = joblib.load("C:/Users/Ashok Kumar/risk_classifier.pkl")
regressor = joblib.load("C:/Users/Ashok Kumar/loan_regressor.pkl")
reference_df = pd.read_csv("C:/Users/Ashok Kumar/reference_data.csv")  # Preprocessed df used during training

# Prediction function
def predict_borrower(user_input):
    input_df = pd.DataFrame([user_input])

    # Drop both target columns from reference_df to avoid them being used as features
    ref_features_df = reference_df.drop(columns=['loan_status', 'loan_amnt'], errors='ignore')

    # Combine input with reference data for proper encoding
    full_df = pd.concat([ref_features_df, input_df], axis=0)

    # One-hot encoding
    full_encoded = pd.get_dummies(full_df)

    # Align feature columns with model
    full_encoded = full_encoded.reindex(columns=clf.feature_names_in_, fill_value=0)

    # Predict
    risk = clf.predict(full_encoded.tail(1))[0]
    loan_amt = regressor.predict(full_encoded.tail(1))[0]

    return int(risk), round(loan_amt, 2)

# Streamlit UI
st.title("Borrower Risk & Loan Amount Predictor")

# Input form
with st.form("borrower_form"):
    term = st.selectbox("Loan Term", [' 36 months', ' 60 months'])
    grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    annual_inc = st.number_input("Annual Income", min_value=10000, step=1000)
    verification_status = st.selectbox("Verification Status", ['Verified', 'Not Verified', 'Source Verified'])
    purpose = st.selectbox("Purpose", ['credit_card', 'debt_consolidation', 'home_improvement', 'major_purchase'])
    dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 40.0, step=0.1)
    open_acc = st.number_input("Open Accounts", min_value=0)
    pub_rec = st.number_input("Public Records", min_value=0)
    revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, step=0.1)
    total_acc = st.number_input("Total Accounts", min_value=1)
    initial_list_status = st.selectbox("Initial List Status", ['w', 'f'])
    application_type = st.selectbox("Application Type", ['Individual', 'Joint App'])

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = {
        'term': term,
        'grade': grade,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': verification_status,
        'purpose': purpose,
        'dti': dti,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'initial_list_status': initial_list_status,
        'application_type': application_type
    }

    risk, max_loan = predict_borrower(user_input)

    st.markdown("### 🧠 Prediction")
    st.write("🔴 **High Risk**" if risk else "🟢 **Low Risk**")
    st.write(f"💰 **Recommended Loan Amount**: ${max_loan}")
