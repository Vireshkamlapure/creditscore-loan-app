import streamlit as st
import pandas as pd
import joblib 

# Page configuration
st.set_page_config(page_title="CrediScore Loan Predictor", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .big-title {
        font-size: 38px;
        font-weight: bold;
        background: linear-gradient(90deg, #0066ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .sub-title {
        font-size: 18px;
        color: #666;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<p class='big-title'>🚀 CrediScore Loan Approval Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Instant AI-based prediction for loan approval with confidence score</p>", unsafe_allow_html=True)
st.markdown("---")

# Ask user's name first
user_name = st.text_input("👤 Enter your name")

# Load Model
try:
    model = joblib.load('loan_model.joblib')
except FileNotFoundError:
    st.error("❌ The required model is not found")
    st.stop()
except Exception as e:
    st.error(f"⚠ An error occured : {e}")
    st.stop()

# Form Section
with st.container():
    with st.form(key='loan_form'):

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Personal Details")
            gender = st.selectbox('Gender', ('Male', 'Female'))
            married = st.selectbox('Married', ('Yes', 'No'))
            dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))
            education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
            self_employed = st.selectbox('Self Employed', ('Yes', 'No'))

        with col2:
            st.subheader("💰 Financial Details")
            applicant_income = st.number_input('Applicant Income (₹)', min_value=0, value=5000)
            coapplicant_income = st.number_input('Co-Applicant Income (₹)', min_value=0.0, value=0.0)
            loan_amount = st.number_input('Loan Amount (in thousands)', min_value=0.0, value=100.0)
            loan_amount_term = st.number_input('Loan Term (months)', min_value=0.0, value=360.0)
            credit_history = st.selectbox('Credit History Available', (1.0, 0.0), format_func=lambda x: 'Yes' if x == 1.0 else 'No')
            property_area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

        submit_button = st.form_submit_button("🔍 Predict Loan Eligibility")

st.markdown("---")

# Prediction Logic
if submit_button:

    if not user_name.strip():
        st.warning("⚠ Please enter your name before proceeding!")
        st.stop()

    input_data = pd.DataFrame({
        'Gender':[gender],
        'Married':[married],
        'Dependents':[dependents],
        'Education':[education],
        'Self_Employed':[self_employed],
        'ApplicantIncome':[applicant_income],
        'CoapplicantIncome':[coapplicant_income],
        'LoanAmount':[loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.markdown("## 🎯 Prediction Result")

        if prediction[0] == 1:
            st.balloons()
            st.success(f"### ✅ Congratulations **{user_name}!** Your loan is **APPROVED** 🎉")
            st.markdown(f"**💡 Approval Confidence:** `{prediction_proba[0][1] * 100:.2f}%`")

            st.info("🔎 Tip: Maintain your good credit score for better interest rates!")

        else:
            st.error(f"### ❌ Sorry **{user_name},** your loan is **NOT APPROVED**")
            st.markdown(f"**💡 Rejection Confidence:** `{prediction_proba[0][0] * 100:.2f}%`")

            st.warning("📌 Tip: Try improving credit history or reducing loan amount and try again.")

    except Exception as e:
        st.error(f"⚠ Prediction failed due to error: {e}")

st.markdown("---")
st.caption("💻 Built with ❤️ by Viresh Kamlapure")
