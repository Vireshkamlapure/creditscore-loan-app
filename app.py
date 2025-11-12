import streamlit as st
import pandas as pd
import joblib 

# Page config
st.set_page_config(page_title="CrediScore", layout="centered")

# ---- App Title Styling ----
st.markdown("""
<style>
.app-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    color: #1E90FF;
}
.line {
    width: 110px;
    height: 3px;
    background-color: #1E90FF;
    margin: 6px auto 20px auto;
    border-radius: 50px;
}
</style>
""", unsafe_allow_html=True)

# Title UI
st.markdown('<div class="app-title">CrediScore</div>', unsafe_allow_html=True)
st.markdown('<div class="line"></div>', unsafe_allow_html=True)
st.write("### AI Based Loan Approval Prediction")
st.markdown("---")

# User name input
user_name = st.text_input("👤 Enter your Name")

# Load Model
try:
    model = joblib.load('loan_model.joblib')
except:
    st.error("❌ Model not found!")
    st.stop()

# ---- Form ----
with st.form("loan_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Personal Information")
        gender = st.selectbox('Gender', ('Male', 'Female'))
        married = st.selectbox('Married', ('Yes', 'No'))
        dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))
        education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
        self_employed = st.selectbox('Self Employed', ('Yes', 'No'))

    with col2:
        st.subheader("💰 Financial & Loan Details")
        applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
        coapplicant_income = st.number_input('Co-Applicant Income', min_value=0.0, value=0.0)
        loan_amount = st.number_input('Loan Amount (in thousands)', min_value=0.0, value=100.0)
        loan_amount_term = st.number_input('Loan Term (Months)', min_value=0.0, value=360.0)
        credit_history = st.selectbox('Credit History Available', (1.0, 0.0), format_func=lambda x: 'Yes' if x else 'No')
        property_area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

    submit = st.form_submit_button("✅ Predict Eligibility")

st.markdown("---")

# ---- Prediction ----
if submit:

    if not user_name.strip():
        st.warning("⚠ Please enter your name above before predicting.")
        st.stop()

    data = pd.DataFrame([{
        'Gender': gender, 'Married': married, 'Dependents': dependents, 'Education': education,
        'Self_Employed': self_employed, 'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income, 'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term, 'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    try:
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0]

        st.subheader("🎯 Prediction Result")

        if pred == 1:
            st.success(f"✅ **Congratulations {user_name}!** Your loan is **APPROVED** 🥳")
            st.write(f"**Confidence Score:** {prob[1] * 100:.2f}%")
            st.balloons()
        else:
            st.error(f"❌ **Sorry {user_name},** your loan is **REJECTED**")
            st.write(f"**Confidence Score:** {prob[0] * 100:.2f}%")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built with ❤️ by Viresh Kamlapure")
