import streamlit as st
import pandas as pd
import joblib 

# Page Config
st.set_page_config(page_title="CrediScore", layout="centered")

# Modern UI Styling
st.markdown("""
<style>
/* App Title */
.app-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 2px;
    text-align: center;
    background: linear-gradient(90deg, #1E90FF, #00C2FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: -10px;
}
.app-line {
    width: 140px;
    height: 4px;
    background: linear-gradient(90deg, #1E90FF, #00C2FF);
    margin: auto;
    border-radius: 20px;
}

.sub-text {
    font-size: 16px;
    color: #6e6e6e;
    text-align: center;
    margin-top: 8px;
}

/* Form Box */
div[data-testid="stForm"] {
    background: #ffffff;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e0e0e0;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
}

/* Result Box */
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 18px;
    margin-top: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# Header UI
st.markdown('<div class="app-title">CrediScore</div>', unsafe_allow_html=True)
st.markdown('<div class="app-line"></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI Powered Loan Approval Prediction System</div>', unsafe_allow_html=True)
st.markdown("---")

# User Name Input
user_name = st.text_input("👋 Enter your name")

# Load Model
try:
    model = joblib.load('loan_model.joblib')
except:
    st.error("❌ Model not found")
    st.stop()

# Form UI
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Personal Info")
        gender = st.selectbox('Gender', ('Male', 'Female'))
        married = st.selectbox('Married', ('Yes', 'No'))
        dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))
        education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
        self_employed = st.selectbox('Self Employed', ('Yes', 'No'))

    with col2:
        st.subheader("💸 Financial Info")
        applicant_income = st.number_input('Applicant Income (₹)', min_value=0, value=5000)
        coapplicant_income = st.number_input('Co-Applicant Income (₹)', min_value=0.0, value=0.0)
        loan_amount = st.number_input('Loan Amount (in thousands)', min_value=0.0, value=100.0)
        loan_amount_term = st.number_input('Loan Term (Months)', min_value=0.0, value=360.0)
        credit_history = st.selectbox('Credit History', (1.0, 0.0), format_func=lambda x: 'Yes' if x else 'No')
        property_area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

    submit = st.form_submit_button("🚀 Check Loan Eligibility")

# Prediction
if submit:

    if not user_name.strip():
        st.warning("Please enter your name first 👆")
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

        st.markdown("### 🎯 Prediction Result")

        if pred == 1:
            st.markdown(f"""
            <div class="result-box" style="background:#e6f9ed; border:1px solid #2ecc71;">
            ✅ <b>Congrats {user_name}!</b><br>
            Your Loan is <b style="color:#27ae60;">APPROVED</b><br>
            <br>Confidence: <b>{prob[1]*100:.2f}%</b>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()

        else:
            st.markdown(f"""
            <div class="result-box" style="background:#fdecea; border:1px solid #e74c3c;">
            ❌ <b>Sorry {user_name},</b><br>
            Your Loan is <b style="color:#e74c3c;">REJECTED</b><br>
            <br>Confidence: <b>{prob[0]*100:.2f}%</b>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown('<div class="footer">Built with ❤️ by Viresh Kamlapure</div>', unsafe_allow_html=True)
