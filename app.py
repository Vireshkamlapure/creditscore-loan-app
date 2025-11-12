import streamlit as st
import pandas as pd
import joblib

# 1. Load Model
try:
    model = joblib.load('loan_model.joblib')
except FileNotFoundError:
    st.error("The required model file ('loan_model.joblib') was not found.")
    st.info("Please make sure the model file is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# 2. UI Configuration
st.set_page_config(page_title="CrediScore Loan Predictor", page_icon="🏦", layout="centered")

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    /* --- Base & Body --- */
    .stApp {
        /* A subtle gradient background */
        background: linear-gradient(to right top, #d3e6f9, #f0f2f6, #ffffff);
    }
    
    /* --- Main Content Card --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #ffffff;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.05);
        max-width: 750px; /* Constrain width for a cleaner look */
        margin: 2rem auto; /* Center the card */
    }
    
    /* --- Title --- */
    h1 {
        color: #0c4a6e; /* Deep blue */
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* --- Introduction Text --- */
    .stAlert[data-baseweb="alert"] {
        border-radius: 10px;
        background-color: #f0f9ff; /* Light blue info box */
        border: 1px solid #bae6fd;
    }
    
    /* --- Subheaders --- */
    h3 {
        color: #1e3a8a; /* Strong blue */
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
        margin-top: 1.5rem;
    }

    /* --- Form Submit Button --- */
    .stButton > button {
        width: 100%;
        background-color: #059669; /* Green */
        color: white;
        border: none;
        padding: 12px 0;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        background-color: #047857; /* Darker green on hover */
        transform: scale(1.01);
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 3px #05966960; /* Focus ring */
    }

    /* --- Input Widgets --- */
    .stSelectbox, .stNumberInput {
        border-radius: 8px;
    }
    
    /* --- Result Boxes --- */
    .stAlert[data-baseweb="alert"][data-kind="success"] {
        background-color: #f0fdf4;
        border: 1px solid #16a34a;
        border-radius: 10px;
    }
    .stAlert[data-baseweb="alert"][data-kind="error"] {
        background-color: #fef2f2;
        border: 1px solid #dc2626;
        border-radius: 10px;
    }
    
    /* --- Metric --- */
    [data-testid="stMetric"] {
        background-color: #f8f8f8;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* --- Footer --- */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 2.5rem;
    }
    
    /* --- Markdown Ruler --- */
    hr {
        border-top: 1px solid #e0e0e0;
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.title("🚀💹 CrediScore: Loan Approval Predictor")
st.info(
    "ℹ️ Enter the applicant's details below to get a real-time prediction "
    "on their loan approval status.",
    icon="ℹ️"
)

# 3. Input Form
with st.form(key='loan_form'):
    
    # --- Create columns for a cleaner layout ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        gender = st.selectbox('Gender', ('Male', 'Female'), index=0)
        married = st.selectbox('Married', ('Yes', 'No'), index=0)
        dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'), index=0)
        education = st.selectbox('Education', ('Graduate', 'Not Graduate'), index=0)
        self_employed = st.selectbox('Self Employed', ('Yes', 'No'), index=0)

    with col2:
        st.subheader("💰 Financial & Loan Details")
        applicant_income = st.number_input('Applicant Income ($)', min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input('Coapplicant Income ($)', min_value=0.0, value=0.0, step=100.0)
        loan_amount = st.number_input('Loan Amount (in thousands $)', min_value=0.0, value=100.0, step=10.0)
        loan_amount_term = st.number_input('Loan Amount Term (in months)', min_value=0.0, value=360.0, step=12.0)
        
        # Credit History: 1.0 means "Yes", 0.0 means "No"
        credit_history = st.selectbox(
            'Credit History Available', 
            (1.0, 0.0), 
            format_func=lambda x: 'Yes' if x == 1.0 else 'No'
        )
        property_area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'), index=0)

    # --- Submit Button ---
    st.markdown("<br>", unsafe_allow_html=True) # Adds a little space
    submit_button = st.form_submit_button(label='Predict Eligibility')

st.markdown("---")

# 4. Prediction Logic & Display
if submit_button:
    # --- Create DataFrame for the model ---
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

    # --- Get the prediction ---
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # --- Display the Result ---
        st.subheader('📊 Prediction Result')
        
        if prediction[0] == 1:
            # --- Approved ---
            st.success('**Congratulations!** Your loan is likely to be **APPROVED.**', icon="✅")
            
            prob = prediction_proba[0][1] * 100
            st.metric(label="Approval Confidence Score", value=f"{prob:.2f}%")
            
            # Add a celebratory emoji
            st.balloons()
            
        else:
            # --- Rejected ---
            st.error('**Unfortunately,** your loan is likely to be **REJECTED.**', icon="❌")
            
            prob = prediction_proba[0][0] * 100
            st.metric(label="Rejection Confidence Score", value=f"{prob:.2f}%")
            st.warning(
                "**Disclaimer:** This is a prediction based on a machine learning model. "
                "Final approval is subject to a full review by the financial institution.",
                icon="⚠️"
            )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown('<div class="footer">By Viresh Kamlapure</div>', unsafe_allow_html=True)