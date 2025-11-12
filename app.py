import streamlit as st
import pandas as pd
import joblib 

#1.Load Model 
try:
    model = joblib.load('loan_model.joblib')
except FileNotFoundError:
    st.error("The required model is not found")
    st.stop()
except Exception as e:
    st.error("An error occured : {e}")
    st.stop()

#2.UI 
#Input Fields 
st.set_page_config(page_title="CrediScore Loan Predictor" , layout="centered")
st.title("🚀💹 CrediScore: Real-Time Loan Approval Predictor")
st.write(
    "Enter the applicant's details below to get a real-time prediction "
    "on their loan approval status."
)
st.markdown("---")
with st.form(key='loan_form'):
    
    # --- Create columns for a cleaner layout ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox('Gender', ('Male', 'Female'), index=0)
        married = st.selectbox('Married', ('Yes', 'No'), index=0)
        dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'), index=0)
        education = st.selectbox('Education', ('Graduate', 'Not Graduate'), index=0)
        self_employed = st.selectbox('Self Employed', ('Yes', 'No'), index=0)

    with col2:
        st.subheader("Financial & Loan Details")
        applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
        coapplicant_income = st.number_input('Coapplicant Income', min_value=0.0, value=0.0)
        loan_amount = st.number_input('Loan Amount (in thousands)', min_value=0.0, value=100.0)
        loan_amount_term = st.number_input('Loan Amount Term (in months)', min_value=0.0, value=360.0)
        
        # Credit History: 1.0 means "Yes", 0.0 means "No"
        credit_history = st.selectbox('Credit History Available', (1.0, 0.0), format_func=lambda x: 'Yes' if x == 1.0 else 'No')
        property_area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'), index=0)

    # --- Submit Button ---
    submit_button = st.form_submit_button(label='Predict Eligibility')

st.markdown("---")
st.write("By Viresh Kamlapure")

#3.Logic Of Working 
if submit_button:
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

#Get the prediction 
    try:
        prediction = model.predict(input_data)

        prediction_proba = model.predict_proba(input_data)

    # --- 5. Display the Result ---
        st.subheader('Prediction Result:')
        
        # prediction[0] will be 1 (Approved) or 0 (Rejected)
        if prediction[0] == 1:
            st.success('**Congratulations!** Your loan is likely to be **APPROVED.**')
            
            # Show confidence score
            st.write(f"**Confidence Score:** {prediction_proba[0][1] * 100:.2f}%")
            
            # Add a celebratory emoji
            st.balloons()
            
        else:
            st.error('**Unfortunately,** your loan is likely to be **REJECTED.**')
            
            # Show confidence score for rejection
            st.write(f"**Confidence Score:** {prediction_proba[0][0] * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")