import streamlit as st
import time
import numpy as np
import pandas as pd
import joblib


@st.cache_resource(show_spinner = "Loading model")
def load_model():
	model = joblib.load("final_lr_model.pkl")
	return model

@st.cache_resource(show_spinner = "Loading scaler")
def load_scaler():
	scaler = joblib.load("lr_scaler.pkl")
	return scaler

@st.cache_resource(show_spinner = "Loading col")
def load_col():
	col_name  = joblib.load("col_name.pkl")
	return col_name

	

@st.cache_data(show_spinner = "Predicting...")
def make_prediction(_model, _scaler, _col_name, X_pred):

	cat_df = pd.DataFrame([[education, employmenttype, maritalstatus, hasmortgage, hasdependent, loanpurpose, hascosigner]],
                          columns=['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'])
	cat_encoded = pd.get_dummies(cat_df, drop_first=True)

	num_df = pd.DataFrame([[age, income, loan_amount, credit_score, monthemployed, numofcl, interest, loanterm, dtiratio]],
                          columns=['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                                   'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'])
	combined_df = pd.concat([num_df, cat_encoded], axis=1)
	combined_df = combined_df.reindex(columns = col_name, fill_value=0)
	# Extract values from combined DataFrame
	loan = combined_df.values
	num_scaled = scaler.transform(loan[:, :9])
	loan = np.hstack([num_scaled, loan[:, 9:]])
	prediction = model.predict(loan)
	if prediction == 1:
		return 'Sorry! Loan not approved for you'
	else:
		return 'Loan Approved! Don’t let us regret this'


if __name__ == '__main__':
	st.title('CHECK YOUR LOAN ELIGIBILITY NOW')

	st.divider()
	col1, col2, col3 = st.columns(3)
	with col1:
		income = st.number_input('INCOME', min_value = 0, max_value = 1000000000, value = 0, step = 1)
		age = st.slider('AGE', min_value = 18, max_value = 100, value = 18, step = 1)
		hasmortgage = st.checkbox('Has_Mortgage')
		interest = st.slider('INTEREST RATE %', min_value = 1, max_value = 100, value = 2, step = 1)
		education = st.selectbox('EDUCATION', ("B - Bachelor's", 'P - Phd', "M - Master's", 'H - High School'))
		employmenttype = st.selectbox('EMPLOYMENT TYPE', ('F - Full-time', 'U - Unemployed', 'S - Self-employed', 'P - Part-time'))

	with col2:
		loan_amount = st.number_input('LOAN AMOUNT', min_value = 0, max_value = 1000000, value = 0, step = 1)
		credit_score = st.slider('CREDIT SCORE', min_value = 100, max_value = 1000, value = 100, step = 1)
		hascosigner = st.checkbox('Has_Cosigner')
		loanterm = st.slider('LOAN TERM', min_value = 1, max_value = 100, value = 1, step = 1)
		maritalstatus = st.selectbox('Marital Status', ('M - Married', 'D - Divorced', 'S - Single'))
		
			
	with col3:
		monthemployed = st.number_input('MONTHS EMPLOYED', min_value = 0, max_value = 200, value = 1, step = 1)
		dtiratio = st.slider('DTI RATIO', min_value = 0.000, max_value = 30.0000, value = 0.10000, step = 0.000001)
		hasdependent = st.checkbox('Has_Dependent')
		numofcl = st.slider('NUMBER OF CREDIT LINE', min_value = 0, max_value = 25, value = 0, step = 1)	
		loanpurpose = st.selectbox('Loan Purpose', ('A - Auto', 'B - Business', 'E - Education', 'H - Home', 'O - Other'))

	pred_btn = st.button('APPLY', type = 'primary')
	if pred_btn:
		model = load_model()
		scaler = load_scaler()
		col_name = load_col()

		X_pred = [
            education, employmenttype, maritalstatus, 
            'yes' if hasmortgage else 'no', 'yes' if hasdependent else 'no', 
            loanpurpose, 'yes' if hascosigner else 'no',
            age, income, loan_amount, credit_score, monthemployed, numofcl, interest, loanterm, dtiratio
        ]
		pred = make_prediction(model, scaler, col_name, X_pred)
		st.write(pred)