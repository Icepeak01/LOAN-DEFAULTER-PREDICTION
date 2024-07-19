from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd


def return_prediction(model, scaler, col_name, sample_json):
    age = sample_json["Age"]
    income = sample_json["Income"]
    l_amount = sample_json["LoanAmount"]
    cd_score = sample_json["CreditScore"]
    month_emp= sample_json["MonthsEmployed"]
    num_cl = sample_json["NumCreditLines"]
    int_rate = sample_json["InterestRate"]
    loan_term = sample_json["LoanTerm"]
    dti_ratio = sample_json["DTIRatio"]
    edu = sample_json["Education"]
    emp_type = sample_json["EmploymentType"]
    marital_s = sample_json["MaritalStatus"]
    has_mort = sample_json["HasMortgage"]
    has_depd = sample_json["HasDependents"]
    loan_purp = sample_json["LoanPurpose"]
    has_cosg = sample_json["HasCoSigner"]


    cat_df = pd.DataFrame([[edu, emp_type, marital_s, has_mort, has_depd, loan_purp, has_cosg]],
                          columns=['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'])

    # One-hot encode categorical features
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)


    num_df = pd.DataFrame([[age, income, l_amount, cd_score, month_emp, num_cl, int_rate, loan_term, dti_ratio]],
                          columns=['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                                   'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'])
    # Combine numerical and categorical data
    combined_df = pd.concat([num_df, cat_encoded], axis=1)
     # Reindex combined_df to match the training columns, filling missing columns with 0
    combined_df = combined_df.reindex(columns = col_name, fill_value=0)

    # Extract values from combined DataFrame
    loan = combined_df.values

    # Scale numerical features (assumes scaler only expects the numerical part)
    num_scaled = scaler.transform(loan[:, :9])

    # Combine scaled numerical features and categorical features
    loan = np.hstack([num_scaled, loan[:, 9:]])

    prediction = model.predict(loan)

    if prediction == 1:
        return 'Thief! Loan not approved for you'
    else:
        return 'Loan Approved! Don’t let us regret this'




loan = Flask(__name__)

@loan.route("/")
def index():
    return '<h1>FLASK IS RUNNING</h1>'


lr_model = joblib.load('final_lr_model.pkl')
lr_scaler = joblib.load("lr_scaler.pkl")
col_name  = joblib.load("col_name.pkl")


@loan.route('/loan_predict', methods = ['POST'])
def loan_prediction():
	content = request.json
	results = return_prediction(lr_model, lr_scaler, content)
	return jsonify(results)



if __name__=='__main__':
	loan.run()