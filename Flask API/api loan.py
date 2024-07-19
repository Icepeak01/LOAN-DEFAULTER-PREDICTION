from flask import Flask, render_template, session, url_for, redirect
from wtforms  import StringField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
import pandas as pd
from flask_wtf import FlaskForm
import joblib


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
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    num_df = pd.DataFrame([[age, income, l_amount, cd_score, month_emp, num_cl, int_rate, loan_term, dti_ratio]],
                          columns=['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                                   'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'])
    combined_df = pd.concat([num_df, cat_encoded], axis=1)
    combined_df = combined_df.reindex(columns = col_name, fill_value=0)

    loan = combined_df.values
    num_scaled = scaler.transform(loan[:, :9])
    loan = np.hstack([num_scaled, loan[:, 9:]])

    prediction = model.predict(loan)

    if prediction == 1:
        return 'Declined, kindly retry some other time'
    else:
        return 'Approved! Don’t let us regret this'
#----------------------------------------------------------------------------------------------------------------------------


loan = Flask(__name__)
loan.config["SECRET_KEY"] = 'my secretkey'

class FlowerForm(FlaskForm):
    ag = StringField("AGE", validators=[DataRequired()])
    inc = StringField("INCOME", validators=[DataRequired()])
    l_amt = StringField("LOAN AMOUNT", validators=[DataRequired()])
    c_sco = StringField("CREDIT SCORE", validators=[DataRequired()])
    m_emp = StringField("MONTHS EMPLOYED", validators=[DataRequired()])
    n_cl = StringField("NUMBER OF CREDIT LINES", validators=[DataRequired()])
    i_rate = StringField("INTEREST RATE", validators=[DataRequired()])
    l_term = StringField("LOAN TERM", validators=[DataRequired()])
    dti = StringField("DTI RATIO", validators=[DataRequired()])
    ed = StringField("EDUCATION", validators=[DataRequired()])
    em_typ = StringField("EMPLOYMENT TYPE", validators=[DataRequired()])
    mar_st = StringField("MARITAL STATUS", validators=[DataRequired()])
    mortg = StringField("HAS MORTGAGE?", validators=[DataRequired()])
    deped = StringField("HAS DEPENDENTS?", validators=[DataRequired()])
    l_purp = StringField("LOAN PURPOSE", validators=[DataRequired()])
    cosign = StringField("HAS COSIGNER?", validators=[DataRequired()])

    submit = SubmitField("Apply")


#------------------------------------------------------------------------------------------------------------------------

@loan.route("/", methods = ['GET', 'POST'])
def index():
    
    form = FlowerForm()

    if form.validate_on_submit():
        session['ag'] = form.ag.data
        session['inc'] = form.inc.data
        session['l_amt'] = form.l_amt.data
        session['c_sco'] = form.c_sco.data
        session['m_emp'] = form.m_emp.data
        session['n_cl'] = form.n_cl.data
        session['i_rate'] = form.i_rate.data
        session['l_term'] = form.l_term.data
        session['dti'] = form.dti.data
        session['ed'] = form.ed.data
        session['em_typ'] = form.em_typ.data
        session['mar_st'] = form.mar_st.data
        session['mortg'] = form.mortg.data
        session['deped'] = form.deped.data
        session['l_purp'] = form.l_purp.data
        session['cosign'] = form.cosign.data

        return redirect(url_for('loan_prediction'))
    return render_template('home.html', form = form)
#---------------------------------------------------------------------------------------------------------------------------

lr_model = joblib.load('final_lr_model.pkl')
lr_scaler = joblib.load("lr_scaler.pkl")
col_name  = joblib.load("col_name.pkl")


@loan.route('/loan_prediction')
def loan_prediction():
    content = {}
    content['Age'] = float(session['ag'])
    content['Income'] = float(session['inc'])
    content['LoanAmount'] = float(session['l_amt'])
    content['CreditScore'] = float(session['c_sco'])
    content['MonthsEmployed'] = float(session['m_emp'])
    content['NumCreditLines'] = float(session['n_cl'])
    content['InterestRate'] = float(session['i_rate'])
    content['LoanTerm'] = float(session['l_term'])
    content['DTIRatio'] = float(session['dti'])
    content['Education'] = session['ed']
    content['EmploymentType'] = session['em_typ']
    content['MaritalStatus'] = session['mar_st']
    content['HasMortgage'] = session['mortg']
    content['HasDependents'] = session['deped']
    content['LoanPurpose'] = session['l_purp']
    content['HasCoSigner'] = session['cosign']

    results = return_prediction(lr_model, lr_scaler, col_name, content)
    return render_template('loan_prediction.html', results = results)



if __name__=='__main__':
	loan.run()