from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

# Initializing Flask
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the pickle model
with open("model_file.pkl", "rb") as file:
    loaded_model = pickle.load(file)


@app.route("/")
def home():
    return render_template("home.html")

features = {}

@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        features['encoded_OverTime'] = request.form['encoded_OverTime']
        features['Age'] = request.form['age']
        features['JobInvolvement'] = request.form['job_involvement']
        features['JobLevel'] = request.form['job_level']
        features['MonthlyIncome'] = request.form['monthly_income']
        features['TotalWorkingYears'] = request.form['total_working_years']
        features['YearsAtCompany'] = request.form['years_at_company']
        features['YearsInCurrentRole'] = request.form['years_in_currentRole']
        features['encoded_MaritalStatus'] = request.form['marital_status']
        features['YearsWithCurrManager'] = request.form['years_with_curr_manager']
        p_df = pd.DataFrame(features,index=[0])
        pred_v = 'Yes' if int(loaded_model.predict(p_df)[0]) == 1 else 'No'
        return render_template("output.html",pred=pred_v)
    else:
        return "Method not allowed"


if __name__ == "__main__":
    app.run(debug=True,port=5001)
