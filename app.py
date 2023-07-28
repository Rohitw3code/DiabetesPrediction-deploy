from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

# Initializing Flask
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the pickle model
with open("model_file.pkl", "rb") as file:
    loaded_model = pickle.load(file)


file_path = 'label_encoder.pkl'
# Load the LabelEncoder from the saved file
with open(file_path, 'rb') as file:
    loaded_label_encoder = pickle.load(file)


file_path = 'min_max_scaler.pkl'
# Load the MinMaxScaler from the saved file
with open(file_path, 'rb') as file:
    loaded_scaler = pickle.load(file)



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

        # LabelEncoder
        columns = ['encoded_Attrition', 'encoded_BusinessTravel', 'encoded_Department',
                'encoded_EducationField', 'encoded_Gender', 'encoded_JobRole', 'encoded_MaritalStatus',
                  'encoded_OverTime']

        columns = ['encoded_MaritalStatus','encoded_OverTime']
        
        for col in columns:
            p_df[col] = loaded_label_encoder.fit_transform(p_df[col])
        
        scaled_data = loaded_scaler.fit_transform(p_df)
        new_df = pd.DataFrame(scaled_data, columns=p_df.columns)
        pred_v = 'Yes' if int(loaded_model.predict(new_df)[0]) == 1 else 'No'
        return render_template("output.html",pred=pred_v)
    else:
        return "Method not allowed"


if __name__ == "__main__":
    app.run(debug=True,port=5001)
