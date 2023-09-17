from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

# Initializing Flask
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the pickle model
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

@app.route("/")
def home():
    return render_template("home.html")

features = {}

@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        features['Pregnancies'] = request.form['Pregnancies']
        features['PlasmaGlucose'] = request.form['PlasmaGlucose']
        features['DiastolicBloodPressure'] = request.form['DiastolicBloodPressure']
        features['TricepsThickness'] = request.form['TricepsThickness']
        features['Serumlnsulin'] = request.form['Serumlnsulin']
        features['BMI'] = request.form['BMI']
        features['DiabetesPedigree'] = request.form['DiabetesPedigree']
        features['Age'] = request.form['Age']
        
        data_dict = {key: value for key, value in features.items()}
        
        p_df = pd.DataFrame(data_dict,index=[0],columns=['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure','TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age'])

        p_df = p_df.apply(pd.to_numeric, errors='coerce', downcast='integer')        
        pred_v = 'Yes' if int(loaded_model.predict(p_df)[0]) == 1 else 'No'
        return render_template("output.html",pred=pred_v)
    else:
        return "Method not allowed"


if __name__ == "__main__":
    app.run(port=5001)
