from flask import Flask, render_template, request
import numpy as np
import dummy_model  # Import your ML model code

app = Flask(__name__)

# Define metadata for features

"""
    logistic freatures are :  
   'HighBP', 'GenHlth', 'DiffWalk', 'BMI', 'HighChol', 'Age',
    'PhysHlth', 'HeartDiseaseorAttack', 'NoDocbcCost', 'MentHlth'

"""

metadata = [
    {"name": "HighBP", "description": "0 = no high BP, 1 = high BP"},
    {"name": "HighChol", "description": "0 = High cholesterol, 1 = yes cholesterol check in 5 years"},
    {"name": "Age", "description": "Age category: 1 = 18-24, 9 = 60-64, 13 = 80 or older"},
    {"name": "GenHlth", "description": "General health: scale 1-5 (1 = excellent, 5 = poor)"},
    {"name": "DiffWalk", "description": "Difficulty walking or climbing stairs 0 = no, 1 = yes"},
    {"name": "HeartDiseaseorAttack", "description": "Coronary heart disease (CHD) or myocardial infarction (MI) 0 = no, 1 = yes"},
    {"name": "MentHlth", "description": "Number of days mental health not good in past 30 days (scale 1-30 days)"},
    {"name": "NoDocbcCost", "description": "Could not see a doctor because of cost in the past 12 months 0 = no, 1 = yes"},
    {"name": "PhysHlth", "description": "Number of days physical health not good in past 30 days (scale 1-30 days)"},
    {"name": "BMI", "description": "Body Mass Index (numeric) BMI levels include underweight (<18.5), normal weight (18.5-24.9), overweight (25-29.9), and obese (≥30)"}
]

@app.route("/")
def index():
    return render_template("index.html", metadata=metadata)

@app.route("/predict", methods=["POST"])
def predict():
    # Collect input values from the form
    input_features = [float(request.form.get(feature["name"])) for feature in metadata]
    input_features = np.array(input_features).reshape(1, -1) 

    # Predict using the model
    predicted_class, probabilities = dummy_model.predict(input_features)

    diab = ""

    if predicted_class == 0:
        diab = "Low Risk"
    else:
        diab = "High Risk"


    return render_template(
        "results.html",
        prediction=diab,
        prob_class_0=str(np.round(probabilities[0],2) * 100) + '%',
        prob_class_1=str(np.round(probabilities[1],2)*100) + '%' ,
    )

if __name__ == "__main__":
    app.run(debug=True)
