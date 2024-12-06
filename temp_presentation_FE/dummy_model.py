from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import numpy as np

# Fetch dataset from UCI ML Repository
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# Extract features and targets
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# Combine features and targets into a DataFrame
df = pd.concat([X, y], axis=1)

# Selected features for the model
selected_features = [
    'HighBP', 'GenHlth', 'DiffWalk', 'BMI', 'HighChol', 'Age',
    'PhysHlth', 'HeartDiseaseorAttack', 'NoDocbcCost', 'MentHlth']

# Define features and target variable
X = df[selected_features]
y = df['Diabetes_binary']


best_model = LogisticRegression(C=0.001, class_weight='balanced', penalty='l2', solver='liblinear')


best_model.fit(X, y)

joblib.dump(best_model, "best_logistic_model.pkl")


def load_model():
    """Load the pre-trained model."""
    return joblib.load("best_logistic_model.pkl")

def preprocess_input(input_data):
    """
    Preprocess user input into the format the model expects.
    Convert the input into a numpy array.
    """
    input_list = [float(x) for x in input_data.split(",")]
    return np.array(input_list).reshape(1, -1)  # Reshape to make it a 2D array for the model

def predict(input_features):
    """
    Predict using the pre-trained model.
    """
    model = load_model()
    probabilities = model.predict_proba(input_features)[0]
    predicted_class = model.predict(input_features)[0]
    return predicted_class, probabilities

if __name__ == "__main__":
    # Test the model predictions with a sample input
    test_input = "1,4,1,18,1,11,0,0,0,0"  # Example input with selected features
    processed_input = preprocess_input(test_input)
    prediction, probabilities = predict(processed_input)

    print(f"Predicted Class: {prediction}")
    print(f"Probability of Class 0: {probabilities[0]}")
    print(f"Probability of Class 1: {probabilities[1]}")
