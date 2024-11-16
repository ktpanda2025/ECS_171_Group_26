from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# Save the model for reuse
joblib.dump(logistic_model, "logistic_model.pkl")

# Define functions for your app
def load_model():
    """Load the pre-trained model."""
    return joblib.load("logistic_model.pkl")

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
    test_input = "1,1,1,25,0,0,1,1,0,1,0,1,0,3,0,0,0,1,8,4,6"  # Example input
    processed_input = preprocess_input(test_input)
    prediction, probabilities = predict(processed_input)

    print(f"Predicted Class: {prediction}")
    print(f"Probability of Class 0: {probabilities[0]}")
    print(f"Probability of Class 1: {probabilities[1]}")
