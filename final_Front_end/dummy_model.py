from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# create pandas dataframe
df = pd.concat([X, y], axis=1)

# Define features (X) and target (y)
selected_features = [
    'HighBP', 'GenHlth', 'DiffWalk', 'BMI', 'HighChol', 'Age',
    'PhysHlth', 'HeartDiseaseorAttack', 'NoDocbcCost', 'MentHlth'
]
X = df[selected_features]
y = df['Diabetes_binary']

oversample = SMOTE()
X, y = oversample.fit_resample(X,y)

# define models and parameters
model = LogisticRegression()
solvers = ['liblinear']
penalty = ['l2']
c_values = [100, 10, 1, 0.1, 0.01] #, 0.01, 0.001]
class_weights = ['balanced', None] # , {0:1, 1:1} , {0: 1, 1: 3}  , {0: 1, 1: 6}] #, {0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}]
# class_weights = ['balanced']

# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values, class_weight=class_weights)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1_micro', error_score=0)
grid_result = grid_search.fit(X, y)

# Fit the best model
best_model = grid_result.best_estimator_
best_model.fit(X, y)

# Save the best model
joblib.dump(best_model, "best_logistic_model.pkl")

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
    test_input = "1,4,1,18,1,11,0,0,0,0" # Example input
    processed_input = preprocess_input(test_input)
    prediction, probabilities = predict(processed_input)

    print(f"Predicted Class: {prediction}")
    print(f"Probability of Class 0: {probabilities[0]}")
    print(f"Probability of Class 1: {probabilities[1]}")



