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
x = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# create pandas dataframe
df = pd.concat([x, y], axis=1)

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
joblib.dump(best_model, "best_logistic_model_1.pkl")


############################################################################################################
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     roc_curve,
#     precision_recall_curve,
# )
# import matplotlib.pyplot as plt

# # Create and train the Logistic Regression model

# log_reg = LogisticRegression(
#     penalty = 'l2',
#     C = 1,
#     random_state=42,
#     max_iter=1000,
#     solver='liblinear',
#     class_weight=None
# )
# log_reg.fit(X_train, y_train)

# # Make predictions
# y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
# y_pred = log_reg.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nModel Accuracy: {accuracy:.4f}")

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print(f"\nROC AUC Score: {roc_auc:.4f}")

# # Plot ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.4f})")
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()

# # Plot Precision-Recall Curve
# precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, label="Logistic Regression")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.show()

############################################################################################################


# Define functions for your app
def load_model():
    """Load the pre-trained model."""
    return joblib.load("best_logistic_model_1.pkl")

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



