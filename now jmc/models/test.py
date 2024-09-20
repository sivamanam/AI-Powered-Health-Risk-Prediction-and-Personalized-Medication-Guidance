import joblib
import numpy as np

# Load the saved models using joblib
model1 = joblib.load('naive_bayes_model1.pkl')  # For predicting Health Risk
model2 = joblib.load('naive_bayes_model2.pkl')  # For predicting Health Suggestions

# Function to convert categorical inputs to numeric values
def preprocess_input(gender, genetic_marker):
    # Encode Gender (Male -> 1, Female -> 0)
    gender_encoded = 1 if gender.lower() == 'male' else 0

    # Encode Genetic Marker (Yes -> 1, No -> 0)
    genetic_marker_encoded = 1 if genetic_marker.lower() == 'yes' else 0

    return gender_encoded, genetic_marker_encoded

# Function to take inputs and predict
def predict_health_risk_and_suggestions():
    # Input values
    s_no = input("Enter S.NO (for reference): ")
    gender = input("Enter Gender (Male/Female): ").strip().capitalize()
    age = float(input("Enter Age: "))
    bmi = float(input("Enter BMI: "))
    cholesterol = float(input("Enter Cholesterol Level (mg/dL): "))
    blood_pressure = float(input("Enter Average Blood Pressure (mmHg): "))  # Single input for average BP
    blood_sugar = float(input("Enter Blood Sugar Level (mg/dL): "))
    genetic_marker = input("Genetic Marker for Diabetes (Yes/No): ").strip().capitalize()

    # Preprocess categorical inputs
    gender_encoded, genetic_marker_encoded = preprocess_input(gender, genetic_marker)

    # Prepare the input array with all numerical values
    input_data = np.array([[s_no, gender_encoded, age, bmi, cholesterol, blood_pressure, blood_sugar, genetic_marker_encoded]])

    # Ensure input data is correctly formatted as float32 or float64 (numeric) before passing to the model
    input_data = input_data.astype(float)

    # Predict using model1 (Health Risk)
    health_risk = model1.predict(input_data)

    # Predict using model2 (Health Suggestions)
    suggestions = model2.predict(input_data)

    # Handle both string and integer outputs from the model
    if isinstance(health_risk[0], str):
        print(f"S.NO: {s_no}")
        print(f"Health Risk Prediction: {health_risk[0]}")  # Directly use the string prediction
    else:
        risk_mapping = {0: 'No Risk', 1: 'At Risk', 2: 'Multiple Risks'}
        print(f"S.NO: {s_no}")
        print(f"Health Risk Prediction: {risk_mapping.get(health_risk[0], 'Unknown Risk')}")  # Fallback if not in mapping

    print(f"Health Suggestions: {suggestions[0]}")  # Suggestions based on the risk

# Call the function to run predictions
predict_health_risk_and_suggestions()
