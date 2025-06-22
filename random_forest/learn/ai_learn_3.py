import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# File paths
MODEL_FILE = "rf_model.joblib"
ENCODERS_FILE = "feature_encoders.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"

# Sample dataset
data = {
    "Disease": ["Flu", "Malaria", "Diabetes", "Asthma", "Migraine"],
    "Symptom1": ["Fever", "Fever", "Fatigue", "Shortness of breath", "Headache"],
    "Symptom2": ["Cough", "Chills", "Increased thirst", "Coughing", "Nausea"],
    "Symptom3": ["Body ache", "Sweating", "Frequent urination", "Wheezing", "Sensitivity to light"],
    "Cause1": ["Virus", "Parasite", "Insulin resistance", "Allergens", "Unknown"],
    "Cause2": ["Cold weather", "Mosquito bite", "Genetics", "Air pollution", "Stress"]
}

df = pd.DataFrame(data)

features = ["Symptom1", "Symptom2", "Symptom3", "Cause1", "Cause2"]
target = "Disease"

# Load or create encoders and model
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODERS_FILE) and os.path.exists(LABEL_ENCODER_FILE):
    rf_model = joblib.load(MODEL_FILE)
    feature_encoders = joblib.load(ENCODERS_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    print("loaded model")
else:
    # Encode features
    feature_encoders = {}
    for col in features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        feature_encoders[col] = le

    # Encode target
    label_encoder = LabelEncoder()
    df[target] = label_encoder.fit_transform(df[target])

    # Train model
    X_train = df[features]
    y_train = df[target]
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Save model and encoders
    joblib.dump(rf_model, MODEL_FILE)
    joblib.dump(feature_encoders, ENCODERS_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    print("saved model....")

# --- Prediction ---

input_data = {
    "Symptom1": "Fever",
    "Symptom2": "Cough",
    "Symptom3": "Body ache",
    "Cause1": "Virus",
    "Cause2": "Cold weather"
}

# Encode input
encoded_input = {}
for feature in features:
    le = feature_encoders[feature]
    encoded_input[feature] = le.transform([input_data[feature]])[0]

# Predict
input_df = pd.DataFrame([encoded_input])
probabilities = rf_model.predict_proba(input_df)[0]
class_names = label_encoder.inverse_transform(range(len(probabilities)))

# Display result
for disease, prob in zip(class_names, probabilities):
    print(f"{disease} = {prob:.2f}")

print("*******************************")


# Get class probabilities and names
# probabilities = rf_model.predict_proba(input_df)[0]
# class_names = label_encoder.inverse_transform(range(len(probabilities)))
# # Combine and sort by probability (descending)
# sorted_probs = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
# # Display
# for disease, prob in sorted_probs:
#     print(f"{disease} = {prob:.2f}")
