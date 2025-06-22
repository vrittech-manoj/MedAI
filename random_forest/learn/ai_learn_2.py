# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# # Sample data
# data = {
#     "Disease": ["Flu", "Malaria", "Diabetes", "Asthma", "Migraine"],
#     "Symptom1": ["Fever", "Fever", "Fatigue", "Shortness of breath", "Headache"],
#     "Symptom2": ["Cough", "Chills", "Increased thirst", "Coughing", "Nausea"],
#     "Symptom3": ["Body ache", "Sweating", "Frequent urination", "Wheezing", "Sensitivity to light"],
#     "Cause1": ["Virus", "Parasite", "Insulin resistance", "Allergens", "Unknown"],
#     "Cause2": ["Cold weather", "Mosquito bite", "Genetics", "Air pollution", "Stress"]
# }

# df = pd.DataFrame(data)

# # Define features and target
# features = ["Symptom1", "Symptom2", "Symptom3", "Cause1", "Cause2"]
# target = "Disease"

# # Encode features
# feature_encoders = {}
# for col in features:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     feature_encoders[col] = le

# # Encode target
# label_encoder = LabelEncoder()
# df[target] = label_encoder.fit_transform(df[target])

# # Split into training and testing sets
# X = df[features]
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Random Forest model
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)

# # --- Prediction Section ---

# # Input data for prediction
# input_data = {
#     "Symptom1": "Fever",
#     "Symptom2": "Chills",
#     "Symptom3": "Sweating",
#     "Cause1": "Parasite",
#     "Cause2": "Mosquito bite"
# }

# # Encode input features
# encoded_input = {}
# for feature in features:
#     le = feature_encoders[feature]
#     encoded_input[feature] = le.transform([input_data[feature]])[0]

# # Convert to DataFrame to avoid warning
# input_df = pd.DataFrame([encoded_input])

# # Predict disease
# predicted_label_encoded = rf_model.predict(input_df)
# predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

# print("Predicted Disease:", predicted_label[0])


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

# Features and target
features = ["Symptom1", "Symptom2", "Symptom3", "Cause1", "Cause2"]
target = "Disease"

# Encode features
feature_encoders = {}
for col in features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    feature_encoders[col] = le

# Encode target
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# Use all data for training
X_train = df[features]
y_train = df[target]

# Train model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# --- Prediction ---

# Input data for prediction
input_data = {
    "Symptom1": "Fever",
    "Symptom2": "Chills",
    "Symptom3": "Sweating",
    "Cause1": "Parasite",
    "Cause2": "Mosquito bite"
}
# Encode input
encoded_input = {}
for feature in features:
    le = feature_encoders[feature]
    encoded_input[feature] = le.transform([input_data[feature]])[0]

# Predict using DataFrame to avoid warnings
input_df = pd.DataFrame([encoded_input])
predicted_label_encoded = rf_model.predict(input_df)
predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

print("Predicted Disease:", predicted_label[0])
