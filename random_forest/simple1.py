import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Sample dataset
data = {
    "Disease": ["Flu", "Food Poisoning", "Migraine", "Dengue", "Common Cold"],
    "Symptom_1": ["Fever", "Nausea", "Headache", "Fever", "Sneezing"],
    "Symptom_2": ["Cough", "Vomiting", "Sensitivity", "Rash", "Runny Nose"],
    "Symptom_3": ["Body Ache", "Stomach Pain", "Nausea", "Joint Pain", "Cough"]
}
df = pd.DataFrame(data)

# Encode text to numbers
le_symptom = LabelEncoder()
le_disease = LabelEncoder()

# Fit encoders
symptoms = list(set(df["Symptom_1"]) | set(df["Symptom_2"]) | set(df["Symptom_3"]))
le_symptom.fit(symptoms)
le_disease.fit(df["Disease"])

# Encode columns
df["Symptom_1"] = le_symptom.transform(df["Symptom_1"])
df["Symptom_2"] = le_symptom.transform(df["Symptom_2"])
df["Symptom_3"] = le_symptom.transform(df["Symptom_3"])
df["Disease"] = le_disease.transform(df["Disease"])

# Train model
X = df[["Symptom_1", "Symptom_2", "Symptom_3"]]
y = df["Disease"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Test new input
test_symptoms = ["Cough", "Vomiting", "Sensitivity", "Runny Nose"]
encoded_input = le_symptom.transform(test_symptoms).reshape(1, -1)
prediction = model.predict(encoded_input)
predicted_disease = le_disease.inverse_transform(prediction)

print("Predicted Disease:", predicted_disease[0])
