# Combines all symptoms and causes into a free-text string

# Uses TF-IDF vectorization

# Uses Logistic Regression (simple, fast, well-performing on small datasets)

# Is modular and clean, ready to scale or convert to OOP


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# -----------------------------
# 1. Data Preparation
# -----------------------------

data = {
    "Disease": ["Flu", "Malaria", "Diabetes", "Asthma", "Migraine"],
    "Symptom1": ["Fever", "Fever", "Fatigue", "Shortness of breath", "Headache"],
    "Symptom2": ["Cough", "Chills", "Increased thirst", "Coughing", "Nausea"],
    "Symptom3": ["Body ache", "Sweating", "Frequent urination", "Wheezing", "Sensitivity to light"],
    "Cause1": ["Virus", "Parasite", "Insulin resistance", "Allergens", "Unknown"],
    "Cause2": ["Cold weather", "Mosquito bite", "Genetics", "Air pollution", "Stress"]
}

df = pd.DataFrame(data)

# Combine symptoms and causes into one text column
def combine_text(row):
    fields = ['Symptom1', 'Symptom2', 'Symptom3', 'Cause1', 'Cause2']
    return ' '.join(str(row[field]) for field in fields if pd.notnull(row[field]))

df['text'] = df.apply(combine_text, axis=1)

# -----------------------------
# 2. Train/Test Split
# -----------------------------

X = df['text']
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. TF-IDF + Classifier Pipeline
# -----------------------------

model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

# -----------------------------
# 4. Train the Model
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate the Model
# -----------------------------

y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 6. Save the Model
# -----------------------------

joblib.dump(model, 'disease_predictor_model.joblib')

# -----------------------------
# 7. Predict on New Input
# -----------------------------

def predict_disease(symptoms_and_causes):
    text_input = ' '.join(symptoms_and_causes)
    prediction = model.predict([text_input])[0]
    probabilities = model.predict_proba([text_input])[0]
    labels = model.classes_
    result = {label: round(prob, 2) for label, prob in zip(labels, probabilities)}
    print("Prediction:", prediction)
    print("Probabilities:", result)
    return prediction, result

# Example usage
predict_disease(["Fever", "Cough", "Body ache", "Virus", "Cold weather"])
