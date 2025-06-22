🏥 Disease Prediction System (Random Forest + XGBoost)
📚 Overview
This repository contains a complete Machine Learning-based system that predicts diseases based on:

Patient Symptoms

Possible Causes

Age of Onset

Family History

Genetic Factors

It uses two powerful ensemble models:

XGBoost Classifier (Primary)

Random Forest Classifier (Alternative)

The system can train, predict, evaluate, handle imbalanced datasets using SMOTE, and save/load models easily.

⚙️ Technologies Used
Python 3.8+

scikit-learn

XGBoost

imbalanced-learn (SMOTE)

pandas

NumPy

joblib

matplotlib

seaborn

📥 Input Format
Input to the model must be a dictionary in patient feature format:

python
Copy
Edit
patient = {
    'symptoms': ['fever', 'cough', 'fatigue', 'body ache'],  # List of symptoms
    'causes': ['viral'],                                     # List of causes
    'age_of_onset': 45,                                      # Numerical
    'family_history': 0,                                     # 0 = No, 1 = Yes
    'has_genetic_factors': 0                                 # 0 = No, 1 = Yes
}
Or in processed feature format:

python
Copy
Edit
{
    'has_fever': 1,
    'has_cough': 1,
    'has_fatigue': 1,
    'has_body ache': 1,
    'cause_viral': 1,
    'age_of_onset': 45,
    'family_history': 0,
    'has_genetic_factors': 0
}
✅ The symptoms and causes fields are position-independent.

📤 Output Format
The system returns:

python
Copy
Edit
{
  'predicted_disease': 'Influenza',
  'confidence': 0.82,
  'differential_diagnosis': {
    'Common Cold': 0.10,
    'COVID-19': 0.06,
    'Pneumonia': 0.02
  }
}
Predicted Disease: Most probable diagnosis.

Confidence: Model's certainty.

Differential Diagnosis: Other likely diseases.

🛠️ How It Works
1. Data Preprocessing
Converts symptom and cause columns into binary features.

Converts categorical fields like family history and genetic factors into binary.

Fills missing age values with median age.

Encodes disease labels numerically.

2. Handling Imbalance
Optionally uses SMOTE to oversample minority classes.

3. Model Training
Random Forest and XGBoost with GridSearchCV for hyperparameter tuning.

5-fold Cross-Validation.

Model Evaluation with:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

4. Feature Importance
Generates top 20 important features plot.

Saves the plot as feature_importance.png.

5. Prediction
Patient data transformed into feature vector.

Returns top predictions along with probabilities.

6. Model Saving and Loading
Save trained models using joblib.

Load them anytime without retraining.

🚀 Usage Instructions
Step 1: Install Required Libraries
bash
Copy
Edit
pip install scikit-learn xgboost pandas numpy imbalanced-learn joblib matplotlib seaborn
Step 2: Train the Model
bash
Copy
Edit
python disease_prediction_system.py
✅ If your dataset diseases_dataset.csv is present, it will use that. Otherwise, it generates a sample dataset.

Step 3: Predict Disease for New Patients
python
Copy
Edit
from disease_prediction_system import DiseasePredictionSystem

# Load saved model
model = DiseasePredictionSystem.load_model('xgboost_disease_model.pkl')

# Prepare new patient data
patient = {
    'symptoms': ['fever', 'cough', 'fatigue', 'body ache'],
    'causes': ['viral'],
    'age_of_onset': 45,
    'family_history': 0,
    'has_genetic_factors': 0
}

# Predict
patient_features = {}
for symptom in patient['symptoms']:
    patient_features[f'has_{symptom}'] = 1
for cause in patient['causes']:
    patient_features[f'cause_{cause}'] = 1
patient_features['age_of_onset'] = patient['age_of_onset']
patient_features['family_history'] = patient['family_history']
patient_features['has_genetic_factors'] = patient['has_genetic_factors']

result = model.predict(patient_features)

print(result)
📈 Evaluation Metrics
Accuracy Score

Precision, Recall, F1 Score

Confusion Matrix

Feature Importance Plot

📁 Project Structure
bash
Copy
Edit
disease_prediction_system/
│
├── diseases_dataset.csv           # (your full dataset)
├── xgboost_disease_model.pkl       # (saved XGBoost model)
├── rf_disease_model.pkl            # (saved Random Forest model)
├── feature_importance.png          # (feature importance chart)
├── disease_prediction_system.py    # (main training and prediction code)
├── README.md                       # (this documentation)
└── requirements.txt                # (library list - optional)
📋 Features
✔️ Train models easily

✔️ Predict instantly

✔️ Save/load models

✔️ Handles missing data

✔️ Handles class imbalance (SMOTE)

✔️ Feature importance visualization

✔️ High-accuracy XGBoost prediction

✔️ Differential diagnosis like real clinical systems

