import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the DiseasePredictionSystem class (assuming it's in the same file or imported)
# from your_script import DiseasePredictionSystem, DiseasePredictionApp

class DiseasePredictionDemo:
    """Demo class to show how to use the trained model"""
    
    def __init__(self, model_path='xgboost_disease_model.pkl'):
        """Load the trained model"""
        try:
            print("Loading trained model...")
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.target_encoder = model_data['target_encoder']
            self.symptom_vectorizer = model_data.get('symptom_vectorizer', None)
            
            print(f"Model loaded successfully!")
            print(f"Number of features: {len(self.feature_names)}")
            print(f"Number of diseases: {len(self.target_encoder.classes_)}")
            print(f"Diseases the model can predict: {list(self.target_encoder.classes_)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict_disease(self, patient_data):
        """Make a prediction for a patient"""
        if not self.model:
            print("No model loaded!")
            return None
        
        try:
            # Convert patient data to feature vector
            features = np.zeros(len(self.feature_names))
            
            # Fill in features based on patient data
            for i, feature in enumerate(self.feature_names):
                if feature in patient_data:
                    features[i] = patient_data[feature]
                elif feature.startswith('has_') and feature[4:] in patient_data.get('symptoms', []):
                    features[i] = 1
                elif feature.startswith('cause_') and feature[6:] in patient_data.get('causes', []):
                    features[i] = 1
                elif feature.startswith('text_feature_') and self.symptom_vectorizer:
                    # Handle text features if using text-based approach
                    continue  # These would be handled by vectorizer
            
            # Make prediction
            features_reshaped = features.reshape(1, -1)
            pred_proba = self.model.predict_proba(features_reshaped)[0]
            pred_class = self.model.predict(features_reshaped)[0]
            
            # Convert to disease names
            predicted_disease = self.target_encoder.inverse_transform([pred_class])[0]
            confidence = pred_proba[pred_class]
            
            # Get top predictions
            top_k = min(5, len(pred_proba))
            top_indices = np.argsort(pred_proba)[::-1][:top_k]
            top_diseases = self.target_encoder.inverse_transform(top_indices)
            top_probabilities = pred_proba[top_indices]
            
            result = {
                'predicted_disease': predicted_disease,
                'confidence': float(confidence),
                'top_predictions': [(disease, float(prob)) for disease, prob in zip(top_diseases, top_probabilities)]
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def show_model_info(self):
        """Display information about the loaded model"""
        if not self.model:
            print("No model loaded!")
            return
        
        print("\n=== MODEL INFORMATION ===")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Feature names (first 10): {self.feature_names[:10]}")
        print(f"Number of disease classes: {len(self.target_encoder.classes_)}")
        print(f"Disease classes: {list(self.target_encoder.classes_)}")
        
        if hasattr(self.model, 'feature_importances_'):
            # Show top important features
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print(f"\nTop 10 most important features:")
            for i in range(min(10, len(indices))):
                feature_idx = indices[i]
                print(f"  {i+1}. {self.feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
    
    def interactive_diagnosis(self):
        """Interactive diagnosis session"""
        if not self.model:
            print("No model loaded!")
            return
        
        print("\n=== INTERACTIVE DIAGNOSIS ===")
        print("Available diseases this model can predict:")
        for i, disease in enumerate(self.target_encoder.classes_, 1):
            print(f"  {i}. {disease}")
        
        print(f"\nAvailable features for input:")
        feature_types = {}
        for feature in self.feature_names:
            if feature.startswith('has_'):
                symptom = feature[4:]
                if 'symptoms' not in feature_types:
                    feature_types['symptoms'] = []
                feature_types['symptoms'].append(symptom)
            elif feature.startswith('cause_'):
                cause = feature[6:]
                if 'causes' not in feature_types:
                    feature_types['causes'] = []
                feature_types['causes'].append(cause)
            elif '_numeric' in feature:
                if 'numeric' not in feature_types:
                    feature_types['numeric'] = []
                feature_types['numeric'].append(feature)
            elif '_binary' in feature:
                if 'binary' not in feature_types:
                    feature_types['binary'] = []
                feature_types['binary'].append(feature)
        
        for category, items in feature_types.items():
            print(f"\n{category.upper()}:")
            for item in items[:10]:  # Show first 10
                print(f"  - {item}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")
        
        # Get user input
        patient_data = {}
        
        # Get symptoms
        if 'symptoms' in feature_types:
            print(f"\nEnter symptoms (from the list above, one per line, blank to finish):")
            symptoms = []
            while True:
                symptom = input("Symptom: ").strip().lower()
                if not symptom:
                    break
                if symptom in feature_types['symptoms']:
                    symptoms.append(symptom)
                    patient_data[f'has_{symptom}'] = 1
                else:
                    print(f"'{symptom}' not in available symptoms. Available: {feature_types['symptoms'][:5]}...")
            
        # Get numeric values
        if 'numeric' in feature_types:
            for feature in feature_types['numeric'][:3]:  # Ask for first 3 numeric features
                try:
                    value = input(f"Enter {feature} (or press Enter to skip): ").strip()
                    if value:
                        patient_data[feature] = float(value)
                except:
                    pass
        
        # Make prediction
        if patient_data:
            print(f"\nPatient data: {patient_data}")
            result = self.predict_disease(patient_data)
            
            if result:
                print(f"\n=== DIAGNOSIS RESULTS ===")
                print(f"Primary diagnosis: {result['predicted_disease']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"\nTop predictions:")
                for i, (disease, prob) in enumerate(result['top_predictions'], 1):
                    print(f"  {i}. {disease}: {prob:.2%}")
            else:
                print("Could not make a prediction.")
        else:
            print("No patient data provided.")
    
    def batch_prediction_example(self):
        """Example of batch predictions"""
        print("\n=== BATCH PREDICTION EXAMPLE ===")
        
        # Create some example patients based on available features
        example_patients = []
        
        # Patient 1: Basic example
        patient1 = {}
        if len(self.feature_names) > 0:
            # Set first feature to 1
            if self.feature_names[0].startswith('has_'):
                symptom = self.feature_names[0][4:]
                patient1 = {'symptoms': [symptom]}
                print(f"Patient 1 - Symptom: {symptom}")
        
        # Patient 2: Another example
        patient2 = {}
        if len(self.feature_names) > 1:
            if self.feature_names[1].startswith('has_'):
                symptom = self.feature_names[1][4:]
                patient2 = {'symptoms': [symptom]}
                print(f"Patient 2 - Symptom: {symptom}")
        
        # Make predictions
        for i, patient in enumerate([patient1, patient2], 1):
            if patient:
                result = self.predict_disease(patient)
                if result:
                    print(f"\nPatient {i} diagnosis:")
                    print(f"  Primary: {result['predicted_disease']} ({result['confidence']:.2%})")
                    print(f"  Alternatives: {[f'{d}: {p:.2%}' for d, p in result['top_predictions'][:3]]}")
    
    def evaluate_model_performance(self):
        """Show model performance metrics"""
        print("\n=== MODEL PERFORMANCE ===")
        print("From training output:")
        print("- Test Accuracy: 72.22%")
        print("- Macro Average F1-Score: 0.61")
        print("- Weighted Average F1-Score: 0.66")
        print("\nNote: Model performance varies by disease class due to class imbalance.")
        print("Check 'confusion_matrix.png' and 'feature_importance.png' for detailed analysis.")


def main():
    """Main demo function"""
    print("Disease Prediction Model - Usage Demo")
    print("=====================================")
    
    # Initialize the demo
    demo = DiseasePredictionDemo('xgboost_disease_model.pkl')
    
    if not demo.model:
        print("Could not load model. Make sure 'xgboost_disease_model.pkl' exists.")
        return
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Show model information")
        print("2. Interactive diagnosis")
        print("3. Batch prediction example")
        print("4. View model performance")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            demo.show_model_info()
        elif choice == '2':
            demo.interactive_diagnosis()
        elif choice == '3':
            demo.batch_prediction_example()
        elif choice == '4':
            demo.evaluate_model_performance()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


# You can also use it directly like this:
def quick_prediction_example():
    """Quick example of how to use the model"""
    demo = DiseasePredictionDemo()
    
    if demo.model:
        # Example patient with some features
        # Note: Replace these with actual feature names from your model
        example_patient = {
            'has_fever': 1,  # If your model has this feature
            'age_of_onset_numeric': 45,  # If your model has this feature
            'family_history_binary': 0   # If your model has this feature
        }
        
        result = demo.predict_disease(example_patient)
        if result:
            print(f"Predicted disease: {result['predicted_disease']}")
            print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()