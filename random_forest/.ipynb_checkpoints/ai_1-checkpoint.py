import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')


class DiseasePredictionSystem:
    def __init__(self, use_smote=True, model_type='xgboost'):
        """
        Initialize the disease prediction system
        
        Parameters:
        -----------
        use_smote : bool, default=True
            Whether to use SMOTE for handling class imbalance
        model_type : str, default='xgboost'
            Model to use ('random_forest' or 'xgboost')
        """
        self.use_smote = use_smote
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_encoder = None
        
    def preprocess_data(self, df):
        """
        Preprocess the raw dataset to prepare it for model training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The raw dataset containing disease information
            
        Returns:
        --------
        X : numpy.ndarray
            Processed feature matrix
        y : numpy.ndarray
            Target labels
        """
        print("Preprocessing data...")
        
        # Extract all symptom columns and convert to position-independent format
        symptom_cols = [col for col in df.columns if col.startswith('symptoms_') and col != 'symptoms_description']
        cause_cols = [col for col in df.columns if col.startswith('causes_') and col != 'causes_description']
        
        # Extract unique symptoms and causes
        all_symptoms = set()
        for col in symptom_cols:
            all_symptoms.update(df[col].dropna().unique())
            
        all_causes = set()
        for col in cause_cols:
            all_causes.update(df[col].dropna().unique())
        
        # Remove any None or NaN values
        all_symptoms = {s for s in all_symptoms if isinstance(s, str)}
        all_causes = {c for c in all_causes if isinstance(c, str)}
        
        # Create binary features for symptoms and causes
        for symptom in all_symptoms:
            df[f'has_{symptom}'] = 0
            for col in symptom_cols:
                df.loc[df[col] == symptom, f'has_{symptom}'] = 1
                
        for cause in all_causes:
            df[f'cause_{cause}'] = 0
            for col in cause_cols:
                df.loc[df[col] == cause, f'cause_{cause}'] = 1
        
        # Handle age_of_onset - convert to numeric if possible
        if 'age_of_onset' in df.columns:
            df['age_of_onset'] = pd.to_numeric(df['age_of_onset'], errors='coerce')
            # Fill missing values with median
            df['age_of_onset'].fillna(df['age_of_onset'].median(), inplace=True)
        else:
            # Create dummy age feature if not present
            df['age_of_onset'] = df['age_of_onset'] if 'age_of_onset' in df.columns else 0
            
        # Handle family_history as binary
        if 'family_history' in df.columns:
            df['family_history'] = df['family_history'].apply(
                lambda x: 1 if isinstance(x, str) and x.lower() in ['yes', 'true', '1'] else 0
            )
        else:
            df['family_history'] = 0
            
        # Handle genetic_factors as binary
        if 'genetic_factors' in df.columns:
            df['has_genetic_factors'] = df['genetic_factors'].apply(
                lambda x: 1 if isinstance(x, str) and x.lower() not in ['no', 'none', 'false', '0', ''] and pd.notna(x) else 0
            )
        else:
            df['has_genetic_factors'] = 0
            
        # Select features for model training
        symptom_features = [col for col in df.columns if col.startswith('has_')]
        cause_features = [col for col in df.columns if col.startswith('cause_')]
        context_features = ['age_of_onset', 'family_history', 'has_genetic_factors']
        
        all_features = symptom_features + cause_features + context_features
        self.feature_names = all_features
        
        # Extract features and target
        X = df[all_features]
        y = df['diseases_name'] if 'diseases_name' in df.columns else df['Associated Disease']
        
        # Encode the target variable
        from sklearn.preprocessing import LabelEncoder
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        print(f"Data preprocessing complete. Generated {len(all_features)} features.")
        return X, y_encoded
    
    def handle_imbalance(self, X, y):
        """Apply SMOTE to handle class imbalance"""
        if self.use_smote:
            print("Applying SMOTE to handle class imbalance...")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            print(f"Data shape after SMOTE: {X_res.shape}")
            return X_res, y_res
        return X, y
        
    def train(self, X, y):
        """Train the prediction model"""
        print(f"Training {self.model_type} model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Apply SMOTE only to training data
        X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        if self.model_type == 'random_forest':
            # Random Forest with hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            self.model = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                n_jobs=-1
            )
        else:  # xgboost
            # XGBoost with hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            self.model = GridSearchCV(
                xgb.XGBClassifier(objective='multi:softprob', random_state=42),
                param_grid,
                cv=5,
                n_jobs=-1
            )
            
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model training complete. Test accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot feature importance
        self.plot_feature_importance()
        
        return self.model, X_test, y_test, y_pred
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model_type == 'random_forest':
            importances = self.model.best_estimator_.feature_importances_
        else:  # xgboost
            importances = self.model.best_estimator_.feature_importances_
            
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot the top 20 features
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.barh(range(min(20, len(indices))), importances[indices[:20]], align='center')
        plt.yticks(range(min(20, len(indices))), [self.feature_names[i] for i in indices[:20]])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
    def predict(self, patient_data):
        """
        Predict disease based on patient symptoms
        
        Parameters:
        -----------
        patient_data : dict
            Dictionary containing patient symptoms, causes, and other factors
            
        Returns:
        --------
        dict
            Prediction results with disease and probabilities
        """
        if not self.model:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Convert patient data to feature vector
        features = np.zeros(len(self.feature_names))
        
        for i, feature in enumerate(self.feature_names):
            if feature in patient_data:
                features[i] = patient_data[feature]
            elif feature.startswith('has_') and feature[4:] in patient_data.get('symptoms', []):
                features[i] = 1
            elif feature.startswith('cause_') and feature[6:] in patient_data.get('causes', []):
                features[i] = 1
                
        # Get prediction probabilities
        features_reshaped = features.reshape(1, -1)
        pred_proba = self.model.predict_proba(features_reshaped)[0]
        pred_class = self.model.predict(features_reshaped)[0]
        
        # Convert to disease names
        predicted_disease = self.target_encoder.inverse_transform([pred_class])[0]
        
        # Get top 5 diseases with probabilities
        top_indices = np.argsort(pred_proba)[::-1][:5]
        top_diseases = self.target_encoder.inverse_transform(top_indices)
        top_probabilities = pred_proba[top_indices]
        
        result = {
            'predicted_disease': predicted_disease,
            'confidence': float(pred_proba[pred_class]),
            'differential_diagnosis': {
                disease: float(prob) for disease, prob in zip(top_diseases, top_probabilities)
            }
        }
        
        return result
    
    def save_model(self, filename='disease_prediction_model.pkl'):
        """Save the trained model and preprocessor"""
        if not self.model:
            raise ValueError("No trained model to save")
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
        
    @classmethod
    def load_model(cls, filename='disease_prediction_model.pkl'):
        """Load a saved model"""
        model_data = joblib.load(filename)
        
        system = cls(model_type=model_data['model_type'])
        system.model = model_data['model']
        system.feature_names = model_data['feature_names']
        system.target_encoder = model_data['target_encoder']
        
        return system

# Usage example
if __name__ == "__main__":
    # Step 1: Load your dataset
    # Replace this with your actual data file
    print("Loading dataset...")
    try:
        df = pd.read_csv('good.csv')
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    except FileNotFoundError:
        print("Example dataset not found. This is just a demonstration of how to use the system.")
        # Create a small sample dataset for demonstration
        data = {
            'diseases_name': ['Flu', 'COVID-19', 'Common Cold', 'Flu', 'COVID-19'],
            'symptoms_1': ['fever', 'fever', 'runny nose', 'fever', 'cough'],
            'symptoms_2': ['cough', 'fatigue', 'cough', 'body ache', 'fever'],
            'symptoms_3': ['fatigue', 'cough', 'sore throat', 'headache', 'loss of taste'],
            'symptoms_4': [None, 'loss of taste', None, None, 'fatigue'],
            'causes_1': ['viral', 'viral', 'viral', 'viral', 'viral'],
            'causes_2': [None, 'infectious', None, None, 'infectious'],
            'age_of_onset': [35, 50, 28, 42, 39],
            'family_history': [0, 0, 0, 1, 0],
            'genetic_factors': ['No', 'No', 'No', 'Yes', 'No']
        }
        df = pd.DataFrame(data)
        print("Created example dataset for demonstration.")
    
    # Step 2: Create prediction system - try both models
    print("\n--- XGBoost Model Training ---")
    xgb_system = DiseasePredictionSystem(use_smote=True, model_type='xgboost')
   
    X, y = xgb_system.preprocess_data(df)
    print("ok ok")
    xgb_model, X_test, y_test, y_pred = xgb_system.train(X, y)
    xgb_system.save_model('xgboost_disease_model.pkl')
    
    print("\n--- Random Forest Model Training ---")
    rf_system = DiseasePredictionSystem(use_smote=True, model_type='random_forest')
    X, y = rf_system.preprocess_data(df)
    rf_model, X_test, y_test, y_pred = rf_system.train(X, y)
    rf_system.save_model('rf_disease_model.pkl')
    
    # Step 3: Example prediction
    print("\n--- Example Prediction ---")
    # Example patient data
    patient = {
        'symptoms': ['fever', 'cough', 'fatigue', 'body ache'],
        'causes': ['viral'],
        'age_of_onset': 45,
        'family_history': 0,
        'has_genetic_factors': 0
    }
    
    # Convert to feature format
    patient_features = {}
    for symptom in patient['symptoms']:
        patient_features[f'has_{symptom}'] = 1
    for cause in patient['causes']:
        patient_features[f'cause_{cause}'] = 1
    patient_features['age_of_onset'] = patient['age_of_onset']
    patient_features['family_history'] = patient['family_history']
    patient_features['has_genetic_factors'] = patient['has_genetic_factors']
    
    # Predict using XGBoost
    xgb_prediction = xgb_system.predict(patient_features)
    print("XGBoost Prediction:")
    print(f"Predicted Disease: {xgb_prediction['predicted_disease']}")
    print(f"Confidence: {xgb_prediction['confidence']:.2f}")
    print("Differential Diagnosis:")
    for disease, prob in xgb_prediction['differential_diagnosis'].items():
        print(f"  - {disease}: {prob:.2f}")
        
    # Predict using Random Forest
    rf_prediction = rf_system.predict(patient_features)
    print("\nRandom Forest Prediction:")
    print(f"Predicted Disease: {rf_prediction['predicted_disease']}")
    print(f"Confidence: {rf_prediction['confidence']:.2f}")
    print("Differential Diagnosis:")
    for disease, prob in rf_prediction['differential_diagnosis'].items():
        print(f"  - {disease}: {prob:.2f}")