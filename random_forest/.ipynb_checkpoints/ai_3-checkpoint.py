import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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
    def __init__(self, use_smote=True, model_type='xgboost', min_samples_per_class=5):
        """
        Initialize the disease prediction system
        
        Parameters:
        -----------
        use_smote : bool, default=True
            Whether to use SMOTE for handling class imbalance
        model_type : str, default='xgboost'
            Model to use ('random_forest' or 'xgboost')
        min_samples_per_class : int, default=5
            Minimum number of samples required per class
        """
        self.use_smote = use_smote
        self.model_type = model_type
        self.min_samples_per_class = min_samples_per_class
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
        
        # Extract target variable - check available columns for disease name
        possible_target_cols = ['diseases_name', 'Associated Disease']
        target_col = next((col for col in possible_target_cols if col in df.columns), None)
        
        if not target_col:
            raise ValueError("Could not find disease name column in the dataset")
        
        # Check for class distribution and filter rare classes
        print(f"Original dataset size: {df.shape[0]} records with {df[target_col].nunique()} unique diseases")
        
        # Remove classes with too few samples to allow stratification
        class_counts = df[target_col].value_counts()
        valid_classes = class_counts[class_counts >= self.min_samples_per_class].index
        df_filtered = df[df[target_col].isin(valid_classes)].copy()
        
        print(f"After filtering rare diseases: {df_filtered.shape[0]} records with {len(valid_classes)} unique diseases")
        print(f"Removed {df.shape[0] - df_filtered.shape[0]} records with rare diseases")
        
        # Extract all symptom columns
        symptom_cols = [col for col in df_filtered.columns if col.startswith('symptoms_') and col != 'symptoms_description']
        cause_cols = [col for col in df_filtered.columns if col.startswith('causes_') and col != 'causes_description']
        
        # Extract unique symptoms and causes
        all_symptoms = set()
        for col in symptom_cols:
            all_symptoms.update(df_filtered[col].dropna().unique())
            
        all_causes = set()
        for col in cause_cols:
            all_causes.update(df_filtered[col].dropna().unique())
        
        # Remove any None or NaN values
        all_symptoms = {s for s in all_symptoms if isinstance(s, str)}
        all_causes = {c for c in all_causes if isinstance(c, str)}
        
        print(f"Found {len(all_symptoms)} unique symptoms and {len(all_causes)} unique causes")
        
        # Create binary features for symptoms and causes
        for symptom in all_symptoms:
            df_filtered[f'has_{symptom}'] = 0
            for col in symptom_cols:
                df_filtered.loc[df_filtered[col] == symptom, f'has_{symptom}'] = 1
                
        for cause in all_causes:
            df_filtered[f'cause_{cause}'] = 0
            for col in cause_cols:
                df_filtered.loc[df_filtered[col] == cause, f'cause_{cause}'] = 1
        
        # Handle age_of_onset - convert to numeric if possible
        if 'age_of_onset' in df_filtered.columns:
            df_filtered['age_of_onset'] = pd.to_numeric(df_filtered['age_of_onset'], errors='coerce')
            # Fill missing values with median
            df_filtered['age_of_onset'].fillna(df_filtered['age_of_onset'].median(), inplace=True)
        else:
            # Create dummy age feature if not present
            df_filtered['age_of_onset'] = 0
            
        # Handle family_history as binary
        if 'family_history' in df_filtered.columns:
            df_filtered['family_history'] = df_filtered['family_history'].apply(
                lambda x: 1 if isinstance(x, str) and x.lower() in ['yes', 'true', '1'] else 0
            )
        else:
            df_filtered['family_history'] = 0
            
        # Handle genetic_factors as binary
        if 'genetic_factors' in df_filtered.columns:
            df_filtered['has_genetic_factors'] = df_filtered['genetic_factors'].apply(
                lambda x: 1 if isinstance(x, str) and x.lower() not in ['no', 'none', 'false', '0', ''] and pd.notna(x) else 0
            )
        else:
            df_filtered['has_genetic_factors'] = 0
            
        # Select features for model training
        symptom_features = [col for col in df_filtered.columns if col.startswith('has_')]
        cause_features = [col for col in df_filtered.columns if col.startswith('cause_')]
        context_features = ['age_of_onset', 'family_history', 'has_genetic_factors']
        
        # Additional features from your dataset schema if available
        extra_features = []
        for feature in ['severity_of_disease']:
            if feature in df_filtered.columns:
                # Convert to numeric if possible
                df_filtered[feature] = pd.to_numeric(df_filtered[feature], errors='coerce')
                df_filtered[feature].fillna(df_filtered[feature].median() if not df_filtered[feature].isna().all() else 0, inplace=True)
                extra_features.append(feature)
        
        # Ensure all context features exist
        for feature in context_features:
            if feature not in df_filtered.columns:
                df_filtered[feature] = 0
        
        all_features = symptom_features + cause_features + context_features + extra_features
        self.feature_names = all_features
        
        # Extract features and target
        X = df_filtered[all_features]
        y = df_filtered[target_col]
        
        # Encode the target variable
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
        
        # Check if any features have zero variance
        feature_var = X.var()
        constant_features = feature_var[feature_var == 0].index.tolist()
        
        if constant_features:
            print(f"Removing {len(constant_features)} constant features with zero variance")
            X = X.drop(columns=constant_features)
            # Update feature names
            self.feature_names = [f for f in self.feature_names if f not in constant_features]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
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
                RandomForestClassifier(random_state=42, class_weight='balanced'),
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
            
            # Get class weights to handle imbalance in XGBoost
            classes, counts = np.unique(y_train, return_counts=True)
            class_weights = {c: len(y_train) / (len(classes) * count) for c, count in zip(classes, counts)}
            
            self.model = GridSearchCV(
                xgb.XGBClassifier(
                    objective='multi:softprob', 
                    random_state=42,
                    scale_pos_weight=1,  # XGBoost will use sample_weight instead
                ),
                param_grid,
                cv=5,
                n_jobs=-1
            )
            
        # Train the model
        if self.model_type == 'xgboost':
            # Create sample weights based on class distribution
            sample_weight = np.array([class_weights[y] for y in y_train])
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model training complete. Test accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Plot feature importance
        self.plot_feature_importance()
        
        return self.model, X_test, y_test, y_pred
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix for the top classes"""
        # Get the most common classes
        classes = np.unique(y_test)
        if len(classes) > 15:  # Limit to top 15 classes to keep plot readable
            # Find the most common classes in the test set
            class_counts = pd.Series(y_test).value_counts()
            top_classes = class_counts.head(15).index
            # Filter to only include these classes
            mask_test = np.isin(y_test, top_classes)
            y_test_filtered = y_test[mask_test]
            y_pred_filtered = y_pred[mask_test]
            classes = top_classes
        else:
            y_test_filtered = y_test
            y_pred_filtered = y_pred
            
        cm = confusion_matrix(y_test_filtered, y_pred_filtered)
        
        # Convert numeric labels back to disease names for better visualization
        class_names = self.target_encoder.inverse_transform(classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if not self.feature_names:
            print("Feature names not available. Skipping feature importance plot.")
            return
            
        if self.model_type == 'random_forest':
            importances = self.model.best_estimator_.feature_importances_
        else:  # xgboost
            importances = self.model.best_estimator_.feature_importances_
            
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot the top 20 features (or all if less than 20)
        n_features = min(20, len(indices))
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.barh(range(n_features), importances[indices[:n_features]], align='center')
        plt.yticks(range(n_features), [self.feature_names[i] for i in indices[:n_features]])
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
        
        # Get top 5 diseases with probabilities (or fewer if less than 5 classes)
        top_k = min(5, len(pred_proba))
        top_indices = np.argsort(pred_proba)[::-1][:top_k]
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

    def analyze_dataset(self, df):
        """Analyze the dataset to provide insights about the disease distribution"""
        print("\n--- Dataset Analysis ---")
        
        # Identify the target column
        target_cols = ['diseases_name', 'Associated Disease']
        target_col = next((col for col in target_cols if col in df.columns), None)
        if not target_col:
            raise ValueError("Could not find disease column in dataset")
        
        # Disease distribution
        disease_counts = df[target_col].value_counts()
        print(f"Total number of diseases: {len(disease_counts)}")
        print(f"Top 10 most common diseases:")
        for disease, count in disease_counts.head(10).items():
            print(f"  - {disease}: {count} records ({count/len(df)*100:.1f}%)")
            
        # Check for class imbalance
        min_class = disease_counts.min()
        max_class = disease_counts.max()
        print(f"\nClass imbalance ratio (max/min): {max_class/min_class:.1f}")
        print(f"Diseases with only one sample: {sum(disease_counts == 1)}")
        print(f"Diseases with 2-5 samples: {sum((disease_counts > 1) & (disease_counts <= 5))}")
        
        # Symptom analysis
        symptom_cols = [col for col in df.columns if col.startswith('symptoms_') and col != 'symptoms_description']
        all_symptoms = set()
        for col in symptom_cols:
            all_symptoms.update(df[col].dropna().unique())
        
        all_symptoms = {s for s in all_symptoms if isinstance(s, str)}
        print(f"\nTotal unique symptoms: {len(all_symptoms)}")
        
        # Most common symptoms
        symptom_frequency = {}
        for symptom in all_symptoms:
            count = 0
            for col in symptom_cols:
                count += (df[col] == symptom).sum()
            symptom_frequency[symptom] = count
            
        # Top 10 symptoms
        top_symptoms = sorted(symptom_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most common symptoms:")
        for symptom, count in top_symptoms:
            print(f"  - {symptom}: {count} occurrences")
        
        # Plot disease distribution
        plt.figure(figsize=(10, 6))
        disease_counts.head(20).plot(kind='bar')
        plt.title('Top 20 Diseases by Frequency')
        plt.xlabel('Disease')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('disease_distribution.png')
        plt.close()
        
        return disease_counts, symptom_frequency


class DiseasePredictionApp:
    """Interactive application for disease prediction"""
    
    def __init__(self, model_path=None):
        """Initialize the application"""
        self.model = None
        self.loaded = False
        
        if model_path:
            try:
                self.model = DiseasePredictionSystem.load_model(model_path)
                self.loaded = True
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {str(e)}")
    
    def train_new_model(self, dataset_path, model_type='xgboost', use_smote=True, min_samples=5):
        """Train a new model with the given dataset"""
        try:
            df = pd.read_csv(dataset_path)
            print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Create and train the model
            self.model = DiseasePredictionSystem(use_smote=use_smote, model_type=model_type, min_samples_per_class=min_samples)
            
            # Analyze the dataset
            self.model.analyze_dataset(df)
            
            # Preprocess and train
            X, y = self.model.preprocess_data(df)
            self.model.train(X, y)
            
            # Save the model
            output_path = f"{model_type}_disease_model.pkl"
            self.model.save_model(output_path)
            self.loaded = True
            
            print(f"Model trained and saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return False
    
    def predict_disease(self, patient):
        """Make a disease prediction for the given patient"""
        if not self.loaded or not self.model:
            print("No model loaded. Please load or train a model first.")
            return None
            
        try:
            # Convert to feature format if needed
            if 'symptoms' in patient:
                patient_features = {}
                for symptom in patient.get('symptoms', []):
                    patient_features[f'has_{symptom}'] = 1
                for cause in patient.get('causes', []):
                    patient_features[f'cause_{cause}'] = 1
                patient_features['age_of_onset'] = patient.get('age_of_onset', 0)
                patient_features['family_history'] = patient.get('family_history', 0)
                patient_features['has_genetic_factors'] = patient.get('has_genetic_factors', 0)
                
                # Add any other features from the patient dict
                for key, value in patient.items():
                    if key not in ['symptoms', 'causes']:
                        patient_features[key] = value
            else:
                # Assume already in feature format
                patient_features = patient
            
            # Make prediction
            result = self.model.predict(patient_features)
            return result
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.loaded or not self.model:
            return "No model is currently loaded."
            
        info = {
            "model_type": self.model.model_type,
            "feature_count": len(self.model.feature_names) if self.model.feature_names else 0,
            "disease_count": len(self.model.target_encoder.classes_) if self.model.target_encoder else 0
        }
        
        return info
    
    def interactive_diagnosis(self):
        """Run an interactive diagnosis session"""
        if not self.loaded or not self.model:
            print("No model loaded. Please load or train a model first.")
            return
            
        print("\n--- Interactive Disease Diagnosis ---")
        print("Enter patient information to get a diagnosis.")
        
        # Get patient symptoms
        symptoms = []
        print("\nEnter symptoms (one per line, blank line to finish):")
        while True:
            symptom = input().strip()
            if not symptom:
                break
            symptoms.append(symptom)
            
        # Get causes if known
        causes = []
        print("\nEnter potential causes (one per line, blank line to finish):")
        while True:
            cause = input().strip()
            if not cause:
                break
            causes.append(cause)
            
        # Get other information
        try:
            age = int(input("\nEnter patient age: "))
        except:
            age = 0
        
        family_history = input("\nDoes the patient have family history of this condition? (yes/no): ").lower() == 'yes'
        genetic_factors = input("\nAre there known genetic factors? (yes/no): ").lower() == 'yes'
        
        # Create patient data
        patient = {
            'symptoms': symptoms,
            'causes': causes,
            'age_of_onset': age,
            'family_history': 1 if family_history else 0,
            'has_genetic_factors': 1 if genetic_factors else 0
        }
        
        # Make prediction
        result = self.predict_disease(patient)
        
        if result:
            print("\n--- Diagnosis Results ---")
            print(f"Primary diagnosis: {result['predicted_disease']} (Confidence: {result['confidence']:.2f})")
            print("\nDifferential diagnosis:")
            for disease, prob in result['differential_diagnosis'].items():
                print(f"  - {disease}: {prob:.2f}")
        else:
            print("Could not generate a diagnosis.")


# Example usage of the disease prediction system
def main():
    print("Disease Prediction System")
    print("-------------------------")
    
    # Parse command line arguments if any
    import argparse
    parser = argparse.ArgumentParser(description="Disease Prediction System")
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--predict', action='store_true', help='Make a prediction')
    parser.add_argument('--interactive', action='store_true', help='Interactive diagnosis mode')
    parser.add_argument('--dataset', type=str, help='Path to dataset CSV file')
    parser.add_argument('--model', type=str, default='xgboost_disease_model.pkl', help='Path to model file')
    parser.add_argument('--model_type', type=str, default='xgboost', 
                        choices=['xgboost', 'random_forest'], help='Model type for training')
    args = parser.parse_args()
    
    app = DiseasePredictionApp(args.model if not args.train else None)
    
    if args.train:
        if not args.dataset:
            print("Error: Dataset path is required for training.")
            return
        
        print(f"Training new {args.model_type} model with dataset: {args.dataset}")
        app.train_new_model(args.dataset, model_type=args.model_type)
        
    elif args.predict:
        # Example patient for prediction
        example_patient = {
            'symptoms': ['fever', 'cough', 'fatigue', 'body ache'],
            'causes': ['viral'],
            'age_of_onset': 45,
            'family_history': 0,
            'has_genetic_factors': 0
        }
        
        result = app.predict_disease(example_patient)
        if result:
            print("\n--- Example Prediction ---")
            print(f"Patient symptoms: {', '.join(example_patient['symptoms'])}")
            print(f"Predicted Disease: {result['predicted_disease']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print("Differential Diagnosis:")
            for disease, prob in result['differential_diagnosis'].items():
                print(f"  - {disease}: {prob:.2f}")
    
    elif args.interactive:
        app.interactive_diagnosis()
        
    else:
        # If no specific action, show both model training and prediction examples
        if args.dataset:
            print(f"Training new {args.model_type} model with dataset: {args.dataset}")
            if app.train_new_model(args.dataset, model_type=args.model_type):
                # Example prediction after training
                example_patient = {
                    'symptoms': ['fever', 'cough', 'fatigue', 'body ache'],
                    'causes': ['viral'],
                    'age_of_onset': 45,
                    'family_history': 0,
                    'has_genetic_factors': 0
                }
                
                result = app.predict_disease(example_patient)
                if result:
                    print("\n--- Example Prediction ---")
                    print(f"Patient symptoms: {', '.join(example_patient['symptoms'])}")
                    print(f"Predicted Disease: {result['predicted_disease']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    print("Differential Diagnosis:")
                    for disease, prob in result['differential_diagnosis'].items():
                        print(f"  - {disease}: {prob:.2f}")
                    
                # Interactive session
                app.interactive_diagnosis()
        else:
            print("No action specified. Use --train to train a model, --predict to make a prediction, or --interactive for interactive mode.")
            print("Example command to train a model:")
            print("  python disease_prediction.py --train --dataset your_dataset.csv --model_type xgboost")


if __name__ == "__main__":
    main()