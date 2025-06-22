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


class DiseasePredictionSystem:
    def __init__(self, use_smote=True, model_type='xgboost', min_samples_per_class=10):
        """
        Initialize the disease prediction system
        
        Parameters:
        -----------
        use_smote : bool, default=True
            Whether to use SMOTE for handling class imbalance
        model_type : str, default='xgboost'
            Model to use ('random_forest' or 'xgboost')
        min_samples_per_class : int, default=10
            Minimum number of samples required per class (increased from 5)
        """
        self.use_smote = use_smote
        self.model_type = model_type
        self.min_samples_per_class = min_samples_per_class
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_encoder = None
        self.symptom_vectorizer = None
        
    def inspect_dataset_structure(self, df):
        """Inspect the dataset structure to understand column names and content"""
        print("Dataset Structure Analysis:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Check for text-based symptom descriptions
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10).tolist()
                print(f"\nColumn '{col}' sample values:")
                for val in sample_values:
                    print(f"  - {str(val)[:100]}...")
                    
                # Check if this might contain symptoms
                if any(keyword in col.lower() for keyword in ['symptom', 'sign', 'manifestation', 'feature']):
                    text_columns.append(col)
                    
        return text_columns
        
    def preprocess_data(self, df):
        """
        Preprocess the raw dataset to prepare it for model training
        """
        print("Preprocessing data...")
        
        # Inspect dataset structure first
        text_columns = self.inspect_dataset_structure(df)
        
        # Extract target variable - check available columns for disease name
        possible_target_cols = ['diseases_name', 'Associated Disease', 'disease', 'Disease', 'condition', 'diagnosis']
        target_col = None
        
        for col in possible_target_cols:
            if col in df.columns:
                # Check if this column has meaningful disease names
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 1 and not all(val == 'Associated Disease' for val in unique_vals):
                    target_col = col
                    break
        
        if not target_col:
            print("Available columns:", df.columns.tolist())
            raise ValueError("Could not find disease name column in the dataset")
        
        print(f"Using '{target_col}' as target column")
        
        # Remove rows where target is missing or generic
        df_clean = df.dropna(subset=[target_col]).copy()
        df_clean = df_clean[df_clean[target_col] != 'Associated Disease']
        
        if df_clean.empty:
            raise ValueError("No valid disease data found after cleaning")
        
        # Check for class distribution and filter rare classes
        print(f"Original dataset size: {df_clean.shape[0]} records with {df_clean[target_col].nunique()} unique diseases")
        
        # Remove classes with too few samples
        class_counts = df_clean[target_col].value_counts()
        valid_classes = class_counts[class_counts >= self.min_samples_per_class].index
        df_filtered = df_clean[df_clean[target_col].isin(valid_classes)].copy()
        
        if df_filtered.empty:
            print(f"No diseases have >= {self.min_samples_per_class} samples. Reducing threshold to 5...")
            self.min_samples_per_class = 5
            valid_classes = class_counts[class_counts >= self.min_samples_per_class].index
            df_filtered = df_clean[df_clean[target_col].isin(valid_classes)].copy()
            
            if df_filtered.empty:
                print("Even with threshold of 5, no valid classes. Using threshold of 2...")
                self.min_samples_per_class = 2
                valid_classes = class_counts[class_counts >= self.min_samples_per_class].index
                df_filtered = df_clean[df_clean[target_col].isin(valid_classes)].copy()
        
        print(f"After filtering rare diseases: {df_filtered.shape[0]} records with {len(valid_classes)} unique diseases")
        print(f"Removed {df_clean.shape[0] - df_filtered.shape[0]} records with rare diseases")
        
        if df_filtered.empty:
            raise ValueError("No valid data remains after filtering")
        
        # Feature extraction approach 1: Look for structured symptom/cause columns
        feature_columns = []
        
        # Try to find symptom columns with different naming patterns
        symptom_patterns = ['symptom', 'sign', 'manifestation', 'feature', 'phenotype']
        cause_patterns = ['cause', 'etiology', 'risk_factor', 'trigger']
        
        symptom_cols = []
        cause_cols = []
        
        for col in df_filtered.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in symptom_patterns):
                symptom_cols.append(col)
            elif any(pattern in col_lower for pattern in cause_patterns):
                cause_cols.append(col)
        
        print(f"Found symptom columns: {symptom_cols}")
        print(f"Found cause columns: {cause_cols}")
        
        # Feature extraction approach 2: Use text-based features if structured ones don't work
        all_features = []
        
        # Process structured symptom/cause columns if available
        if symptom_cols or cause_cols:
            all_symptoms = set()
            all_causes = set()
            
            # Extract unique values from symptom columns
            for col in symptom_cols:
                values = df_filtered[col].dropna().astype(str)
                # Split on common delimiters
                for val in values:
                    if pd.notna(val) and val.strip() and val != 'nan':
                        # Split on various delimiters
                        parts = val.replace(';', ',').replace('|', ',').replace('/', ',').split(',')
                        for part in parts:
                            clean_part = part.strip().lower()
                            if clean_part and len(clean_part) > 2:
                                all_symptoms.add(clean_part)
            
            # Extract unique values from cause columns
            for col in cause_cols:
                values = df_filtered[col].dropna().astype(str)
                for val in values:
                    if pd.notna(val) and val.strip() and val != 'nan':
                        parts = val.replace(';', ',').replace('|', ',').replace('/', ',').split(',')
                        for part in parts:
                            clean_part = part.strip().lower()
                            if clean_part and len(clean_part) > 2:
                                all_causes.add(clean_part)
            
            print(f"Extracted {len(all_symptoms)} unique symptoms and {len(all_causes)} unique causes")
            
            # Create binary features for symptoms
            for symptom in all_symptoms:
                feature_name = f'has_{symptom}'
                df_filtered[feature_name] = 0
                for col in symptom_cols:
                    mask = df_filtered[col].astype(str).str.lower().str.contains(symptom, na=False, regex=False)
                    df_filtered.loc[mask, feature_name] = 1
                all_features.append(feature_name)
            
            # Create binary features for causes
            for cause in all_causes:
                feature_name = f'cause_{cause}'
                df_filtered[feature_name] = 0
                for col in cause_cols:
                    mask = df_filtered[col].astype(str).str.lower().str.contains(cause, na=False, regex=False)
                    df_filtered.loc[mask, feature_name] = 1
                all_features.append(feature_name)
        
        # Add numeric features if available
        numeric_features = ['age_of_onset', 'severity_of_disease', 'age', 'onset_age']
        for feature in numeric_features:
            if feature in df_filtered.columns:
                df_filtered[f'{feature}_numeric'] = pd.to_numeric(df_filtered[feature], errors='coerce')
                df_filtered[f'{feature}_numeric'].fillna(df_filtered[f'{feature}_numeric'].median(), inplace=True)
                all_features.append(f'{feature}_numeric')
        
        # Add categorical features
        categorical_features = ['family_history', 'genetic_factors', 'gender', 'sex']
        for feature in categorical_features:
            if feature in df_filtered.columns:
                # Convert to binary
                df_filtered[f'{feature}_binary'] = df_filtered[feature].apply(
                    lambda x: 1 if pd.notna(x) and str(x).lower() in ['yes', 'true', '1', 'positive', 'present'] else 0
                )
                all_features.append(f'{feature}_binary')
        
        # If we still don't have enough features, use text-based approach
        if len(all_features) < 5:
            print("Using text-based feature extraction as fallback...")
            
            # Combine all text columns that might contain symptom information
            text_data = []
            for idx, row in df_filtered.iterrows():
                combined_text = []
                for col in df_filtered.columns:
                    if col != target_col and df_filtered[col].dtype == 'object':
                        val = str(row[col])
                        if val and val != 'nan' and val != 'None':
                            combined_text.append(val)
                text_data.append(' '.join(combined_text))
            
            # Use TF-IDF to extract features
            self.symptom_vectorizer = TfidfVectorizer(
                max_features=100,  # Limit features to prevent overfitting
                min_df=2,  # Feature must appear in at least 2 documents
                max_df=0.8,  # Feature must appear in less than 80% of documents
                stop_words='english',
                ngram_range=(1, 2)  # Use both unigrams and bigrams
            )
            
            if text_data and any(text.strip() for text in text_data):
                text_features = self.symptom_vectorizer.fit_transform(text_data)
                text_feature_names = [f'text_feature_{i}' for i in range(text_features.shape[1])]
                
                # Add text features to dataframe
                text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names, index=df_filtered.index)
                df_filtered = pd.concat([df_filtered, text_df], axis=1)
                all_features.extend(text_feature_names)
                
                print(f"Added {len(text_feature_names)} text-based features")
        
        # Ensure we have features
        if not all_features:
            raise ValueError("Could not extract any meaningful features from the dataset")
        
        # Remove features with zero variance
        X_temp = df_filtered[all_features]
        feature_var = X_temp.var()
        non_constant_features = feature_var[feature_var > 0].index.tolist()
        
        if not non_constant_features:
            raise ValueError("All extracted features have zero variance")
        
        self.feature_names = non_constant_features
        print(f"Using {len(self.feature_names)} features for training")
        
        # Extract features and target
        X = df_filtered[self.feature_names]
        y = df_filtered[target_col]
        
        # Encode the target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        print(f"Data preprocessing complete. Final dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")
        
        return X, y_encoded
    
    def handle_imbalance(self, X, y):
        """Apply SMOTE to handle class imbalance"""
        if self.use_smote and len(np.unique(y)) > 1:
            print("Applying SMOTE to handle class imbalance...")
            try:
                # Use SMOTE with k_neighbors based on minority class size
                min_class_size = min(np.bincount(y))
                k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
                
                sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_res, y_res = sm.fit_resample(X, y)
                print(f"Data shape after SMOTE: {X_res.shape}")
                return X_res, y_res
            except Exception as e:
                print(f"SMOTE failed: {e}. Using original data.")
                return X, y
        return X, y
        
    def train(self, X, y):
        """Train the prediction model"""
        print(f"Training {self.model_type} model...")
        
        # Split the data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
        except ValueError:
            # If stratification fails, use regular split
            print("Stratified split failed, using regular split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
        
        # Apply SMOTE only to training data
        X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        if self.model_type == 'random_forest':
            # Simplified Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        else:  # xgboost
            # Simplified XGBoost
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
        # Train the model
        print("Fitting model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model training complete. Test accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        try:
            print(classification_report(y_test, y_pred, zero_division=0))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
        
        # Plot confusion matrix if not too many classes
        if len(np.unique(y_test)) <= 20:
            self.plot_confusion_matrix(y_test, y_pred)
        
        # Plot feature importance
        self.plot_feature_importance()
        
        return self.model, X_test, y_test, y_pred
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(y_test, y_pred)
            class_names = self.target_encoder.inverse_transform(np.unique(y_test))
            
            plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names))))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45)
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Confusion matrix saved as 'confusion_matrix.png'")
        except Exception as e:
            print(f"Could not create confusion matrix: {e}")
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                return
                
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot the top 20 features (or all if less than 20)
            n_features = min(20, len(indices))
            
            plt.figure(figsize=(12, 8))
            plt.title("Top Feature Importances")
            plt.barh(range(n_features), importances[indices[:n_features]], align='center')
            plt.yticks(range(n_features), [self.feature_names[i] for i in indices[:n_features]])
            plt.xlabel('Relative Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Feature importance plot saved as 'feature_importance.png'")
        except Exception as e:
            print(f"Could not create feature importance plot: {e}")
        
    def predict(self, patient_data):
        """
        Predict disease based on patient symptoms
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
            'model_type': self.model_type,
            'symptom_vectorizer': self.symptom_vectorizer
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
        system.symptom_vectorizer = model_data.get('symptom_vectorizer', None)
        
        return system

    def analyze_dataset(self, df):
        """Analyze the dataset to provide insights"""
        print("\n--- Dataset Analysis ---")
        
        # Show basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        
        # Try to identify target column
        target_cols = ['diseases_name', 'Associated Disease', 'disease', 'Disease']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            disease_counts = df[target_col].value_counts()
            print(f"\nTotal number of unique values in '{target_col}': {len(disease_counts)}")
            print(f"Top 10 most common values:")
            for disease, count in disease_counts.head(10).items():
                print(f"  - {disease}: {count} records ({count/len(df)*100:.1f}%)")
                
            # Class imbalance analysis
            print(f"\nClass distribution analysis:")
            print(f"Values with only 1 sample: {sum(disease_counts == 1)}")
            print(f"Values with 2-5 samples: {sum((disease_counts > 1) & (disease_counts <= 5))}")
            print(f"Values with 6-10 samples: {sum((disease_counts > 5) & (disease_counts <= 10))}")
            print(f"Values with >10 samples: {sum(disease_counts > 10)}")
        
        # Show sample of data
        print(f"\nSample data:")
        print(df.head())
        
        return df


class DiseasePredictionApp:
    """Interactive application for disease prediction"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.loaded = False
        
        if model_path:
            try:
                self.model = DiseasePredictionSystem.load_model(model_path)
                self.loaded = True
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {str(e)}")
    
    def train_new_model(self, dataset_path, model_type='xgboost', use_smote=True, min_samples=10):
        """Train a new model with the given dataset"""
        try:
            print(f"Loading dataset from {dataset_path}...")
            df = pd.read_csv(dataset_path)
            print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Create and train the model
            self.model = DiseasePredictionSystem(
                use_smote=use_smote, 
                model_type=model_type, 
                min_samples_per_class=min_samples
            )
            
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
            import traceback
            traceback.print_exc()
            return False


def main():
    print("Disease Prediction System - Fixed Version")
    print("----------------------------------------")
    
    import argparse
    parser = argparse.ArgumentParser(description="Disease Prediction System")
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--dataset', type=str, help='Path to dataset CSV file')
    parser.add_argument('--model_type', type=str, default='xgboost', 
                        choices=['xgboost', 'random_forest'], help='Model type for training')
    parser.add_argument('--min_samples', type=int, default=10, 
                        help='Minimum samples per class (default: 10)')
    args = parser.parse_args()
    
    if args.train and args.dataset:
        app = DiseasePredictionApp()
        success = app.train_new_model(
            args.dataset, 
            model_type=args.model_type,
            min_samples=args.min_samples
        )
        
        if success:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining failed. Check the error messages above.")
    else:
        print("Usage: python script.py --train --dataset your_file.csv")
        print("Optional: --model_type xgboost --min_samples 10")


if __name__ == "__main__":
    main()