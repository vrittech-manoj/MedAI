import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel , get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import json
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PubMedBERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super(PubMedBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class MedicalDiseaseClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.id_to_label = None
        
    def preprocess_text(self, text):
        """Clean and standardize medical text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s,.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Medical term standardization
        medical_synonyms = {
            'pyrexia': 'fever',
            'dyspnea': 'shortness of breath',
            'cephalgia': 'headache',
            'myalgia': 'muscle pain',
            'arthralgia': 'joint pain'
        }
        
        for synonym, standard in medical_synonyms.items():
            text = text.replace(synonym, standard)
        
        return text
    
    def combine_symptoms(self, row):
        """Combine all symptom columns into structured text"""
        symptom_cols = ['symptoms_1', 'symptoms_2', 'symptoms_3', 'symptoms_4', 
                       'symptoms_5', 'symptoms_6', 'symptoms_7', 'symptoms_8', 'symptoms_9']
        
        symptoms = []
        for col in symptom_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                symptoms.append(str(row[col]).strip())
        
        combined_text = ""
        
        if symptoms:
            combined_text += f"Symptoms: {', '.join(symptoms)}. "
        
        if 'description' in row and pd.notna(row['description']):
            combined_text += f"Description: {row['description']}. "
        
        if 'symptoms_description' in row and pd.notna(row['symptoms_description']):
            combined_text += f"Additional symptoms: {row['symptoms_description']}. "
        
        if 'causes_description' in row and pd.notna(row['causes_description']):
            combined_text += f"Causes: {row['causes_description']}. "
        
        return self.preprocess_text(combined_text)
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the medical dataset"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(csv_path)
        
        # Combine text fields
        df['combined_text'] = df.apply(self.combine_symptoms, axis=1)
        
        # Remove rows with empty text or missing disease names
        df = df[(df['combined_text'].str.len() > 10) & (df['diseases_name'].notna())]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        df['labels'] = self.label_encoder.fit_transform(df['diseases_name'])
        
        # Create id to label mapping
        self.id_to_label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        
        print(f"Dataset loaded: {len(df)} samples, {len(self.label_encoder.classes_)} diseases")
        return df
    
    def create_data_loaders(self, df, test_size=0.2, val_size=0.1, batch_size=16):
        """Create train, validation, and test data loaders"""
        
        # Split data
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['labels'], random_state=42
        )
        
        train_df, val_df = train_test_split(
            train_df, test_size=val_size/(1-test_size), stratify=train_df['labels'], random_state=42
        )
        
        # Create datasets
        train_dataset = MedicalDataset(
            train_df['combined_text'].values,
            train_df['labels'].values,
            self.tokenizer
        )
        
        val_dataset = MedicalDataset(
            val_df['combined_text'].values,
            val_df['labels'].values,
            self.tokenizer
        )
        
        test_dataset = MedicalDataset(
            test_df['combined_text'].values,
            test_df['labels'].values,
            self.tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, train_df
    
    def train_model(self, csv_path, epochs=4, batch_size=16, learning_rate=2e-5):
        """Train the PubMedBERT model"""
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(csv_path)
        
        # Create data loaders
        train_loader, val_loader, test_loader, train_df = self.create_data_loaders(df, batch_size=batch_size)
        
        # Initialize model
        n_classes = len(self.label_encoder.classes_)
        self.model = PubMedBERTClassifier(n_classes)
        self.model.to(self.device)
        
        # Calculate class weights for imbalanced dataset
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['labels']),
            y=train_df['labels']
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Setup training
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_f1 = 0
        patience = 2
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_accuracy, val_f1 = self.evaluate(val_loader)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'label_encoder': self.label_encoder,
                    'id_to_label': self.id_to_label
                }, 'best_medical_model.pt')
                patience_counter = 0
                print("New best model saved!")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        # Load best model and evaluate on test set
        self.load_model('best_medical_model.pt')
        test_accuracy, test_f1 = self.evaluate(test_loader)
        print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test F1 Score: {test_f1:.4f}")
        
        # Save final model components
        self.save_model_components()
        
        return test_accuracy, test_f1
    
    def evaluate(self, data_loader):
        """Evaluate model performance"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return accuracy, f1
    
    def save_model_components(self):
        """Save all model components"""
        # Save tokenizer
        self.tokenizer.save_pretrained('./medical_model')
        
        # Save label mappings
        with open('./medical_model/label_mappings.json', 'w') as f:
            json.dump({
                'id_to_label': self.id_to_label,
                'label_to_id': {v: k for k, v in self.id_to_label.items()}
            }, f)
        
        print("Model components saved successfully!")
    
    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found!")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        
        # Load label encoder and mappings
        self.label_encoder = checkpoint['label_encoder']
        self.id_to_label = checkpoint['id_to_label']
        
        # Load model
        n_classes = len(self.label_encoder.classes_)
        self.model = PubMedBERTClassifier(n_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict_top3_diseases(self, symptoms_text):
        """Predict top 3 diseases with confidence scores"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded! Please train or load a model first.")
        
        # Preprocess input
        processed_text = self.preprocess_text(symptoms_text)
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            
            top3_probs = top3_probs.cpu().numpy()[0]
            top3_indices = top3_indices.cpu().numpy()[0]
        
        # Format results
        results = []
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            disease = self.id_to_label[idx]
            confidence = float(prob)
            results.append({
                'rank': i + 1,
                'disease': disease,
                'confidence_score': round(confidence, 4)
            })
        
        return results

# Example usage and training script
def main():
    # Initialize classifier
    classifier = MedicalDiseaseClassifier()
    
    # Train model (uncomment to train)
    print("Starting training...")
    test_accuracy, test_f1 = classifier.train_model('2.csv', epochs=4, batch_size=16)
    
    # Example predictions after training
    print("\n" + "="*50)
    print("TESTING PREDICTIONS")
    print("="*50)
    
    # Test cases
    test_cases = [
        "Patient has fever, cough, shortness of breath, and chest pain for 3 days",
        "Severe headache, nausea, vomiting, sensitivity to light",
        "Joint pain, morning stiffness, swelling in hands and feet",
        "Chest pain radiating to left arm, sweating, nausea, shortness of breath"
    ]
    
    for i, symptoms in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {symptoms}")
        print("-" * 60)
        
        try:
            predictions = classifier.predict_top3_diseases(symptoms)
            for pred in predictions:
                print(f"Rank {pred['rank']}: {pred['disease']} (Confidence: {pred['confidence_score']:.1%})")
        except Exception as e:
            print(f"Error: {e}")

# Function to load model and make predictions (for inference only)
def load_and_predict(symptoms_text):
    """Load trained model and make prediction"""
    classifier = MedicalDiseaseClassifier()
    
    try:
        classifier.load_model('best_medical_model.pt')
        predictions = classifier.predict_top3_diseases(symptoms_text)
        return predictions
    except Exception as e:
        print(f"Error loading model or making prediction: {e}")
        return None

if __name__ == "__main__":
    # Uncomment the line below to train the model
    main()
    
    # Example of using the trained model for inference
    # symptoms_input = "Patient complains of persistent cough, fever, and difficulty breathing"
    # results = load_and_predict(symptoms_input)
    # if results:
    #     for result in results:
    #         print(f"{result['rank']}. {result['disease']} - {result['confidence_score']:.1%}")