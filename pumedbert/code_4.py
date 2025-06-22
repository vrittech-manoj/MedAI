import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
import re
import os
import pickle
import warnings
from tqdm import tqdm
import logging
from datetime import datetime
import gc
import psutil
import multiprocessing as mp
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class OptimizedMedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, cache_encodings=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_encodings = cache_encodings
        self.encoding_cache = {} if cache_encodings else None
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Use cache for repeated texts (useful for large datasets with duplicates)
        if self.cache_encodings and text in self.encoding_cache:
            encoding = self.encoding_cache[text]
        else:
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
            
            if self.cache_encodings:
                self.encoding_cache[text] = encoding
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class AdvancedPubMedBERTClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.2, hidden_size=768):
        super(AdvancedPubMedBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        
        # Simplified architecture to prevent overfitting on small datasets
        self.dropout1 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Simple forward pass
        x = self.dropout1(pooled_output)
        return self.classifier(x)

class ScalableMedicalDiseaseClassifier:
    def __init__(self, model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.id_to_label = None
        self.training_history = []
        
        # Memory and performance monitoring
        self.memory_usage = []
        self.batch_times = []
        
        logger.info(f"Initialized classifier with device: {self.device}")
        logger.info(f"Available CPU cores: {mp.cpu_count()}")
        logger.info(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        
    def monitor_memory(self):
        """Monitor memory usage for large dataset processing"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / (1024 * 1024)
        self.memory_usage.append(memory_usage_mb)
        return memory_usage_mb
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing for medical domain"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^\w\s,.-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Extended medical term standardization
        medical_synonyms = {
            'pyrexia': 'fever',
            'dyspnea': 'shortness of breath',
            'dyspnoea': 'shortness of breath',
            'cephalgia': 'headache',
            'myalgia': 'muscle pain',
            'arthralgia': 'joint pain',
            'malaise': 'general discomfort',
            'anorexia': 'loss of appetite',
            'hemoptysis': 'coughing blood',
            'epistaxis': 'nosebleed',
            'rhinorrhea': 'runny nose',
            'lacrimation': 'excessive tearing',
            'photophobia': 'sensitivity to light',
            'tinnitus': 'ringing in ears'
        }
        
        for synonym, standard in medical_synonyms.items():
            text = text.replace(synonym, standard)
        
        return text
    
    def combine_symptoms(self, row):
        """Combine symptom columns for the given CSV format"""
        # Get symptom columns (symptoms_1 to symptoms_5)
        symptom_cols = [f'symptoms_{i}' for i in range(1, 6)]
        
        symptoms = []
        for col in symptom_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                symptoms.append(str(row[col]).strip())
        
        combined_text = ""
        
        if symptoms:
            combined_text = f"Patient symptoms: {', '.join(symptoms)}."
        
        return self.preprocess_text(combined_text)
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the medical dataset"""
        logger.info(f"Loading dataset from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} samples from CSV")
            
            # Combine symptoms for each row
            df['combined_text'] = df.apply(self.combine_symptoms, axis=1)
            
            # Filter out invalid rows
            df = df[
                (df['combined_text'].str.len() > 5) & 
                (df['diseases_name'].notna()) &
                (df['diseases_name'].str.strip() != '')
            ]
            
            logger.info(f"After filtering: {len(df)} valid samples")
            
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        df['labels'] = self.label_encoder.fit_transform(df['diseases_name'])
        
        # Create mappings
        self.id_to_label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        
        # Log dataset statistics
        logger.info(f"Number of unique diseases: {len(self.label_encoder.classes_)}")
        logger.info(f"Disease distribution:\n{df['diseases_name'].value_counts()}")
        
        # Check for class imbalance
        min_samples = df['diseases_name'].value_counts().min()
        max_samples = df['diseases_name'].value_counts().max()
        logger.info(f"Class balance - Min: {min_samples}, Max: {max_samples}, Ratio: {max_samples/min_samples:.2f}")
        
        return df
    
    def create_data_loaders(self, df, test_size=0.2, val_size=0.1, batch_size=8, num_workers=0):
        """Create data loaders with proper handling for small datasets"""
        
        logger.info(f"Creating data loaders with batch_size={batch_size}, num_workers={num_workers}")
        
        n_samples = len(df)
        n_classes = df['diseases_name'].nunique()
        
        # For very small datasets, adjust splits
        if n_samples <= 20:
            logger.warning(f"Very small dataset ({n_samples} samples). Using minimal splits.")
            test_size = max(1, int(n_samples * 0.2))  # At least 1 sample for test
            val_size_actual = max(1, int((n_samples - test_size) * 0.2))  # At least 1 for validation
        else:
            test_size = int(n_samples * test_size)
            val_size_actual = int(n_samples * val_size)
        
        # Check if we can use stratification
        min_class_count = df['diseases_name'].value_counts().min()
        
        # For stratification to work, we need at least 2 samples per class in each split
        # With very small datasets, this often isn't possible
        use_stratify = (min_class_count >= 3 and test_size >= n_classes and val_size_actual >= n_classes)
        
        if use_stratify:
            logger.info("Using stratified splits")
            # Stratified split to maintain class distribution
            train_df, test_df = train_test_split(
                df, test_size=test_size, stratify=df['labels'], 
                random_state=42, shuffle=True
            )
            
            # Check if we still have enough samples for validation stratification
            remaining_samples = len(train_df)
            val_min_count = train_df['diseases_name'].value_counts().min()
            
            if val_min_count >= 2 and val_size_actual < remaining_samples - n_classes:
                train_df, val_df = train_test_split(
                    train_df, test_size=val_size_actual, 
                    stratify=train_df['labels'], random_state=42, shuffle=True
                )
            else:
                logger.warning("Cannot stratify validation split, using random split")
                train_df, val_df = train_test_split(
                    train_df, test_size=val_size_actual, 
                    stratify=None, random_state=42, shuffle=True
                )
        else:
            logger.warning("Cannot use stratification due to small dataset size, using random splits")
            
            # Simple random splits
            train_df, test_df = train_test_split(
                df, test_size=test_size, stratify=None, 
                random_state=42, shuffle=True
            )
            
            train_df, val_df = train_test_split(
                train_df, test_size=val_size_actual, 
                stratify=None, random_state=42, shuffle=True
            )
        
        logger.info(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Log class distribution in each split
        logger.info(f"Train distribution:\n{train_df['diseases_name'].value_counts()}")
        logger.info(f"Val distribution:\n{val_df['diseases_name'].value_counts()}")
        logger.info(f"Test distribution:\n{test_df['diseases_name'].value_counts()}")
        
        # Create datasets
        train_dataset = OptimizedMedicalDataset(
            train_df['combined_text'].values,
            train_df['labels'].values,
            self.tokenizer,
            cache_encodings=False
        )
        
        val_dataset = OptimizedMedicalDataset(
            val_df['combined_text'].values,
            val_df['labels'].values,
            self.tokenizer,
            cache_encodings=False
        )
        
        test_dataset = OptimizedMedicalDataset(
            test_df['combined_text'].values,
            test_df['labels'].values,
            self.tokenizer,
            cache_encodings=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader, test_loader, train_df
    
    def train_model(self, csv_path, epochs=8, batch_size=8, learning_rate=2e-5, 
                   max_grad_norm=1.0, warmup_ratio=0.1):
        """Training with settings optimized for small datasets"""
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(csv_path)
        
        # Create data loaders
        train_loader, val_loader, test_loader, train_df = self.create_data_loaders(df, batch_size=batch_size)
        
        # Initialize model
        n_classes = len(self.label_encoder.classes_)
        self.model = AdvancedPubMedBERTClassifier(n_classes, dropout_rate=0.3)
        self.model.to(self.device)
        
        # Calculate class weights (reduced impact for small datasets)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['labels']),
            y=train_df['labels']
        )
        
        # Reduce extreme weights for small datasets
        class_weights = np.clip(class_weights, 0.5, 2.0)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Optimizer with lower weight decay for small datasets
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.001,  # Reduced weight decay
            eps=1e-8
        )
        
        # Reduced label smoothing for small datasets
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_ratio * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_f1 = 0
        patience = 5  # Increased patience for small datasets
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                train_pbar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'Acc': f'{train_correct/train_total:.4f}',
                    'LR': f'{current_lr:.2e}'
                })
            
            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_accuracy, val_f1, val_loss = self.evaluate_detailed(val_loader, criterion)
            
            # Log epoch results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'learning_rate': scheduler.get_last_lr()[0],
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_history.append(epoch_results)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_checkpoint(epoch, val_f1, 'best_medical_model.pt')
                patience_counter = 0
                logger.info(f"New best model saved! F1: {val_f1:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
        
        # Load best model and final evaluation
        self.load_model('best_medical_model.pt')
        test_accuracy, test_f1, test_loss = self.evaluate_detailed(test_loader, criterion)
        
        logger.info(f"Final Test Results - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        
        # Save training history
        self.save_training_history()
        
        return test_accuracy, test_f1
    
    def evaluate_detailed(self, data_loader, criterion=None):
        """Enhanced evaluation with loss calculation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        avg_loss = total_loss / len(data_loader) if criterion is not None else 0
        
        return accuracy, f1, avg_loss
    
    def save_checkpoint(self, epoch, val_f1, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'id_to_label': self.id_to_label,
            'epoch': epoch,
            'val_f1': val_f1,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"Model checkpoint saved to {filename}")

    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found!")
        
        logger.info(f"Loading model from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            logger.info("Loaded model with weights_only=False")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load label encoder and mappings
        self.label_encoder = checkpoint['label_encoder']
        self.id_to_label = checkpoint['id_to_label']
        
        # Load model
        n_classes = len(self.label_encoder.classes_)
        self.model = AdvancedPubMedBERTClassifier(n_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
    def save_training_history(self):
        """Save training history for analysis"""
        with open('training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info("Training history saved to training_history.json")
    
    def predict_top3_diseases(self, symptoms_text, temperature=1.0):
        """Enhanced prediction with temperature scaling"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded! Please train or load a model first.")
        
        # Handle both comma-separated and direct input
        if isinstance(symptoms_text, str) and ',' in symptoms_text:
            # Convert comma-separated to natural text
            symptoms_list = [s.strip() for s in symptoms_text.split(',')]
            processed_text = f"Patient symptoms: {', '.join(symptoms_list)}."
        else:
            processed_text = f"Patient symptoms: {symptoms_text}."
        
        processed_text = self.preprocess_text(processed_text)
        
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
            
            # Apply temperature scaling
            scaled_outputs = outputs / temperature
            probabilities = torch.nn.functional.softmax(scaled_outputs, dim=1)
            
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
                'confidence_score': round(confidence, 4),
                'confidence_percentage': f"{confidence*100:.2f}%"
            })
        
        return results

# Production-ready training and inference functions
def train_production_model(csv_path, **kwargs):
    """Train model with production settings"""
    classifier = ScalableMedicalDiseaseClassifier()
    
    # Default production settings optimized for small datasets
    default_settings = {
        'epochs': 10,
        'batch_size': 2,  # Even smaller batch size for very small datasets
        'learning_rate': 5e-6,  # Very low learning rate
        'warmup_ratio': 0.1
    }
    
    # Override with user settings
    settings = {**default_settings, **kwargs}
    
    logger.info(f"Training with settings: {settings}")
    
    try:
        test_accuracy, test_f1 = classifier.train_model(csv_path, **settings)
        logger.info(f"Training completed successfully! Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        return classifier, test_accuracy, test_f1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def load_and_predict_production(symptoms_text, model_path='best_medical_model.pt'):
    """Production inference function"""
    classifier = ScalableMedicalDiseaseClassifier()
    
    try:
        classifier.load_model(model_path)
        predictions = classifier.predict_top3_diseases(symptoms_text)
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

# Example usage
if __name__ == "__main__":
    logger.info("Starting medical disease classification system...")
    
    # Training (uncomment to train)
    classifier, test_acc, test_f1 = train_production_model('2.csv')
    
    # Test symptoms
    test_symptoms = [
        "Loss of appetite,Fatigue,Shallow breathing,Headache,Muscle ache",
        "Rash,Back pain,Chills,Low platelet,Weakness", 
        "Noisy breathing,Exercise-induced symptoms,Tight chest,Night cough,Trouble exhaling",
        "Severe throbbing headache with nausea and sensitivity to light",
        "Persistent dry cough with fever and loss of taste"
    ]
    
    # Test predictions
    for i, symptoms in enumerate(test_symptoms, 1):
        print(f"\nTest Case {i}: {symptoms}")
        print("-" * 60)
        
        predictions = load_and_predict_production(symptoms)
        if predictions:
            for pred in predictions:
                print(f"Rank {pred['rank']}: {pred['disease']} ({pred['confidence_percentage']})")
    
    logger.info("System ready for production use!")