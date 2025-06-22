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
    def __init__(self, n_classes, dropout_rate=0.3, hidden_size=768):
        super(AdvancedPubMedBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        
        # Advanced architecture with multiple dropout layers and batch normalization
        self.dropout1 = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.intermediate = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        self.classifier = nn.Linear(hidden_size // 2, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Advanced forward pass with intermediate layers
        x = self.dropout1(pooled_output)
        x = self.batch_norm(x)
        x = self.intermediate(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
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
        """Intelligently combine symptom columns with prioritization"""
        # Get all symptom columns dynamically
        symptom_cols = [col for col in row.index if col.startswith('symptoms_')]
        
        symptoms = []
        for col in symptom_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                symptoms.append(str(row[col]).strip())
        
        combined_text = ""
        
        if symptoms:
            combined_text += f"Primary symptoms: {', '.join(symptoms)}. "
        
        # Add additional fields if available
        additional_fields = ['description', 'symptoms_description', 'causes_description', 
                           'risk_factors', 'complications', 'treatment']
        
        for field in additional_fields:
            if field in row and pd.notna(row[field]):
                field_name = field.replace('_', ' ').title()
                combined_text += f"{field_name}: {row[field]}. "
        
        return self.preprocess_text(combined_text)
    
    def load_and_preprocess_data(self, csv_path, chunk_size=10000):
        """Efficiently load and preprocess large datasets"""
        logger.info(f"Loading dataset from {csv_path}")
        
        # Read in chunks for memory efficiency
        chunks = []
        total_rows = 0
        
        try:
            for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
                # Process each chunk
                chunk_df['combined_text'] = chunk_df.apply(self.combine_symptoms, axis=1)
                
                # Filter out invalid rows
                chunk_df = chunk_df[
                    (chunk_df['combined_text'].str.len() > 5) & 
                    (chunk_df['diseases_name'].notna()) &
                    (chunk_df['diseases_name'].str.strip() != '')
                ]
                
                if len(chunk_df) > 0:
                    chunks.append(chunk_df)
                    total_rows += len(chunk_df)
                
                # Memory monitoring
                memory_usage = self.monitor_memory()
                logger.info(f"Processed chunk, total rows: {total_rows}, Memory: {memory_usage:.2f} MB")
                
                # Garbage collection for large datasets
                gc.collect()
        
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
        
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Total dataset size: {len(df)} samples")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        df['labels'] = self.label_encoder.fit_transform(df['diseases_name'])
        
        # Create mappings
        self.id_to_label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        
        # Log dataset statistics
        logger.info(f"Number of unique diseases: {len(self.label_encoder.classes_)}")
        logger.info(f"Disease distribution:\n{df['diseases_name'].value_counts().head(10)}")
        
        return df
    
    def create_data_loaders(self, df, test_size=0.2, val_size=0.1, batch_size=16, num_workers=None):
        """Create optimized data loaders for large datasets"""
        
        if num_workers is None:
            num_workers = min(4, mp.cpu_count())
        
        logger.info(f"Creating data loaders with batch_size={batch_size}, num_workers={num_workers}")
        
        # Stratified split to maintain class distribution
        # train_df, test_df = train_test_split(
        #     df, test_size=test_size, stratify=df['labels'], 
        #     random_state=42, shuffle=True
        # )
        
        # train_df, val_df = train_test_split(
        #     train_df, test_size=val_size/(1-test_size), 
        #     stratify=train_df['labels'], random_state=42, shuffle=True
        # )

        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=None, 
            random_state=42, shuffle=True
        )
        
        train_df, val_df = train_test_split(
            train_df, test_size=val_size/(1-test_size), 
            stratify=None, random_state=42, shuffle=True
        )
        
        logger.info(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets with caching for large datasets
        cache_encodings = len(df) > 50000  # Enable caching for large datasets
        
        train_dataset = OptimizedMedicalDataset(
            train_df['combined_text'].values,
            train_df['labels'].values,
            self.tokenizer,
            cache_encodings=cache_encodings
        )
        
        val_dataset = OptimizedMedicalDataset(
            val_df['combined_text'].values,
            val_df['labels'].values,
            self.tokenizer,
            cache_encodings=cache_encodings
        )
        
        test_dataset = OptimizedMedicalDataset(
            test_df['combined_text'].values,
            test_df['labels'].values,
            self.tokenizer,
            cache_encodings=cache_encodings
        )
        
        # Create optimized data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size * 2,  # Larger batch for inference
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return train_loader, val_loader, test_loader, train_df
    
    def train_model(self, csv_path, epochs=4, batch_size=16, learning_rate=2e-5, 
                   accumulation_steps=1, max_grad_norm=1.0, warmup_ratio=0.1):
        """Advanced training with gradient accumulation and mixed precision"""
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(csv_path)
        
        # Adjust batch size for large datasets
        if len(df) > 100000:
            batch_size = max(8, batch_size // 2)
            accumulation_steps = accumulation_steps * 2
            logger.info(f"Large dataset detected. Adjusted batch_size to {batch_size} with accumulation_steps={accumulation_steps}")
        
        # Create data loaders
        train_loader, val_loader, test_loader, train_df = self.create_data_loaders(df, batch_size=batch_size)
        
        # Initialize model
        n_classes = len(self.label_encoder.classes_)
        self.model = AdvancedPubMedBERTClassifier(n_classes)
        self.model.to(self.device)
        
        # Mixed precision training for efficiency
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['labels']),
            y=train_df['labels']
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Advanced optimizer settings
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        total_steps = len(train_loader) * epochs // accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_ratio * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop with advanced features
        best_val_f1 = 0
        patience = 3
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
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, attention_mask)
                        loss = criterion(outputs, labels) / accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels) / accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                
                train_loss += loss.item() * accumulation_steps
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
                
                # Memory monitoring for large datasets
                if batch_idx % 100 == 0:
                    memory_usage = self.monitor_memory()
                    if memory_usage > 8000:  # > 8GB
                        logger.warning(f"High memory usage: {memory_usage:.2f} MB")
                        gc.collect()
                        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
            
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
        
        # Save training history and model components
        self.save_training_history()
        self.save_model_components()
        
        return test_accuracy, test_f1
    
    def evaluate_detailed(self, data_loader, criterion=None):
        """Enhanced evaluation with loss calculation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
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
        """Save comprehensive model checkpoint with better PyTorch 2.6+ compatibility"""
        # Save model weights separately (weights_only compatible)
        model_weights = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'val_f1': val_f1,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save non-tensor objects separately using pickle
        metadata = {
            'label_encoder': self.label_encoder,
            'id_to_label': self.id_to_label,
            'epoch': epoch,
            'val_f1': val_f1,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save both together for backward compatibility
        full_checkpoint = {**model_weights, **metadata}
        
        try:
            # Try saving with the full checkpoint first
            torch.save(full_checkpoint, filename)
        except Exception as e:
            logger.warning(f"Failed to save full checkpoint: {e}")
            # Fallback: save separately
            torch.save(model_weights, filename)
            
            # Save metadata separately
            metadata_file = filename.replace('.pt', '_metadata.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved model weights to {filename} and metadata to {metadata_file}")

    def load_model(self, model_path):
        """Load trained model with improved error handling for PyTorch 2.6+"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found!")
        
        logger.info(f"Loading model from {model_path}")
        
        # Try different loading strategies
        checkpoint = None
        
        # Strategy 1: Try loading with weights_only=False (backward compatibility)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            logger.info("Loaded model with weights_only=False")
        except Exception as e1:
            logger.warning(f"weights_only=False failed: {e1}")
            
            # Strategy 2: Try loading weights only and metadata separately
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                
                # Try to load metadata from separate file
                metadata_file = model_path.replace('.pt', '_metadata.pkl')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    checkpoint.update(metadata)
                    logger.info("Loaded model with separate metadata file")
                else:
                    logger.error("Metadata file not found and weights_only=True doesn't contain label_encoder")
                    raise ValueError("Cannot load model: missing label encoder information")
                    
            except Exception as e2:
                logger.error(f"All loading strategies failed. Error: {e2}")
                raise e2
        
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
        
        # Save memory usage statistics
        if self.memory_usage:
            memory_stats = {
                'max_memory_mb': max(self.memory_usage),
                'avg_memory_mb': np.mean(self.memory_usage),
                'memory_timeline': self.memory_usage
            }
            with open('memory_stats.json', 'w') as f:
                json.dump(memory_stats, f, indent=2)
    
    def save_model_components(self):
        """Save all model components for deployment"""
        os.makedirs('./medical_model_production', exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained('./medical_model_production')
        
        # Save label mappings
        with open('./medical_model_production/label_mappings.json', 'w') as f:
            json.dump({
                'id_to_label': self.id_to_label,
                'label_to_id': {v: k for k, v in self.id_to_label.items()},
                'num_classes': len(self.id_to_label)
            }, f, indent=2)
        
        # Save model configuration
        model_config = {
            'model_name': self.model_name,
            'num_classes': len(self.id_to_label),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('./medical_model_production/config.json', 'w') as f:
            json.dump(model_config, f, indent=2)
        
        logger.info("Model components saved to ./medical_model_production/")
    

    def predict_top3_diseases(self, symptoms_text, temperature=1.0):
        """Enhanced prediction with temperature scaling"""
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
    
    # Default production settings
    default_settings = {
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'accumulation_steps': 2,
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
    # Example with your small dataset format
    logger.info("Starting medical disease classification system...")
    
    # For training (uncomment when ready)
    # classifier, test_acc, test_f1 = train_production_model('2.csv')
    
    # For inference only
    test_symptoms = [
        "Rash,Back pain,Chills,Low platelet,Weakness",
        "Noisy breathing,Exercise-induced symptoms,Tight chest,Night cough,Trouble exhaling",
        # "Severe throbbing headache with nausea and sensitivity to light",
        # "Persistent dry cough with fever and loss of taste",
        # "Runny nose, sneezing, and sore throat"
    ]
    
    # Uncomment when model is trained
    for i, symptoms in enumerate(test_symptoms, 1):
        print(f"\nTest Case {i}: {symptoms}")
        print("-" * 60)
        
        predictions = load_and_predict_production(symptoms)
        if predictions:
            for pred in predictions:
                print(f"Rank {pred['rank']}: {pred['disease']} ({pred['confidence_percentage']})")
    
    logger.info("System ready for production use!")

# resourse gap,methodology,tool