#!/usr/bin/env python3
"""
BioBERT Fine-tuning for Biomedical Disease Data
Scalable sequence-by-sequence processing approach
Designed for easy conversion to OOP later
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AdamW, get_linear_schedule_with_warmup
)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import logging
from typing import Dict, List, Tuple, Optional
import gc
import time

# Setup logging for monitoring training progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters - centralized for easy tuning
CONFIG = {
    'model_name': 'dmis-lab/biobert-base-cased-v1.1',  # BioBERT model
    'max_length': 512,           # Maximum sequence length
    'batch_size': 8,             # Batch size (adjust based on GPU memory)
    'learning_rate': 2e-5,       # Learning rate for fine-tuning
    'num_epochs': 3,             # Number of training epochs
    'warmup_steps': 100,         # Warmup steps for learning rate scheduler
    'weight_decay': 0.01,        # Weight decay for regularization
    'gradient_accumulation_steps': 2,  # For larger effective batch size
    'max_grad_norm': 1.0,        # Gradient clipping
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_and_prepare_data(csv_path: str, sample_size: int = 4) -> pd.DataFrame:
    """
    Load CSV data and prepare it for BioBERT fine-tuning
    Focus on specified columns only
    """
    logger.info(f"Loading data from: {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Take only first 4 rows as requested
    df = df.head(sample_size)
    
    # Select only the specified columns
    columns_to_use = [
        "Associated Disease",
        "Disease Ontology Description", 
        "UniProt Description",  # Assuming this is the "Description" mentioned
        "symptoms_1",
        "symptoms_2", 
        "causes_1",
        "causes_2"
    ]
    
    # Check which columns exist in the dataframe
    existing_columns = [col for col in columns_to_use if col in df.columns]
    logger.info(f"Using columns: {existing_columns}")
    
    # Create processed dataframe
    processed_df = df[existing_columns].copy()
    
    # Handle missing values - fill with empty strings
    processed_df = processed_df.fillna('')
    
    logger.info(f"Data shape: {processed_df.shape}")
    logger.info(f"Sample data:\n{processed_df.head()}")
    
    return processed_df

def create_text_pairs_for_similarity(df: pd.DataFrame) -> List[Dict]:
    """
    Create text pairs for similarity learning
    Combines different aspects of disease information
    """
    text_pairs = []
    
    for idx, row in df.iterrows():
        # Create comprehensive disease description
        disease_info = f"{row.get('Associated Disease', '')} "
        disease_info += f"{row.get('Disease Ontology Description', '')} "
        disease_info += f"{row.get('UniProt Description', '')}"
        
        # Create symptoms description
        symptoms = []
        for i in [1, 2]:
            symptom = row.get(f'symptoms_{i}', '')
            if symptom and symptom.strip():
                symptoms.append(symptom.strip())
        symptoms_text = ". ".join(symptoms)
        
        # Create causes description
        causes = []
        for i in [1, 2]:
            cause = row.get(f'causes_{i}', '')
            if cause and cause.strip():
                causes.append(cause.strip())
        causes_text = ". ".join(causes)
        
        # Create different text pair combinations for training
        if disease_info.strip() and symptoms_text.strip():
            text_pairs.append({
                'text_a': disease_info.strip(),
                'text_b': symptoms_text,
                'label': 1,  # Related (disease-symptoms)
                'pair_type': 'disease_symptoms'
            })
        
        if disease_info.strip() and causes_text.strip():
            text_pairs.append({
                'text_a': disease_info.strip(),
                'text_b': causes_text,
                'label': 1,  # Related (disease-causes)
                'pair_type': 'disease_causes'
            })
        
        if symptoms_text.strip() and causes_text.strip():
            text_pairs.append({
                'text_a': symptoms_text,
                'text_b': causes_text,
                'label': 1,  # Related (symptoms-causes)
                'pair_type': 'symptoms_causes'
            })
    
    logger.info(f"Created {len(text_pairs)} text pairs for training")
    return text_pairs

def tokenize_text_pair(tokenizer, text_a: str, text_b: str, max_length: int = 512) -> Dict:
    """
    Tokenize a pair of texts for BioBERT input
    Returns tokenized inputs ready for model
    """
    # Tokenize the text pair
    encoded = tokenizer(
        text_a,
        text_b,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'].squeeze(),
        'attention_mask': encoded['attention_mask'].squeeze(),
        'token_type_ids': encoded['token_type_ids'].squeeze()
    }

class BioBERTSimilarityModel(torch.nn.Module):
    """
    BioBERT model for similarity/classification tasks
    Simple architecture for high accuracy
    """
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        
        # Load BioBERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.biobert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get BioBERT outputs
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

def process_batch_sequential(batch_data: List[Dict], tokenizer, model, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a batch of data sequentially (sequence by sequence)
    Memory efficient approach
    """
    batch_inputs = []
    batch_labels = []
    
    # Process each sequence in the batch
    for item in batch_data:
        # Tokenize the text pair
        tokenized = tokenize_text_pair(
            tokenizer, 
            item['text_a'], 
            item['text_b'], 
            CONFIG['max_length']
        )
        
        batch_inputs.append(tokenized)
        batch_labels.append(item['label'])
    
    # Stack inputs for batch processing
    input_ids = torch.stack([item['input_ids'] for item in batch_inputs]).to(device)
    attention_mask = torch.stack([item['attention_mask'] for item in batch_inputs]).to(device)
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch_inputs]).to(device)
    labels = torch.tensor(batch_labels).to(device)
    
    # Forward pass through model
    logits = model(input_ids, attention_mask, token_type_ids)
    
    return logits, labels

def train_model_sequential(model, train_data: List[Dict], val_data: List[Dict], tokenizer) -> Dict:
    """
    Train BioBERT model using sequential processing
    High accuracy focused training loop
    """
    device = CONFIG['device']
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_data) // CONFIG['batch_size'] * CONFIG['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training metrics
    training_stats = []
    best_val_accuracy = 0.0
    
    logger.info("Starting BioBERT fine-tuning...")
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        logger.info(f"{'='*50}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_true_labels = []
        
        # Process training data in batches sequentially
        for i in range(0, len(train_data), CONFIG['batch_size']):
            batch = train_data[i:i + CONFIG['batch_size']]
            
            # Sequential batch processing
            logits, labels = process_batch_sequential(batch, tokenizer, model, device)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_train_loss += loss.item()
            
            # Backward pass with gradient accumulation
            loss = loss / CONFIG['gradient_accumulation_steps']
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            
            # Optimizer step
            if (i // CONFIG['batch_size'] + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Store predictions for metrics
            predictions = torch.argmax(logits, dim=-1)
            train_predictions.extend(predictions.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())
            
            # Memory cleanup
            del logits, labels, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate training metrics
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        train_f1 = f1_score(train_true_labels, train_predictions, average='weighted')
        avg_train_loss = total_train_loss / (len(train_data) // CONFIG['batch_size'])
        
        logger.info(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for i in range(0, len(val_data), CONFIG['batch_size']):
                batch = val_data[i:i + CONFIG['batch_size']]
                
                # Sequential batch processing
                logits, labels = process_batch_sequential(batch, tokenizer, model, device)
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                # Store predictions
                predictions = torch.argmax(logits, dim=-1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
                
                # Memory cleanup
                del logits, labels, loss
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
        avg_val_loss = total_val_loss / (len(val_data) // CONFIG['batch_size'])
        
        logger.info(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_biobert_model.pth')
            logger.info(f"New best model saved! Validation accuracy: {best_val_accuracy:.4f}")
        
        # Store epoch statistics
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        })
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'training_stats': training_stats,
        'best_val_accuracy': best_val_accuracy
    }

def main_training_pipeline(csv_path: str):
    """
    Main training pipeline - scalable and sequential processing
    """
    logger.info("="*60)
    logger.info("BioBERT Fine-tuning Pipeline Started")
    logger.info("="*60)
    
    # Step 1: Load and prepare data
    logger.info("Step 1: Loading and preparing data...")
    df = load_and_prepare_data(csv_path, sample_size=4)
    
    # Step 2: Create text pairs for training
    logger.info("Step 2: Creating text pairs...")
    text_pairs = create_text_pairs_for_similarity(df)
    
    # Add some negative examples (unrelated pairs) for better training
    # For demonstration, we'll create some synthetic negative pairs
    negative_pairs = []
    for i in range(len(text_pairs)//2):  # Create half as many negative examples
        if i < len(text_pairs) - 1:
            negative_pairs.append({
                'text_a': text_pairs[i]['text_a'],
                'text_b': text_pairs[i+1]['text_b'],
                'label': 0,  # Unrelated
                'pair_type': 'negative'
            })
    
    # Combine positive and negative examples
    all_pairs = text_pairs + negative_pairs
    logger.info(f"Total training pairs: {len(all_pairs)} (Positive: {len(text_pairs)}, Negative: {len(negative_pairs)})")
    
    # Step 3: Split data
    logger.info("Step 3: Splitting data...")
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42, stratify=[p['label'] for p in all_pairs])
    
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")
    
    # Step 4: Initialize tokenizer and model
    logger.info("Step 4: Loading BioBERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = BioBERTSimilarityModel(CONFIG['model_name'], num_classes=2)
    
    logger.info(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Step 5: Train the model
    logger.info("Step 5: Starting model training...")
    start_time = time.time()
    
    results = train_model_sequential(model, train_pairs, val_pairs, tokenizer)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Step 6: Report results
    logger.info("="*60)
    logger.info("Training Completed!")
    logger.info("="*60)
    logger.info(f"Total training time: {training_time:.2f} seconds")
    logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    
    # Print detailed training statistics
    logger.info("\nDetailed Training Statistics:")
    logger.info("-" * 80)
    logger.info(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Train F1':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12}")
    logger.info("-" * 80)
    
    for stat in results['training_stats']:
        logger.info(f"{stat['epoch']:<8} {stat['train_loss']:<12.4f} {stat['train_accuracy']:<12.4f} "
                   f"{stat['train_f1']:<12.4f} {stat['val_loss']:<12.4f} {stat['val_accuracy']:<12.4f} {stat['val_f1']:<12.4f}")
    
    logger.info("-" * 80)
    logger.info("Model saved as: best_biobert_model.pth")
    logger.info("Ready for conversion to OOP architecture!")
    
    return model, tokenizer, results

# Example usage and testing
if __name__ == "__main__":
    # Configuration check
    logger.info("Configuration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    
    # Check device availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CUDA not available. Using CPU.")
    
    # Example run (uncomment to execute)
    # model, tokenizer, results = main_training_pipeline('your_dataset.csv')
    
    print("\n" + "="*60)
    print("BioBERT Fine-tuning Script Ready!")
    print("="*60)
    print("Key Features:")
    print("✓ Scalable sequence-by-sequence processing")
    print("✓ Memory efficient batch handling")
    print("✓ High accuracy focused architecture")
    print("✓ Comprehensive logging and monitoring")
    print("✓ Ready for OOP conversion")
    print("✓ BioBERT optimized for biomedical text")
    print("\nTo run: main_training_pipeline('your_csv_file.csv')")