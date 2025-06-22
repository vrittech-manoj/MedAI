# ===============================
# GOOGLE COLAB MEDICAL DISEASE CLASSIFIER
# Complete code - just run cells in order
# ===============================

# CELL 1: Install required packages
!pip install transformers torch scikit-learn pandas numpy tqdm

# CELL 2: Import libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re
import warnings
from tqdm import tqdm
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CELL 3: Create sample dataset (replace with your CSV upload)
# Create sample medical data
sample_data = {
    'symptoms_1': ['high fever', 'severe headache', 'dry cough', 'runny nose', 'high fever'],
    'symptoms_2': ['muscle pain', 'nausea', 'fatigue', 'sore throat', 'stomach pain'],
    'symptoms_3': ['fatigue', 'light sensitivity', 'shortness of breath', 'sneezing', 'diarrhea'],
    'diseases_name': ['Dengue', 'Migraine', 'Pneumonia', 'Asthma', 'Typhoid']
}

# Create DataFrame
df = pd.DataFrame(sample_data)
print("Sample dataset created:")
print(df.head())

# CELL 4: Data preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s,.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def combine_symptoms(row):
    symptoms = []
    for col in row.index:
        if col.startswith('symptoms_') and pd.notna(row[col]):
            symptoms.append(str(row[col]).strip())
    return preprocess_text(', '.join(symptoms))

# Process data
df['combined_text'] = df.apply(combine_symptoms, axis=1)
print("\nProcessed data:")
print(df[['combined_text', 'diseases_name']].head())

# CELL 5: Dataset class
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

# CELL 6: Model definition
class MedicalClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3):
        super(MedicalClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# CELL 7: Prepare data for training
# Encode labels
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['diseases_name'])

# Create mappings
id_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
print(f"\nDisease mappings: {id_to_label}")

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['combined_text'].values, 
    df['labels'].values, 
    test_size=0.2, 
    random_state=42
)

print(f"\nData split - Train: {len(train_texts)}, Test: {len(test_texts)}")

# CELL 8: Initialize model and tokenizer
model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
tokenizer = AutoTokenizer.from_pretrained(model_name)

n_classes = len(label_encoder.classes_)
model = MedicalClassifier(n_classes)
model.to(device)

print(f"Model initialized with {n_classes} classes")

# CELL 9: Create data loaders
batch_size = 2  # Small batch for small dataset

train_dataset = MedicalDataset(train_texts, train_labels, tokenizer)
test_dataset = MedicalDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Data loaders created with batch size: {batch_size}")

# CELL 10: Training setup
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
epochs = 3

print("Training setup complete")

# CELL 11: Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

print("\nTraining completed!")

# CELL 12: Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'label_encoder': label_encoder,
    'id_to_label': id_to_label,
    'tokenizer_name': model_name
}, 'medical_model.pt', _use_new_zipfile_serialization=False)

print("Model saved as 'medical_model.pt'")

# CELL 13: Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    return accuracy, f1, all_predictions, all_labels

# CELL 14: Evaluate model
test_accuracy, test_f1, predictions, true_labels = evaluate_model(model, test_loader, device)

print(f"\nTest Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# Show detailed results
print(f"\nDetailed Results:")
for i, (pred, true) in enumerate(zip(predictions, true_labels)):
    pred_disease = id_to_label[pred]
    true_disease = id_to_label[true]
    print(f"Sample {i+1}: Predicted={pred_disease}, Actual={true_disease}")

# CELL 15: Prediction function
def predict_disease(symptoms_text, model, tokenizer, label_encoder, id_to_label, device):
    model.eval()
    
    # Preprocess text
    processed_text = preprocess_text(symptoms_text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
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
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, min(3, len(id_to_label)), dim=1)
        
        top3_probs = top3_probs.cpu().numpy()[0]
        top3_indices = top3_indices.cpu().numpy()[0]
    
    # Format results
    results = []
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
        disease = id_to_label[idx]
        confidence = float(prob)
        results.append({
            'rank': i + 1,
            'disease': disease,
            'confidence': f"{confidence*100:.2f}%"
        })
    
    return results

# CELL 16: Test predictions
test_symptoms = [
    "high fever and muscle pain",
    "severe headache with nausea",
    "dry cough and fatigue",
    "runny nose and sore throat"
]

print("\nTesting predictions:")
for i, symptoms in enumerate(test_symptoms):
    print(f"\nTest {i+1}: '{symptoms}'")
    predictions = predict_disease(symptoms, model, tokenizer, label_encoder, id_to_label, device)
    for pred in predictions:
        print(f"  {pred['rank']}. {pred['disease']} ({pred['confidence']})")

# CELL 17: Load model function (for future use)
def load_model(model_path):
    """Load saved model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
    
    # Load label encoder and mappings
    label_encoder = checkpoint['label_encoder']
    id_to_label = checkpoint['id_to_label']
    
    # Load model
    n_classes = len(label_encoder.classes_)
    model = MedicalClassifier(n_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, label_encoder, id_to_label

# CELL 18: Example of loading and using saved model
print("\n" + "="*50)
print("LOADING SAVED MODEL EXAMPLE")
print("="*50)

# Load the saved model
loaded_model, loaded_tokenizer, loaded_label_encoder, loaded_id_to_label = load_model('medical_model.pt')

# Test with the loaded model
test_symptom = "high fever and stomach pain"
print(f"\nTesting with loaded model: '{test_symptom}'")

prediction = predict_disease(
    test_symptom, 
    loaded_model, 
    loaded_tokenizer, 
    loaded_label_encoder, 
    loaded_id_to_label, 
    device
)

for pred in prediction:
    print(f"  {pred['rank']}. {pred['disease']} ({pred['confidence']})")

print("\n" + "="*50)
print("COMPLETE! Model trained, saved, and tested successfully!")
print("="*50)


# https://claude.ai/chat/80449a84-dfb3-49fe-bce6-49669ec439b9