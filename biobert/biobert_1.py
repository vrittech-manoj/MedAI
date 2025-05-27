# Fine-tuned BioBERT for Disease Prediction
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. LOAD AND PREPARE YOUR DATASET
# Your CSV should have columns: disease, symptoms1, symptoms2, cause1, description
df = pd.read_csv('your_disease_dataset.csv')

# Combine all text features for each disease
def combine_disease_info(row):
    text = f"{row['symptoms1']} {row['symptoms2']} {row['cause1']} {row['description']}"
    return text.strip()

df['combined_text'] = df.apply(combine_disease_info, axis=1)

# 2. CREATE TRAINING DATASET
class DiseaseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 3. PREPARE DATA FOR TRAINING
# Create labels (disease name to ID mapping)
disease_to_id = {disease: idx for idx, disease in enumerate(df['disease'].unique())}
id_to_disease = {idx: disease for disease, idx in disease_to_id.items()}

df['disease_id'] = df['disease'].map(disease_to_id)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['combined_text'].values,
    df['disease_id'].values,
    test_size=0.2,
    random_state=42
)

# 4. LOAD PRE-TRAINED BIOBERT
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

# Add classification head
class BioBERTClassifier(torch.nn.Module):
    def __init__(self, biobert_model, num_classes):
        super().__init__()
        self.biobert = biobert_model
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_classes)  # 768 is BioBERT hidden size
    
    def forward(self, input_ids, attention_mask):
        outputs = self.biobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# Initialize model
num_classes = len(disease_to_id)
classifier = BioBERTClassifier(model, num_classes)

# 5. CREATE DATA LOADERS
train_dataset = DiseaseDataset(X_train, y_train, tokenizer)
test_dataset = DiseaseDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 6. TRAINING SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)

optimizer = AdamW(classifier.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# 7. TRAINING LOOP
def train_model(model, train_loader, optimizer, criterion, epochs=3):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(train_loader):.4f}')

# Train the model
print("Training Fine-tuned BioBERT...")
train_model(classifier, train_loader, optimizer, criterion, epochs=3)

# 8. PREDICTION FUNCTION
def predict_disease(symptoms_text, model, tokenizer, id_to_disease, top_k=3):
    model.eval()
    
    # Tokenize input
    encoding = tokenizer(
        symptoms_text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    results = []
    for i in range(top_k):
        disease_id = top_indices[0][i].item()
        disease_name = id_to_disease[disease_id]
        confidence = top_probs[0][i].item()
        
        results.append({
            'disease': disease_name,
            'confidence': confidence * 100  # Convert to percentage
        })
    
    return results

# 9. EXAMPLE USAGE
user_input = "I have fever and cough and feel tired"
predictions = predict_disease(user_input, classifier, tokenizer, id_to_disease)

print(f"\nUser Input: '{user_input}'")
print("Top Disease Predictions:")
for i, pred in enumerate(predictions, 1):
    print(f"{i}. {pred['disease']}: {pred['confidence']:.2f}% confidence")

# 10. SAVE THE FINE-TUNED MODEL
torch.save({
    'model_state_dict': classifier.state_dict(),
    'disease_to_id': disease_to_id,
    'id_to_disease': id_to_disease
}, 'finetuned_biobert_disease_classifier.pth')

print("\nModel saved as 'finetuned_biobert_disease_classifier.pth'")