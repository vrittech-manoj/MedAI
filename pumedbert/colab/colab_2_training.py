# Install necessary packages
!pip install -q transformers datasets torch scikit-learn pandas kagglehub

# Imports
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the Symptom2Disease dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "niyarrbarman/symptom2disease",
    file_path=""
)
print(f"Loaded dataset with {df.shape[0]} rows and {df['label'].nunique()} diseases")  # 1200 rows, 24 labels :contentReference[oaicite:1]{index=1}

# Map string labels to integer IDs
labels = sorted(df['label'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
df['label_id'] = df['label'].map(label2id)

# Prepare Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df[['text', 'label_id']].rename(columns={'label_id': 'labels'}))

# Load tokenizer
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

hf_dataset = hf_dataset.map(tokenize_function, batched=True)
hf_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load model
num_labels = len(labels)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./pubmedbert-symptom2disease',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    seed=42,
    fp16=torch.cuda.is_available(),
)

# Trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train
print("Training started…")
trainer.train()
print("Training completed!")

# Optional: compute basic metrics via cross-validation or train-eval split
metrics = trainer.evaluate()
print("Evaluation Results:", metrics)

# Prediction function
def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        cls = torch.argmax(probs, dim=-1).item()
        conf = probs[0][cls].item()
    return id2label[cls], conf

# Test predictions
test_texts = [
    "I have a red, scaly rash on my arms and scalp.",
    "Experiencing high fever, chills, and a severe headache.",
    "My joints are swollen and painful with stiffness."
]
print("\nTest Predictions:")
for t in test_texts:
    pred, conf = predict_text(t)
    print(f"Input: {t}\nPredicted: {pred} (confidence: {conf:.3f})\n")
