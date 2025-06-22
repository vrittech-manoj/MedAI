# Install necessary packages
!pip install -q transformers datasets torch scikit-learn pandas

# Imports
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model & tokenizer
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample dataset
sample_data = {
    'text': [
        # General (0)
        "The patient was diagnosed with type 2 diabetes mellitus and prescribed metformin.",
        "Hypertension management requires lifestyle modifications and medication adherence.",
        "Cardiovascular exercise reduces the risk of heart disease significantly.",
        "Proper sleep hygiene contributes to better overall health and mental clarity.",
        "Regular health checkups help in early disease detection and prevention.",
        "Nutritional counseling is essential for promoting lifelong healthy habits.",
        "Vaccination programs have improved public health outcomes globally.",
        "Primary care physicians play a key role in ongoing health maintenance.",
        # Acute Care (1)
        "Acute respiratory distress syndrome was observed in the ICU patient.",
        "COVID-19 symptoms include fever, cough, and difficulty breathing.",
        "The antibiotic treatment was effective against the bacterial infection.",
        "Emergency intubation was required due to severe respiratory failure.",
        "The trauma patient was stabilized and transferred to the surgical ICU.",
        "Rapid administration of epinephrine reversed the anaphylactic reaction.",
        "The ER team responded immediately to the cardiac arrest call.",
        "Sepsis protocol was initiated for the patient with high fever and hypotension.",
        # Research (2)
        "The study shows promising results for cancer immunotherapy treatments.",
        "Gene therapy shows potential for treating inherited genetic disorders.",
        "Clinical trials are ongoing for a new Alzheimer’s medication.",
        "Researchers identified novel biomarkers for early cancer detection.",
        "A new vaccine platform showed strong immune response in lab animals.",
        "Machine learning models were used to predict disease progression.",
        "The trial tested the efficacy of stem cell therapy in spinal injury patients.",
        # Chronic Care (3)
        "Alzheimer's disease progression can be slowed with early intervention.",
        "Mental health awareness is crucial for overall patient wellbeing.",
        "Long-term asthma control requires consistent medication use and monitoring.",
        "Regular dialysis sessions are essential for end-stage renal disease patients.",
        "Patients with rheumatoid arthritis benefit from biologic therapy.",
        "Chronic pain management often includes both physical and psychological support.",
        "Diabetes management includes glucose monitoring and dietary planning.",
        "Adherence to HIV therapy significantly improves long-term outcomes.",
    ],
    'label': [
        0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3
    ]
}

label_names = {0: "General", 1: "Acute Care", 2: "Research", 3: "Chronic Care"}

# Create dataset
df = pd.DataFrame(sample_data)
train_texts = df['text'].tolist()
train_labels = df['label'].tolist()

train_dataset = Dataset.from_dict({
    'text': train_texts,
    'labels': train_labels
})

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load model
num_labels = len(set(train_labels))
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={i: label_names[i] for i in range(num_labels)},
    label2id={label_names[i]: i for i in range(num_labels)}
)
model.to(device)

# Training arguments (old version compatible)
training_args = TrainingArguments(
    output_dir='./pubmedbert-classifier',
    num_train_epochs=10,
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
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train
print("Training started...")
trainer.train()
print("Training completed!")

# Prediction function
def predict_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

# Test predictions
test_texts = [
    "The patient underwent emergency cardiac catheterization for acute STEMI.",
    "Recent clinical trials demonstrate efficacy of novel immunotherapy approaches.",
    "Chronic kidney disease management requires multidisciplinary care coordination.",
    "The patient was diagnosed with type 2 diabetes mellitus and prescribed metformin.",
]

print("\nTest Predictions:")
print("-" * 50)
for text in test_texts:
    pred_class, confidence = predict_text(text, model, tokenizer)
    pred_label = label_names[pred_class]
    print(f"Text: {text}")
    print(f"Predicted: {pred_label} (confidence: {confidence:.3f})\n")



# ✅ Your Current Setup: Fine-Tuning Classification
# You're doing this:

# 🔍 Tokenizing biomedical text

# 🧠 Fine-tuning PubMedBERT (AutoModelForSequenceClassification)

# 🎯 Training on labeled examples: text → label

# 🧪 Using Trainer to optimize cross-entropy loss

# 🔁 Predicting new examples with softmax + class label

# MIMIC-III and MIMIC-II datasets