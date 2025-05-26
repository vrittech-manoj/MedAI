from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=2)

# Example disease-symptom data
data = [
    {"disease": "Flu", "symptoms": "fever, chills, sore throat, muscle aches"},
    {"disease": "Diabetes", "symptoms": "frequent urination, increased thirst, fatigue"},
    {"disease": "Hypertension", "symptoms": "headache, dizziness, blurred vision"},
    {"disease": "Asthma", "symptoms": "shortness of breath, chest tightness, wheezing"},
    {"disease": "Migraine", "symptoms": "throbbing headache, nausea, sensitivity to light"},
]

# Symptom query
query = "fever,chills"

# Predict disease match
scores = []
for item in data:
    inputs = tokenizer(query, item["symptoms"], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prob = F.softmax(logits, dim=1)[0][1].item()  # Probability of being a match
        scores.append((item["disease"], prob))

# Sort diseases by predicted match probability
predicted = sorted(scores, key=lambda x: x[1], reverse=True)

# Show top prediction
print("Predicted disease:", predicted[0][0])
