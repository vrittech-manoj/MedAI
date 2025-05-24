import pandas as pd

# Sample data: 5 rows with Disease, Symptoms, and Causes
data = {
    "Disease": ["Flu", "Malaria", "Diabetes", "Asthma", "Migraine"],
    "Symptom1": ["Fever", "Fever", "Fatigue", "Shortness of breath", "Headache"],
    "Symptom2": ["Cough", "Chills", "Increased thirst", "Coughing", "Nausea"],
    "Symptom3": ["Body ache", "Sweating", "Frequent urination", "Wheezing", "Sensitivity to light"],
    "Cause1": ["Virus", "Parasite", "Insulin resistance", "Allergens", "Unknown"],
    "Cause2": ["Cold weather", "Mosquito bite", "Genetics", "Air pollution", "Stress"]
}

df = pd.DataFrame(data)
print(df)
