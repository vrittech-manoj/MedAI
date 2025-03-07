

import requests
import json

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyB6gsoALkD0gEFba7OXCCSGgRUyO4GeYGs"
headers = {
    'Content-Type': 'application/json'
}



message = """
"You are a best medical data assistant with have information. you have all the information on diseases,medical and health. Process the provided raw disease data into a CSV with the following columns:

id, Associated Disease, Disease Ontology Description, UniProt Description, Protein Count, Direct Association Count, Mondo ID, GARD Rare, diseases_name, symptoms_1, symptoms_2, symptoms_3, symptoms_4, symptoms_5, symptoms_6, symptoms_7, symptoms_8, symptoms_9, description, symptoms_description, causes_description, causes_1, causes_2, causes_3, causes_4, causes_5, treatment_1, treatment_2, treatment_3, prevention_1, prevention_2, prevention_3, risk_factor, age_of_onset, genetic_factors, family_history, severity_of_disease, diagnosis_methods, complications, epidemiology, prognosis.

For missing or unknown data, use NAN. Ensure each disease entry is processed as a row, with each field separated by commas. The CSV should contain no headers. If any data is not available, fill it with NAN.
it is better if you fill all fields with best value. just only give me csv formatted data.
Input: 7,"16q24.3 microdeletion syndrome",,,6,6,"MONDO:0016838",1"
"""
data = {
    "contents": [{
        "parts": [{"text": message}]
    }]
}

# response = requests.post(url, headers=headers, data=json.dumps(data))
# data = response.json()['candidates'][0]['content']['parts'][0]['text']

# # data.replace
# data = data.replace("```csv", "")
# data = data.replace("```","")

# with open("uncleaned_data",'a') as file_obj:
#     file_obj.write(data)
