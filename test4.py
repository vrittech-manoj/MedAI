import requests
import json

URL = "https://api.scaleway.ai/cf98fe2a-a994-4401-aab9-54498913f51b/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer e5c0eb69-75e5-41ac-b0dc-42cecab58269" # Replace SCW_SECRET_KEY with your IAM API key
}

message = """"You are a best medical data assistant with have information. you have all the information on diseases,medical and health. Process the provided raw disease data into a CSV with the following columns:

            "id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare","Symbol","UniProt","Disease Data Source","JensenLab TextMining zscore","JensenLab Confidence","Expression Atlas Log2 Fold Change","DisGeNET Score","Associated Disease Evidence","Associated Disease Drug Name","Associated Disease P-value","Associated Disease Source","Associated Disease Source ID","Monarch S2O", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis".

            For missing or unknown data, use NAN. Ensure each disease entry is processed as a row, with each field separated by commas. The CSV should contain no headers. If any data is not available, fill it with NAN.
            it is better if you fill all fields with best value. just only give me csv formatted data.
            Input: 13,"18-Hydroxylase deficiency",,,1,1,,,"CYP11B2","P19099","CTD",,,,,"marker/mechanism",,,,"MESH:C537806","""

PAYLOAD = {
    "model": "llama-3.3-70b-instruct",
    "messages": [
            { "role": "system", "content": "You are a best diseases assistant" },
            { "role": "user", "content": message },
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0,
        "stream": False,
    }

response = requests.post(URL, headers=HEADERS, data=json.dumps(PAYLOAD), stream=False)

for line in response.iter_lines():
    print(line)
    if line:
        decoded_line = line.decode('utf-8').strip()
        if decoded_line == "data: [DONE]":
            break
        if decoded_line.startswith("data: "):
            try:
                data = json.loads(decoded_line[len("data: "):])
                print(data)
                if data.get("choices") and data["choices"][0]["delta"].get("content"):
                    print(data["choices"][0]["delta"]["content"], end="")
            except json.JSONDecodeError:
                continue