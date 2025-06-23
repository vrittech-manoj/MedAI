

# bad_data_file_obj = open("errors/bad_data.csv","a")
with open("good_data/good.csv","r",encoding="utf-8") as file_obj:
    datas = file_obj.readlines()


import pandas as pd
# Load CSV file (adjust path as needed)
df = pd.read_csv("good_data/good.csv")
# print(df.head())
# print(df.columns)


# Strip whitespace from column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
# Fill missing with empty string or suitable default
# df.fillna('', inplace=True)
# print(df)


# Convert symptoms into list columns
symptom_cols = [f"symptoms_{i}" for i in range(1, 10)]
df["symptoms_list"] = df[symptom_cols].values.tolist()

# Same for causes or treatments
cause_cols = [f"causes_{i}" for i in range(1, 6)]
df["causes_list"] = df[cause_cols].values.tolist()

treatment_cols = [f"treatment_{i}" for i in range(1, 4)]
df["treatments_list"] = df[treatment_cols].values.tolist()

print(df.columns)

for _, row in df.iterrows():
    id = row['id']
    associated_disease = row['associated_disease']
    disease_ontology_description = row['disease_ontology_description']
    uniprot_description = row['uniprot_description']
    protein_count = row['protein_count']
    direct_association_count = row['direct_association_count']
    mondo_id = row['mondo_id']
    gard_rare = row['gard_rare']
    diseases_name = row['diseases_name']
    description = row['description']
    symptoms_description = row['symptoms_description']
    causes_description = row['causes_description']
    risk_factor = row['risk_factor']
    age_of_onset = row['age_of_onset']
    genetic_factors = row['genetic_factors']
    family_history = row['family_history']
    severity_of_disease = row['severity_of_disease']
    diagnosis_methods = row['diagnosis_methods']
    complications = row['complications']
    epidemiology = row['epidemiology']
    prognosis = row['prognosis']
    symptoms_list = row['symptoms_list']
    causes_list = row['causes_list']
    treatments_list = row['treatments_list']
    print(treatments_list)
    break
    # Now you can process each row as needed using these variables


