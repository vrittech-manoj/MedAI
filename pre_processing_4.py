# file_name = input("Enter file name#>>>...")
# print(file_name)
# file_path = "extracted_data/"+"1"+".csv"
# print(file_path)

header = ["id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis"]
print(f"total header:{len(header)}")

# bad_data_file_obj = open("errors/bad_data.csv","a")
with open("good_data/good.csv","r",encoding="utf-8") as file_obj:
    datas = file_obj.readlines()

for data in datas:
    splitted_data = data.split(',')
    # print(splitted_data)
    print(len(splitted_data))
    # break


