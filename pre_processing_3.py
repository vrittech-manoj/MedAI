# file_name = input("Enter file name#>>>...")
# print(file_name)
file_path = "extracted_data/"+"1"+".csv"
print(file_path)

header = ["sn","id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare","Symbol","UniProt","Disease Data Source","JensenLab TextMining zscore","JensenLab Confidence","Expression Atlas Log2 Fold Change","DisGeNET Score","Associated Disease Evidence","Associated Disease Drug Name","Associated Disease P-value","Associated Disease Source","Associated Disease Source ID","Monarch S2O", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis"]
print(f"total header:{len(header)}")

# print(header)

with open(file_path,"r") as file_obj:
    rows = file_obj.readlines()


    
def get_list(row):
    my_row = row.replace('","',"md5")
    my_row = my_row.replace('"','')
    my_row = my_row.replace(',','@')
    my_row = my_row.replace("md5",",")
    my_row = my_row.split(",")
    # print(my_row)
    return my_row

# print(header)
for row in rows:
    my_row = get_list(row)
    if len(my_row) == len(header):
        print(f"[✅ Success: Row {my_row[0]} , length:{len(my_row)}")
        if my_row[0] == '1':
            print("issues")
    else:
        print(f"[x bad: Row {my_row[0]},{my_row[1]}, length {len(my_row)}")


