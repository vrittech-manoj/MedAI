# file_name = input("Enter file name#>>>...")
# print(file_name)
# file_path = "extracted_data/"+"1"+".csv"
# print(file_path)

header = ["sn","id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare","Symbol","UniProt","Disease Data Source","JensenLab TextMining zscore","JensenLab Confidence","Expression Atlas Log2 Fold Change","DisGeNET Score","Associated Disease Evidence","Associated Disease Drug Name","Associated Disease P-value","Associated Disease Source","Associated Disease Source ID","Monarch S2O", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis"]
# print(f"total header:{len(header)}")

# print(header)



    
def get_list(row):
    my_row = row.replace('",',"md5")
    my_row = my_row.replace('", ',"md5")
    my_row = my_row.replace('" ,"',"md5")
    my_row = my_row.replace('"','')
    my_row = my_row.replace(',','@')
    my_row = my_row.replace("md5",",")
    my_row = my_row.split(",")
    # print(my_row)
    return my_row

# def get_list(row):
#     my_row = row.replace('"',"md59")
#     my_row = my_row.replace('md59,','md59')
#     my_row = my_row.replace(',md59','md59')
#     my_row = my_row.replace('','md59')
#     return my_row

total_good = 0
total_bad = 0
total_data = 0
total_one_added_error = 0
total_fifty_six_data = 0
total_fifty_seven_data = 0

# bad_data_file_obj = open("errors/bad_data.csv","a")
good_file_object = open("extracted_data/good.csv","w",encoding="utf-8")
one_error_data_file_obj = open("errors/one_error_data.csv","w",encoding="utf-8")

for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]:
    file_path = "extracted_data/"+f"{num}"+".csv"
    with open(file_path,"r", encoding="utf-8") as file_obj:
        rows = file_obj.readlines()
    # print(header)
    good_data = 0
    bad_data = 0
    one_added_error = 0
    fifty_six_data = 0
    fifty_seven_data = 0
    for row in rows:
        my_row = get_list(row)
        if len(my_row) == len(header) and  my_row[0] != "sn" and my_row[0] != "1":
            good_data +=1
            good_file_object.write(','.join(my_row))
            # print(f"[✅ Success: Row {my_row[0]} , length:{len(my_row)}")
        else:
            
            if my_row[0] == "1":
                one_added_error = one_added_error + 1
                one_error_data_file_obj.write(','.join(my_row[1:]))
                # rows.remove(row)
                # print(f"[one added error : Row {my_row[0]},{my_row[1]}, length {len(my_row)}")
            elif len(my_row) == 56:
                fifty_six_data = fifty_six_data + 1
            elif len(my_row) == 57:
                fifty_seven_data = fifty_seven_data + 1
            else:
                # bad_data_file_obj.write(','.join(my_row))
                bad_data +=1
                # print(f"[x bad: Row {my_row[0]},{my_row[1]}, length {len(my_row)}")
    total_good = total_good + good_data
    total_data = total_data + len(rows)
    total_fifty_six_data = fifty_six_data +total_fifty_six_data
    total_one_added_error = total_one_added_error + one_added_error
    total_fifty_seven_data = total_fifty_seven_data + fifty_seven_data
    total_bad = total_bad+bad_data

    print(f"[✅ Success: total good: {good_data} ,one added error: {one_added_error} ,total fifty_six_data: {fifty_six_data} , total fifty_seven_data: {total_fifty_seven_data}, total bad:{bad_data}, total data:{len(rows)}")

print("*********************************************")
print(f"[✅ Success: total good: {total_good} ,total bad:{total_bad},total one added error:{total_one_added_error} ,total fifty six:{total_fifty_six_data},total fifty_seven_data:{total_fifty_seven_data} total data:{total_data}")

