# file_name = input("Enter file name#>>>...")
# print(file_name)
# file_path = "extracted_data/"+"1"+".csv"
# print(file_path)

header = ["id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis"]
print(f"total header:{len(header)}")

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
total_fourty = 0
total_fourty_two = 0
total_fourty_three = 0

# bad_data_file_obj = open("errors/bad_data.csv","a")
good_file_object = open("good_data/good.csv","w",encoding="utf-8")
one_error_data_file_obj = open("errors/one_error_data.csv","w",encoding="utf-8")

for num in [2,3,4,5,6,7,8,9,"10_completed_10",11,12,13,14,15,16,17,18,19,"20_completed_second_phase",21,22,23,24,25,26,27,"28_next",29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]:
    file_path = "good_data/"+f"{num}"+".csv"
    with open(file_path,"r", encoding="utf-8") as file_obj:
        rows = file_obj.readlines()
    # print(header)
    good_data = 0
    bad_data = 0
    one_added_error = 0
    fourty_two = 0
    fourty_three = 0
    for row in rows:
        my_row = get_list(row)
        if len(my_row) == len(header):
            good_data +=1
            good_file_object.write(','.join(my_row))
            # print(f"[✅ Success: Row {my_row[0]} , length:{len(my_row)}")
        else:
            
            if len(my_row) == 40:
                one_added_error = one_added_error + 1
                one_error_data_file_obj.write(','.join(my_row[1:]))
                # rows.remove(row)
                # print(f"[one added error : Row {my_row[0]},{my_row[1]}, length {len(my_row)}")
            elif len(my_row) == 42:
                fourty_two = fourty_two + 1
            elif len(my_row) == 43:
                fourty_three = fourty_three + 1
            else:
                # bad_data_file_obj.write(','.join(my_row))
                bad_data +=1
                # print(f"[x bad: Row {my_row[0]},{my_row[1]}, length {len(my_row)}")
    total_good = total_good + good_data
    total_data = total_data + len(rows)
    total_fourty_two = fourty_two +total_fourty_two
    total_fourty = total_fourty + one_added_error
    total_fourty_three = total_fourty_three + fourty_three
    total_bad = total_bad+bad_data

    print(f"[✅ Success: total good: {good_data} ,one fourty error: {one_added_error} ,total fourty_two: {fourty_two} , total fourty_three: {total_fourty_three}, total bad:{bad_data}, total data:{len(rows)}")

print("*********************************************")
print(f"[✅ Success: total good: {total_good} ,total bad:{total_bad},total fourty error:{total_fourty} ,total fourty two:{total_fourty_two},total fourty_three:{total_fourty_three} total data:{total_data}")

