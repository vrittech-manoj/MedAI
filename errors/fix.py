
import pandas as pd

df = pd.read_csv("second_main_datasets.csv")

disease_name = '12q14 microdeletion syndrome'
result = df[df['Associated Disease'] == disease_name]
# result = df[df['Associated Disease'].str.contains(disease_name, case=False, na=False)]
# print(result)

with open("one_error_data.csv","r", encoding="utf-8") as file_obj:
    rows = file_obj.readlines()

good = 0
fifty_four = 0
fifty_six = 0

for item in rows:
    my_row = item.split(",")
    # for index,item in enumerate(my_row):
    #     print(index,item)
    print(len(my_row))
    result = df[df['Associated Disease'] == my_row[1]]
    if not result.empty:
        diseases_id = result.iloc[0]['id']
        my_row.insert(1,int(diseases_id))
        print(diseases_id)
    else:
        print("No exact match found.")
    last_occurance_diseases_name = len(my_row) - 1 - my_row[::-1].index(my_row[2])
    print(my_row, len(my_row),my_row[2],last_occurance_diseases_name)
   
    # print(diseases_id)
    if len(my_row) == 55:
        good = good+1
    if len(my_row) == 54:
        fifty_four = fifty_four+1
    if len(my_row) == 56:
        fifty_six = fifty_six+1

    input("pause")

print(good,fifty_four,fifty_six)
