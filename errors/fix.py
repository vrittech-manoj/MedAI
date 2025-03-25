
with open("one_error_data.csv","r", encoding="utf-8") as file_obj:
    rows = file_obj.readlines()

good = 0
fifty_four = 0
fifty_six = 0

for item in rows:
    my_row = item.split(",")
    print(len(my_row))
    if len(my_row) == 55:
        good = good+1
    if len(my_row) == 54:
        fifty_four = fifty_four+1
    if len(my_row) == 56:
        fifty_six = fifty_six+1

    input("pause")

print(good,fifty_four,fifty_six)
