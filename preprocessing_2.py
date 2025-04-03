import pandas as pd

df = pd.read_csv("main_datasets.csv")
# df.to_csv("second_main_datasets.csv", index=True)


# print(df)

# print(df.describe())
all_diseases = df["Associated Disease"].unique()
print(len(all_diseases))
input("pause")
# if "11-Beta-hydroxylase deficiency" in all_diseases:
#     print("match .")
# print(len(all_diseases))
# print(df['Disease'].unique())
# Filter rows where Disease is 'Fungal infection'
# fungal_infection_rows = df[df["Disease"] == "GERD"].drop_duplicates()

# Display the result
# print(fungal_infection_rows)

# df = df.drop_duplicates()
# print(df)