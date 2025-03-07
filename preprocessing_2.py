import pandas as pd

df = pd.read_csv("dataset.csv")
# print(df)

# print(df.describe())
# print(df['Disease'].unique())
# Filter rows where Disease is 'Fungal infection'
# fungal_infection_rows = df[df["Disease"] == "GERD"].drop_duplicates()

# Display the result
# print(fungal_infection_rows)

# df = df.drop_duplicates()
# print(df)