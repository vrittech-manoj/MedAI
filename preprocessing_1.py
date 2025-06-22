import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)

# # Read data from a CSV file
# # df = pd.read_csv('path/to/your/file.csv')
# # print("CSV DataFrame:")
# # print(df.head())

# # Basic operations
# print("Names:")
# print(df['Name'],"----")

# print("Filtered DataFrame (Age > 30):")
print(df[df['Age'] > 30],"------")

# # Adding a new column
df['Salary'] = [50000, 60000, 70000]
# print("DataFrame with Salary:")
print(df)

# # Descriptive statistics
# print("Descriptive statistics:")
print(df.describe())