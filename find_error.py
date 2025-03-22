import pandas as pd

file_path = "uncleaned_data.csv"  # Replace with your file path

# Read file as raw text
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Get the expected number of columns from the header
expected_columns = len(lines[0].strip().split(","))

# Find incorrect rows
wrong_sn_values = []
print(f"🔍 Expected number of columns: {expected_columns}\n")

for i, line in enumerate(lines[1:], start=2):  # Start from line 2 (skip header)
    row_values = line.strip().split(",")
    actual_columns = len(row_values)

    if actual_columns != expected_columns:
        bad_sn = row_values[0]  # Extract 'sn' (first column)
        wrong_sn_values.append(bad_sn)
        print(f"⚠️ Bad Row {i}: SN={bad_sn} (Columns: {actual_columns} instead of {expected_columns})\n{line.strip()}\n")

# Print final list of bad SN values
print("\n🚨 **List of Wrong Format `sn` Values:**")
print(", ".join(wrong_sn_values) if wrong_sn_values else "✅ No formatting issues found.")
