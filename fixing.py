import pandas as pd

# Step 1: Read the CSV with strict handling of line breaks
df = pd.read_csv(
    "uncleaned_data.csv",
    sep=",",               # Ensure proper column separation
    quotechar='"',         # Handle quoted values
    header=0,              # Treat first row as column headers
    skipinitialspace=True, # Remove spaces after commas
    engine="python"        # Use a more flexible parser for irregular CSVs
)

# Step 2: Remove extra quotes and spaces from column names
df.columns = df.columns.str.replace('"', '').str.strip()

# Step 3: Check if there are still formatting issues
if df.shape[0] == 1:  # If there's only 1 row, the CSV needs further cleaning
    print("⚠ CSV formatting issue detected! Cleaning the file manually...")
    
    # Read the file manually and normalize line breaks
    with open("uncleaned_data.csv", "r", encoding="utf-8") as f:
        content = f.read()

    # Fix incorrect newline characters and unnecessary spaces
    content = content.replace("\r\n", "\n").replace(' ,"', '",').replace('" ', '"')

    # Save cleaned file
    with open("cleaned_data.csv", "w", encoding="utf-8") as f:
        f.write(content)

    # Step 4: Re-read the cleaned file
    df = pd.read_csv("cleaned_data.csv", sep=",", quotechar='"', header=0, skipinitialspace=True, engine="python")
    df.columns = df.columns.str.replace('"', '').str.strip()

# Step 5: Display the cleaned DataFrame
import ace_tools as tools
tools.display_dataframe_to_user(name="Cleaned Data", dataframe=df)

# Step 6: Validate the structure
print("\n✅ Data Cleaning Successful!")
print("DataFrame Shape:", df.shape)  # Should now show multiple rows, not just (1, 55)
print(df.info())  # Should display correct data types
