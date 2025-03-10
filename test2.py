import requests
import json
import pandas as pd
import time
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# List of API keys to rotate
TOKEN_LIST = [
    # "AIzaSyB6gsoALkD0gEFba7OXCCSGgRUyO4GeYGs",
    "AIzaSyBL5tmxogE2gX_Ge5NaQGswt-c9m64ugh0",
    "AIzaSyBMwo8HaGSZK_r7Yf8DAQO6f0IIfKDyFPY",
    "AIzaSyBDn2M069ar1TV3nQd5Gl_jHDnCtw7Ut4s",
    "AIzaSyDZz7Xs0iiAVoVZ9_u2HuLsGMrtPe8X9s8",
    "AIzaSyBAhFSS4sOhXH8_pRm0Shp6kIs_FTPajY0",
    "AIzaSyAO69ZzDhcKa8JxZiPe9aRi6PIUwfxhqsk",
    "AIzaSyAhUvz6UVb2cT-p8Dl3An9OegDbrCs9BY0",
    "AIzaSyDHWMdntqQVq1r-BpGcyXQOY0e93E3NxrQ",
    "AIzaSyAJilBfiR2V1mXQ1JCaRIVA8oCLvkTCL6M",
    "AIzaSyAWz1kzRkrPOtwnqu_T-n90iSN1R3nbtxA",
    "AIzaSyDs1RC8XFWbNn4J_gm4wCn0Abx0iDy9Tr4",
    "AIzaSyC_G60jv63lXUyIQQpQMbB8sdqLt2quBJA",
    "AIzaSyAdPQwj4w_mxyj785z33BjHJy-oC0Hr8f8",
    "AIzaSyBbuFT6eE856dU1FCJMgaiPLasJwBgAFVU",
    "AIzaSyBluSDtYMdfkIB9yfgRL1v5XeU9Et0uJFw",
    "AIzaSyCj9QixCfsmEDArEFlsYaLH-gGKWBntqdQ",
    "AIzaSyCab5BIJn7hQ3gLUIh7Dsy35VAfHXXNXak",
    "AIzaSyBv_AJLMYrDhCSOcJGh3qiVVWZrpHIPCT8",
]

HEADERS = {'Content-Type': 'application/json'}

# Global Counters
success_num = 0
failure_num = 0
task_queue = Queue()

# Read and load dataset into queue
def load_data():
    dataframe = pd.read_csv("second_main_datasets.csv")
    for index, row in dataframe.iterrows():
        task_queue.put((index, ','.join(row.astype(str))))  # ✅ Save row index and data
    print(f"📊 Total Tasks Loaded: {task_queue.qsize()}")

# Function to process API request
def process_format(index, input_row):
    global success_num, failure_num  # ✅ Fix: Declare global variables
     # Load previous dataset to avoid duplicate entries
    if not os.path.exists("uncleaned_data.csv") or os.stat("uncleaned_data.csv").st_size == 0:
        print("⚠️ Error: The uncleaned_data.csv file is empty. No data to parse.")
    else:
        try:
            # input("pause")
            old_storage = pd.read_csv("uncleaned_data.csv", encoding="utf-8", dtype=str)
            # print(int(input_row.split(',')[0]),old_storage["sn"].values)
            if str(input_row.split(',')[0]) in old_storage["sn"].values:
                    print(f"[🔄 skipping: Row at {index})] save over hited")
                    return
            else:
                pass
      
        except pd.errors.EmptyDataError:
            print("⚠️ Error: No data found in uncleaned_data.csv")
        except pd.errors.ParserError:
            print("⚠️ Parsing Error: The file may be corrupted.")
    

    message = f"""
            "You are a best medical data assistant with have information. you have all the information on diseases,medical and health. Process the provided raw disease data into a CSV with the following columns:

            "sn","id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare","Symbol","UniProt","Disease Data Source","JensenLab TextMining zscore","JensenLab Confidence","Expression Atlas Log2 Fold Change","DisGeNET Score","Associated Disease Evidence","Associated Disease Drug Name","Associated Disease P-value","Associated Disease Source","Associated Disease Source ID","Monarch S2O", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis".

            For missing or unknown data, use NAN. The CSV should contain no headers. If any data is not available, fill it with NAN ,all value should be in double quotes "value" with commas seperated. and use @ in case of commas with in value. 
            it is better if you fill all fields with best describe , i will very happy if no any value have NAN . just only give me csv formatted data so that pandas can easly process.please only give me output as csv format as given rule, and panda can process it.
            Input: {input_row}
            """

    data = {"contents": [{"parts": [{"text": message}]}]}
    max_retries = 5  # Maximum retry attempts
    delay = 1  # Start with 1 second delay
    # input("go for next round yes no? #>>>")
    for attempt in range(max_retries):
        try:
            api_key = random.choice(TOKEN_LIST)  # ✅ Randomly choose an API key
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

            response = requests.post(url, headers=HEADERS, data=json.dumps(data))
            
            # If we get a 429 error, wait and retry with another key
            if response.status_code == 429:
                print(f"⚠️ Rate limit hit (attempt {attempt+1}/{max_retries}) with API key: {api_key}. Retrying in {delay} seconds with another key...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff (1s -> 2s -> 4s -> 8s)
                continue  # Retry with a different key
            
            response.raise_for_status()  # Raise exception for non-200 status codes

            # Try to extract CSV data
            csv_data = response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', None)

            # If CSV data is missing or empty, log error
            if not csv_data:
                raise ValueError("Invalid or empty response received from API")

            csv_data = csv_data.replace("```csv", "").replace("```", "").strip()

            # Batch write to file (buffering for better performance)
            with open("uncleaned_data.csv", 'a', encoding="utf-8") as file_obj:
                file_obj.write(csv_data + "\n")

            success_num += 1  # ✅ Fix: Now modifies the global variable
            print(f"[✅ Success: Row {index} (Total: {success_num})] using API key: {api_key}")
            time.sleep(0.5)  # ✅ Slow down API requests to prevent hitting rate limits
            return  # Exit the function on success

        except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as http_err:
            print(f"❌ HTTP Error: {http_err}")
            if response.status_code == 429:  # Rate limit handling
                continue  # Retry with a different key

        except Exception as e:
            failure_num += 1  # ✅ Fix: Now modifies the global variable
            print(f"[❌ Error: Row {index} (Total: {failure_num})]: {e}")

            # ✅ Save failed data to `errors.csv`
            with open("errors.csv", 'a', encoding="utf-8") as error_file:
                error_file.write(f"{index},{input_row}, ERROR: {str(e)}\n")

            return  # Exit function on failure

# Function to run processing using ThreadPoolExecutor
def main(num_threads=1):  # ✅ Reduce threads to avoid hitting rate limits
    """Start multiple worker threads for processing"""
    print(f"🚀 Starting processing with {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {executor.submit(process_format, index, row): (index, row) for index, row in list(task_queue.queue)}

        for future in as_completed(future_to_task):
            index, input_row = future_to_task[future]
            try:
                future.result()  # Wait for each thread to complete
            except Exception as e:
                print(f"❌ Error in thread execution for Row {index}: {e}")

                # ✅ Save failed data to `errors.csv`
                with open("errors.csv", 'a', encoding="utf-8") as error_file:
                    error_file.write(f"{index},{input_row}, ERROR: {str(e)}\n")

    print("✅ Processing completed!")

# Execute the script
if __name__ == "__main__":
    load_data()
    main(num_threads=2)  # ✅ Reduce threads to avoid rate limits
