import requests
import json
import requests
import json
import pandas as pd
import time
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

URL = "https://api.scaleway.ai/cf98fe2a-a994-4401-aab9-54498913f51b/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer e5c0eb69-75e5-41ac-b0dc-42cecab58269" # Replace SCW_SECRET_KEY with your IAM API key
}

# Global Counters
success_num = 0
failure_num = 0
task_queue = Queue()

def load_data():
    dataframe = pd.read_csv("main_datasets.csv")
    for index, row in dataframe.iterrows():
        task_queue.put((index, ','.join(row.astype(str))))  # ✅ Save row index and data
    print(f"📊 Total Tasks Loaded: {task_queue.qsize()}")


def process_format(index, input_row):
    global success_num, failure_num  # ✅ Fix: Declare global variables
    max_retries = 5  # Maximum retry attempts
    delay = 5  # Start with 1 second delay
    message = f"""
            "You are a best medical data assistant with have information. you have all the information on diseases,medical and health. Process the provided raw disease data into a CSV with the following columns:

            "id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare","Symbol","UniProt","Disease Data Source","JensenLab TextMining zscore","JensenLab Confidence","Expression Atlas Log2 Fold Change","DisGeNET Score","Associated Disease Evidence","Associated Disease Drug Name","Associated Disease P-value","Associated Disease Source","Associated Disease Source ID","Monarch S2O", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis".

            For missing or unknown data, use NAN. Ensure each disease entry is processed as a row, with each field separated by commas. The CSV should contain no headers. If any data is not available, fill it with NAN.
            it is better if you fill all fields with best value. just only give me csv formatted data.
            Input: {input_row}
            """

    PAYLOAD = {
        "model": "llama-3.3-70b-instruct",
        "messages": [
                { "role": "system", "content": "You are a best medical data assistant with have information" },
                { "role": "user", "content": message },
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0,
            "stream": True,
        }
    print("going to loop")
    for attempt in range(max_retries):
        # try:
            print("in response ",attempt)
            time.sleep(1)
            response = requests.post(URL, headers=HEADERS, data=json.dumps(PAYLOAD), stream=False)
            print(response.status_code,"***********status code")
            if response.status_code == 429:
                print(f"🚨 429 Rate Limit Hit for Row {index}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue  # Retry the request
            response.raise_for_status()  # Raise exception for non-200 status codes
            for line in response.iter_lines():
                # print(line)
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line == "data: [DONE]":
                        print(f"❌ Data done: ")
                        continue
                    if decoded_line.startswith("data: "):
                        try:
                            data = json.loads(decoded_line[len("data: "):])
                            
                            if data.get("choices") and data["choices"][0]["delta"].get("content"):
                                # print(data["choices"][0]["delta"]["content"], end="")
                                            # Batch write to file (buffering for better performance)
                                csv_data = data["choices"][0]["delta"]["content"]
                                with open("uncleaned_data.csv", 'a', encoding="utf-8") as file_obj:
                                    file_obj.write(csv_data + "\n")

                                success_num += 1  # ✅ Fix: Now modifies the global variable
                                print(f"[✅ Success: Row {index} (Total: {success_num})] ")
                                
                                return  # Exit the function on success
                            else:
                                print("******",data,"*****************")
                        except json.JSONDecodeError as e:
                            print(f"❌ HTTP Error: {e}")
                            continue 
        # except:                   
        #     failure_num += 1  # ✅ Fix: Now modifies the global variable
        #     print(f"[❌ Error: Row {index} (Total: {failure_num})]:, Error:{e} ")

        #     # ✅ Save failed data to `errors.csv`
        #     with open("errors.csv", 'a', encoding="utf-8") as error_file:
        #         error_file.write(f"{index},{input_row}, ERROR:{e} \n")

            return  # Exit function on failure
 
        
        

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



if __name__ == "__main__":
    load_data()
    main(num_threads=1)  # ✅ Reduce threads to avoid rate limits




