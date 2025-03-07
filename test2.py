

import requests
import json
import pandas as pd

import asyncio
import threading
from queue import Queue
import time
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyB6gsoALkD0gEFba7OXCCSGgRUyO4GeYGs"
headers = {
    'Content-Type': 'application/json'
}
task_queue = Queue()  # Multi-threaded queue


def processing_queue():
    dataframe = pd.read_csv("main_datasets.csv")
    
    batch_size = 1000  # Batch size for insertion
    batch = []
    
    start_time = time.time()  # Start timer

    for index, row in dataframe.iterrows():
        input_row = ','.join(row.astype(str))
        batch.append(input_row)

        # Insert in batches
        if len(batch) >= batch_size:
            for item in batch:
                task_queue.put(item)  # Using non-async queue put
            batch.clear()

    # Insert remaining data in queue
    for item in batch:
        task_queue.put(item)

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    return

processing_queue()
print(task_queue.qsize())


class DataProcessing:
    def process_format(self,input_row):
        message = f"""
            "You are a best medical data assistant with have information. you have all the information on diseases,medical and health. Process the provided raw disease data into a CSV with the following columns:

            "id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare","Symbol","UniProt","Disease Data Source","JensenLab TextMining zscore","JensenLab Confidence","Expression Atlas Log2 Fold Change","DisGeNET Score","Associated Disease Evidence","Associated Disease Drug Name","Associated Disease P-value","Associated Disease Source","Associated Disease Source ID","Monarch S2O", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis".

            For missing or unknown data, use NAN. Ensure each disease entry is processed as a row, with each field separated by commas. The CSV should contain no headers. If any data is not available, fill it with NAN.
            it is better if you fill all fields with best value. just only give me csv formatted data.
            Input: {input_row}
            """
        data = {
                "contents": [{
                    "parts": [{"text": message}]
                }]
            }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        data = response.json()['candidates'][0]['content']['parts'][0]['text']

        # data.replace
        data = data.replace("```csv", "")
        data = data.replace("```","")

        with open("uncleaned_data",'a',encoding="utf-8", errors="ignore") as file_obj:
            # print(input_row,"\n")
            file_obj.write(data+"\n")


    def worker(self):
        """Worker thread to process queue items"""
        while not task_queue.empty():
            item = task_queue.get()
            self.process_format(item)
            task_queue.task_done()  # Mark task as done


    def main(self, num_threads=5):
        """Start multiple worker threads for processing"""
        print(f"🚀 Starting processing with {num_threads} threads...")
        threads = []

        for _ in range(num_threads):
            thread = threading.Thread(target=self.worker)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        print("✅ Processing completed!")

# Start multi-threaded processing
obj = DataProcessing()
obj.main(num_threads=10)  # Adjust number of threads based on system performance