import asyncio
from ollama import AsyncClient

# async def chat_with_deepseek_async(model, messages):
#     async for part in await AsyncClient().chat(model=model, messages=messages, stream=True):
#         print(part['message']['content'], end='', flush=True)

# asyncio.run(chat_with_deepseek_async("deepseek-r1:7b", "who are you ?"))
# ollama pull deepseek-r1:1.5b

import asyncio
from ollama import AsyncClient


async def chat():
  messages = f"""
            "You are a best medical data assistant with have information. you have all the information on diseases,medical and health. Process the provided raw disease data into a CSV with the following columns:

            "id","Associated Disease","Disease Ontology Description","UniProt Description","Protein Count","Direct Association Count","Mondo ID","GARD Rare","Symbol","UniProt","Disease Data Source","JensenLab TextMining zscore","JensenLab Confidence","Expression Atlas Log2 Fold Change","DisGeNET Score","Associated Disease Evidence","Associated Disease Drug Name","Associated Disease P-value","Associated Disease Source","Associated Disease Source ID","Monarch S2O", "diseases_name", "symptoms_1", "symptoms_2", "symptoms_3", "symptoms_4", "symptoms_5", "symptoms_6", "symptoms_7", "symptoms_8", "symptoms_9", "description", "symptoms_description", "causes_description", "causes_1", "causes_2", "causes_3", "causes_4", "causes_5", "treatment_1", "treatment_2", "treatment_3", "prevention_1", "prevention_2", "prevention_3", "risk_factor", "age_of_onset", "genetic_factors", "family_history", "severity_of_disease", "diagnosis_methods", "complications", "epidemiology", "prognosis".

            For missing or unknown data, use NAN. Ensure each disease entry is processed as a row, with each field separated by commas. The CSV should contain no headers. If any data is not available, fill it with NAN.
            it is better if you fill all fields with best value. just only give me csv formatted data.
            Input: 10084,"oropharynx cancer","A pharynx cancer that is located_in the  oropharynx.",,35,35,"MONDO:0004608",1,"HLA-DRB1","P13761","CTD",,,,,"marker/mechanism",,,,"MESH:D009959",
            """
  message = {'role': 'user', 'content': messages}
  response = await AsyncClient().chat(model='deepseek-r1:1.5b', messages=[message])
  print(response)

asyncio.run(chat())
asyncio.run(chat())
asyncio.run(chat())