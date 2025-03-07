import requests
import json

URL = "https://api.scaleway.ai/cf98fe2a-a994-4401-aab9-54498913f51b/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer SCW_SECRET_KEY" # Replace SCW_SECRET_KEY with your IAM API key
}

PAYLOAD = {
  "model": "llama-3.3-70b-instruct",
  "messages": [
        { "role": "system", "content": "You are a helpful assistant" },
		{ "role": "user", "content": "" },
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.7,
    "presence_penalty": 0,
    "stream": True,
}

response = requests.post(URL, headers=HEADERS, data=json.dumps(PAYLOAD), stream=True)

for line in response.iter_lines():
    if line:
        decoded_line = line.decode('utf-8').strip()
        if decoded_line == "data: [DONE]":
            break
        if decoded_line.startswith("data: "):
            try:
                data = json.loads(decoded_line[len("data: "):])
                if data.get("choices") and data["choices"][0]["delta"].get("content"):
                    print(data["choices"][0]["delta"]["content"], end="")
            except json.JSONDecodeError:
                continue