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
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='deepseek-r1:1.5b', messages=[message])
  print(response)

asyncio.run(chat())
asyncio.run(chat())
asyncio.run(chat())