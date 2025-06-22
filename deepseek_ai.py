import vllm
from vllm import LLM

# Load the model using vLLM
model = LLM(model="deepseek-ai/DeepSeek-V3")

# Set up the pipeline for text generation
prompt = "Enter your input text here"
output = model.generate(prompt)

print(output)
