# DialoGPT - Best CPU conversational model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load DialoGPT-medium (good balance of quality and speed)
model_name = "microsoft/DialoGPT-large"  # or "microsoft/DialoGPT-large" for better quality

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Your input message (conversation starter)
input_text = "Here is information: The capitol of France is Paris. What is the capital of Germany?"

# Encode input with special tokens for conversation
input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

# Generate response
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=60,
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

# Extract just the response
new_tokens = outputs[0][input_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print("INPUT TEXT:")
print(input_text)
print()
print("MODEL RESPONSE:")
print(response)