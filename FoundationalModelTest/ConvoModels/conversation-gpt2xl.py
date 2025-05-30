# GPT-2 model inference
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load GPT-2 model
model_name = "gpt2-xl"  # You can also use "gpt2-medium", "gpt2-large", or "gpt2-xl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Your input text
input_text = "Make this into a more coherent and gramatically correct sentence: hi my name is joe, how are you today, tell me abt urself and how you are"

# Tokenize
inputs = tokenizer(input_text, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        tokenizer=tokenizer
    )

# Extract just the new part
new_tokens = outputs[0][inputs.input_ids.shape[1]:]
generated_only = tokenizer.decode(new_tokens, skip_special_tokens=True)
print("Input text:", input_text)
print("MODEL GENERATED:")
print(generated_only)