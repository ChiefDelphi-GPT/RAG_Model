# Qwen2.5-3B-Instruct - Much better conversational model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Qwen2.5-72B-Instruct (largest version - excellent conversation quality)
model_name = "Qwen/Qwen2.5-14B-Instruct" # you can use "Qwen/Qwen2.5-3B-Instruct", # "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"
# Determine device
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32  # MPS works better with float32
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map=None  # Don't use auto device mapping
).to(device)

# Your input message
input_text = "Hi how are you doing. What is ur name? What are your model specifications?"

# Format as a proper conversation using Qwen's chat template
messages = [
    {"role": "user", "content": input_text}
]

# Apply the chat template
formatted_input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Encode the formatted input and move to correct device
input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)

# Create attention mask to avoid the warning
attention_mask = torch.ones_like(input_ids).to(device)

# Generate response with better parameters for conversation
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.8,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Extract and decode the response
response_ids = outputs[0][input_ids.shape[1]:]
response = tokenizer.decode(response_ids, skip_special_tokens=True)

print("INPUT TEXT:")
print(input_text)
print()
print("MODEL RESPONSE:")
print(response.strip())

# Optional: For continued conversation
# def chat_with_model(messages_history):
#     """Function to continue the conversation"""
#     formatted_input = tokenizer.apply_chat_template(
#         messages_history,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     
#     input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)
#     attention_mask = torch.ones_like(input_ids).to(device)
#     
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=150,
#             temperature=0.7,
#             do_sample=True,
#             top_p=0.8,
#             repetition_penalty=1.1,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
#     
#     response_ids = outputs[0][input_ids.shape[1]:]
#     return tokenizer.decode(response_ids, skip_special_tokens=True).strip()

# Example of continued conversation (commented out):
# print("\n" + "="*50)
# print("CONTINUED CONVERSATION EXAMPLE:")
# conversation = [
#     {"role": "user", "content": "Here is information: The capitol of France is Paris. What is the capital of Germany?"},
#     {"role": "assistant", "content": response.strip()},
#     {"role": "user", "content": "What about Italy?"}
# ]
# 
# follow_up_response = chat_with_model(conversation)
# print("Follow-up question: What about Italy?")
# print("Model response:", follow_up_response)