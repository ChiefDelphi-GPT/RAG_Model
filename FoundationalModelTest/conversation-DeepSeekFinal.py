# DeepSeek R1 - Reasoning conversational model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Available DeepSeek R1 model sizes (from smallest to largest):
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"    - 1.5B parameters (~3GB RAM)
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"      - 7B parameters (~14GB RAM) ← Recommended
# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"     - 8B parameters (~16GB RAM)
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"     - 14B parameters (~28GB RAM)
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"     - 32B parameters (~64GB RAM)
# "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"    - 70B parameters (~140GB RAM)
# "deepseek-ai/DeepSeek-R1"                       - 671B parameters (~1.3TB RAM - Main flagship model)

# Choose your model size based on available RAM:
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # Smallest - fastest loading
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"    # Good balance of quality and resource usage
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   # Alternative 8B option
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"   # Larger for better reasoning
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   # High-end consumer hardware
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # Very large - needs lots of RAM
# model_name = "deepseek-ai/DeepSeek-R1"                     # Main flagship - enterprise only (671B)

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
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map=None,  # Don't use auto device mapping
    trust_remote_code=True  # DeepSeek models may require this
).to(device)

# Your input message
input_text = "Hi how are you doing. What is ur name? What are your model specifications?"

# Format as a proper conversation using DeepSeek's chat template
messages = [
    {"role": "user", "content": input_text}
]

# Apply the chat template (DeepSeek models use similar format to Qwen)
try:
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
except:
    # Fallback if chat template not available
    formatted_input = f"User: {input_text}\nAssistant:"
start_time = time.time()

# Encode the formatted input and move to correct device
input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)

# Create attention mask to avoid the warning
attention_mask = torch.ones_like(input_ids).to(device)

# Generate response with parameters optimized for reasoning models
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1500,  # Increased for reasoning responses
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    )
# Extract and decode the response
response_ids = outputs[0][input_ids.shape[1]:]
response = tokenizer.decode(response_ids, skip_special_tokens=True)
end_time = time.time()
print("INPUT TEXT:")
print(input_text)
print()
print("MODEL RESPONSE:")
print(response.strip())
print()
print()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

# Optional: For continued conversation
# def chat_with_model(messages_history):
#     """Function to continue the conversation"""
#     try:
#         formatted_input = tokenizer.apply_chat_template(
#             messages_history,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#     except:
#         # Fallback formatting
#         conversation_text = ""
#         for msg in messages_history:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             conversation_text += f"{role}: {msg['content']}\n"
#         formatted_input = conversation_text + "Assistant:"
#     
#     input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)
#     attention_mask = torch.ones_like(input_ids).to(device)
#     
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=200,
#             temperature=0.7,
#             do_sample=True,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
#         )
#     
#     response_ids = outputs[0][input_ids.shape[1]:]
#     return tokenizer.decode(response_ids, skip_special_tokens=True).strip()

# Example of continued conversation (commented out):
# print("\n" + "="*50)
# print("CONTINUED CONVERSATION EXAMPLE:")
# conversation = [
#     {"role": "user", "content": "Hi how are you doing. What is ur name? What are your model specifications?"},
#     {"role": "assistant", "content": response.strip()},
#     {"role": "user", "content": "Can you solve this math problem: What is 127 * 83?"}
# ]
# 
# follow_up_response = chat_with_model(conversation)
# print("Follow-up question: Can you solve this math problem: What is 127 * 83?")
# print("Model response:", follow_up_response)