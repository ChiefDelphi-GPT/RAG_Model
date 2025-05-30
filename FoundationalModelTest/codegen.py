# Use a proper causal language model that works on CPU
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Try CodeGen - a proper causal LM for code generation
model_name = "Salesforce/codegen-350M-mono"  # Small code generation model

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Input for code generation
    input_text = "write me a function that computes the nth Fibonacci number in Python"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate with better parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=80,
            temperature=0.1,  # Very low temperature for deterministic code
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            tokenizer=tokenizer  # Add tokenizer for stop strings
        )
    
    # Decode and show only the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the new part
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_only = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("MODEL GENERATED:")
    print(generated_only)

except Exception as e:
    print(f"CodeGen failed: {e}")
    