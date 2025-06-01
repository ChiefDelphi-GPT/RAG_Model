# Qwen2.5-Coder model inference - Ask coding question - FIXED VERSION
# Available model sizes:
# - Qwen/Qwen2.5-Coder-0.5B-Instruct
# - Qwen/Qwen2.5-Coder-1.5B-Instruct  
# - Qwen/Qwen2.5-Coder-3B-Instruct
# - Qwen/Qwen2.5-Coder-7B-Instruct
# - Qwen/Qwen2.5-Coder-14B-Instruct
# - Qwen/Qwen2.5-Coder-32B-Instruct

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import time
import traceback

def ask_coding_question():
    # Load Qwen2.5-Coder model
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    print(f"Loading model: {model_name}")
    print("This may take a few minutes...")
    
    # Determine device (copied from tester)
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal)")
    else:
        device = "cpu"
        print("Using CPU")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token properly to avoid attention mask issues (from tester)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or "<pad>"
            tokenizer.pad_token_id = tokenizer.unk_token_id or 0
        
        # Load model with proper device handling (from tester)
        try:
            if device == "mps":
                # For MPS, load to CPU first then move to MPS
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                if device == "cpu":
                    model = model.to(device)
        except Exception as e:
            print(f"Error loading with float16, trying float32: {e}")
            # Fallback to float32 if float16 fails (from tester)
            if device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise
    
    # Coding question
    question = """How do you code the fibonacci function in Python? 
Please provide a clean, efficient implementation with comments explaining the approach."""
    
    # Format as instruction for Qwen (from tester)
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Provide clear, well-commented code examples."},
        {"role": "user", "content": question}
    ]
    
    try:
        # Apply chat template (from tester)
        formatted_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print("Question asked:", question)
        print("\n" + "="*50)
        print("MODEL GENERATING RESPONSE:")
        print("="*50)
        
        # Tokenize input with proper parameters (from tester)
        inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to the same device as model (from tester)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create text streamer for real-time token generation
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generate response with streaming (improved parameters from tester)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4000,
                temperature=0.3,  # Lower temperature for more consistent code
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                streamer=streamer
            )
        
        generation_time = time.time() - start_time
        
        # Extract and display the generated response
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print("\n" + "="*50)
        print(f"Generation completed in {generation_time:.2f} seconds")
        print(f"Generated {len(new_tokens)} tokens")
        print("="*50)
        
        return generated_response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error generating response: {e}"

if __name__ == "__main__":
    try:
        response = ask_coding_question()
        print(f"\nFull generated response:\n{response}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Make sure you have sufficient GPU memory and the transformers library installed.")
        print("You may need to use a smaller model size if you encounter memory issues.")