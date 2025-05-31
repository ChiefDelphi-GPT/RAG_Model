# Qwen2.5-Coder model inference - Ask coding question
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

def ask_coding_question():
    # Load Qwen2.5-Coder model
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    print(f"Loading model: {model_name}")
    print("This may take a few minutes...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Coding question
    question = """How do you code the fibonacci function in Python? 
Please provide a clean, efficient implementation with comments explaining the approach."""
    
    # Format as instruction for Qwen
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Provide clear, well-commented code examples."},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("Question asked:", question)
    print("\n" + "="*50)
    print("MODEL GENERATING RESPONSE:")
    print("="*50)
    
    # Tokenize input
    inputs = tokenizer(formatted_input, return_tensors="pt")
    
    # Create text streamer for real-time token generation
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate response with streaming
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=4000,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            streamer=streamer
        )
    
    generation_time = time.time() - start_time
    
    # Extract and display the generated response
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print("\n" + "="*50)
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Generated {len(new_tokens)} tokens")
    print("="*50)
    
    return generated_response

if __name__ == "__main__":
    try:
        response = ask_coding_question()
        print(f"\nFull generated response:\n{response}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have sufficient GPU memory and the transformers library installed.")
        print("You may need to use a smaller model size if you encounter memory issues.")