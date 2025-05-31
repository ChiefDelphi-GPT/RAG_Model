# Enhanced Code Generation Models Template - 2025 Edition (Fixed Testing & Timing)
import os
# Set tokenizers parallelism to false to avoid fork issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import sys
import subprocess
import tempfile
import re
import json
import importlib.util
import traceback
from contextlib import redirect_stdout, redirect_stderr
import io

# Define multiple model options with their specifications
MODELS = {
    # DeepSeek Coder V2 - Top performing open-source model (2025)
    "deepseek_v2_lite": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    
    # DeepSeek Coder V1 - Reliable and well-tested
    "deepseek_v1": "deepseek-ai/deepseek-coder-6.7b-instruct",
    
    # Code Llama - Meta's specialized code model
    "code_llama": "codellama/CodeLlama-7b-Instruct-hf",
    
    # StarCoder2 - Latest version of the StarCoder series
    "starcoder2": "bigcode/starcoder2-7b",
    
    # Codestral - Mistral's code-specialized model
    "codestral": "mistralai/Codestral-22B-v0.1",
    
    # WizardCoder - Enhanced version with better instruction following
    "wizardcoder": "WizardLM/WizardCoder-Python-7B-V1.0",
    
    # Original CodeGen for comparison (your current model)
    "codegen_original": "Salesforce/codegen-350M-mono"
}

# Coding test problems with expected outputs and comprehensive test cases
CODING_TESTS = [
    {
        "name": "Fibonacci with Memoization",
        "prompt": "Write a Python function that computes the nth Fibonacci number recursively with memoization. You can name the function whatever you like (e.g., fibonacci, fib, etc.).",
        "test_cases": [
            {"input": [5], "expected": 5},
            {"input": [10], "expected": 55},
            {"input": [15], "expected": 610},
            {"input": [0], "expected": 0},
            {"input": [1], "expected": 1}
        ],
        "description": "Classic dynamic programming problem"
    },
    {
        "name": "Two Sum Problem",
        "prompt": "Write a Python function that finds two numbers in an array that add up to a target value. The function should take two parameters: a list/array of numbers and a target number. Return the indices of the two numbers as a list. You can name the function whatever you like.",
        "test_cases": [
            {"input": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"input": [[3, 2, 4], 6], "expected": [1, 2]},
            {"input": [[3, 3], 6], "expected": [0, 1]}
        ],
        "description": "Hash table/array problem"
    },
    {
        "name": "Valid Parentheses",
        "prompt": "Write a Python function 'is_valid(s)' that determines if the input string has valid parentheses. Consider '()', '[]', and '{}'. Return True if valid, False otherwise.",
        "test_cases": [
            {"input": ["()"], "expected": True},
            {"input": ["()[]{}"], "expected": True},
            {"input": ["(]"], "expected": False},
            {"input": ["([)]"], "expected": False},
            {"input": ["{[]}"], "expected": True}
        ],
        "description": "Stack-based problem"
    },
    {
        "name": "Binary Search",
        "prompt": "Write a Python function 'binary_search(nums, target)' that performs binary search on a sorted array. Return the index if found, -1 otherwise.",
        "test_cases": [
            {"input": [[1, 2, 3, 4, 5], 3], "expected": 2},
            {"input": [[1, 2, 3, 4, 5], 6], "expected": -1},
            {"input": [[1], 1], "expected": 0},
            {"input": [[], 1], "expected": -1}
        ],
        "description": "Divide and conquer algorithm"
    },
    {
        "name": "Maximum Subarray Sum",
        "prompt": "Write a Python function 'max_subarray(nums)' that finds the contiguous subarray with the largest sum using Kadane's algorithm. Return the maximum sum.",
        "test_cases": [
            {"input": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"input": [[1]], "expected": 1},
            {"input": [[5, 4, -1, 7, 8]], "expected": 23},
            {"input": [[-1]], "expected": -1}
        ],
        "description": "Dynamic programming - Kadane's algorithm"
    },
    {
        "name": "Merge Two Sorted Lists",
        "prompt": "Write a Python function 'merge_lists(list1, list2)' that merges two sorted arrays into one sorted array. Return the merged array.",
        "test_cases": [
            {"input": [[1, 2, 4], [1, 3, 4]], "expected": [1, 1, 2, 3, 4, 4]},
            {"input": [[], [1, 2]], "expected": [1, 2]},
            {"input": [[1], []], "expected": [1]},
            {"input": [[], []], "expected": []}
        ],
        "description": "Two-pointer technique"
    },
    {
        "name": "Palindrome Check",
        "prompt": "Write a Python function 'is_palindrome(s)' that checks if a string is a palindrome, ignoring case and non-alphanumeric characters. Return True if palindrome, False otherwise.",
        "test_cases": [
            {"input": ["A man, a plan, a canal: Panama"], "expected": True},
            {"input": ["race a car"], "expected": False},
            {"input": [""], "expected": True},
            {"input": ["a"], "expected": True}
        ],
        "description": "String manipulation with two pointers"
    }
]

# Base directory for temporary files
BASE_TEMP_DIR = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/FoundationalModelTest"

def extract_python_code(text):
    """Extract Python code from generated text, handling various formats"""
    text = text.strip()
    
    # Handle markdown code blocks
    if "```python" in text:
        pattern = r'```python\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    elif "```" in text:
        pattern = r'```\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # Look for function definitions and extract complete functions
    lines = text.split('\n')
    code_lines = []
    found_function = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Start of a function
        if line.strip().startswith('def '):
            found_function = True
            code_lines.append(line)
            i += 1
            
            # Continue collecting lines that belong to this function
            while i < len(lines):
                current_line = lines[i]
                # If it's indented or empty, it belongs to the function
                if (current_line.startswith(' ') or current_line.startswith('\t') or 
                    current_line.strip() == ''):
                    code_lines.append(current_line)
                # If we hit another function, start over
                elif current_line.strip().startswith('def '):
                    break
                # If we hit non-indented code that's not a function, we might be done
                elif current_line.strip() and not current_line.startswith(' ') and not current_line.startswith('\t'):
                    # Check if it looks like it could be part of the function (comments, etc.)
                    if current_line.strip().startswith('#'):
                        code_lines.append(current_line)
                    else:
                        break
                else:
                    code_lines.append(current_line)
                i += 1
        else:
            i += 1
    
    if found_function:
        return '\n'.join(code_lines)
    
    # Fallback - return everything if it looks like code
    if 'def ' in text:
        return text
    
    return text

def find_function_name(code):
    """Find the function name in the generated code"""
    lines = code.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('def '):
            # Extract function name - handle spaces around parentheses
            func_part = line.split('(')[0].replace('def ', '').strip()
            return func_part
    return None

def safe_execute_code(code, timeout=10):
    """Safely execute code with timeout and error handling"""
    try:
        # Create a restricted namespace
        namespace = {
            '__builtins__': {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'int': int,
                'str': str,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'float': float,
                'sorted': sorted,
                'reversed': reversed,
                'zip': zip,
                'any': any,
                'all': all,
                'ord': ord,
                'chr': chr,
                'isinstance': isinstance,
                'type': type,
                'print': print,  # Allow print for debugging
            }
        }
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)
        
        return namespace, None, stdout_capture.getvalue(), stderr_capture.getvalue()
        
    except Exception as e:
        return None, str(e), "", str(e)

def test_function_safely(code, function_name, test_cases):
    """Test the function safely with comprehensive error handling"""
    try:
        # Execute the code safely
        namespace, error, stdout, stderr = safe_execute_code(code)
        
        if error:
            return [f"❌ Code execution failed: {error}"], 0
        
        # Get the function from namespace
        if function_name not in namespace:
            available_funcs = [k for k in namespace.keys() if k.startswith('def ') or callable(namespace.get(k))]
            return [f"❌ Function '{function_name}' not found. Available: {available_funcs}"], 0
        
        func = namespace[function_name]
        
        if not callable(func):
            return [f"❌ '{function_name}' is not callable"], 0
        
        passed_tests = 0
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                args = test_case['input']
                expected = test_case['expected']
                
                # Call the function with the arguments
                if len(args) == 1:
                    result = func(args[0])
                elif len(args) == 2:
                    result = func(args[0], args[1])
                else:
                    result = func(*args)
                
                # Special handling for two_sum problem
                if (isinstance(expected, list) and len(expected) == 2 and 
                    all(isinstance(x, int) for x in expected) and
                    len(args) == 2 and isinstance(args[0], list)):
                    
                    # This looks like a two_sum problem
                    nums, target = args[0], args[1]
                    if (isinstance(result, list) and len(result) == 2 and
                        all(isinstance(x, int) for x in result) and
                        0 <= result[0] < len(nums) and 0 <= result[1] < len(nums) and 
                        nums[result[0]] + nums[result[1]] == target and result[0] != result[1]):
                        passed_tests += 1
                        results.append(f"✅ Test {i+1}: PASS")
                        continue
                
                # Regular comparison
                if result == expected:
                    passed_tests += 1
                    results.append(f"✅ Test {i+1}: PASS")
                else:
                    results.append(f"❌ Test {i+1}: FAIL - got {result}, expected {expected}")
                    
            except Exception as e:
                results.append(f"❌ Test {i+1}: ERROR - {type(e).__name__}: {str(e)[:100]}")
        
        return results, passed_tests
        
    except Exception as e:
        return [f"❌ Testing failed: {type(e).__name__}: {str(e)[:100]}"], 0

def load_model_safely(model_name):
    """Load model with better error handling and compatibility"""
    try:
        print(f"    Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"    Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            # Add compatibility options
            use_cache=True,
            attn_implementation="eager"  # Use eager attention to avoid cache issues
        )
        
        return tokenizer, model, None
        
    except Exception as e:
        return None, None, str(e)

def generate_code_safely(model, tokenizer, input_text, max_retries=3):
    """Generate code with better error handling and timing"""
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Tokenize input with attention mask
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            )
            
            # Move to device if using GPU
            if torch.cuda.is_available() and hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with optimized parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1500,  # Reduced for faster generation
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    use_cache=True
                )
            
            # Extract generated code
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            return generated_text, generation_time, None
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Generation attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
            else:
                return None, 0, str(e)
    
    return None, 0, "Max retries exceeded"

def test_code_model(model_name, model_key, output_file):
    """Test a specific code generation model with timing"""
    print(f"\n🧪 Testing {model_key.upper()}: {model_name}")
    
    # Load model
    tokenizer, model, load_error = load_model_safely(model_name)
    
    if load_error:
        error_msg = f"❌ {model_key} failed to load: {load_error}"
        print(f"  {error_msg}")
        output_file.write(f"{error_msg}\n")
        return {
            'success': False,
            'model_key': model_key,
            'error': load_error,
            'total_passed_tests': 0,
            'total_possible_tests': 0,
            'success_rate': 0.0,
            'total_generation_time': 0.0
        }
    
    model_results = []
    total_passed_tests = 0
    total_possible_tests = 0
    total_generation_time = 0.0
    
    # Test each coding problem
    for i, test_info in enumerate(CODING_TESTS):
        print(f"  📝 Test {i+1}: {test_info['name']}")
        
        # Different prompt styles for different models
        if "instruct" in model_name.lower() or "wizard" in model_name.lower():
            input_text = f"### Instruction:\n{test_info['prompt']}\n\n### Response:"
        else:
            input_text = f"# {test_info['prompt']}\n"
        
        # Generate code
        generated_text, generation_time, gen_error = generate_code_safely(model, tokenizer, input_text)
        total_generation_time += generation_time
        
        if gen_error:
            print(f"    ❌ Generation failed: {gen_error}")
            output_file.write(f"\n--- Test {i+1}: {test_info['name']} ---\n")
            output_file.write(f"❌ Generation failed: {gen_error}\n")
            continue
        
        print(f"    ⏱️  Generated in {generation_time:.2f}s")
        
        # Write detailed output to file
        output_file.write(f"\n--- Test {i+1}: {test_info['name']} ---\n")
        output_file.write(f"INPUT PROMPT:\n{input_text}\n\n")
        output_file.write(f"MODEL OUTPUT (Generated in {generation_time:.2f}s):\n{generated_text.strip()}\n\n")
        
        # Extract and test the generated code
        clean_code = extract_python_code(generated_text)
        
        if "def " not in clean_code:
            result_msg = "❌ No function definition found"
            print(f"    {result_msg}")
            output_file.write(f"{result_msg}\n")
            continue
        
        # Find function name
        function_name = find_function_name(clean_code)
        
        if not function_name:
            result_msg = "❌ Could not extract function name"
            print(f"    {result_msg}")
            output_file.write(f"{result_msg}\n")
            continue
        
        print(f"    🔍 Found function: {function_name}")
        
        # Test the generated code
        test_results, passed_tests = test_function_safely(clean_code, function_name, test_info['test_cases'])
        
        # Write test results to file
        output_file.write(f"EXTRACTED CODE:\n{clean_code}\n\n")
        output_file.write(f"TEST RESULTS:\n")
        for result in test_results:
            output_file.write(f"{result}\n")
        
        # Print concise results to console
        for result in test_results:
            print(f"    {result}")
        
        # Track results
        model_results.append({
            'test_name': test_info['name'],
            'passed_tests': passed_tests,
            'total_tests': len(test_info['test_cases']),
            'generation_time': generation_time
        })
        
        total_passed_tests += passed_tests
        total_possible_tests += len(test_info['test_cases'])
    
    # Calculate overall score for this model
    success_rate = (total_passed_tests / total_possible_tests) if total_possible_tests > 0 else 0
    
    print(f"  📊 Model Summary: {total_passed_tests}/{total_possible_tests} tests passed ({success_rate:.1%})")
    print(f"  ⏱️  Total generation time: {total_generation_time:.2f}s")
    
    # Write summary to file
    output_file.write(f"\nMODEL SUMMARY FOR {model_key.upper()}:\n")
    output_file.write(f"Total tests passed: {total_passed_tests}/{total_possible_tests}\n")
    output_file.write(f"Success rate: {success_rate:.2%}\n")
    output_file.write(f"Total generation time: {total_generation_time:.2f}s\n")
    output_file.write(f"Average generation time per test: {total_generation_time/len(CODING_TESTS):.2f}s\n\n")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    return {
        'success': True,
        'model_key': model_key,
        'total_passed_tests': total_passed_tests,
        'total_possible_tests': total_possible_tests,
        'success_rate': success_rate,
        'total_generation_time': total_generation_time,
        'results': model_results
    }

def main():
    """Test multiple code generation models"""
    # Ensure base directory exists
    os.makedirs(BASE_TEMP_DIR, exist_ok=True)
    
    output_file_path = os.path.join(BASE_TEMP_DIR, 'CodeGenModelOut.txt')
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        print("🚀 Code Generation Models Comparison - 2025 Edition (Fixed & Timed)")
        print(f"💾 Results will be saved to: {output_file_path}")
        
        header = "🚀 Code Generation Models Comparison - 2025 Edition (Fixed Testing & Timing)\n"
        header += f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Number of coding tests: {len(CODING_TESTS)}\n"
        header += f"PyTorch CUDA available: {torch.cuda.is_available()}\n"
        header += f"Temporary files directory: {BASE_TEMP_DIR}\n\n"
        
        output_file.write(header)
        
        # Test models in order of recommendation (best first)
        recommended_order = [
            "deepseek_v2_lite",  # Best overall performance
            "deepseek_v1",       # Reliable fallback
            "code_llama",        # Meta's solid option
            "starcoder2",        # Good multilingual support
            "wizardcoder",       # Enhanced instruction following
            "codegen_original"   # Your original model
        ]
        
        all_results = []
        total_test_time = time.time()
        
        for model_key in recommended_order:
            if model_key in MODELS:
                model_start_time = time.time()
                result = test_code_model(MODELS[model_key], model_key, output_file)
                model_end_time = time.time()
                
                result['total_test_time'] = model_end_time - model_start_time
                all_results.append(result)
                
                print(f"  🕐 Total time for {model_key}: {result['total_test_time']:.2f}s")
                
                # Add delay between model tests
                time.sleep(2)
        
        total_test_time = time.time() - total_test_time
        
        # Find the best performing model
        successful_results = [r for r in all_results if r['success']]
        
        if successful_results:
            # Sort by success rate, then by total passed tests
            best_model = max(successful_results, key=lambda x: (x['success_rate'], x['total_passed_tests']))
            
            print(f"\n🏆 WINNER: {best_model['model_key'].upper()}")
            print(f"📊 Success Rate: {best_model['success_rate']:.1%}")
            print(f"✅ Tests Passed: {best_model['total_passed_tests']}/{best_model['total_possible_tests']}")
            print(f"⏱️  Generation Time: {best_model['total_generation_time']:.2f}s")
            
            # Write final summary to file
            final_summary = f"\n{'='*60}\n"
            final_summary += "FINAL RESULTS\n"
            final_summary += f"{'='*60}\n"
            final_summary += f"WINNER: {best_model['model_key'].upper()}\n"
            final_summary += f"Success Rate: {best_model['success_rate']:.2%}\n"
            final_summary += f"Tests Passed: {best_model['total_passed_tests']}/{best_model['total_possible_tests']}\n"
            final_summary += f"Total Generation Time: {best_model['total_generation_time']:.2f}s\n"
            
            final_summary += f"\nALL MODEL RANKINGS:\n"
            sorted_results = sorted(successful_results, 
                                   key=lambda x: (x['success_rate'], x['total_passed_tests']), 
                                   reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                final_summary += f"{i}. {result['model_key']}: {result['success_rate']:.1%} ({result['total_passed_tests']}/{result['total_possible_tests']}) - {result['total_generation_time']:.2f}s\n"
            
            final_summary += f"\nTotal testing time: {total_test_time:.2f}s\n"
            final_summary += f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            output_file.write(final_summary)
            
            return best_model['model_key']
        else:
            print("❌ No models were successfully tested.")
            output_file.write("❌ No models were successfully tested.\n")
            return None

if __name__ == "__main__":
    start_time = time.time()
    best_model = main()
    end_time = time.time()
    
    if best_model:
        print(f"\n🎯 The best performing model is: {best_model.upper()}")
    else:
        print("\n❌ No successful model found.")
    
    print(f"🕐 Total execution time: {end_time - start_time:.2f} seconds")