# Qwen2.5-Coder Comprehensive Test Suite - FIXED VERSION
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
import os
import tempfile
import subprocess
import sys
import re
import traceback

class QwenCodeTester:
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_dir = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/FoundationalModelTest/varTest"
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        print(f"Loading model: {model_name}")
        print("This may take a few minutes...")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Using MPS (Metal)")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token properly to avoid attention mask issues
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token or "<pad>"
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id or 0
            
            # Load model with proper device handling
            try:
                if self.device == "mps":
                    # For MPS, load to CPU first then move to MPS
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                    self.model = self.model.to(self.device)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.device == "cuda" else None,
                        low_cpu_mem_usage=True
                    )
                    if self.device == "cpu":
                        self.model = self.model.to(self.device)
            except Exception as e:
                print(f"Error loading with float16, trying float32: {e}")
                # Fallback to float32 if float16 fails
                if self.device == "mps":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.model = self.model.to(self.device)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def get_coding_problems(self):
        return {
            "fibonacci": {
                "prompt": "Write a Python function called 'fibonacci' that takes an integer n and returns the nth Fibonacci number. Use an efficient approach.",
                "test_cases": [
                    (0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 8), (10, 55)
                ]
            },
            "two_sum": {
                "prompt": "Write a Python function called 'two_sum' that takes a list of integers and a target sum, and returns the indices of two numbers that add up to the target.",
                "test_cases": [
                    ([2, 7, 11, 15], 9, [0, 1]),
                    ([3, 2, 4], 6, [1, 2]),
                    ([3, 3], 6, [0, 1])
                ]
            },
            "valid_parentheses": {
                "prompt": "Write a Python function called 'is_valid_parentheses' that takes a string containing parentheses '(', ')', '{', '}', '[', ']' and returns True if they are valid (properly nested and closed).",
                "test_cases": [
                    ("()", True), ("()[]{}", True), ("(]", False), ("([)]", False), ("{[]}", True), ("", True)
                ]
            },
            "binary_search": {
                "prompt": "Write a Python function called 'binary_search' that takes a sorted list and a target value, and returns the index of the target (or -1 if not found).",
                "test_cases": [
                    ([1, 2, 3, 4, 5], 3, 2), ([1, 2, 3, 4, 5], 6, -1), ([1], 1, 0), ([], 1, -1)
                ]
            },
            "max_subarray": {
                "prompt": "Write a Python function called 'max_subarray_sum' that takes a list of integers and returns the maximum sum of any contiguous subarray (Kadane's algorithm).",
                "test_cases": [
                    ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),
                    ([1], 1),
                    ([5, 4, -1, 7, 8], 23),
                    ([-1], -1)
                ]
            },
            "merge_sorted_lists": {
                "prompt": "Write a Python function called 'merge_sorted_lists' that takes two sorted lists and returns a new sorted list containing all elements from both lists.",
                "test_cases": [
                    ([1, 2, 4], [1, 3, 4], [1, 1, 2, 3, 4, 4]),
                    ([], [], []),
                    ([], [0], [0]),
                    ([1], [], [1])
                ]
            },
            "palindrome_check": {
                "prompt": "Write a Python function called 'is_palindrome' that takes a string and returns True if it's a palindrome (reads the same forwards and backwards), ignoring case and non-alphanumeric characters.",
                "test_cases": [
                    ("A man a plan a canal Panama", True),
                    ("race a car", False),
                    ("", True),
                    ("Madam", True),
                    ("No 'x' in Nixon", True)
                ]
            }
        }
    
    def generate_code(self, problem_name, prompt):
        """Generate code for a given problem using the model"""
        
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Write clean, efficient Python code. Only provide the function implementation, no additional explanation or example usage."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            formatted_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            print(f"\n{'='*60}")
            print(f"GENERATING CODE FOR: {problem_name.upper()}")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"\nModel generating response...")
            print("-" * 40)
            
            inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True)
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4000,
                    temperature=0.3,  # Lower temperature for more consistent code
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    streamer=streamer
                )
            
            generation_time = time.time() - start_time
            
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_code = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            print(f"\n{'-'*40}")
            print(f"Generation completed in {generation_time:.2f} seconds")
            print(f"Generated {len(new_tokens)} tokens")
            
            return generated_code, generation_time
            
        except Exception as e:
            print(f"Error generating code: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"# Error generating code: {e}", 0.0
    
    def extract_function_code(self, generated_text):
        """Extract Python function code from generated text"""
        try:
            # Look for Python code blocks
            code_patterns = [
                r'```python\n(.*?)\n```',
                r'```\n(.*?)\n```',
                r'def\s+\w+.*?(?=\n\n|\n$|\Z)',
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, generated_text, re.DOTALL)
                if matches:
                    return matches[0].strip()
            
            # If no code blocks found, try to extract function definitions
            lines = generated_text.split('\n')
            code_lines = []
            in_function = False
            
            for line in lines:
                if line.strip().startswith('def '):
                    in_function = True
                if in_function:
                    code_lines.append(line)
                    # Stop if we encounter a line that's not indented and not a function definition
                    if (line.strip() and 
                        not line.startswith(' ') and 
                        not line.startswith('\t') and 
                        not line.strip().startswith('def ') and
                        len(code_lines) > 1):  # Make sure we have at least function definition + one line
                        break
            
            return '\n'.join(code_lines) if code_lines else generated_text
        except Exception as e:
            print(f"Error extracting function code: {e}")
            return generated_text
    
    def create_test_file(self, problem_name, function_code, test_cases):
        """Create a test file for the generated function"""
        
        try:
            # Extract just the function code
            clean_code = self.extract_function_code(function_code)
            
            # Create test script
            test_script = f"""# Generated code for {problem_name}
{clean_code}

# Test cases
def run_tests():
    test_results = []
    
"""
            
            # Add specific test cases for each problem
            if problem_name == "fibonacci":
                test_script += """    for n, expected in test_cases:
        try:
            result = fibonacci(n)
            test_results.append((f"fibonacci({n})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"fibonacci({n})", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "two_sum":
                test_script += """    for nums, target, expected in test_cases:
        try:
            result = two_sum(nums, target)
            # Check if result indices are correct (order doesn't matter)
            is_correct = (result == expected or result == expected[::-1]) if result else False
            test_results.append((f"two_sum({nums}, {target})", result, expected, is_correct))
        except Exception as e:
            test_results.append((f"two_sum({nums}, {target})", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "valid_parentheses":
                test_script += """    for s, expected in test_cases:
        try:
            result = is_valid_parentheses(s)
            test_results.append((f"is_valid_parentheses('{s}')", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"is_valid_parentheses('{s}')", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "binary_search":
                test_script += """    for arr, target, expected in test_cases:
        try:
            result = binary_search(arr, target)
            test_results.append((f"binary_search({arr}, {target})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"binary_search({arr}, {target})", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "max_subarray":
                test_script += """    for arr, expected in test_cases:
        try:
            result = max_subarray_sum(arr)
            test_results.append((f"max_subarray_sum({arr})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"max_subarray_sum({arr})", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "merge_sorted_lists":
                test_script += """    for list1, list2, expected in test_cases:
        try:
            result = merge_sorted_lists(list1, list2)
            test_results.append((f"merge_sorted_lists({list1}, {list2})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"merge_sorted_lists({list1}, {list2})", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "palindrome_check":
                test_script += """    for s, expected in test_cases:
        try:
            result = is_palindrome(s)
            test_results.append((f"is_palindrome('{s}')", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"is_palindrome('{s}')", f"ERROR: {e}", expected, False))
"""
            
            test_script += f"""
    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = {test_cases}
    try:
        results = run_tests()
        
        all_passed = True
        for test_name, result, expected, passed in results:
            status = "✓" if passed else "✗"
            print(f"{{status}} {{test_name}} -> {{result}} (expected: {{expected}})")
            if not passed:
                all_passed = False
        
        overall_status = 'PASS' if all_passed else 'FAIL'
        print(f"\\nOverall: {{overall_status}}")
        exit(0 if all_passed else 1)
    except Exception as e:
        print(f"Error running tests: {{e}}")
        print(f"Traceback: {{traceback.format_exc()}}")
        exit(1)
"""
            
            # Write to file
            file_path = os.path.join(self.base_dir, f"{problem_name}.py")
            with open(file_path, 'w') as f:
                f.write(test_script)
            
            return file_path
            
        except Exception as e:
            print(f"Error creating test file: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def run_test_file(self, file_path):
        """Run the test file and return results"""
        if file_path is None:
            return False, "", "Test file creation failed"
            
        try:
            result = subprocess.run(
                [sys.executable, file_path], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Test timed out"
        except Exception as e:
            return False, "", f"Error running test: {str(e)}"
    
    def run_full_test_suite(self):
        """Run the complete test suite"""
        problems = self.get_coding_problems()
        results = {}
        
        print(f"\n{'='*80}")
        print(f"RUNNING FULL TEST SUITE FOR MODEL: {self.model_name}")
        print(f"{'='*80}")
        
        for problem_name, problem_data in problems.items():
            print(f"\n{'*'*60}")
            print(f"TESTING PROBLEM: {problem_name.upper()}")
            print(f"{'*'*60}")
            
            try:
                # Generate code
                generated_code, gen_time = self.generate_code(problem_name, problem_data["prompt"])
                
                # Create test file
                test_file_path = self.create_test_file(
                    problem_name, 
                    generated_code, 
                    problem_data["test_cases"]
                )
                
                # Run tests
                passed, stdout, stderr = self.run_test_file(test_file_path)
                
                # Store results
                results[problem_name] = {
                    "passed": passed,
                    "generation_time": gen_time,
                    "generated_code": self.extract_function_code(generated_code),
                    "test_output": stdout,
                    "error_output": stderr
                }
                
                # Print test results
                test_status = "✓ PASS" if passed else "✗ FAIL"
                print(f"\nTEST RESULT: {test_status}")
                print(f"Generation Time: {gen_time:.2f} seconds")
                if stdout:
                    print(f"Test Output:\n{stdout}")
                if stderr:
                    print(f"Error Output:\n{stderr}")
                    
            except Exception as e:
                print(f"Error testing problem {problem_name}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                results[problem_name] = {
                    "passed": False,
                    "generation_time": 0.0,
                    "generated_code": "Error generating code",
                    "test_output": "",
                    "error_output": str(e)
                }
        
        return results
    
    def print_summary(self, results):
        """Print a summary of all test results"""
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY FOR MODEL: {self.model_name}")
        print(f"{'='*80}")
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r["passed"])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\nDetailed Results:")
        print("-" * 40)
        
        for problem, result in results.items():
            test_status = "✓" if result["passed"] else "✗"
            print(f"{test_status} {problem:<20} | {result['generation_time']:>6.2f}s")

def main():
    """Run tests for both 7B and 3B models sequentially"""
    
    models_to_test = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        #"Qwen/Qwen2.5-Coder-3B-Instruct"
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'#'*100}")
        print(f"STARTING TESTS FOR MODEL: {model_name}")
        print(f"{'#'*100}")
        
        try:
            tester = QwenCodeTester(model_name)
            results = tester.run_full_test_suite()
            tester.print_summary(results)
            all_results[model_name] = results
            
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            all_results[model_name] = {"error": str(e)}
        
        # Clear GPU memory between models
        try:
            if 'tester' in locals():
                del tester.model
                del tester.tokenizer
                del tester
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")
    
    # Final comparison
    print(f"\n{'='*100}")
    print("FINAL COMPARISON")
    print(f"{'='*100}")
    
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name}: ERROR - {results['error']}")
        else:
            passed = sum(1 for r in results.values() if r["passed"])
            total = len(results)
            print(f"{model_name}: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")

if __name__ == "__main__":
    main()