# Qwen2.5-Coder Advanced Test Suite - CHALLENGING PROBLEMS
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

class QwenAdvancedCodeTester:
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_dir = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/FoundationalModelTest/"
        
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
            "trie_autocomplete": {
                "prompt": "Write a Python class called 'TrieAutocomplete' that implements a trie data structure for autocompletion. Include methods: insert(word), search(word), starts_with(prefix), and get_all_words_with_prefix(prefix). get_all_words_with_prefix should return a list of all words that start with the given prefix.",
                "test_cases": [
                    (["hello", "help", "hero", "her", "world"], "he", ["hello", "help", "hero", "her"]),
                    (["apple", "app", "application"], "app", ["apple", "app", "application"]),
                    (["test"], "testing", []),
                    ([], "any", [])
                ]
            },
            "lru_cache": {
                "prompt": "Write a Python class called 'LRUCache' that implements a Least Recently Used cache with get(key) and put(key, value) methods. The cache should have a maximum capacity. When capacity is exceeded, remove the least recently used item. Both operations should be O(1).",
                "test_cases": [
                    # (capacity, operations, expected_results)
                    (2, [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2), ("get", 3), ("get", 1)], [None, None, 1, None, -1, 3, -1]),
                    (1, [("put", 2, 1), ("get", 2), ("put", 3, 2), ("get", 2), ("get", 3)], [None, 1, None, -1, 2])
                ]
            },
            "expression_evaluator": {
                "prompt": "Write a Python function called 'evaluate_expression' that takes a string representing a mathematical expression with +, -, *, /, parentheses, and handles operator precedence correctly. Return the result as a float. Handle division by zero by returning 'Error'.",
                "test_cases": [
                    ("2 + 3 * 4", 14.0),
                    ("(2 + 3) * 4", 20.0),
                    ("10 / 2 + 3", 8.0),
                    ("2 * (3 + 4) / 2", 7.0),
                    ("10 / 0", "Error"),
                    ("((1 + 2) * 3 - 4) / 2", 2.5)
                ]
            },
            "graph_shortest_path": {
                "prompt": "Write a Python function called 'dijkstra_shortest_path' that takes a weighted graph represented as a dictionary of adjacency lists (graph[node] = [(neighbor, weight), ...]) and returns the shortest distances from a start node to all other nodes. Return a dictionary where keys are nodes and values are shortest distances. Use float('inf') for unreachable nodes.",
                "test_cases": [
                    ({"A": [("B", 4), ("C", 2)], "B": [("D", 3)], "C": [("D", 1), ("E", 5)], "D": [("E", 1)], "E": []}, "A", {"A": 0, "B": 4, "C": 2, "D": 3, "E": 4}),
                    ({"1": [("2", 1)], "2": [("3", 2)], "3": [], "4": []}, "1", {"1": 0, "2": 1, "3": 3, "4": float('inf')})
                ]
            },
            "advanced_dp": {
                "prompt": "Write a Python function called 'longest_common_subsequence' that takes two strings and returns the length of their longest common subsequence using dynamic programming. Then write 'reconstruct_lcs' that returns the actual LCS string.",
                "test_cases": [
                    ("ABCDGH", "AEDFHR", 3, "ADH"),
                    ("AGGTAB", "GXTXAYB", 4, "GTAB"),
                    ("", "ABC", 0, ""),
                    ("ABC", "", 0, ""),
                    ("ABCDEF", "ABCDEF", 6, "ABCDEF")
                ]
            },
            "regex_matcher": {
                "prompt": "Write a Python function called 'regex_match' that implements basic regex matching supporting '.' (any character) and '*' (zero or more of preceding character). Return True if the entire string matches the pattern. Do not use the re module.",
                "test_cases": [
                    ("aa", "a", False),
                    ("aa", "a*", True),
                    ("ab", ".*", True),
                    ("aab", "c*a*b", True),
                    ("mississippi", "mis*is*p*.", False),
                    ("mississippi", "mis*is*ip*.", True)
                ]
            },
            "interval_scheduler": {
                "prompt": "Write a Python function called 'merge_intervals' that takes a list of intervals (each interval is [start, end]) and merges overlapping intervals. Return the merged intervals sorted by start time.",
                "test_cases": [
                    ([[1,3],[2,6],[8,10],[15,18]], [[1,6],[8,10],[15,18]]),
                    ([[1,4],[4,5]], [[1,5]]),
                    ([[1,2]], [[1,2]]),
                    ([], []),
                    ([[1,4],[0,4]], [[0,4]])
                ]
            },
            "thread_safe_counter": {
                "prompt": "Write a Python class called 'ThreadSafeCounter' that implements a thread-safe counter with methods: increment(), decrement(), get_value(), and reset(). Use threading.Lock to ensure thread safety. Include a method batch_operation(operations) that takes a list of operations ['inc', 'dec', 'reset'] and performs them atomically.",
                "test_cases": [
                    # Test basic operations
                    (["inc", "inc", "dec", "get"], 1),
                    (["inc", "reset", "get"], 0),
                    (["dec", "dec", "inc", "get"], -1),
                    (["batch", ["inc", "inc", "dec"], "get"], 1)
                ]
            },
            "serialize_deserialize_tree": {
                "prompt": "Write Python functions 'serialize_binary_tree' and 'deserialize_binary_tree' that can serialize a binary tree to a string and deserialize it back. Use a TreeNode class with attributes: val, left, right. Handle None nodes appropriately. The serialization format should be compact and unambiguous.",
                "test_cases": [
                    # Tree structure: TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
                    # Should serialize and deserialize correctly
                    ("manual_tree_test", True)  # Special test case handled separately
                ]
            },
            "memory_efficient_fibonacci": {
                "prompt": "Write a Python function called 'fibonacci_matrix' that computes the nth Fibonacci number using matrix exponentiation for O(log n) time complexity. Also write 'fibonacci_generator' that yields Fibonacci numbers indefinitely using O(1) space per number generated.",
                "test_cases": [
                    # (n, expected_fib_n)
                    (0, 0), (1, 1), (10, 55), (20, 6765), (50, 12586269025),
                    # Generator test: first 10 numbers
                    ("generator_test", [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
                ]
            }
        }
    
    def generate_code(self, problem_name, prompt):
        """Generate code for a given problem using the model"""
        
        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write clean, efficient, and well-structured code. Include all necessary imports. Provide complete implementations with proper error handling where appropriate."},
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
                    max_new_tokens=6000,  # Increased for more complex code
                    temperature=0.2,  # Lower temperature for more consistent code
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
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, generated_text, re.DOTALL)
                if matches:
                    return matches[0].strip()
            
            # If no code blocks found, return the entire text (it might be pure code)
            return generated_text.strip()
            
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
import threading
import time
from collections import defaultdict
import heapq

{clean_code}

# Test cases
def run_tests():
    test_results = []
    
"""
            
            # Add specific test cases for each problem
            if problem_name == "trie_autocomplete":
                test_script += """    for words, prefix, expected in test_cases:
        try:
            trie = TrieAutocomplete()
            for word in words:
                trie.insert(word)
            result = sorted(trie.get_all_words_with_prefix(prefix))
            expected_sorted = sorted(expected)
            test_results.append((f"TrieAutocomplete({words}).get_all_words_with_prefix('{prefix}')", result, expected_sorted, result == expected_sorted))
        except Exception as e:
            test_results.append((f"TrieAutocomplete test", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "lru_cache":
                test_script += """    for capacity, operations, expected in test_cases:
        try:
            cache = LRUCache(capacity)
            results = []
            for op in operations:
                if op[0] == "put":
                    results.append(cache.put(op[1], op[2]))
                elif op[0] == "get":
                    results.append(cache.get(op[1]))
            test_results.append((f"LRUCache({capacity}) operations", results, expected, results == expected))
        except Exception as e:
            test_results.append((f"LRUCache test", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "expression_evaluator":
                test_script += """    for expr, expected in test_cases:
        try:
            result = evaluate_expression(expr)
            is_correct = (result == expected) or (str(result) == str(expected))
            test_results.append((f"evaluate_expression('{expr}')", result, expected, is_correct))
        except Exception as e:
            test_results.append((f"evaluate_expression('{expr}')", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "graph_shortest_path":
                test_script += """    for graph, start, expected in test_cases:
        try:
            result = dijkstra_shortest_path(graph, start)
            test_results.append((f"dijkstra_shortest_path({start})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"dijkstra_shortest_path test", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "advanced_dp":
                test_script += """    for s1, s2, expected_len, expected_lcs in test_cases:
        try:
            length = longest_common_subsequence(s1, s2)
            try:
                actual_lcs = reconstruct_lcs(s1, s2)
                is_correct = (length == expected_len) and (actual_lcs == expected_lcs)
                test_results.append((f"LCS('{s1}', '{s2}')", (length, actual_lcs), (expected_len, expected_lcs), is_correct))
            except:
                # If reconstruct_lcs fails, just test length
                is_correct = length == expected_len
                test_results.append((f"LCS length('{s1}', '{s2}')", length, expected_len, is_correct))
        except Exception as e:
            test_results.append((f"LCS test", f"ERROR: {e}", (expected_len, expected_lcs), False))
"""
            elif problem_name == "regex_matcher":
                test_script += """    for text, pattern, expected in test_cases:
        try:
            result = regex_match(text, pattern)
            test_results.append((f"regex_match('{text}', '{pattern}')", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"regex_match('{text}', '{pattern}')", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "interval_scheduler":
                test_script += """    for intervals, expected in test_cases:
        try:
            result = merge_intervals(intervals)
            test_results.append((f"merge_intervals({intervals})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"merge_intervals({intervals})", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "thread_safe_counter":
                test_script += """    for operations, expected in test_cases:
        try:
            counter = ThreadSafeCounter()
            result = None
            for op in operations:
                if op == "inc":
                    counter.increment()
                elif op == "dec":
                    counter.decrement()
                elif op == "reset":
                    counter.reset()
                elif op == "get":
                    result = counter.get_value()
                elif op == "batch":
                    # Next item should be the batch operations
                    batch_ops = operations[operations.index(op) + 1]
                    counter.batch_operation(batch_ops)
            test_results.append((f"ThreadSafeCounter operations", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"ThreadSafeCounter test", f"ERROR: {e}", expected, False))
"""
            elif problem_name == "serialize_deserialize_tree":
                test_script += """    # Manual tree construction test
    try:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right
        
        # Create test tree: 1(2, 3(4, 5))
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.right.left = TreeNode(4)
        root.right.right = TreeNode(5)
        
        # Serialize and deserialize
        serialized = serialize_binary_tree(root)
        deserialized = deserialize_binary_tree(serialized)
        
        # Check if trees are equal (simple structure check)
        def trees_equal(t1, t2):
            if not t1 and not t2:
                return True
            if not t1 or not t2:
                return False
            return (t1.val == t2.val and 
                   trees_equal(t1.left, t2.left) and 
                   trees_equal(t1.right, t2.right))
        
        result = trees_equal(root, deserialized)
        test_results.append(("serialize/deserialize tree", result, True, result))
    except Exception as e:
        test_results.append(("serialize/deserialize tree", f"ERROR: {e}", True, False))
"""
            elif problem_name == "memory_efficient_fibonacci":
                test_script += """    # Test matrix fibonacci
    for n, expected in test_cases[:-1]:  # All except generator test
        try:
            result = fibonacci_matrix(n)
            test_results.append((f"fibonacci_matrix({n})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"fibonacci_matrix({n})", f"ERROR: {e}", expected, False))
    
    # Test generator
    try:
        gen = fibonacci_generator()
        generated = [next(gen) for _ in range(10)]
        expected = test_cases[-1][1]  # Last test case is generator test
        test_results.append(("fibonacci_generator first 10", generated, expected, generated == expected))
    except Exception as e:
        test_results.append(("fibonacci_generator", f"ERROR: {e}", expected, False))
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
                timeout=60  # Increased timeout for complex operations
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
        print(f"RUNNING ADVANCED TEST SUITE FOR MODEL: {self.model_name}")
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
        print(f"ADVANCED TEST SUMMARY FOR MODEL: {self.model_name}")
        print(f"{'='*80}")
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r["passed"])
        
        print(f"Total Advanced Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\nDetailed Results:")
        print("-" * 60)
        
        for problem, result in results.items():
            test_status = "✓" if result["passed"] else "✗"
            print(f"{test_status} {problem:<25} | {result['generation_time']:>6.2f}s")

def main():
    """Run advanced tests for Qwen models"""
    
    models_to_test = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'#'*100}")
        print(f"STARTING ADVANCED TESTS FOR MODEL: {model_name}")
        print(f"{'#'*100}")
        
        try:
            tester = QwenAdvancedCodeTester(model_name)
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
    print("FINAL ADVANCED TEST COMPARISON")
    print(f"{'='*100}")
    
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name}: ERROR - {results['error']}")
        else:
            passed = sum(1 for r in results.values() if r["passed"])
            total = len(results)
            print(f"{model_name}: {passed}/{total} advanced tests passed ({(passed/total)*100:.1f}%)")

if __name__ == "__main__":
    main()