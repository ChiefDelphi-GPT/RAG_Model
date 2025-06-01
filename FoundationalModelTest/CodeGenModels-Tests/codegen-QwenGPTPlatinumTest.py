# USACO Platinum Level Advanced Test Suite - IMPROVED VERSION
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
from typing import List, Tuple, Dict, Set
import random

class USACOPlatinumTester:
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_dir = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/FoundationalModelTest/varPlatinumTest"

        
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
    
    def get_usaco_platinum_problems(self):
        """Return USACO Platinum level problems - improved and clearer"""
        return {
            "shortest_path_dag": {
                "title": "Shortest Path in DAG with Modified Edges",
                "prompt": """
=== PROBLEM: Shortest Path in Directed Acyclic Graph ===

BACKGROUND:
You have a Directed Acyclic Graph (DAG) where you can modify edge weights.
Given a DAG with N vertices and M directed edges, find the shortest path from source to destination.
However, you can choose to halve the weight of exactly ONE edge on your path.

TASK:
Write a function shortest_path_with_discount(graph, source, destination) that returns the minimum possible distance from source to destination when you can halve exactly one edge weight on the chosen path.

INPUT FORMAT:
- graph: Dictionary where graph[u] = [(v, weight), ...] representing edges from u to v with given weight
- source: Starting vertex (integer)
- destination: Target vertex (integer)

OUTPUT:
- Return the minimum distance as a float/int

ALGORITHM HINT:
1. This is a DAG, so you can use topological sorting
2. For each possible edge to discount, calculate the shortest path
3. Use dynamic programming on the topologically sorted vertices

EXAMPLE:
Graph: {0: [(1, 10), (2, 20)], 1: [(3, 5)], 2: [(3, 3)], 3: []}
Source: 0, Destination: 3
- Path 0→1→3 costs 10+5=15, with discount on edge (0,1): 5+5=10, or discount on (1,3): 10+2.5=12.5
- Path 0→2→3 costs 20+3=23, with discount on edge (0,2): 10+3=13, or discount on (2,3): 20+1.5=21.5
- Best result: 10 (discount the 0→1 edge)

CONSTRAINTS:
- 2 ≤ N ≤ 1000
- 1 ≤ M ≤ 5000
- All edge weights are positive integers ≤ 100
- Graph is guaranteed to be a DAG
- Path from source to destination is guaranteed to exist
""",
                "test_cases": [
                    # (graph, source, destination, expected_min_distance)
                    (
                        {0: [(1, 10), (2, 20)], 1: [(3, 5)], 2: [(3, 3)], 3: []},
                        0, 3, 10.0
                    ),
                    (
                        {0: [(1, 8)], 1: [(2, 12)], 2: []},
                        0, 2, 14.0  # 8/2 + 12 = 16, or 8 + 12/2 = 14
                    ),
                    (
                        {0: [(1, 6), (2, 4)], 1: [(2, 2), (3, 8)], 2: [(3, 10)], 3: []},
                        0, 3, 9.0  # Path 0→1→3 with discount on (1,3): 6 + 8/2 = 10, or 0→2→3 with discount: 4/2 + 10 = 12, or 4 + 10/2 = 9
                    )
                ]
            },
            
            "binary_lifting_lca": {
                "title": "Lowest Common Ancestor with Binary Lifting",
                "prompt": """
=== PROBLEM: Lowest Common Ancestor Queries ===

BACKGROUND:
Given a rooted tree, answer multiple queries asking for the Lowest Common Ancestor (LCA) of two nodes.
Use Binary Lifting technique for O(log N) query time after O(N log N) preprocessing.

TASK:
Implement a class TreeLCA that supports:
- __init__(self, tree, root): Initialize with tree structure and root node
- lca(self, u, v): Return the LCA of nodes u and v

ALGORITHM EXPLANATION:
Binary Lifting precomputes ancestors at distances 2^0, 2^1, 2^2, ... for each node.
- parent[node][i] = ancestor of 'node' at distance 2^i
- To find LCA, first make both nodes at same depth, then lift them together

DETAILED STEPS:
1. Run DFS to compute depth of each node
2. Build binary lifting table: parent[node][i] = parent[parent[node][i-1]][i-1]
3. For LCA query:
   a) Make both nodes at same depth by lifting the deeper one
   b) If they're the same, return it
   c) Otherwise, lift both simultaneously until their parents become the same

INPUT FORMAT:
- tree: Dictionary where tree[u] = [v1, v2, ...] (undirected edges)
- root: Root node of the tree
- All nodes are integers starting from 0

EXAMPLE:
Tree: {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]}
Root: 0
Tree structure:
    0
   / \\
  1   2
 /|   |
3 4   5

Queries:
- lca(3, 4) = 1 (parent of both 3 and 4)
- lca(3, 5) = 0 (must go up to root)
- lca(1, 2) = 0 (siblings under root)

CONSTRAINTS:
- 1 ≤ N ≤ 100,000
- Tree is connected
- All queries should run in O(log N) time
""",
                "test_cases": [
                    # (tree, root, queries, expected_results)
                    (
                        {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]},
                        0,
                        [(3, 4), (3, 5), (1, 2), (0, 5), (3, 3)],
                        [1, 0, 0, 0, 3]
                    ),
                    (
                        {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]},  # Path tree
                        0,
                        [(4, 0), (3, 1), (4, 2), (1, 4)],
                        [0, 1, 2, 1]
                    ),
                    (
                        {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]},  # Star tree
                        0,
                        [(1, 2), (1, 3), (2, 3), (0, 1)],
                        [0, 0, 0, 0]
                    )
                ]
            },
            
            "segment_tree": {
                "title": "Range Sum Query with Point Updates",
                "prompt": """
=== PROBLEM: Range Sum Queries with Updates ===

BACKGROUND:
You have an array of N integers. Support two operations:
1. Update a single element at position i to value x
2. Query the sum of elements in range [left, right] (inclusive)

TASK:
Implement a class SegmentTree that supports:
- __init__(self, arr): Initialize with array
- update(self, index, value): Set arr[index] = value
- query(self, left, right): Return sum of arr[left:right+1]

ALGORITHM EXPLANATION:
Segment Tree is a binary tree where:
- Each leaf represents one array element
- Each internal node represents the sum of its children
- Both operations work in O(log N) time

IMPLEMENTATION GUIDE:
1. Build tree: tree[node] = tree[2*node] + tree[2*node+1]
2. Update: modify leaf and propagate changes upward
3. Query: combine results from relevant segments

DETAILED STRUCTURE:
- Use array of size 4*N to store tree
- Node i has children at 2*i and 2*i+1
- For array of size N, leaves start at index N

EXAMPLE:
Array: [1, 3, 5, 7, 9, 11]
Initial queries:
- query(0, 2) = 1+3+5 = 9
- query(1, 4) = 3+5+7+9 = 24

After update(1, 10):  # Change arr[1] from 3 to 10
- query(0, 2) = 1+10+5 = 16
- query(1, 4) = 10+5+7+9 = 31

CONSTRAINTS:
- 1 ≤ N ≤ 100,000
- All values fit in 32-bit integers
- All indices are 0-based
""",
                "test_cases": [
                    # (initial_array, operations, expected_results)
                    (
                        [1, 3, 5, 7, 9, 11],
                        [("query", 0, 2), ("query", 1, 4), ("update", 1, 10), ("query", 0, 2), ("query", 1, 4)],
                        [9, 24, None, 16, 31]
                    ),
                    (
                        [2, 4, 6, 8],
                        [("query", 0, 3), ("update", 2, 0), ("query", 0, 3), ("query", 2, 3), ("update", 0, 5)],
                        [20, None, 14, 8, None]
                    ),
                    (
                        [10],
                        [("query", 0, 0), ("update", 0, 20), ("query", 0, 0)],
                        [10, None, 20]
                    )
                ]
            }
        }
    
    def generate_code(self, problem_name, problem_data):
        """Generate code for a given USACO Platinum problem using the model"""
        
        messages = [
            {"role": "system", "content": """You are an expert competitive programmer. Focus on implementing clean, correct solutions.

For the given problem:
1. Read the problem statement carefully
2. Understand the algorithm hints provided
3. Implement the required function/class exactly as specified
4. Use standard algorithms and data structures
5. Handle all edge cases properly
6. Write efficient code suitable for competitive programming

Your response should contain ONLY the implementation code - no explanations, no markdown formatting, just clean Python code that can be executed directly."""},
            {"role": "user", "content": f"**{problem_data['title']}**\n\n{problem_data['prompt']}\n\nImplement the required function/class with the exact signature specified in the problem."}
        ]
        
        try:
            formatted_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            print(f"\n{'='*80}")
            print(f"GENERATING CODE FOR: {problem_data['title'].upper()}")
            print(f"{'='*80}")
            print(f"This is a USACO Platinum level problem - extremely challenging!")
            print(f"\nModel generating response...")
            print("-" * 50)
            
            inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True)
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4000,  # Reasonable limit for focused solutions
                    temperature=0.1,  # Very low temperature for precision
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
            
            print(f"\n{'-'*50}")
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
    
    def create_test_file(self, problem_name, problem_data, function_code, test_cases):
        """Create a test file for the generated function"""
        
        try:
            # Extract just the function code
            clean_code = self.extract_function_code(function_code)
            
            # Create test script
            test_script = f"""# Generated code for USACO Platinum: {problem_data['title']}
import sys
import heapq
from collections import defaultdict, deque
import math
from typing import List, Dict, Set, Tuple, Optional

{clean_code}

# Test cases
def run_tests():
    test_results = []
    
"""
            
            # Add specific test cases for each problem
            if problem_name == "shortest_path_dag":
                test_script += """    for graph, source, destination, expected in test_cases:
        try:
            result = shortest_path_with_discount(graph, source, destination)
            # Allow small floating point differences
            passed = abs(result - expected) < 1e-6
            test_results.append((f"Shortest path DAG: {source}→{destination}", result, expected, passed))
        except Exception as e:
            test_results.append((f"Shortest path DAG test", f"ERROR: {e}", expected, False))
"""
            
            elif problem_name == "binary_lifting_lca":
                test_script += """    for tree, root, queries, expected_results in test_cases:
        try:
            lca_solver = TreeLCA(tree, root)
            results = []
            for u, v in queries:
                result = lca_solver.lca(u, v)
                results.append(result)
            
            test_results.append((f"Binary Lifting LCA", results, expected_results, results == expected_results))
        except Exception as e:
            test_results.append((f"Binary Lifting LCA test", f"ERROR: {e}", expected_results, False))
"""
            
            elif problem_name == "segment_tree":
                test_script += """    for initial_array, operations, expected_results in test_cases:
        try:
            seg_tree = SegmentTree(initial_array[:])  # Make a copy
            results = []
            for op in operations:
                if op[0] == "query":
                    result = seg_tree.query(op[1], op[2])
                    results.append(result)
                elif op[0] == "update":
                    seg_tree.update(op[1], op[2])
                    results.append(None)
            
            test_results.append((f"Segment Tree operations", results, expected_results, results == expected_results))
        except Exception as e:
            test_results.append((f"Segment Tree test", f"ERROR: {e}", expected_results, False))
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
            print(f"{{status}} {{test_name}}")
            print(f"    Result: {{result}}")
            print(f"    Expected: {{expected}}")
            print(f"    Status: {{'PASS' if passed else 'FAIL'}}")
            print()
            if not passed:
                all_passed = False
        
        overall_status = 'PASS' if all_passed else 'FAIL'
        print(f"Overall Result: {{overall_status}}")
        print(f"Difficulty Level: USACO PLATINUM (Advanced)")
        exit(0 if all_passed else 1)
    except Exception as e:
        print(f"Error running tests: {{e}}")
        print(f"Traceback: {{traceback.format_exc()}}")
        exit(1)
"""
            
            # Write to file
            file_path = os.path.join(self.base_dir, f"usaco_platinum_{problem_name}.py")
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
                timeout=60  # Reasonable timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Test timed out (algorithm may be inefficient or infinite loop)"
        except Exception as e:
            return False, "", f"Error running test: {str(e)}"
    
    def run_full_test_suite(self):
        """Run the complete USACO Platinum test suite"""
        problems = self.get_usaco_platinum_problems()
        results = {}
        
        print(f"\n{'='*100}")
        print(f"RUNNING IMPROVED USACO PLATINUM TEST SUITE FOR MODEL: {self.model_name}")
        print(f"🏆 DIFFICULTY LEVEL: ADVANCED - COMPETITIVE PROGRAMMING")
        print(f"{'='*100}")
        
        for problem_name, problem_data in problems.items():
            print(f"\n{'*'*80}")
            print(f"🎯 USACO PLATINUM PROBLEM: {problem_data['title'].upper()}")
            print(f"{'*'*80}")
            
            try:
                # Generate code
                generated_code, gen_time = self.generate_code(problem_name, problem_data)
                
                # Create test file
                test_file_path = self.create_test_file(
                    problem_name, 
                    problem_data,
                    generated_code, 
                    problem_data["test_cases"]
                )
                
                # Run tests
                passed, stdout, stderr = self.run_test_file(test_file_path)
                
                # Store results
                results[problem_name] = {
                    "title": problem_data["title"],
                    "passed": passed,
                    "generation_time": gen_time,
                    "generated_code": self.extract_function_code(generated_code),
                    "test_output": stdout,
                    "error_output": stderr
                }
                
                # Print test results
                test_status = "🏆 PLATINUM PASS" if passed else "❌ PLATINUM FAIL"
                print(f"\n{test_status}")
                print(f"⏱️  Generation Time: {gen_time:.2f} seconds")
                if stdout:
                    print(f"📊 Test Output:\n{stdout}")
                if stderr:
                    print(f"🚨 Error Output:\n{stderr}")
                    
            except Exception as e:
                print(f"Error testing problem {problem_name}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                results[problem_name] = {
                    "title": problem_data["title"],
                    "passed": False,
                    "generation_time": 0.0,
                    "generated_code": "Error generating code",
                    "test_output": "",
                    "error_output": str(e)
                }
        
        return results
    
    def print_summary(self, results):
        """Print a summary of all USACO Platinum test results"""
        print(f"\n{'='*100}")
        print(f"🏆 USACO PLATINUM TEST SUMMARY FOR MODEL: {self.model_name}")
        print(f"{'='*100}")
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r["passed"])
        
        print(f"Total USACO Platinum Problems: {total_tests}")
        print(f"🏆 Solved: {passed_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}")
        print(f"🎯 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Difficulty assessment
        if passed_tests == total_tests:
            print(f"🌟 RATING: COMPETITIVE PROGRAMMING EXPERT - Outstanding algorithmic mastery!")
        elif passed_tests >= total_tests * 0.67:
            print(f"🥇 RATING: ADVANCED PROGRAMMER - Strong algorithmic understanding!")
        elif passed_tests >= total_tests * 0.33:
            print(f"🥈 RATING: INTERMEDIATE PROGRAMMER - Good foundation with room to grow!")
        else:
            print(f"🥉 RATING: DEVELOPING SKILLS - Keep practicing advanced algorithms!")
        
        print(f"\nDetailed Results:")
        print("-" * 80)
        
        for problem, result in results.items():
            test_status = "🏆" if result["passed"] else "❌"
            print(f"{test_status} {result['title']:<50} | {result['generation_time']:>8.2f}s")
        
        print(f"\n{'='*100}")
        print("📚 ALGORITHM CATEGORIES TESTED:")
        print("   • Dynamic Programming on DAGs")
        print("   • Binary Lifting for Tree Queries") 
        print("   • Segment Trees for Range Operations")
        print("🎓 These problems test fundamental competitive programming techniques!")

def main():
    """Run USACO Platinum level tests for Qwen models"""
    
    models_to_test = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        #"Qwen/Qwen2.5-Coder-32B-Instruct",  # Uncomment if you have access
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'#'*120}")
        print(f"🚀 STARTING IMPROVED USACO PLATINUM TESTS FOR MODEL: {model_name}")
        print(f"{'#'*120}")
        
        try:
            tester = USACOPlatinumTester(model_name)
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
    print(f"\n{'='*120}")
    print("🏆 FINAL USACO PLATINUM CHAMPIONSHIP RESULTS")
    print(f"{'='*120}")
    
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"❌ {model_name}: ERROR - {results['error']}")
        else:
            passed = sum(1 for r in results.values() if r["passed"])
            total = len(results)
            percentage = (passed/total)*100
            
            if percentage == 100:
                rating = "🏆 GRANDMASTER"
            elif percentage >= 67:
                rating = "🥇 EXPERT" 
            elif percentage >= 33:
                rating = "🥈 ADVANCED"
            else:
                rating = "🥉 DEVELOPING"
                
            print(f"{rating} {model_name}: {passed}/{total} USACO Platinum problems solved ({percentage:.1f}%)")
    
    print(f"\n{'='*120}")
    print("🎓 These improved problems focus on core competitive programming algorithms.")
    print("   Success indicates strong understanding of advanced data structures and algorithms.")
    print("   Each problem includes detailed explanations and implementation guidance!")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()