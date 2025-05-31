# USACO Platinum Level Advanced Test Suite - EXTREMELY CHALLENGING PROBLEMS
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
        """Return USACO Platinum level problems - extremely challenging"""
        return {
            "max_flow_min_cut": {
                "title": "Network Flow with Vertex Capacities",
                "prompt": """
USACO Platinum Problem: Network Flow with Vertex Capacities

You are given a directed graph with N vertices (numbered 1 to N) and M edges. Each vertex has a capacity limit (maximum flow that can pass through it), and each edge has a capacity limit as well. You need to find the maximum flow from source vertex S to sink vertex T.

This is different from standard max flow because vertices also have capacity constraints, not just edges.

Write a function 'max_flow_vertex_capacity(graph, vertex_capacities, source, sink)' that:
- graph: dict where graph[u] = [(v, edge_capacity), ...] representing edges from u to v
- vertex_capacities: dict where vertex_capacities[v] = capacity of vertex v
- source, sink: source and sink vertices
- Returns: maximum flow value from source to sink

Constraints:
- 1 ≤ N ≤ 100
- 1 ≤ M ≤ 1000  
- All capacities are positive integers ≤ 1000

Example:
Graph: {1: [(2, 10), (3, 10)], 2: [(4, 10)], 3: [(4, 10)], 4: []}
Vertex capacities: {1: 100, 2: 5, 3: 8, 4: 100}
Source: 1, Sink: 4
Answer: 13 (5 through vertex 2, 8 through vertex 3)

Hint: Transform the problem by splitting vertices with capacity constraints.
""",
                "test_cases": [
                    # (graph, vertex_capacities, source, sink, expected_max_flow)
                    (
                        {1: [(2, 10), (3, 10)], 2: [(4, 10)], 3: [(4, 10)], 4: []},
                        {1: 100, 2: 5, 3: 8, 4: 100},
                        1, 4, 13
                    ),
                    (
                        {1: [(2, 20)], 2: [(3, 20)], 3: []},
                        {1: 100, 2: 5, 3: 100},
                        1, 3, 5
                    ),
                    (
                        {1: [(2, 10), (3, 15)], 2: [(4, 20)], 3: [(4, 10)], 4: []},
                        {1: 100, 2: 12, 3: 8, 4: 100},
                        1, 4, 18
                    )
                ]
            },
            
            "heavy_light_decomposition": {
                "title": "Tree Path Queries with Heavy-Light Decomposition",
                "prompt": """
USACO Platinum Problem: Tree Path Queries

You are given a tree with N vertices. Each vertex has a value. You need to support two types of operations:
1. Update the value of a vertex
2. Query the maximum value on the path between two vertices

Use Heavy-Light Decomposition with a segment tree to achieve O(log²N) time complexity per operation.

Write a class 'HeavyLightDecomposition' with methods:
- __init__(self, tree, values): Initialize with tree structure and initial vertex values
  - tree: dict where tree[u] = [v1, v2, ...] (undirected tree edges)
  - values: list where values[i] is the value of vertex i (0-indexed)
- update(self, vertex, new_value): Update vertex value
- query_path_max(self, u, v): Return maximum value on path from u to v

Constraints:
- 1 ≤ N ≤ 100,000
- All values are integers in range [-10^9, 10^9]
- All vertices are 0-indexed

Example:
Tree: {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
Values: [1, 4, 2, 8]
query_path_max(3, 2) should return 4 (path: 3-1-0-2, values: 8,4,1,2, max=8... wait, let me recalculate)
Actually: path 3-1-0-2 has values [8,4,1,2], so max is 8.

This requires implementing:
1. Heavy-Light Decomposition of the tree
2. Segment tree for range maximum queries
3. Path queries by decomposing paths into heavy/light edges
""",
                "test_cases": [
                    # (tree, initial_values, operations, expected_results)
                    (
                        {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]},
                        [1, 4, 2, 8],
                        [("query", 3, 2), ("update", 1, 10), ("query", 3, 2), ("query", 0, 3)],
                        [8, None, 10, 10]
                    ),
                    (
                        {0: [1], 1: [0, 2], 2: [1, 3, 4], 3: [2], 4: [2]},
                        [5, 3, 7, 1, 9],
                        [("query", 3, 4), ("query", 0, 4), ("update", 2, 15), ("query", 0, 4)],
                        [7, 7, None, 15]
                    )
                ]
            },
            
            "centroid_decomposition": {
                "title": "Tree Distance Queries with Centroid Decomposition",
                "prompt": """
USACO Platinum Problem: Tree Distance Queries

You are given a tree with N vertices. You need to support two types of operations:
1. Mark a vertex as "important"
2. Query the distance to the nearest "important" vertex from a given vertex

Use Centroid Decomposition to preprocess the tree and answer queries efficiently.

Write a class 'CentroidDecomposition' with methods:
- __init__(self, tree): Initialize with tree structure
  - tree: dict where tree[u] = [v1, v2, ...] (undirected tree edges)
- mark_important(self, vertex): Mark vertex as important
- query_nearest_distance(self, vertex): Return distance to nearest important vertex (or -1 if none marked)

The key insight is to build a centroid tree and maintain distance information from each centroid to all vertices in its subtree.

Constraints:
- 1 ≤ N ≤ 100,000
- All queries and updates should run in O(log N) time after O(N log N) preprocessing
- All vertices are 0-indexed

Example:
Tree: {0: [1, 2], 1: [0, 3, 4], 2: [0], 3: [1], 4: [1]}
Operations:
1. mark_important(3)
2. query_nearest_distance(4) → should return 2 (path: 4-1-3)
3. mark_important(0)  
4. query_nearest_distance(4) → should return 2 (still 4-1-0, but 4-1-3 is also distance 2)
5. query_nearest_distance(2) → should return 1 (path: 2-0)

This requires implementing:
1. Centroid decomposition of the tree
2. Distance computation from centroids
3. Efficient nearest marked vertex queries
""",
                "test_cases": [
                    # (tree, operations, expected_results)
                    (
                        {0: [1, 2], 1: [0, 3, 4], 2: [0], 3: [1], 4: [1]},
                        [
                            ("mark", 3), 
                            ("query", 4), 
                            ("mark", 0), 
                            ("query", 4), 
                            ("query", 2),
                            ("query", 1)
                        ],
                        [None, 2, None, 2, 1, 1]
                    ),
                    (
                        {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]},  # Path graph
                        [
                            ("mark", 0),
                            ("query", 4),
                            ("mark", 4), 
                            ("query", 2),
                            ("query", 1)
                        ],
                        [None, 4, None, 2, 1]
                    ),
                    (
                        {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]},  # Star graph
                        [
                            ("mark", 1),
                            ("query", 3),
                            ("mark", 0),
                            ("query", 3),
                            ("query", 2)
                        ],
                        [None, 2, None, 1, 1]
                    )
                ]
            }
        }
    
    def generate_code(self, problem_name, problem_data):
        """Generate code for a given USACO Platinum problem using the model"""
        
        messages = [
            {"role": "system", "content": """You are an expert competitive programmer specializing in advanced algorithms and data structures. You have mastery of:
- Network flows (Ford-Fulkerson, Dinic's algorithm, push-relabel)
- Advanced tree algorithms (Heavy-Light Decomposition, Centroid Decomposition)
- Segment trees, Fenwick trees, and other range query structures
- Complex graph algorithms and optimization techniques

Write clean, efficient, and mathematically correct implementations. Include all necessary imports and helper functions. Your code should handle edge cases and be optimized for competitive programming constraints."""},
            {"role": "user", "content": f"**{problem_data['title']}**\n\n{problem_data['prompt']}"}
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
                    max_new_tokens=8000,  # Even more tokens for complex algorithms
                    temperature=0.1,  # Very low temperature for precision
                    do_sample=True,
                    top_p=0.85,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
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
            if problem_name == "max_flow_min_cut":
                test_script += """    for graph, vertex_capacities, source, sink, expected in test_cases:
        try:
            result = max_flow_vertex_capacity(graph, vertex_capacities, source, sink)
            test_results.append((f"Max Flow: {graph} from {source} to {sink}", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"Max Flow test", f"ERROR: {e}", expected, False))
"""
            
            elif problem_name == "heavy_light_decomposition":
                test_script += """    for tree, initial_values, operations, expected_results in test_cases:
        try:
            hld = HeavyLightDecomposition(tree, initial_values)
            results = []
            for i, op in enumerate(operations):
                if op[0] == "query":
                    result = hld.query_path_max(op[1], op[2])
                    results.append(result)
                elif op[0] == "update":
                    hld.update(op[1], op[2])
                    results.append(None)
            
            test_results.append((f"HLD operations", results, expected_results, results == expected_results))
        except Exception as e:
            test_results.append((f"Heavy-Light Decomposition test", f"ERROR: {e}", expected_results, False))
"""
            
            elif problem_name == "centroid_decomposition":
                test_script += """    for tree, operations, expected_results in test_cases:
        try:
            cd = CentroidDecomposition(tree)
            results = []
            for i, op in enumerate(operations):
                if op[0] == "mark":
                    cd.mark_important(op[1])
                    results.append(None)
                elif op[0] == "query":
                    result = cd.query_nearest_distance(op[1])
                    results.append(result)
            
            test_results.append((f"Centroid Decomposition operations", results, expected_results, results == expected_results))
        except Exception as e:
            test_results.append((f"Centroid Decomposition test", f"ERROR: {e}", expected_results, False))
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
        print(f"Difficulty Level: USACO PLATINUM (Extremely Advanced)")
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
                timeout=120  # Longer timeout for complex algorithms
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
        print(f"RUNNING USACO PLATINUM TEST SUITE FOR MODEL: {self.model_name}")
        print(f"🏆 DIFFICULTY LEVEL: EXTREMELY ADVANCED - COMPETITIVE PROGRAMMING MASTERY")
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
            print(f"🌟 RATING: COMPETITIVE PROGRAMMING MASTER - This model demonstrates exceptional algorithmic expertise!")
        elif passed_tests >= total_tests * 0.67:
            print(f"🥇 RATING: ADVANCED ALGORITHMIC COMPETENCY - Strong performance on expert-level problems!")
        elif passed_tests >= total_tests * 0.33:
            print(f"🥈 RATING: INTERMEDIATE ALGORITHMIC UNDERSTANDING - Some advanced concepts understood!")
        else:
            print(f"🥉 RATING: BASIC PROGRAMMING ABILITY - USACO Platinum problems are extremely challenging!")
        
        print(f"\nDetailed Results:")
        print("-" * 80)
        
        for problem, result in results.items():
            test_status = "🏆" if result["passed"] else "❌"
            print(f"{test_status} {result['title']:<50} | {result['generation_time']:>8.2f}s")
        
        print(f"\n{'='*100}")
        print("📚 ALGORITHM CATEGORIES TESTED:")
        print("   • Network Flow with Vertex Constraints")
        print("   • Heavy-Light Decomposition + Segment Trees") 
        print("   • Centroid Decomposition + Distance Queries")
        print("🎓 These represent some of the most advanced algorithmic concepts in competitive programming!")

def main():
    """Run USACO Platinum level tests for Qwen models"""
    
    models_to_test = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        # "Qwen/Qwen2.5-Coder-32B-Instruct",  # Uncomment if you have access
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'#'*120}")
        print(f"🚀 STARTING USACO PLATINUM TESTS FOR MODEL: {model_name}")
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
                rating = "🥉 NOVICE"
                
            print(f"{rating} {model_name}: {passed}/{total} USACO Platinum problems solved ({percentage:.1f}%)")
    
    print(f"\n{'='*120}")
    print("🎓 USACO Platinum problems represent the pinnacle of competitive programming difficulty.")
    print("   These test advanced graph algorithms, complex data structures, and mathematical optimization.")
    print("   Success on even 1-2 problems indicates exceptional algorithmic competency!")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()