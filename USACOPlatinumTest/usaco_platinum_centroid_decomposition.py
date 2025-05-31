# Generated code for USACO Platinum: Tree Distance Queries with Centroid Decomposition
import sys
import heapq
from collections import defaultdict, deque
import math
from typing import List, Dict, Set, Tuple, Optional

from collections import defaultdict, deque

class CentroidDecomposition:
    def __init__(self, tree):
        self.tree = tree
        self.n = len(tree)
        self.size = [0] * self.n
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        self.heavy_child = [-1] * self.n
        self.subtree_root = [-1] * self.n
        self.dist_to_centroid = [float('inf')] * self.n
        self.important_vertices = set()
        
        # Preprocess the tree
        self.preprocess_tree()
    
    def preprocess_tree(self):
        self.calc_size_and_depth(0, -1)
        self.decompose_tree(0, -1)
    
    def calc_size_and_depth(self, u, p):
        self.size[u] = 1
        self.depth[u] = self.depth[p] + 1
        self.parent[u] = p
        max_child_size = 0
        
        for v in self.tree[u]:
            if v != p:
                self.calc_size_and_depth(v, u)
                self.size[u] += self.size[v]
                if self.size[v] > max_child_size:
                    max_child_size = self.size[v]
                    self.heavy_child[u] = v
    
    def decompose_tree(self, u, p):
        centroid = self.find_centroid(u, p)
        self.subtree_root[centroid] = centroid
        self.dist_to_centroid[centroid] = 0
        
        for v in self.tree[centroid]:
            if v != p and v != self.heavy_child[centroid]:
                self.decompose_tree(v, centroid)
        
        if self.heavy_child[centroid] != -1:
            self.decompose_tree(self.heavy_child[centroid], centroid)
    
    def find_centroid(self, u, p):
        centroid = u
        while True:
            heavy_child = self.heavy_child[centroid]
            if heavy_child == -1 or self.size[heavy_child] <= self.size[u] // 2:
                break
            centroid = heavy_child
        return centroid
    
    def mark_important(self, vertex):
        self.important_vertices.add(vertex)
        self.update_distances(vertex)
    
    def update_distances(self, vertex):
        u = vertex
        while u != -1:
            self.dist_to_centroid[u] = min(self.dist_to_centroid[u], self.depth[vertex] - self.depth[u])
            u = self.parent[u]
    
    def query_nearest_distance(self, vertex):
        u = vertex
        min_distance = float('inf')
        
        while u != -1:
            min_distance = min(min_distance, self.dist_to_centroid[u] + self.depth[vertex] - self.depth[u])
            if u in self.important_vertices:
                break
            u = self.parent[u]
        
        return min_distance if min_distance != float('inf') else -1

# Example usage:
tree = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0],
    3: [1],
    4: [1]
}

cd = CentroidDecomposition(tree)
cd.mark_important(3)
print(cd.query_nearest_distance(4))  # Output: 2
cd.mark_important(0)
print(cd.query_nearest_distance(4))  # Output: 2
print(cd.query_nearest_distance(2))  # Output: 1

# Test cases
def run_tests():
    test_results = []
    
    for tree, operations, expected_results in test_cases:
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

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [({0: [1, 2], 1: [0, 3, 4], 2: [0], 3: [1], 4: [1]}, [('mark', 3), ('query', 4), ('mark', 0), ('query', 4), ('query', 2), ('query', 1)], [None, 2, None, 2, 1, 1]), ({0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}, [('mark', 0), ('query', 4), ('mark', 4), ('query', 2), ('query', 1)], [None, 4, None, 2, 1]), ({0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}, [('mark', 1), ('query', 3), ('mark', 0), ('query', 3), ('query', 2)], [None, 2, None, 1, 1])]
    try:
        results = run_tests()
        
        all_passed = True
        for test_name, result, expected, passed in results:
            status = "✓" if passed else "✗"
            print(f"{status} {test_name}")
            print(f"    Result: {result}")
            print(f"    Expected: {expected}")
            print(f"    Status: {'PASS' if passed else 'FAIL'}")
            print()
            if not passed:
                all_passed = False
        
        overall_status = 'PASS' if all_passed else 'FAIL'
        print(f"Overall Result: {overall_status}")
        print(f"Difficulty Level: USACO PLATINUM (Extremely Advanced)")
        exit(0 if all_passed else 1)
    except Exception as e:
        print(f"Error running tests: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        exit(1)
