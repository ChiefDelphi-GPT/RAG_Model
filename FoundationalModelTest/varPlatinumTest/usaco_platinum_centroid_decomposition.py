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
        self.parent = [-1] * self.n
        self.size = [1] * self.n
        self.marked = [False] * self.n
        self.dist_to_centroid = [float('inf')] * self.n
        self.centroid_tree = defaultdict(list)
        
        # Step 1: Compute sizes of all subtrees
        self.compute_sizes(0)
        
        # Step 2: Find centroids and build centroid tree
        self.find_centroids(0)
        
        # Step 3: Precompute distances from centroids to all vertices in their subtree
        self.precompute_distances()
    
    def compute_sizes(self, node):
        for child in self.tree[node]:
            if self.parent[child] == -1:
                self.parent[child] = node
                self.size[node] += self.compute_sizes(child)
        return self.size[node]
    
    def find_centroids(self, node):
        centroid = node
        max_size = self.size[node] // 2
        
        for child in self.tree[node]:
            if self.parent[child] != -1 and self.size[child] > max_size:
                centroid = self.find_centroids(child)
                max_size = self.size[centroid] // 2
        
        self.centroid_tree[self.parent[centroid]].append(centroid)
        self.centroid_tree[centroid].append(self.parent[centroid])
        return centroid
    
    def precompute_distances(self):
        queue = deque([node for node in range(self.n) if self.parent[node] == -1])
        
        while queue:
            node = queue.popleft()
            for child in self.tree[node]:
                if self.parent[child] != -1:
                    self.dist_to_centroid[child] = self.dist_to_centroid[node] + 1
                    queue.append(child)
    
    def mark_important(self, vertex):
        self.marked[vertex] = True
    
    def query_nearest_distance(self, vertex):
        if self.marked[vertex]:
            return 0
        
        min_dist = float('inf')
        current = vertex
        
        while current != -1:
            min_dist = min(min_dist, self.dist_to_centroid[current])
            current = self.parent[current]
        
        return min_dist if min_dist != float('inf') else -1

# Example usage:
tree = {0: [1, 2], 1: [0, 3, 4], 2: [0], 3: [1], 4: [1]}
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
