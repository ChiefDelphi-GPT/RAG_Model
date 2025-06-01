# Generated code for USACO Platinum: Lowest Common Ancestor with Binary Lifting
import sys
import heapq
from collections import defaultdict, deque
import math
from typing import List, Dict, Set, Tuple, Optional

class TreeLCA:
    def __init__(self, tree, root):
        self.n = len(tree)
        self.max_log_n = self._compute_max_log_n()
        self.depth = [-1] * self.n
        self.parent = [[-1] * self.max_log_n for _ in range(self.n)]
        
        self._dfs(tree, root, -1, 0)
        self._build_lifting_table()

    def _compute_max_log_n(self):
        return int.bit_length(self.n) + 1

    def _dfs(self, tree, node, par, d):
        self.depth[node] = d
        self.parent[node][0] = par
        for child in tree[node]:
            if child != par:
                self._dfs(tree, child, node, d + 1)

    def _build_lifting_table(self):
        for i in range(1, self.max_log_n):
            for node in range(self.n):
                if self.parent[node][i - 1] != -1:
                    self.parent[node][i] = self.parent[self.parent[node][i - 1]][i - 1]

    def lca(self, u, v):
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        
        # Lift v to the same depth as u
        diff = self.depth[v] - self.depth[u]
        for i in range(diff.bit_length()):
            if diff & (1 << i):
                v = self.parent[v][i]
        
        if u == v:
            return u
        
        # Lift both u and v until their parents are the same
        for i in range(self.max_log_n - 1, -1, -1):
            if self.parent[u][i] != self.parent[v][i]:
                u = self.parent[u][i]
                v = self.parent[v][i]
        
        return self.parent[u][0]

# Test cases
def run_tests():
    test_results = []
    
    for tree, root, queries, expected_results in test_cases:
        try:
            lca_solver = TreeLCA(tree, root)
            results = []
            for u, v in queries:
                result = lca_solver.lca(u, v)
                results.append(result)
            
            test_results.append((f"Binary Lifting LCA", results, expected_results, results == expected_results))
        except Exception as e:
            test_results.append((f"Binary Lifting LCA test", f"ERROR: {e}", expected_results, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [({0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]}, 0, [(3, 4), (3, 5), (1, 2), (0, 5), (3, 3)], [1, 0, 0, 0, 3]), ({0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}, 0, [(4, 0), (3, 1), (4, 2), (1, 4)], [0, 1, 2, 1]), ({0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}, 0, [(1, 2), (1, 3), (2, 3), (0, 1)], [0, 0, 0, 0])]
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
        print(f"Difficulty Level: USACO PLATINUM (Advanced)")
        exit(0 if all_passed else 1)
    except Exception as e:
        print(f"Error running tests: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        exit(1)
