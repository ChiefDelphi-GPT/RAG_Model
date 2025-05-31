# Generated code for USACO Platinum: Tree Path Queries with Heavy-Light Decomposition
import sys
import heapq
from collections import defaultdict, deque
import math
from typing import List, Dict, Set, Tuple, Optional

class HeavyLightDecomposition:
    def __init__(self, tree, values):
        self.n = len(values)
        self.tree = tree
        self.values = values
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        self.size = [1] * self.n
        self.head = [0] * self.n
        self.pos = [0] * self.n
        self.current_pos = 0
        
        # Step 1: Perform DFS to compute size, depth, and parent
        self.dfs(0, -1)
        
        # Step 2: Decompose the tree using HLD
        self.hld(0, -1, 0)
        
        # Step 3: Build the segment tree
        self.seg_tree = SegmentTree(self.n)
        for i in range(self.n):
            self.seg_tree.update(self.pos[i], self.values[i])
    
    def dfs(self, node, par):
        self.parent[node] = par
        for nei in self.tree[node]:
            if nei != par:
                self.depth[nei] = self.depth[node] + 1
                self.dfs(nei, node)
                self.size[node] += self.size[nei]
    
    def hld(self, node, par, h):
        self.head[node] = h
        self.pos[node] = self.current_pos
        self.current_pos += 1
        max_child = -1
        for nei in self.tree[node]:
            if nei != par and (max_child == -1 or self.size[nei] > self.size[max_child]):
                max_child = nei
        if max_child != -1:
            self.hld(max_child, node, h)
        for nei in self.tree[node]:
            if nei != par and nei != max_child:
                self.hld(nei, node, nei)
    
    def update(self, vertex, new_value):
        self.values[vertex] = new_value
        self.seg_tree.update(self.pos[vertex], new_value)
    
    def query_path_max(self, u, v):
        res = float('-inf')
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            res = max(res, self.seg_tree.query(self.pos[self.head[u]], self.pos[u]))
            u = self.parent[self.head[u]]
        if self.pos[u] > self.pos[v]:
            u, v = v, u
        res = max(res, self.seg_tree.query(self.pos[u], self.pos[v]))
        return res

class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.tree = [float('-inf')] * (4 * n)
    
    def update(self, idx, val):
        self._update(1, 0, self.n - 1, idx, val)
    
    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if start <= idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query(self, l, r):
        return self._query(1, 0, self.n - 1, l, r)
    
    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return float('-inf')
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        left = self._query(2 * node, start, mid, l, r)
        right = self._query(2 * node + 1, mid + 1, end, l, r)
        return max(left, right)

# Example usage:
tree = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
values = [1, 4, 2, 8]
hld = HeavyLightDecomposition(tree, values)
print(hld.query_path_max(3, 2))  # Output: 8
hld.update(2, 5)
print(hld.query_path_max(3, 2))  # Output: 8

# Test cases
def run_tests():
    test_results = []
    
    for tree, initial_values, operations, expected_results in test_cases:
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

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [({0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}, [1, 4, 2, 8], [('query', 3, 2), ('update', 1, 10), ('query', 3, 2), ('query', 0, 3)], [8, None, 10, 10]), ({0: [1], 1: [0, 2], 2: [1, 3, 4], 3: [2], 4: [2]}, [5, 3, 7, 1, 9], [('query', 3, 4), ('query', 0, 4), ('update', 2, 15), ('query', 0, 4)], [7, 7, None, 15])]
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
