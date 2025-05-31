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
        self.head = [0] * self.n
        self.pos = [0] * self.n
        self.chain = [0] * self.n
        self.chain_size = [0] * self.n
        self.t = 0
        self.seg_tree = [0] * (4 * self.n)

        # Build the tree
        self.dfs(0, -1)
        self.build_hld(0, -1)
        self.build_segment_tree(0, self.n - 1, 1)

    def dfs(self, node, par):
        self.parent[node] = par
        self.depth[node] = self.depth[par] + 1 if par != -1 else 0
        self.chain[node] = node
        self.chain_size[node] = 1
        max_child_size = 0
        for child in self.tree[node]:
            if child != par:
                self.dfs(child, node)
                self.chain_size[node] += self.chain_size[child]
                if self.chain_size[child] > max_child_size:
                    max_child_size = self.chain_size[child]
                    self.chain[node] = child

    def build_hld(self, node, par):
        self.head[node] = self.chain[node]
        self.pos[node] = self.t
        self.t += 1
        for child in self.tree[node]:
            if child != par and child == self.chain[node]:
                self.build_hld(child, node)
        for child in self.tree[node]:
            if child != par and child != self.chain[node]:
                self.build_hld(child, node)

    def build_segment_tree(self, l, r, rt):
        if l == r:
            self.seg_tree[rt] = self.values[l]
        else:
            mid = (l + r) // 2
            self.build_segment_tree(l, mid, rt << 1)
            self.build_segment_tree(mid + 1, r, rt << 1 | 1)
            self.seg_tree[rt] = max(self.seg_tree[rt << 1], self.seg_tree[rt << 1 | 1])

    def update(self, node, new_value):
        pos = self.pos[node]
        self.seg_tree[pos] = new_value
        while pos > 0:
            pos >>= 1
            self.seg_tree[pos] = max(self.seg_tree[pos << 1], self.seg_tree[pos << 1 | 1])

    def query_path_max(self, u, v):
        res = float('-inf')
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            res = max(res, self.query_range(self.pos[self.head[u]], self.pos[u]))
            u = self.parent[self.head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        res = max(res, self.query_range(self.pos[u], self.pos[v]))
        return res

    def query_range(self, l, r):
        res = float('-inf')
        while l <= r:
            if l & 1:
                res = max(res, self.seg_tree[l])
                l += 1
            if not (r & 1):
                res = max(res, self.seg_tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

# Example usage:
tree = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
values = [1, 4, 2, 8]
hld = HeavyLightDecomposition(tree, values)
print(hld.query_path_max(3, 2))  # Output should be 8

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
