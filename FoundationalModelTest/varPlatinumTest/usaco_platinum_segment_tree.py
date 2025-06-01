# Generated code for USACO Platinum: Range Sum Query with Point Updates
import sys
import heapq
from collections import defaultdict, deque
import math
from typing import List, Dict, Set, Tuple, Optional

class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build_tree(arr, 0, 0, self.n - 1)

    def _build_tree(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build_tree(arr, 2 * node + 1, start, mid)
            self._build_tree(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def update(self, index, value):
        self._update_util(0, 0, self.n - 1, index, value)

    def _update_util(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                self._update_util(2 * node + 1, start, mid, idx, val)
            else:
                self._update_util(2 * node + 2, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, left, right):
        return self._query_util(0, 0, self.n - 1, left, right)

    def _query_util(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        p1 = self._query_util(2 * node + 1, start, mid, l, r)
        p2 = self._query_util(2 * node + 2, mid + 1, end, l, r)
        return p1 + p2

# Test cases
def run_tests():
    test_results = []
    
    for initial_array, operations, expected_results in test_cases:
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

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [([1, 3, 5, 7, 9, 11], [('query', 0, 2), ('query', 1, 4), ('update', 1, 10), ('query', 0, 2), ('query', 1, 4)], [9, 24, None, 16, 31]), ([2, 4, 6, 8], [('query', 0, 3), ('update', 2, 0), ('query', 0, 3), ('query', 2, 3), ('update', 0, 5)], [20, None, 14, 8, None]), ([10], [('query', 0, 0), ('update', 0, 20), ('query', 0, 0)], [10, None, 20])]
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
