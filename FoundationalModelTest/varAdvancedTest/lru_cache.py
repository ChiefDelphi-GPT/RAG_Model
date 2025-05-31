# Generated code for lru_cache
import threading
import time
from collections import defaultdict
import heapq

from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Move the accessed key to the end to mark it as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update the value and move the key to the end
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove the first item (least recently used)
            self.cache.popitem(last=False)
        # Add or update the key-value pair
        self.cache[key] = value

# Example usage:
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))       # returns 1
cache.put(3, 3)          # evicts key 2
print(cache.get(2))       # returns -1 (not found)
cache.put(4, 4)          # evicts key 1
print(cache.get(1))       # returns -1 (not found)
print(cache.get(3))       # returns 3
print(cache.get(4))       # returns 4

# Test cases
def run_tests():
    test_results = []
    
    for capacity, operations, expected in test_cases:
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

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [(2, [('put', 1, 1), ('put', 2, 2), ('get', 1), ('put', 3, 3), ('get', 2), ('get', 3), ('get', 1)], [None, None, 1, None, -1, 3, -1]), (1, [('put', 2, 1), ('get', 2), ('put', 3, 2), ('get', 2), ('get', 3)], [None, 1, None, -1, 2])]
    try:
        results = run_tests()
        
        all_passed = True
        for test_name, result, expected, passed in results:
            status = "✓" if passed else "✗"
            print(f"{status} {test_name} -> {result} (expected: {expected})")
            if not passed:
                all_passed = False
        
        overall_status = 'PASS' if all_passed else 'FAIL'
        print(f"\nOverall: {overall_status}")
        exit(0 if all_passed else 1)
    except Exception as e:
        print(f"Error running tests: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        exit(1)
