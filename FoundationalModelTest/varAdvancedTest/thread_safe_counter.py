# Generated code for thread_safe_counter
import threading
import time
from collections import defaultdict
import heapq

import threading

class ThreadSafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        """Increment the counter by 1."""
        with self.lock:
            self.value += 1

    def decrement(self):
        """Decrement the counter by 1."""
        with self.lock:
            self.value -= 1

    def get_value(self):
        """Get the current value of the counter."""
        with self.lock:
            return self.value

    def reset(self):
        """Reset the counter to 0."""
        with self.lock:
            self.value = 0

    def batch_operation(self, operations):
        """
        Perform a batch of operations atomically.
        
        :param operations: List of operations ['inc', 'dec', 'reset']
        """
        with self.lock:
            for operation in operations:
                if operation == 'inc':
                    self.increment()
                elif operation == 'dec':
                    self.decrement()
                elif operation == 'reset':
                    self.reset()
                else:
                    raise ValueError(f"Invalid operation: {operation}")

# Example usage:
if __name__ == "__main__":
    counter = ThreadSafeCounter()
    
    # Increment the counter
    counter.increment()
    print("After increment:", counter.get_value())  # Output: 1
    
    # Decrement the counter
    counter.decrement()
    print("After decrement:", counter.get_value())  # Output: 0
    
    # Reset the counter
    counter.reset()
    print("After reset:", counter.get_value())  # Output: 0
    
    # Batch operations
    counter.batch_operation(['inc', 'inc', 'dec'])
    print("After batch operations:", counter.get_value())  # Output: 0

# Test cases
def run_tests():
    test_results = []
    
    for operations, expected in test_cases:
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

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [(['inc', 'inc', 'dec', 'get'], 1), (['inc', 'reset', 'get'], 0), (['dec', 'dec', 'inc', 'get'], -1), (['batch', ['inc', 'inc', 'dec'], 'get'], 1)]
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
