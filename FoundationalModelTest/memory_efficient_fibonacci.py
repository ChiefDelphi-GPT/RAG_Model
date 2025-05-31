# Generated code for memory_efficient_fibonacci
import threading
import time
from collections import defaultdict
import heapq

import numpy as np

def fibonacci_matrix(n):
    """
    Compute the nth Fibonacci number using matrix exponentiation.
    
    Args:
    n (int): The position in the Fibonacci sequence (0-indexed).
    
    Returns:
    int: The nth Fibonacci number.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Transformation matrix for Fibonacci sequence
    F = np.array([[1, 1], [1, 0]], dtype=object)
    
    # Function to multiply two matrices
    def matrix_multiply(A, B):
        return np.dot(A, B) % 10**9 + 7  # Using modulo to prevent overflow
    
    # Function to perform matrix exponentiation
    def matrix_power(matrix, power):
        result = np.identity(len(matrix), dtype=object)
        base = matrix
        
        while power > 0:
            if power % 2 == 1:
                result = matrix_multiply(result, base)
            base = matrix_multiply(base, base)
            power //= 2
        
        return result
    
    # Raise the transformation matrix to the (n-1)th power
    result_matrix = matrix_power(F, n - 1)
    
    # The top left cell of the resulting matrix contains the nth Fibonacci number
    return result_matrix[0][0]

def fibonacci_generator():
    """
    Generator function that yields Fibonacci numbers indefinitely.
    
    Yields:
    int: The next Fibonacci number.
    """
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Example usage:
if __name__ == "__main__":
    # Test fibonacci_matrix
    print(fibonacci_matrix(10))  # Output: 55
    
    # Test fibonacci_generator
    fib_gen = fibonacci_generator()
    for _ in range(10):
        print(next(fib_gen))  # Output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# Test cases
def run_tests():
    test_results = []
    
    # Test matrix fibonacci
    for n, expected in test_cases[:-1]:  # All except generator test
        try:
            result = fibonacci_matrix(n)
            test_results.append((f"fibonacci_matrix({n})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"fibonacci_matrix({n})", f"ERROR: {e}", expected, False))
    
    # Test generator
    try:
        gen = fibonacci_generator()
        generated = [next(gen) for _ in range(10)]
        expected = test_cases[-1][1]  # Last test case is generator test
        test_results.append(("fibonacci_generator first 10", generated, expected, generated == expected))
    except Exception as e:
        test_results.append(("fibonacci_generator", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [(0, 0), (1, 1), (10, 55), (20, 6765), (50, 12586269025), ('generator_test', [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])]
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
