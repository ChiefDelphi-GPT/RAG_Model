# Generated code for expression_evaluator
import threading
import time
from collections import defaultdict
import heapq

import re

def evaluate_expression(expression):
    # Define a regular expression pattern for valid characters in the expression
    valid_chars_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if the expression contains only valid characters
    if not re.match(valid_chars_pattern, expression):
        return "Error: Invalid characters in expression."
    
    try:
        # Evaluate the expression safely
        result = eval(expression)
        
        # If the result is a complex number (e.g., due to division by zero), return 'Error'
        if isinstance(result, complex) or result == float('inf') or result == float('-inf'):
            return "Error"
        
        return float(result)
    
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage:
print(evaluate_expression("3 + 5 * (10 - 4)"))  # Output: 33.0
print(evaluate_expression("10 / 0"))           # Output: Error: Division by zero.
print(evaluate_expression("2 + 2 * 2"))        # Output: 6.0
print(evaluate_expression("invalid input!"))   # Output: Error: Invalid characters in expression.

# Test cases
def run_tests():
    test_results = []
    
    for expr, expected in test_cases:
        try:
            result = evaluate_expression(expr)
            is_correct = (result == expected) or (str(result) == str(expected))
            test_results.append((f"evaluate_expression('{expr}')", result, expected, is_correct))
        except Exception as e:
            test_results.append((f"evaluate_expression('{expr}')", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [('2 + 3 * 4', 14.0), ('(2 + 3) * 4', 20.0), ('10 / 2 + 3', 8.0), ('2 * (3 + 4) / 2', 7.0), ('10 / 0', 'Error'), ('((1 + 2) * 3 - 4) / 2', 2.5)]
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
