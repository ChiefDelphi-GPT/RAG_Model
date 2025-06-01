# Generated code for expression_evaluator
import threading
import time
from collections import defaultdict
import heapq

import ast
import operator

# Define supported operations
operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def evaluate_node(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        op_type = type(node.op)
        if op_type in operators:
            try:
                return operators[op_type](left, right)
            except ZeroDivisionError:
                return 'Error'
        else:
            raise TypeError(f"Unsupported operation: {op_type}")
    elif isinstance(node, ast.Paren):  # Parentheses
        return evaluate_node(node.value)
    else:
        raise TypeError(f"Unsupported type: {type(node)}")

def evaluate_expression(expression):
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Evaluate the AST
        result = evaluate_node(tree.body)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage:
print(evaluate_expression("3 + 5 * (10 - 4)"))  # Output: 28.0
print(evaluate_expression("10 / 0"))           # Output: Error

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
