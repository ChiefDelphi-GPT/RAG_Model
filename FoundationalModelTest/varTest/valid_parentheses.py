# Generated code for valid_parentheses
def is_valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            return False
    
    return not stack

# Test cases
def run_tests():
    test_results = []
    
    for s, expected in test_cases:
        try:
            result = is_valid_parentheses(s)
            test_results.append((f"is_valid_parentheses('{s}')", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"is_valid_parentheses('{s}')", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [('()', True), ('()[]{}', True), ('(]', False), ('([)]', False), ('{[]}', True), ('', True)]
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
