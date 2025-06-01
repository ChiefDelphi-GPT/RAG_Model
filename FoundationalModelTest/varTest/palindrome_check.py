# Generated code for palindrome_check
import re

def is_palindrome(s: str) -> bool:
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    # Check if the cleaned string is equal to its reverse
    return cleaned == cleaned[::-1]

# Test cases
def run_tests():
    test_results = []
    
    for s, expected in test_cases:
        try:
            result = is_palindrome(s)
            test_results.append((f"is_palindrome('{s}')", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"is_palindrome('{s}')", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [('A man a plan a canal Panama', True), ('race a car', False), ('', True), ('Madam', True), ("No 'x' in Nixon", True)]
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
