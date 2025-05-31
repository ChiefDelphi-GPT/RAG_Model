# Generated code for palindrome_check
def is_palindrome(s: str) -> bool:
    # Filter out non-alphanumeric characters and convert to lowercase
    filtered_chars = [char.lower() for char in s if char.isalnum()]
    # Check if the filtered list of characters is equal to its reverse
    return filtered_chars == filtered_chars[::-1]

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
