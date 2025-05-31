# Generated code for two_sum
def two_sum(nums, target):
    num_to_index = {}
    for index, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], index]
        num_to_index[num] = index
    return []

# Test cases
def run_tests():
    test_results = []
    
    for nums, target, expected in test_cases:
        try:
            result = two_sum(nums, target)
            # Check if result indices are correct (order doesn't matter)
            is_correct = (result == expected or result == expected[::-1]) if result else False
            test_results.append((f"two_sum({nums}, {target})", result, expected, is_correct))
        except Exception as e:
            test_results.append((f"two_sum({nums}, {target})", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [([2, 7, 11, 15], 9, [0, 1]), ([3, 2, 4], 6, [1, 2]), ([3, 3], 6, [0, 1])]
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
