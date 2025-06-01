# Generated code for binary_search
def binary_search(sorted_list, target):
    left, right = 0, len(sorted_list) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Test cases
def run_tests():
    test_results = []
    
    for arr, target, expected in test_cases:
        try:
            result = binary_search(arr, target)
            test_results.append((f"binary_search({arr}, {target})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"binary_search({arr}, {target})", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [([1, 2, 3, 4, 5], 3, 2), ([1, 2, 3, 4, 5], 6, -1), ([1], 1, 0), ([], 1, -1)]
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
