# Generated code for merge_sorted_lists
def merge_sorted_lists(list1, list2):
    result = []
    i, j = 0, 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    while i < len(list1):
        result.append(list1[i])
        i += 1
    
    while j < len(list2):
        result.append(list2[j])
        j += 1
    
    return result

# Test cases
def run_tests():
    test_results = []
    
    for list1, list2, expected in test_cases:
        try:
            result = merge_sorted_lists(list1, list2)
            test_results.append((f"merge_sorted_lists({list1}, {list2})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"merge_sorted_lists({list1}, {list2})", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [([1, 2, 4], [1, 3, 4], [1, 1, 2, 3, 4, 4]), ([], [], []), ([], [0], [0]), ([1], [], [1])]
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
