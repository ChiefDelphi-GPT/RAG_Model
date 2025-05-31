# Generated code for interval_scheduler
import threading
import time
from collections import defaultdict
import heapq

from typing import List

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []

    # Sort intervals based on the start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        last_merged_start, last_merged_end = merged[-1]

        if current_start <= last_merged_end:  # Overlapping intervals
            # Merge the current interval with the last merged interval
            merged[-1] = [last_merged_start, max(last_merged_end, current_end)]
        else:
            # No overlap, add the current interval to the merged list
            merged.append([current_start, current_end])

    return merged

# Example usage:
if __name__ == "__main__":
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    print(merge_intervals(intervals))  # Output: [[1, 6], [8, 10], [15, 18]]

# Test cases
def run_tests():
    test_results = []
    
    for intervals, expected in test_cases:
        try:
            result = merge_intervals(intervals)
            test_results.append((f"merge_intervals({intervals})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"merge_intervals({intervals})", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]), ([[1, 4], [4, 5]], [[1, 5]]), ([[1, 2]], [[1, 2]]), ([], []), ([[1, 4], [0, 4]], [[0, 4]])]
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
