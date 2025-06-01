# Generated code for regex_matcher
import threading
import time
from collections import defaultdict
import heapq

def regex_match(text: str, pattern: str) -> bool:
    def match_here(text_index: int, pattern_index: int) -> bool:
        # If the pattern is exhausted, check if the text is also exhausted
        if pattern_index == len(pattern):
            return text_index == len(text)
        
        # Check for '*' in the pattern
        if pattern_index + 1 < len(pattern) and pattern[pattern_index + 1] == '*':
            # '*' means zero or more of the preceding element
            # Try matching zero occurrences of the current character
            if match_here(text_index, pattern_index + 2):
                return True
            # Try matching one or more occurrences of the current character
            while text_index < len(text) and (pattern[pattern_index] == '.' or text[text_index] == pattern[pattern_index]):
                if match_here(text_index + 1, pattern_index + 2):
                    return True
                text_index += 1
            return False
        
        # Check for '.' or exact character match
        if pattern[pattern_index] == '.' or (text_index < len(text) and text[text_index] == pattern[pattern_index]):
            return match_here(text_index + 1, pattern_index + 1)
        
        return False

    return match_here(0, 0)

# Example usage:
print(regex_match("aab", "c*a*b"))  # True
print(regex_match("mississippi", "mis*is*p*."))  # False
print(regex_match("ab", ".*"))  # True
print(regex_match("abc", "a.c"))  # True
print(regex_match("abcd", "d*"))  # False

# Test cases
def run_tests():
    test_results = []
    
    for text, pattern, expected in test_cases:
        try:
            result = regex_match(text, pattern)
            test_results.append((f"regex_match('{text}', '{pattern}')", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"regex_match('{text}', '{pattern}')", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [('aa', 'a', False), ('aa', 'a*', True), ('ab', '.*', True), ('aab', 'c*a*b', True), ('mississippi', 'mis*is*p*.', False), ('mississippi', 'mis*is*ip*.', True)]
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
