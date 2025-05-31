# Generated code for regex_matcher
import threading
import time
from collections import defaultdict
import heapq

def regex_match(text: str, pattern: str) -> bool:
    def match_here(s: str, p: str) -> bool:
        """Match text starting at s against pattern starting at p."""
        if not p:
            return not s
        
        first_match = bool(s) and (p[0] == s[0] or p[0] == '.')
        
        if len(p) >= 2 and p[1] == '*':
            # Two cases:
            # 1. Zero occurrences of the preceding element
            # 2. One or more occurrences of the preceding element
            return (match_here(s, p[2:]) or 
                    (first_match and match_here(s[1:], p)))
        else:
            return first_match and match_here(s[1:], p[1:])
    
    return match_here(text, pattern)

# Example usage:
print(regex_match("aa", "a"))       # False
print(regex_match("aa", "a*"))      # True
print(regex_match("ab", ".*"))     # True
print(regex_match("aab", "c*a*b"))   # True
print(regex_match("mississippi", "mis*is*p*."))  # False

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
