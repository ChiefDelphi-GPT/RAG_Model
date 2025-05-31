# Generated code for advanced_dp
import threading
import time
from collections import defaultdict
import heapq

def longest_common_subsequence(str1: str, str2: str) -> int:
    """
    Returns the length of the longest common subsequence between two strings.
    
    :param str1: First input string
    :param str2: Second input string
    :return: Length of the longest common subsequence
    """
    m, n = len(str1), len(str2)
    # Create a 2D array to store lengths of longest common subsequence.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build the dp array from bottom up
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def reconstruct_lcs(str1: str, str2: str) -> str:
    """
    Reconstructs the longest common subsequence between two strings.
    
    :param str1: First input string
    :param str2: Second input string
    :return: The longest common subsequence as a string
    """
    m, n = len(str1), len(str2)
    # Create a 2D array to store lengths of longest common subsequence.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build the dp array from bottom up
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct the LCS from the dp array
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    # The reconstructed LCS is built backwards, so reverse it
    lcs.reverse()
    return ''.join(lcs)

# Example usage:
str1 = "AGGTAB"
str2 = "GXTXAYB"
print("Length of LCS:", longest_common_subsequence(str1, str2))  # Output: 4
print("LCS:", reconstruct_lcs(str1, str2))  # Output: "GTAB"

# Test cases
def run_tests():
    test_results = []
    
    for s1, s2, expected_len, expected_lcs in test_cases:
        try:
            length = longest_common_subsequence(s1, s2)
            try:
                actual_lcs = reconstruct_lcs(s1, s2)
                is_correct = (length == expected_len) and (actual_lcs == expected_lcs)
                test_results.append((f"LCS('{s1}', '{s2}')", (length, actual_lcs), (expected_len, expected_lcs), is_correct))
            except:
                # If reconstruct_lcs fails, just test length
                is_correct = length == expected_len
                test_results.append((f"LCS length('{s1}', '{s2}')", length, expected_len, is_correct))
        except Exception as e:
            test_results.append((f"LCS test", f"ERROR: {e}", (expected_len, expected_lcs), False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [('ABCDGH', 'AEDFHR', 3, 'ADH'), ('AGGTAB', 'GXTXAYB', 4, 'GTAB'), ('', 'ABC', 0, ''), ('ABC', '', 0, ''), ('ABCDEF', 'ABCDEF', 6, 'ABCDEF')]
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
