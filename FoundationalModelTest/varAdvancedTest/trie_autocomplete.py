# Generated code for trie_autocomplete
import threading
import time
from collections import defaultdict
import heapq

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class TrieAutocomplete:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Inserts a word into the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        """Searches for a word in the trie."""
        node = self._search_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix):
        """Checks if there's any word in the trie that starts with the given prefix."""
        return self._search_node(prefix) is not None

    def get_all_words_with_prefix(self, prefix):
        """Returns a list of all words that start with the given prefix."""
        node = self._search_node(prefix)
        if not node:
            return []
        
        result = []
        self._collect_words(node, prefix, result)
        return result

    def _search_node(self, prefix):
        """Helper method to find the node corresponding to the end of the prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect_words(self, node, current_prefix, result):
        """Recursively collects all words starting from the given node."""
        if node.is_end_of_word:
            result.append(current_prefix)
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_prefix + char, result)

# Example usage:
trie = TrieAutocomplete()
words = ["hello", "helium", "hero", "heron", "world"]
for word in words:
    trie.insert(word)

print(trie.search("hello"))  # True
print(trie.starts_with("he"))   # True
print(trie.get_all_words_with_prefix("he"))  # ['hello', 'helium', 'hero', 'heron']

# Test cases
def run_tests():
    test_results = []
    
    for words, prefix, expected in test_cases:
        try:
            trie = TrieAutocomplete()
            for word in words:
                trie.insert(word)
            result = sorted(trie.get_all_words_with_prefix(prefix))
            expected_sorted = sorted(expected)
            test_results.append((f"TrieAutocomplete({words}).get_all_words_with_prefix('{prefix}')", result, expected_sorted, result == expected_sorted))
        except Exception as e:
            test_results.append((f"TrieAutocomplete test", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [(['hello', 'help', 'hero', 'her', 'world'], 'he', ['hello', 'help', 'hero', 'her']), (['apple', 'app', 'application'], 'app', ['apple', 'app', 'application']), (['test'], 'testing', []), ([], 'any', [])]
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
