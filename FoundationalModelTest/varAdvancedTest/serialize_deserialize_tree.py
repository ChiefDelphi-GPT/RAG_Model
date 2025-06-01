# Generated code for serialize_deserialize_tree
import threading
import time
from collections import defaultdict
import heapq

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def serialize_binary_tree(root):
    """Serialize a binary tree to a string."""
    def serialize_helper(node):
        if not node:
            return '#'
        return f"{node.val},{serialize_helper(node.left)},{serialize_helper(node.right)}"
    
    return serialize_helper(root)

def deserialize_binary_tree(data):
    """Deserialize a string to a binary tree."""
    def deserialize_helper(nodes):
        value = next(nodes)
        if value == '#':
            return None
        node = TreeNode(int(value))
        node.left = deserialize_helper(nodes)
        node.right = deserialize_helper(nodes)
        return node
    
    # Split the data string into an iterator of values
    node_values = iter(data.split(','))
    return deserialize_helper(node_values)

# Example usage:
if __name__ == "__main__":
    # Create a sample binary tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.right.left = TreeNode(4)
    root.right.right = TreeNode(5)

    # Serialize the binary tree
    serialized_data = serialize_binary_tree(root)
    print("Serialized:", serialized_data)

    # Deserialize the binary tree
    deserialized_root = deserialize_binary_tree(serialized_data)
    print("Deserialized Root Value:", deserialized_root.val if deserialized_root else "None")

# Test cases
def run_tests():
    test_results = []
    
    # Manual tree construction test
    try:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right
        
        # Create test tree: 1(2, 3(4, 5))
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.right.left = TreeNode(4)
        root.right.right = TreeNode(5)
        
        # Serialize and deserialize
        serialized = serialize_binary_tree(root)
        deserialized = deserialize_binary_tree(serialized)
        
        # Check if trees are equal (simple structure check)
        def trees_equal(t1, t2):
            if not t1 and not t2:
                return True
            if not t1 or not t2:
                return False
            return (t1.val == t2.val and 
                   trees_equal(t1.left, t2.left) and 
                   trees_equal(t1.right, t2.right))
        
        result = trees_equal(root, deserialized)
        test_results.append(("serialize/deserialize tree", result, True, result))
    except Exception as e:
        test_results.append(("serialize/deserialize tree", f"ERROR: {e}", True, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [('manual_tree_test', True)]
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
