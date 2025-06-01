# Generated code for USACO Platinum: Shortest Path in DAG with Modified Edges
import sys
import heapq
from collections import defaultdict, deque
import math
from typing import List, Dict, Set, Tuple, Optional

from collections import defaultdict, deque

def topo_sort(graph):
    n = len(graph)
    indegree = [0] * n
    for u in graph:
        for v, _ in graph[u]:
            indegree[v] += 1
    
    queue = deque([u for u in range(n) if indegree[u] == 0])
    order = []
    
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, _ in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    
    return order

def shortest_path_with_discount(graph, source, destination):
    n = len(graph)
    order = topo_sort(graph)
    
    # Initialize distances
    dist = [[float('inf')] * 2 for _ in range(n)]
    dist[source][0] = 0
    dist[source][1] = 0
    
    # Process nodes in topological order
    for u in order:
        for v, w in graph[u]:
            if dist[u][0] + w < dist[v][0]:
                dist[v][0] = dist[u][0] + w
                dist[v][1] = dist[u][1] + w
            if dist[u][1] + w / 2 < dist[v][1]:
                dist[v][1] = dist[u][1] + w / 2
    
    return min(dist[destination])

# Example usage:
graph = {
    0: [(1, 10), (2, 20)],
    1: [(3, 5)],
    2: [(3, 3)],
    3: []
}
source = 0
destination = 3
print(shortest_path_with_discount(graph, source, destination))  # Output: 10

# Test cases
def run_tests():
    test_results = []
    
    for graph, source, destination, expected in test_cases:
        try:
            result = shortest_path_with_discount(graph, source, destination)
            # Allow small floating point differences
            passed = abs(result - expected) < 1e-6
            test_results.append((f"Shortest path DAG: {source}→{destination}", result, expected, passed))
        except Exception as e:
            test_results.append((f"Shortest path DAG test", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [({0: [(1, 10), (2, 20)], 1: [(3, 5)], 2: [(3, 3)], 3: []}, 0, 3, 10.0), ({0: [(1, 8)], 1: [(2, 12)], 2: []}, 0, 2, 14.0), ({0: [(1, 6), (2, 4)], 1: [(2, 2), (3, 8)], 2: [(3, 10)], 3: []}, 0, 3, 9.0)]
    try:
        results = run_tests()
        
        all_passed = True
        for test_name, result, expected, passed in results:
            status = "✓" if passed else "✗"
            print(f"{status} {test_name}")
            print(f"    Result: {result}")
            print(f"    Expected: {expected}")
            print(f"    Status: {'PASS' if passed else 'FAIL'}")
            print()
            if not passed:
                all_passed = False
        
        overall_status = 'PASS' if all_passed else 'FAIL'
        print(f"Overall Result: {overall_status}")
        print(f"Difficulty Level: USACO PLATINUM (Advanced)")
        exit(0 if all_passed else 1)
    except Exception as e:
        print(f"Error running tests: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        exit(1)
