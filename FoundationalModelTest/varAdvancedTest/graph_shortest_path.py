# Generated code for graph_shortest_path
import threading
import time
from collections import defaultdict
import heapq

import heapq

def dijkstra_shortest_path(graph, start):
    """
    Find the shortest path from a start node to all other nodes in a weighted graph.
    
    :param graph: A dictionary representing the graph where keys are nodes and values are lists of tuples (neighbor, weight).
    :param start: The starting node.
    :return: A dictionary where keys are nodes and values are the shortest distances from the start node.
    """
    # Priority queue to store the nodes to be explored
    pq = []
    heapq.heappush(pq, (0, start))
    
    # Dictionary to store the shortest distance from the start node to each node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            # Only consider this new path if it's better
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example usage:
if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 5)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 5), ('C', 1)]
    }
    start_node = 'A'
    print(dijkstra_shortest_path(graph, start_node))

# Test cases
def run_tests():
    test_results = []
    
    for graph, start, expected in test_cases:
        try:
            result = dijkstra_shortest_path(graph, start)
            test_results.append((f"dijkstra_shortest_path({start})", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"dijkstra_shortest_path test", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [({'A': [('B', 4), ('C', 2)], 'B': [('D', 3)], 'C': [('D', 1), ('E', 5)], 'D': [('E', 1)], 'E': []}, 'A', {'A': 0, 'B': 4, 'C': 2, 'D': 3, 'E': 4}), ({'1': [('2', 1)], '2': [('3', 2)], '3': [], '4': []}, '1', {'1': 0, '2': 1, '3': 3, '4': inf})]
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
