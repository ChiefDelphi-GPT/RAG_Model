# Generated code for USACO Platinum: Network Flow with Vertex Capacities
import sys
import heapq
from collections import defaultdict, deque
import math
from typing import List, Dict, Set, Tuple, Optional

from collections import defaultdict, deque

def bfs_capacity(graph, source, sink, parent):
    visited = set()
    queue = deque([source])
    visited.add(source)
    
    while queue:
        u = queue.popleft()
        
        for v, capacity in graph[u]:
            if v not in visited and capacity > 0:
                queue.append(v)
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return True
    return False

def ford_fulkerson(graph, source, sink):
    parent = {}
    max_flow = 0
    
    while bfs_capacity(graph, source, sink, parent):
        path_flow = float('Inf')
        s = sink
        
        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s][1])
            s = parent[s]
        
        max_flow += path_flow
        
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v][1] -= path_flow
            graph[v][u][1] += path_flow
            v = parent[v]
    
    return max_flow

def max_flow_vertex_capacity(graph, vertex_capacities, source, sink):
    n = len(vertex_capacities)
    
    # Create new nodes for each vertex
    new_nodes = {v: (v + n, v + 2 * n) for v in vertex_capacities}
    
    # Create the transformed graph
    transformed_graph = defaultdict(list)
    
    # Add edges from original graph
    for u, edges in graph.items():
        for v, capacity in edges:
            transformed_graph[new_nodes[u][1]].append((new_nodes[v][0], capacity))
    
    # Add edges between new nodes
    for v, capacity in vertex_capacities.items():
        transformed_graph[new_nodes[v][0]].append((new_nodes[v][1], capacity))
    
    # Add source and sink connections
    for v in vertex_capacities:
        transformed_graph[source].append((new_nodes[v][0], float('Inf')))
        transformed_graph[new_nodes[v][1]].append((sink, float('Inf')))
    
    # Run Ford-Fulkerson algorithm
    return ford_fulkerson(transformed_graph, source, sink)

# Example usage
graph = {1: [(2, 10), (3, 10)], 2: [(4, 10)], 3: [(4, 10)], 4: []}
vertex_capacities = {1: 100, 2: 5, 3: 8, 4: 100}
source = 1
sink = 4
print(max_flow_vertex_capacity(graph, vertex_capacities, source, sink))  # Output: 13

# Test cases
def run_tests():
    test_results = []
    
    for graph, vertex_capacities, source, sink, expected in test_cases:
        try:
            result = max_flow_vertex_capacity(graph, vertex_capacities, source, sink)
            test_results.append((f"Max Flow: {graph} from {source} to {sink}", result, expected, result == expected))
        except Exception as e:
            test_results.append((f"Max Flow test", f"ERROR: {e}", expected, False))

    return test_results

if __name__ == "__main__":
    import traceback
    test_cases = [({1: [(2, 10), (3, 10)], 2: [(4, 10)], 3: [(4, 10)], 4: []}, {1: 100, 2: 5, 3: 8, 4: 100}, 1, 4, 13), ({1: [(2, 20)], 2: [(3, 20)], 3: []}, {1: 100, 2: 5, 3: 100}, 1, 3, 5), ({1: [(2, 10), (3, 15)], 2: [(4, 20)], 3: [(4, 10)], 4: []}, {1: 100, 2: 12, 3: 8, 4: 100}, 1, 4, 18)]
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
        print(f"Difficulty Level: USACO PLATINUM (Extremely Advanced)")
        exit(0 if all_passed else 1)
    except Exception as e:
        print(f"Error running tests: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        exit(1)
