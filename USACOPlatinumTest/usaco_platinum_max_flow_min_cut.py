# Generated code for USACO Platinum: Network Flow with Vertex Capacities
import sys
import heapq
from collections import defaultdict, deque
import math
from typing import List, Dict, Set, Tuple, Optional

from collections import defaultdict, deque

def bfs(capacity, flow, source, sink):
    parent = {}
    visited = set()
    queue = deque([source])
    
    while queue:
        u = queue.popleft()
        if u == sink:
            break
        
        for v, cap in capacity[u].items():
            if v not in visited and cap - flow[u][v] > 0:
                visited.add(v)
                parent[v] = u
                queue.append(v)
    
    return parent if sink in visited else None

def ford_fulkerson(capacity, source, sink):
    n = len(capacity)
    flow = [[0] * n for _ in range(n)]
    
    max_flow = 0
    while True:
        parent = bfs(capacity, flow, source, sink)
        if not parent:
            break
        
        path_flow = float('Inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, capacity[parent[s]][s] - flow[parent[s]][s])
            s = parent[s]
        
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = parent[v]
        
        max_flow += path_flow
    
    return max_flow

def max_flow_vertex_capacity(graph, vertex_capacities, source, sink):
    # Transform the graph
    new_graph = defaultdict(list)
    capacity = defaultdict(lambda: defaultdict(int))
    
    # Create new nodes for each vertex with non-zero capacity
    node_map = {}
    for v in vertex_capacities:
        if vertex_capacities[v] > 0:
            v_in = v + 1000  # Assuming N <= 100, so v_in = v + 1000 won't overlap
            v_out = v + 2000
            node_map[v] = (v_in, v_out)
            capacity[v_in][v_out] = vertex_capacities[v]
    
    # Add edges to the new graph
    for u in graph:
        for v, edge_capacity in graph[u]:
            if u in node_map:
                u = node_map[u][1]
            if v in node_map:
                v = node_map[v][0]
            new_graph[u].append(v)
            capacity[u][v] = edge_capacity
    
    # Adjust source and sink if they are split
    if source in node_map:
        source = node_map[source][0]
    if sink in node_map:
        sink = node_map[sink][1]
    
    # Calculate max flow
    return ford_fulkerson(capacity, source, sink)

# Example usage
graph = {
    1: [(2, 10), (3, 10)],
    2: [(4, 10)],
    3: [(4, 10)],
    4: []
}
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
