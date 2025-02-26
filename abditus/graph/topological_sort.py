from typing import List

graph = [
    [1, 2],    # Node 0 -> Nodes 1, 2
    [3, 4],    # Node 1 -> Nodes 3, 4
    [4, 5],    # Node 2 -> Nodes 4, 5
    [6],       # Node 3 -> Node 6
    [6, 7],    # Node 4 -> Nodes 6, 7
    [7, 8],    # Node 5 -> Nodes 7, 8
    [8],       # Node 6 -> Node 8
    [8],       # Node 7 -> Node 8
    [],        # Node 8 has no outgoing edges
    [10, 11],  # Node 9  -> Nodes 10, 11 (second disconnected component)
    [12],      # Node 10 -> Node 12
    [12, 13],  # Node 11 -> Nodes 12, 13
    [13],      # Node 12 -> Node 13
    []         # Node 13 has no outgoing edges
]



def topological_sort(graph: List[List[int]]) -> List[int]:
    visited = set()
    order = []

    def dfs(node):
        if id(node) not in visited:
            visited.add(id(node))
            for neighbour in graph[node]:
                dfs(neighbour)
            order.append(node)

    for node in range(len(graph)):
        dfs(node)
    return order


print(topological_sort(graph))
