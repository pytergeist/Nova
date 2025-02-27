from typing import List


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
