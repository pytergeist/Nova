from typing import List


def depth_first_search(graph: List[List[int]], start: int = 0) -> List[int]:
    visited = set()
    result = []

    def dfs_visit(node):
        if id(node) not in visited:
            visited.add(id(node))
            result.append(node)
            for neighbour in graph[node]:
                dfs_visit(neighbour)
        return result

    result = dfs_visit(start)
    return result
