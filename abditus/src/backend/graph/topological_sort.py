from typing import List

from abditus.src.backend.autodiff import Node


class TopologicalSort:
    def __init__(self, start_node: Node) -> None:
        self.start_node = start_node
        self.visited = set()
        self.order = []

    def dfs(
        self, node: Node
    ):  # TODO: Make this iterative DFS - Recursive can cause max depth issues
        if id(node) not in self.visited:
            self.visited.add(id(node))
            for parent in node.parents:
                self.dfs(parent)
            self.order.append(node)

    def sort(self):
        self.dfs(self.start_node)
        return self.order[::-1]


if __name__ == "__main__":
    import numpy as np

    from abditus.src.backend.core import Tensor
    from abditus.src.backend.graph import print_graph

    A = Tensor(np.ones_like([1, 1, 1]), requires_grad=True)
    B = Tensor(np.ones_like([1, 1, 1]), requires_grad=True)
    C = Tensor(np.ones_like([1, 1, 1]), requires_grad=True)

    D = A + B + C

    D.backward()

    order = TopologicalSort(D._node).sort()

    print([type(n) for n in order])

    print_graph(D._node)
