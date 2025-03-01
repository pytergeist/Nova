from typing import List, Set

from abditus.src.backend.autodiff import Node


class TopologicalSort:
    """Topological sort of the nodes in the graph.

    This class is used to perform a topological sort of nodes in the computation graph for backpropagation.

    Attributes:
        visited (set[int]): A set of visited nodes, in-built id function used to assign unique int to node instances.
        order (List[Node]): The topological sort order of the nodes in the graph.
    """

    def __init__(self) -> None:
        self.visited: Set[int] = set()
        self.order: list = []

    def dfs(
        self, node: Node
    ):  # TODO: Make this iterative DFS - Recursive can cause max depth issues
        """Depth-first search traversal of the graph. Currently implemented recursively.

        Args:
            node (Node): The current node in the traversal.
        """
        if id(node) not in self.visited:
            self.visited.add(id(node))
            for parent in node.parents:
                self.dfs(parent)
            self.order.append(node)

    def sort(self, start_node: Node, reverse: bool = True) -> List[Node]:
        """Sort the nodes in the graph topologically.

        The Engine class is used to build the computational graph, the starting node is the node that
        the .backward function is called on (last node in the graph). Therefore, the topological sort needs to be
        reversed for backpropagation.

        Args:
            start_node (Node): The node to start the topological sort from.
            reverse (bool): True if the order should be reversed, False otherwise.

        returns:
            List[Node]: The topological sort order of the nodes in the graph.
        """
        self.dfs(start_node)
        if reverse:
            return self.order[::-1]
        return self.order


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
