# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from typing import List, Set

from nova.src.backend.autodiff import Node


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

    def dfs_recursive(self, node: Node):
        """Depth-first search traversal of the graph. Currently implemented recursively.

        Args:
            node (Node): The current node in the traversal.
        """
        if id(node) not in self.visited:
            self.visited.add(id(node))
            for parent in node.parents:
                self.dfs(parent)
            self.order.append(node)

    def dfs_iterative(self, node: Node):
        """Depth-first search traversal of the graph. Currently implemented iteratively.

        Args:
            node (Node): The current node in the traversal.
        """
        stack = [node]
        while stack:
            current_node = stack.pop()
            if id(current_node) not in self.visited:
                self.visited.add(id(current_node))
                for parent in current_node.parents:
                    stack.append(parent)
                self.order.append(current_node)

    def sort(
        self, start_node: Node, mode: str = "iterative", reverse: bool = False
    ) -> List[Node]:
        """Sort the nodes in the graph topologically.

        The Engine class is used to build the computational graph, the starting node is the node that
        the .backward function is called on (last node in the graph). Therefore, the topological sort needs to be
        reversed for backpropagation.

        IMPORTANT NOTE: The graph for backpropagation needs to be returned in reverse order, the behaviour of the
        two DFS algorithms is different. The recursive DFS will add child nodes after there parents, returning the
        graph sorted from the first created node to the last; therefore the order needs to be reversed. The iterative
        DFS will add child nodes before there parents, returning the graph sorted from the last created node to the
        first; therefore the order does not need to be reversed.

        Args:
            start_node (Node): The node to start the topological sort from.
            mode (str): The mode to use for the depth-first search traversal, either iterative or recursive
            reverse (bool): True if the order should be reversed, False otherwise.

        returns:
            List[Node]: The topological sort order of the nodes in the graph.
        """
        assert mode in [
            "iterative",
            "recursive",
        ], f"Invalid mode: {mode}, please choose between iterative or recursive"

        dfs_fn = getattr(self, "".join(["dfs_", mode]))
        dfs_fn(start_node)

        if reverse:
            return self.order[::-1]
        return self.order
