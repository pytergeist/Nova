from typing import List, Optional, Set, Tuple

import numpy as np

from abditus.src.backend.autodiff._node import Node
from abditus.src.backend.graph import TopologicalSort
from abditus.src.backend.operations import Operation


class Engine:
    """The autodiff engine used to build the computational graph.

    This class represents the autodiff engine used to build the computational graph.
    It is used to create nodes in the graph, update the node index, set the node index,
    and add created nodes to a list for debugging purposes.

    Attributes:
        node_idx_counter (int): The counter used to keep track of the node index.
        created_nodes (List[Node]): A list of created nodes for debugging purposes.
    """

    def __init__(self) -> None:
        self.node_idx_counter = 0  # TODO: This is in here for dev/debug purposes
        self.created_nodes = (
            []
        )  # TODO: Why node_idx starting at 8 in the print_graph function

    def _update_node_idx(self) -> None:
        """Updates the node index counter."""
        self.node_idx_counter += 1

    def _set_node_idx(self, node: Node) -> None:
        """Sets the node index attribute on the node.
            Used in the build_node method to set the node index on the node instance.

        Args:
            node (Node): The node to set the index on.
        """
        setattr(node, "idx", self.node_idx_counter)

    def _add_created_node(self, node: Node) -> None:
        """Adds the created node to the list of created nodes."""
        self.created_nodes.append(node)

    def build_node(
        self,
        data: np.ndarray,
        operation: Optional[Operation] = None,
        parents: Tuple["Node", ...] = (),
        requires_grad: bool = False,
    ) -> "Node":
        """Builds a node in the computational graph.

        Args:
            data (np.ndarray): The data for the node.
            operation (Optional[Operation]): The operation that created the node.
            parents (Tuple["Node", ...]): The parent nodes of the node.
            requires_grad (bool): Flag to indicate if gradients should be computed.

        Returns:
            Node: The created node.
        """
        self._update_node_idx()
        node = Node(
            value=data,
            operation=operation,
            parents=parents,
            requires_grad=requires_grad,
        )
        self._set_node_idx(node)
        self._add_created_node(node)
        return node

    def build_leaf_node(self, data, requires_grad) -> "Node":
        """Builds a leaf node in the computational graph.

        Args:
            data (np.ndarray): The data for the node.
            requires_grad (bool): Flag to indicate if gradients should be computed.

        Returns:
            Node: The created leaf node (a node with no children).
        """
        return self.build_node(
            data=data, operation=None, parents=(), requires_grad=requires_grad
        )

    @staticmethod
    def set_node_gradient_if_none(node: Node, grad_output) -> np.ndarray:
        if grad_output is None:
            grad_output = np.ones_like(node._value, dtype=node._value.dtype)
        return grad_output

    @staticmethod
    def _zero_grad(sorted_nodes: List[Node]) -> List[Node]:
        """Recursively zero out gradients for parents.

        Args:
            visited (Optional[Set[int]]): A set of visited node ids to prevent infinite loops.
        """
        [setattr(node, "grad", np.zeros_like(node.value)) for node in sorted_nodes]
        return sorted_nodes

    def backward(self, start_node: Node, start_grad: np.ndarray):
        """Performs the backward pass through the computational graph.

        Args:
            start_node (Node): The node to start the backward pass from.
            start_grad (np.ndarray): The gradient of the output tensor.
        """
        sorted_nodes = TopologicalSort().sort(
            start_node, mode="iterative", reverse=False
        )
        sorted_nodes = self._zero_grad(sorted_nodes)
        start_grad = self.set_node_gradient_if_none(start_node, start_grad)
        start_node.update_node_gradient(start_grad)
        for node in sorted_nodes:
            if node.operation is None:
                continue
            parent_grads = node.operation.backward_func(node, *node.parents, node.grad)
            for parent, pgrad in zip(node.parents, parent_grads):
                if parent.requires_grad:
                    pgrad = self.set_node_gradient_if_none(parent, pgrad)
                    parent.update_node_gradient(pgrad)

    def __enter__(self) -> "Engine":
        """Enter method for context manager pattern."""
        return self

    def __del__(self) -> None:
        """Destructor method for the Engine."""
        print("Engine deleted")

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Exit method for context manager pattern."""
        return False

    def current(self) -> "Engine":
        """Returns the current engine instance.
        Designed to be used with the context manager pattern, e.g. with Engine.current() as engine:
        similar to with Gradient.tape() as tape: in TensorFlow.
        """
        return self


# if __name__ == "__main__":
#     import time
#     with Engine() as engine:
#         print(engine)
#         time.sleep(5)
#         print(engine.current())
