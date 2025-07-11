from typing import List, Literal, Optional, Tuple, Type

import numpy as np

from nova.src.backend.autodiff._node import Node
from nova.src.backend.graph import TopologicalSort
from nova.src.backend.operations import Operation


class Engine:
    """The autodiff engine used to build the computational graph.

    This class represents the autodiff engine used to build the computational graph.
    It is used to create nodes in the graph, update the node index, set the node index,
    and add created nodes to a list for debugging purposes.

    Attributes:
        node_idx_counter (int): The counter used to keep track of the node index.
        created_nodes (List[Node]): A list of created nodes for debugging purposes.
    """

    def __init__(
        self, sorter_cls: Type[TopologicalSort] = TopologicalSort
    ) -> None:  # TODO: Replace typehint with better sort typehint
        self.node_idx_counter = 0  # TODO: This is in here for dev/debug purposes
        self.created_nodes = []
        self.sorter_cls = sorter_cls

    def _update_node_idx(self) -> None:
        """Updates the node index counter."""
        self.node_idx_counter += 1

    def _set_node_idx(self, node: Node) -> None:
        """Sets the node index attribute on the node. Used in the build_node method to
        set the node index on the node instance.

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
        parents: Tuple[Node, ...] = (),
        requires_grad: bool = False,
        role: Optional[Literal["kernel", "bias"]] = None,
    ) -> Node:
        """Builds a node in the computational graph.

        Args:
            data (np.ndarray): The data for the node.
            operation (Optional[Operation]): The operation that created the node.
            parents (Tuple["Node", ...]): The parent nodes of the node.
            requires_grad (bool): Flag to indicate if finite_difference should be computed.
            role (Optional[Literal["kernel", "bias"]]): The role of the node (used for weights).

        Returns:
            Node: The created node.
        """
        self._update_node_idx()
        node = Node(
            value=data,
            operation=operation,
            parents=parents,
            requires_grad=requires_grad,
            role=role,
        )
        self._set_node_idx(node)
        self._add_created_node(node)
        return node

    def build_leaf_node(  # TODO: why does this not index the node??
        self, data, requires_grad, role: Optional[Literal["kernel", "bias"]] = None
    ) -> Node:
        """Builds a leaf node in the computational graph.

        Args:
            data (np.ndarray): The data for the node.
            requires_grad (bool): Flag to indicate if finite_difference should be computed.
            role (Optional[Literal["kernel", "bias"]]): The role of the node (used for weights).

        Returns:
            Node: The created leaf node (a node with no children).
        """
        return self.build_node(
            data=data,
            operation=None,
            parents=(),
            requires_grad=requires_grad,
            role=role,
        )

    @staticmethod
    def set_node_gradient_if_none(node: Node, grad_output) -> np.ndarray:
        if grad_output is None:
            grad_output = np.ones_like(node.value, dtype=node.value.dtype)
        return grad_output

    @staticmethod
    def _zero_grad(
        sorted_nodes: List[Node],
    ) -> List[Node]:  # TODO: why are we using settattr here?
        """Recursively zero out finite_difference for parents.

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
        sorter = self.sorter_cls()
        sorted_nodes = sorter.sort(start_node, mode="iterative", reverse=False)
        sorted_nodes = self._zero_grad(
            sorted_nodes
        )  # TODO: should this be done in the backward step or live seperately?
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
        # This is a placeholder for any cleanup logic if needed.
        # Currently, it does nothing but will be extended in the future.
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Exit method for context manager pattern."""
        return False

    def get_current(self) -> "Engine":
        """Returns the current engine instance.

        Designed to be used with the context manager pattern, e.g. with Engine.current()
        as engine: similar to with Gradient.tape() as tape: in TensorFlow.
        """
        return self
