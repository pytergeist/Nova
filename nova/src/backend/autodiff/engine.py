# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from typing import List, Literal, Optional, Tuple, Type

import numpy as np

from nova.src.backend.autodiff._node import Node
from nova.src.backend.graph import TopologicalSort
from nova.src.backend.operations import Operation


class Engine:
    """
    The engine used to manage the nodes in, and build, the computational autodiff graph.

    This class represents the autodiff engine used to build the computational graph.
    It is used to create nodes in the graph, update the node index, set the node index,
    and add created nodes to a list for debugging purposes.

    Attributes:
        node_idx_counter (int): The counter used to keep track of the node index.
        created_nodes (List[Node]): A list of created nodes for debugging purposes.
        sorter_cls (TopologicalSort): A graph sorting class that implements recursive and iterative dfs
    """

    def __init__(
        self, sorter_cls: Type[TopologicalSort] = TopologicalSort
    ) -> None:  # TODO: Replace typehint with better sort typehint
        self.node_idx_counter = 0  # TODO: This is in here for dev/debug purposes
        self.created_nodes = []
        self.sorter_cls = sorter_cls

    def _update_node_idx(self) -> None:
        self.node_idx_counter += 1

    def _set_node_idx(self, node: Node) -> None:
        """
        Sets the node index attribute on the node. Used in the build_node method to
        set the node index on the node instance at run time.

        Args:
            node (Node): The node to set the index on.
        """
        setattr(node, "idx", self.node_idx_counter)

    def _add_created_node(self, node: Node) -> None:
        self.created_nodes.append(node)

    def build_node(
        self,
        data: np.ndarray,
        operation: Optional[Operation] = None,
        parents: Tuple[Node, ...] = (),
        requires_grad: bool = False,
        role: Optional[Literal["kernel", "bias"]] = None,
    ) -> Node:
        """
        Builds a node in the computational graph. This method is used in the Tensor class,
        inside the _apply_op and _apply_unary_op method to build a node in the autodiff
        graph with an operation registered to it. The build_node operation is called in
        the _apply(_unary)_op fn's, which sets the value of the new node as the result
        from the unary/binary operation and sets the new nodes parents as the Tensor._node
        value(s) that created that resultant data. This is followed by
        '_create_new_fusion_wrapped_tensor', which creates a new Fusion (C++) Tensor and
        sets the _node attr of that Tensor to the newly created node. Code examples below:

        ```python
        data = ...
        t1 = Tensor(v1) # This Tensor will be a root node, (see build_leaf_node doc str)
        t2 = Tensor(v1) # This Tensor will be a root node, (see build_leaf_node doc str)
        t3 = t1 + t2 # This Tensor will node the newly created node

        # The value of t3's node will be the direct result of the operation applied to t1-/t2.node.value
        t3.node.value  = t1.node.value + t2.node.value
        t3.node.parents = (t1.node, t2.node)

        # Autodiff graph will then look like below:
        t1 (root node)    t2(root node)
        |                 |
        ->   (Add op)   <-
                |
                t3
        ```

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
        """
        Builds a leaf node in the computational graph. This method is used in the Tensor class
        on initialisation when the _node parameter passed to the Tensor is None. The reason for
        this is that this method sets the parents (e.g. Tensors that created the "current" Tensor)
        to an empty tuple. This is done as when a user manually initialises a new Tensor in code, this
        Tensor is likely to be the root for further operation, e.g. below:

        ```python
        data = ...
        t1 = Tensor(v1) # This Tensor will be a root node, invoking build_leaf_node
        t2 = Tensor(v1) # This Tensor will be a root node, invoking build_leaf_node
        t3 = t1 + t2 # This Tensor will not be a root node,
                     # instead the build_node method is invoked at run time
                     # see build_node for worked examples

        # Autodiff graph will then look like below:
        t1 (root node)    t2(root node)
        |                 |
        ->   (Add op)   <-
                |
                t3

        ```

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
    def _zero_grad(sorted_nodes: List[Node]) -> List[Node]:
        [setattr(node, "grad", np.zeros_like(node.value)) for node in sorted_nodes]
        return sorted_nodes

    def backward(self, start_node: Node, start_grad: np.ndarray):
        """
        Performs the backward pass through the computational graph.

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
        """Enter method for context manager pattern.
        Invoked with the with statement.
        """
        return self

    def __del__(self) -> None:
        # This is a placeholder for any cleanup logic if needed.
        # Currently, it does nothing but delete when out of scope
        # will be extended in the future.
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        # Current logic just exits when out of scope
        # will be extended in the future
        return False

    def get_current(self) -> "Engine":
        """
        Returns the current engine instance.

        Designed to be used with the context manager pattern, e.g. with Engine.get_current()
        as engine: similar to with Gradient.tape() as tape: in TensorFlow etc.
        """
        return self
