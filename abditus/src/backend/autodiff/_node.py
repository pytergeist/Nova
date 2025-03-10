from typing import Optional, Tuple

import numpy as np

from abditus.src.backend.operations import Operation


class Node:
    """A Node in the computational graph, used for autodiff operations.

    This class represents a node in the computational graph used for autodiff operations.
    It is used to store the value of the node, the operation that created the node,
    the parents of the node, and whether the node requires finite_difference.

    Attributes:
        value (np.ndarray): The value (data) of the node.
        operation (Optional[Operation]): The operation that created the node (found in the operations module).
        parents (Tuple["Node", ...]): The parent nodes of the node (makes up the graph representation).
        requires_grad (bool): True if finite_difference should be computed, False otherwise.
        grad (Optional[np.ndarray]): The gradient of the node.
    """

    def __init__(
        self,
        value: np.ndarray,
        operation: Optional[Operation] = None,
        parents: Tuple["Node", ...] = (),
        requires_grad: bool = False,
        role: Optional[str] = None,
    ):
        self._value = value
        self._operation = operation
        self._parents = parents
        self._requires_grad = requires_grad
        self._grad = None
        self._role = role

    @property
    def value(self) -> np.ndarray:
        """Value property getter. There is no corresponding setter as the value
        attribute is read-only after initialisation.

        Returns:
            np.ndarray: The array of data or scalar stored in node.
        """
        return self._value

    @property
    def grad(self) -> Optional[np.ndarray]:
        """Grad property getter.

        Returns:
            np.ndarray: The array of data or scalar stored in node.
        """
        return self._grad

    @grad.setter
    def grad(self, grad_value: np.ndarray) -> None:
        """Grad property setter.

        Updates the grad property of the node.
        """
        self._grad = grad_value

    @property
    def requires_grad(self) -> bool:
        """Grad property getter. There is no corresponding setter as the grad attribute
        is read-only after initialisation.

        Returns:
            bool: True if finite_difference should be computed, False otherwise.
        """
        return self._requires_grad

    @property
    def role(self) -> Optional[str]:
        """The role of this node (e.g., 'bias', 'kernel')."""
        return self._role

    @role.setter
    def role(self, role: Optional[str]) -> None:
        self._role = role

    @property
    def operation(self) -> Optional[Operation]:
        """Operation property getter. There is no corresponding setter as the operation
        attribute is read-only after initialisation.

        Returns:
            Optional[Operation]: The operation that created the node.
        """
        return self._operation

    @property
    def parents(self) -> Tuple["Node", ...]:
        """Parents property getter. There is no corresponding setter as the operation
        attribute is read-only after initialisation.

        Returns:
            Tuple["Node", ...]: The parent nodes of the node.
        """
        return self._parents

    def update_node_gradient(self, grad_output: Optional[np.ndarray] = None) -> None:
        """Update the gradient of the node.

        Args:
            grad_output (Optional[np.ndarray]): The gradient of the output tensor.
        """
        if self._grad is None:
            self._grad = grad_output
        else:
            self._grad += grad_output

    def check_node_requires_grad_comp(self):
        """Check if the node requires finite_difference computation."""
        if not self._requires_grad:
            return

    def __repr__(self):
        role_str = f", role={self._role}" if self._role else ""
        return (
            f"Node(value={self.value}, operation={self.operation}, "
            f"parents={self.parents}, requires_grad={self.requires_grad}{role_str})"
        )
