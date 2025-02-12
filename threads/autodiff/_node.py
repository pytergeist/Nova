from typing import Optional, Tuple

import numpy as np

from threads.operations.operation import Operation


class GraphNode:
    def __init__(
        self,
        value: np.ndarray,  # TODO: Should this be a tensor: "Tensor"? is it ever not a tensor?
        operation: Optional[Operation] = None,
        parents: Tuple["GraphNode", ...] = (),
        requires_grad: bool = False,
    ):
        self._value = value
        self._operation = operation
        self._parents = parents
        self._requires_grad = requires_grad
        self._grad = None

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def grad(self) -> Optional[np.ndarray]:
        return self._grad

    @grad.setter
    def grad(self, grad_value: np.ndarray) -> None:
        self._grad = grad_value

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def operation(self) -> Optional[Operation]:
        return self._operation

    @property
    def parents(self) -> Tuple["GraphNode", ...]:
        return self._parents

    def backward(self, grad_output: Optional[np.ndarray] = None):
        """
        Recursively compute gradients for parents.
        """
        if not self._requires_grad:
            return

        if grad_output is None:
            grad_output = np.ones_like(self._value, dtype=self._value.dtype)

        if self._grad is None:
            self._grad = grad_output
        else:
            self._grad += grad_output

        if self._operation is None:
            return

        parent_grads = self._operation.backward_func(self, *self._parents, grad_output)
        for parent, pgrad in zip(self.parents, parent_grads):
            if parent.requires_grad:
                parent.backward(pgrad)

    def __repr__(self):
        return f"""
        Node(value={self.value},
        operation={self.operation}, parents={self.parents},
        requires_grad={self.requires_grad})
        """


if __name__ == "__main__":
    root_node = GraphNode(np.array([1, 2, 3]))
    print(root_node)
