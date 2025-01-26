import numpy as np
from typing import Optional

from threads.operations.operation import Operation


class Node:
    def __init__(
        self,
        value: np.ndarray,
        operation: Optional[Operation] = None,
        parents: tuple = (),
        requires_grad: bool = False,
    ):
        self.value = value
        self.operation = operation
        self.parents = parents
        self.requires_grad = requires_grad
        self.grad = None
        self.name = None

    def backward(self, grad_output: Optional[np.ndarray] = None):
        """
        Recursively compute gradients for parents.
        """
        if grad_output is None:
            grad_output = np.ones_like(self.value, dtype=self.value.dtype)

        if self.grad is None:
            self.grad = grad_output
        else:
            self.grad += grad_output

        if self.operation is None or not self.requires_grad:
            return

        parent_grads = self.operation.backward_func(self, *self.parents, grad_output)
        for parent, pgrad in zip(self.parents, parent_grads):
            if parent.requires_grad:
                parent.backward(pgrad)

    def __repr__(self):
        return f"""Node(name={self.name}, value={self.value}, 
        operation={self.operation}, parents={self.parents},
        requires_grad={self.requires_grad})"""


if __name__ == "__main__":
    root_node = Node(np.array([1, 2, 3]))
    print(root_node)
