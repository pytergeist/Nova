# tensor

from threading import Lock
from typing import Optional

import numpy as np

from threads.graph.node import Node
from threads.operations.registry import (add_op, divide_op, right_subtract_op,
                                         subtract_op)


class Tensor:
    def __init__(self, data, requires_grad=False, node: Optional[Node] = None) -> None:
        data = self._convert_to_numpy(data)
        self.node = self._build_leaf_node(data, requires_grad, node)
        self.lock = Lock()

    def _convert_to_numpy(self, data: list) -> np.ndarray:
        return np.asarray(data, dtype=np.float32)  # TODO: Needs editing

    @staticmethod
    def _build_leaf_node(data: np.ndarray, requires_grad: bool, node: Node) -> Node:
        # TODO: write validation to ensure data is np array
        if node is None:
            node = Node(
                value=data,
                operation=None,  # leaf nodes hold no op
                parents=(),
                requires_grad=requires_grad,
            )
        return node

    @property
    def data(self) -> np.ndarray:
        return self.node.value

    @property
    def grad(self) -> np.ndarray:
        return self.node.grad

    @grad.setter
    def grad(self, new_grad: np.ndarray):
        self.node.grad = new_grad

    @property
    def requires_grad(self) -> np.ndarray:
        return self.node.requires_grad

    @property
    def value(self):
        return self.node.value

    def backward(self, grad_output: Optional[np.ndarray] = None):
        self.node.backward(grad_output)

    def _apply_op(self, other, operation):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_value = operation.forward_func(self, other)

        out_node = Node(
            value=out_value,
            operation=operation,
            parents=(self.node, other.node),
            requires_grad=self.node.requires_grad or other.node.requires_grad,
        )
        return Tensor(data=None, node=out_node)

    def __add__(self, other):
        return self._apply_op(other, add_op)

    def __sub__(self, other):
        return self._apply_op(other, subtract_op)

    def __rsub__(self, other):
        return self._apply_op(other, right_subtract_op)

    def __truediv__(self, other):
        return self._apply_op(other, divide_op)


if __name__ == "__main__":
    a = Tensor(np.array([1, 2, 3]), requires_grad=True)
    b = Tensor(np.array([4, 5, 7]), requires_grad=True)

    c = a / b
    print(c.data)  # [0.25, 0.4, 0.42857143]

    c.backward()  # Compute gradients
    print(a.grad)  # [0.25,0.2,0.142857]
    print(b.grad)  # [−0.0625,−0.08,−0.061224]
