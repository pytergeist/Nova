# tensor

from threading import Lock
from typing import Optional

import numpy as np

from threads.autodiff._node import GraphNode
from threads.operations.registry import (
    add_op,
    divide_op,
    matmul_op,
    right_subtract_op,
    subtract_op,
    sum_op,
)


class Tensor:
    def __init__(
        self,
        data,
        requires_grad=False,
        dtype=np.float32,
        node: Optional[GraphNode] = None,
    ) -> None:
        if data is not None and not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)

        # If no node is provided, build a leaf GraphNode.
        if node is None:
            node = GraphNode(
                value=data, operation=None, parents=(), requires_grad=requires_grad
            )
        self._node = node

    @property
    def data(self) -> np.ndarray:
        return self._node.value

    @property
    def grad(self) -> np.ndarray:
        return self._node.grad

    @grad.setter
    def grad(self, new_grad: np.ndarray):
        self._node.grad = new_grad

    @property
    def requires_grad(self) -> bool:
        return self._node.requires_grad

    @property
    def value(self):
        return self._node.value

    @staticmethod
    def standardise_dtype(dtype):  # TODO: add more dtypes, create dtype maps
        if dtype is None:
            return np.float32

    def backward(self, grad_output: Optional[np.ndarray] = None) -> None:
        self._node.backward(grad_output)

    def _apply_op(self, other, operation):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_value = operation.forward_func(self, other)

        out_node = GraphNode(
            value=out_value,
            operation=operation,
            parents=(self._node, other._node),
            requires_grad=self._node.requires_grad or other._node.requires_grad,
        )
        return Tensor(data=None, node=out_node)

    def _apply_unary_op(self, operation):
        """
        Like _apply_op, but for single-input (unary) operations.
        """
        out_value = operation.forward_func(self)  # forward pass
        out_node = GraphNode(
            value=out_value,
            operation=operation,
            parents=(self._node,),
            requires_grad=self._node.requires_grad,
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

    def __matmul__(self, other):
        return self._apply_op(other, matmul_op)

    def sum(self):
        return self._apply_unary_op(sum_op)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"


if __name__ == "__main__":
    A = Tensor(np.random.randn(2, 3), requires_grad=True)
    B = Tensor(np.random.randn(3, 4), requires_grad=True)

    C = A @ B
    print("Forward: ", C.data)

    loss = C.sum()

    loss.backward()

    print("dLoss/dA = ", A.grad)
    print("dLoss/dB = ", B.grad)
