# tensor

from threading import Lock

import numpy as np

from neurothread.autodiff.autodiff import AutoDiff
from neurothread.operations.ops import add_op, subtract_op


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self.lock = Lock()

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("Cannot backward a Tensor with requires_grad=False")
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        self._backward()

    def _apply_op(self, other, operation):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = operation.forward_func(self, other)
        result = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_output = np.ones_like(result.data)
            operation.backward_func(self, other, grad_output)

        result._backward = _backward
        return result

    def __add__(self, other):
        return self._apply_op(other, add_op)

    def __sub__(self, other):
        return self._apply_op(other, subtract_op)


if __name__ == "__main__":
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 7], requires_grad=True)

    c = a - b  # Element-wise addition
    print(c.data)  # [5, 7, 9]

    c.backward()  # Compute gradients
    print(a.grad)  # [1, 1, 1] (since d(c)/d(a) = 1)
    print(b.grad)  # [1, 1, 1] (since d(c)/d(b) = 1)
