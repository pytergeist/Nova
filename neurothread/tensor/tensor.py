# tensor

from threading import Lock

import numpy as np

from neurothread.tensor import AutoDiff, add, subtract


class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        Initializes a Tensor object.

        Args:
            data: Numerical data (NumPy array or Python list).
            requires_grad: If True, tracks gradients for this tensor.
        """
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self.lock = Lock()

    def backward(self):
        """
        Computes gradients for the current tensor and its dependencies.
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot backward a Tensor with requires_grad=False")

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        self._backward()

    def __add__(self, other):
        """
        Performs element-wise addition of two tensors with operator overloading autograd support.

        Args:
            other (Tensor): The tensor to add to `self`.

        Returns:
            Tensor: A new tensor representing the result of the addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        data, requires_grad = add(self, other)
        result = Tensor(data, requires_grad=requires_grad)

        def _backward():
            grad_output = np.ones_like(result.data)
            AutoDiff.add_backward(self, other, grad_output)

        result._backward = _backward
        return result

    def __sub__(self, other):
        """
        Performs element-wise subtraction of two tensors with operator overloading autograd support.

        Args:
            other (Tensor): The tensor to add to `self`.

        Returns:
            Tensor: A new tensor representing the result of the addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        data, requires_grad = subtract(self, other)
        result = Tensor(data, requires_grad=requires_grad)

        def _backward():
            grad_output = np.ones_like(result.data)
            AutoDiff.subtract_backward(self, other, grad_output)

        result._backward = _backward
        return result

    def __mul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __floordiv__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __abs__(self):
        raise NotImplementedError




if __name__ == "__main__":
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 7], requires_grad=True)

    c = a - b  # Element-wise addition
    print(c.data)  # [5, 7, 9]

    c.backward()  # Compute gradients
    print(a.grad)  # [1, 1, 1] (since d(c)/d(a) = 1)
    print(b.grad)  # [1, 1, 1] (since d(c)/d(b) = 1)
