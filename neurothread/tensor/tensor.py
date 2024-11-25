# tensor

import numpy as np
from threading import Lock


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
        Performs element-wise addition of two tensors with autograd support.

        Args:
            other (Tensor): The tensor to add to `self`.

        Returns:
            Tensor: A new tensor representing the result of the addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            with self.lock:
                if self.requires_grad:
                    self.grad = (
                        np.ones_like(self.data)
                        if self.grad is None
                        else self.grad + np.ones_like(self.data)
                    )

            with other.lock:
                if other.requires_grad:
                    other.grad = (
                        np.ones_like(other.data)
                        if other.grad is None
                        else other.grad + np.ones_like(other.data)
                    )
        result._backward = _backward
        return result

if __name__ == "__main__":
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)

    c = a + b  # Element-wise addition
    print(c.data)  # [5, 7, 9]

    c.backward()  # Compute gradients
    print(a.grad)  # [1, 1, 1] (since d(c)/d(a) = 1)
    print(b.grad)  # [1, 1, 1] (since d(c)/d(b) = 1)

