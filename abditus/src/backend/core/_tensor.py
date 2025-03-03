from typing import Optional, Union

import numpy as np

from abditus.src.backend.autodiff import Engine, Node
from abditus.src.backend.operations import (
    Operation,
    add_op,
    divide_op,
    matmul_op,
    maximum_op,
    right_subtract_op,
    subtract_op,
    sum_op,
)


class Tensor:
    """A Tensor data-structure that uses operator overloading to perform element-wise
    operations.

    This class represents a tensor that uses operator overloading, registered operations (from the operations
     module) and the autodiff engine to perform element-wise operations on tensors in the forward pass and track
     their finite_difference in the backward pass. It is used as a base class for the Variable class, which
      is used as a model parameter in the high-level neural network API.

    Attributes:
        data (np.ndarray): The underlying data of the tensor.
        requires_grad (bool): True if finite_difference should be computed, False otherwise.
        dtype (np.dtype): The data type of the tensor.
        engine (Engine): The autodiff engine used to build the computational graph.
        node (Node): The node in the computational graph that represents the tensor.
    """

    def __init__(
        self,
        data,
        requires_grad=False,
        dtype=np.float32,
        engine: Engine = Engine(),
        # TODO: add engine=None, add get_current() for use with context manager pattern - is this correct?
        node: Optional[
            Node
        ] = None,  # TODO: should this be here? as the engine manages all nodes
    ) -> None:
        self.engine = engine
        if data is not None and not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)

        # If no node is provided, build a leaf GraphNode.
        if node is None:
            node = self.engine.build_leaf_node(data, requires_grad)
        self._node = node

    @property
    def data(self) -> np.ndarray:
        """Data property getter. There is no corresponding setter as the data attribute
        is read-only.

        Returns:
            np.ndarray: The underlying data of the tensor.
        """
        return self._node.value

    @property
    def grad(self) -> np.ndarray:
        """Grad (gradient) property getter.

        Returns:
            np.ndarray: The gradient of the tensor.
        """
        return self._node.grad

    @grad.setter
    def grad(self, new_grad: np.ndarray):
        """Grad (gradient) property setter.

        Updates the gradient of the tensor.
        """
        self._node.grad = new_grad

    @property
    def requires_grad(self) -> bool:
        """Grad (gradient) property getter. There is no corresponding setter as the
        requires_grad attribute is read-only.

        Returns:
            bool: True if finite_difference should be computed, False otherwise.
        """
        return self._node.requires_grad

    @staticmethod
    def standardise_dtype(
        dtype,
    ):  # TODO: This needs to be abstracted into a separate module
        if dtype is None:
            return np.float32

    def backward(self, grad_output: Optional[np.ndarray] = None) -> None:
        """Backward pass to compute finite_difference.

        Args:
            grad_output (np.ndarray, optional): The gradient of the output tensor. Defaults to None.

        Zero's out the finite_difference of the current node to prevent double counting
         and recursively computes finite_difference for parents.
        """
        self.engine.backward(self._node, grad_output)

    def _apply_op(
        self, other: Union["Tensor", np.ndarray, float, int], operation: Operation
    ):
        """Apply a binary operation to this tensor and another tensor or scalar.

        Args:
            other (Union[Tensor, np.ndarray, float, int]): The other tensor or scalar to apply the operation to.
            operation (Operation): The operation to apply.

        Returns:
            Tensor: The result of the operation (a third Tensor).
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_value = operation.forward_func(self, other)

        out_node = self.engine.build_node(
            data=out_value,
            operation=operation,
            parents=(self._node, other._node),
            requires_grad=self._node.requires_grad or other._node.requires_grad,
        )
        return Tensor(data=None, node=out_node)

    def _apply_unary_op(self, operation: Operation) -> "Tensor":
        """Like _apply_op, but for single-input (unary) operations."""
        out_value = operation.forward_func(self)  # forward pass
        out_node = self.engine.build_node(
            data=out_value,
            operation=operation,
            parents=(self._node,),
            requires_grad=self._node.requires_grad,
        )
        return Tensor(data=None, node=out_node)

    def __add__(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        """Addition operator overload.

        args:
            other (Union[Tensor, np.ndarray, float, int]): The other tensor or scalar to add.

        Returns:
            Tensor: The result of the addition.
        """
        return self._apply_op(other, add_op)

    def __sub__(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        """Subtraction operator overload.

        args:
            other (Union[Tensor, np.ndarray, float, int]): The other tensor or scalar to subtract.

        Returns:
            Tensor: The result of the subtraction.
        """
        return self._apply_op(other, subtract_op)

    def __rsub__(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        """Right subtraction operator overload.

        args:
            other (Union[Tensor, np.ndarray, float, int]): The other tensor or scalar to subtract.

        Returns:
            Tensor: The result of the subtraction.
        """
        return self._apply_op(other, right_subtract_op)

    def __truediv__(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        """Division operator overload.

        args:
            other (Union[Tensor, np.ndarray, float, int]): The other tensor or scalar to divide.

        Returns:
            Tensor: The result of the division.
        """
        return self._apply_op(other, divide_op)

    def __matmul__(self, other: Union["Tensor", np.ndarray]) -> "Tensor":
        """Matrix multiplication operator overload.

        args:
            other (Union[Tensor, np.ndarray): The other tensor or array.

        Returns:
            Tensor: The result of the matrix multiplication
        """
        return self._apply_op(other, matmul_op)

    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype)

    def sum(self):
        """Sum the elements of the tensor.

        Returns:
            Tensor: The sum of the tensor elements.
        """
        return self._apply_unary_op(sum_op)

    def maximum(self, other: Union["Tensor", np.ndarray]):
        """Elementwise max between this tensor and `other`.

        Mimics np.maximum(self, other).
        """
        return self._apply_op(other, maximum_op)

    def __mul__(self, other):
        raise NotImplementedError("Multiplication not yet implemented")

    def __rmul__(self, other):
        raise NotImplementedError("Multiplication not yet implemented")

    def __pow__(self, other):
        raise NotImplementedError("Power not yet implemented")

    def __rpow__(self, other):
        raise NotImplementedError("Power not yet implemented")

    def __neg__(self):
        raise NotImplementedError("Negation not yet implemented")

    def __floordiv__(self, other):
        raise NotImplementedError("Floor division not yet implemented")

    def __rfloordiv__(self, other):
        raise NotImplementedError("Floor division not yet implemented")

    def __radd__(self, other):
        raise NotImplementedError("Addition not yet implemented")

    def __rtruediv__(self, other):
        raise NotImplementedError("Division not yet implemented")

    def exp(self):
        raise NotImplementedError("Exponential not yet implemented")

    def log(self):
        raise NotImplementedError("Logarithm not yet implemented")

    def sqrt(self):
        raise NotImplementedError("Square root not yet implemented")

    def sin(self):
        raise NotImplementedError("Sine not yet implemented")

    def cos(self):
        raise NotImplementedError("Cosine not yet implemented")

    def tan(self):
        raise NotImplementedError("Tangent not yet implemented")

    def __abs__(self):
        raise NotImplementedError("Absolute value not yet implemented")

    def mean(self):
        raise NotImplementedError("Mean not yet implemented")

    def min(self):
        raise NotImplementedError("Minimum not yet implemented")

    def argmax(self):
        raise NotImplementedError("Argmax not yet implemented")

    def argmin(self):
        raise NotImplementedError("Argmin not yet implemented")

    def prod(self):
        raise NotImplementedError("Product not yet implemented")

    def shape(self):
        raise NotImplementedError("Shape not yet implemented")

    def reshape(self):
        raise NotImplementedError("Reshape not yet implemented")

    def flatten(self):
        raise NotImplementedError("Flatten not yet implemented")

    def transpose(self):
        raise NotImplementedError("Transpose not yet implemented")

    def expand_dims(self):
        raise NotImplementedError("Expand dims not yet implemented")

    def squeeze(self):
        raise NotImplementedError("Squeeze not yet implemented")

    def __getitem__(self, item):
        raise NotImplementedError("Indexing not yet implemented")

    def __setitem__(self, key, value):
        raise NotImplementedError("Indexing not yet implemented")

    def astype(self):
        raise NotImplementedError("astype not yet implemented")

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"


if __name__ == "__main__":

    A = Tensor(np.array([1.0, -1.0, 1.0]), requires_grad=True)
    A = A.maximum(0.0)

    print(A)

    A.backward()

    print(A)
    print(A.grad)
