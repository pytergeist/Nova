"""
Fusion Tensor module exposing Tensor<float> (for composition)
"""

from __future__ import annotations
import numpy
import typing
from . import autodiff
from . import factory

__all__ = ["Random", "Tensor", "autodiff", "factory", "grad_tape"]

class Random:
    @typing.overload
    def __init__(self, arg0: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    def uniform_cpp(self, shape: list[int], min: float, max: float) -> Tensor:
        """
        Create a uniform distribution of a shape between min and max values
        """

class Tensor:
    @typing.overload
    def __add__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __add__(self, arg0: float) -> Tensor: ...
    @typing.overload
    def __ge__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __ge__(self, arg0: float) -> Tensor: ...
    def __gt__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __init__(self, shape: list[int], requires_grad: bool) -> None:
        """
        Construct a Tensor of given shape, zero-initialized. Optionally set requires_grad.
        """

    @typing.overload
    def __init__(
        self, shape: list[int], data: list[float], requires_grad: bool
    ) -> None:
        """
        Construct a Tensor from a shape list and a flat data list. Optionally set requires_grad.
        """

    def __isub__(self, arg0: Tensor) -> Tensor: ...
    def __matmul__(self, arg0: Tensor) -> Tensor:
        """
        Matrix multiplication (A @ B)
        """

    @typing.overload
    def __mul__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __mul__(self, arg0: float) -> Tensor: ...
    def __neg__(self) -> Tensor: ...
    @typing.overload
    def __pow__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __pow__(self, arg0: float) -> Tensor: ...
    @typing.overload
    def __pow__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __pow__(self, arg0: float) -> Tensor: ...
    def __repr__(self) -> str: ...
    @typing.overload
    def __sub__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __sub__(self, arg0: float) -> Tensor: ...
    @typing.overload
    def __truediv__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __truediv__(self, arg0: float) -> Tensor: ...
    def backward(self) -> None: ...
    def diag(self) -> Tensor: ...
    def exp(self) -> Tensor: ...
    def get_grad(self) -> Tensor: ...
    def log(self) -> Tensor: ...
    @typing.overload
    def maximum(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def maximum(self, arg0: float) -> Tensor: ...
    def mean(self) -> Tensor:
        """
        Return the global mean of the Tensor.
        """

    def ones_like(self) -> Tensor:
        """
        Ones with same shape.
        """

    def set_values(self, values: list[float]) -> None:
        """
        Fill the Tensor with a flat list of length prod(shape).
        """

    def sqrt(self) -> Tensor: ...
    def sum(self) -> Tensor: ...
    def swapaxes(self, axis1: int, axis2: int) -> Tensor: ...
    def to_numpy(self) -> numpy.ndarray[numpy.float32]:
        """
        Return a NumPy array view of the Tensor’s contents.
        """

    def transpose(self) -> Tensor:
        """
        Return the transpose.
        """

    def zeros_like(self) -> Tensor:
        """
        Zeros with same shape.
        """

    @property
    def dtype(self) -> numpy.dtype[typing.Any]:
        """
        NumPy dtype of the tensor.
        """

    @property
    def ndim(self) -> int:
        """
        Number of dimensions.
        """

    @property
    def requires_grad(self) -> bool:
        """
        Requires grad flag
        """

    @requires_grad.setter
    def requires_grad(self, arg1: bool) -> None: ...
    @property
    def shape(self) -> list[int]:
        """
        Returns the shape as a list of ints.
        """

    @property
    def size(self) -> int:
        """
        Total number of elements (product of shape).
        """

class grad_tape:
    def __enter__(self) -> grad_tape: ...
    def __exit__(
        self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any
    ) -> bool: ...
    def __init__(self) -> None: ...
