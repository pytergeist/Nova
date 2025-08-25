"""
Fusion Tensor module exposing Tensor<float> (for composition)
"""
from __future__ import annotations
import numpy
import typing
from . import factory
__all__ = ['Random', 'Tensor', 'factory']
class Random:
    @typing.overload
    def __init__(self, arg0: int) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    def uniform_cpp(self, shape: list[int], min: float, max: float) -> Tensor:
        """
        Create a uniform distribution of a shape between min and max values
        """
class Tensor:
    def __add__(self, arg0: Tensor) -> Tensor:
        ...
    def __ge__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __init__(self, shape: list[int]) -> None:
        """
        Construct a Tensor of given shape, zero‐initialized.
        """
    @typing.overload
    def __init__(self, shape: list[int], data: list[float]) -> None:
        """
        Construct a Tensor from a shape list and a flat data list.
        """
    def __isub__(self, arg0: Tensor) -> Tensor:
        ...
    def __matmul__(self, arg0: Tensor) -> Tensor:
        """
        Matrix multiplication (A @ B)
        """
    def __mul__(self, arg0: Tensor) -> Tensor:
        ...
    def __pow__(self, other: Tensor) -> Tensor:
        """
        Elementwise power: each element raised to corresponding element of other tensor.
        """
    def __repr__(self) -> str:
        ...
    def __rsub__(self, arg0: Tensor) -> Tensor:
        ...
    def __sub__(self, arg0: Tensor) -> Tensor:
        ...
    def __truediv__(self, arg0: Tensor) -> Tensor:
        ...
    def exp(self) -> Tensor:
        ...
    def log(self) -> Tensor:
        ...
    def maximum(self, other: Tensor) -> Tensor:
        ...
    def ones_like(self) -> Tensor:
        """
        Return a Tensor of zeros with the same shape as this one.
        """
    def set_values(self, values: list[float]) -> None:
        """
        Fill the Tensor with a flat list of length prod(shape).
        """
    def sqrt(self) -> Tensor:
        ...
    def sum(self) -> Tensor:
        ...
    def to_numpy(self) -> numpy.ndarray[numpy.float32]:
        """
        Return a NumPy array view of the Tensor’s contents.
        """
    def transpose(self) -> Tensor:
        """
        Return a new Tensor that is the transpose of this one.
        """
    def zeros_like(self) -> Tensor:
        """
        Return a Tensor of zeros with the same shape as this one.
        """
    @property
    def shape(self) -> list[int]:
        """
        Returns the shape as a list of ints.
        """
    @property
    def size(self) -> int:
        """
        Returns total number of elements (product of shape).
        """
