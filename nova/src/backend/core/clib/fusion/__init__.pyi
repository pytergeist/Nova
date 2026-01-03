"""
Fusion Tensor module exposing Tensor<float> (for composition)
"""

from __future__ import annotations
import numpy
import typing
from . import autodiff
from . import factory

__all__ = [
    "CppDType",
    "CppDevice",
    "CppDeviceType",
    "Grad",
    "Random",
    "Tensor",
    "autodiff",
    "factory",
]

class CppDType:
    """
    Members:

      FLOAT32

      FLOAT64

      INT32

      INT64

      BOOL
    """

    BOOL: typing.ClassVar[CppDType]  # value = <CppDType.BOOL: 4>
    FLOAT32: typing.ClassVar[CppDType]  # value = <CppDType.FLOAT32: 0>
    FLOAT64: typing.ClassVar[CppDType]  # value = <CppDType.FLOAT64: 1>
    INT32: typing.ClassVar[CppDType]  # value = <CppDType.INT32: 2>
    INT64: typing.ClassVar[CppDType]  # value = <CppDType.INT64: 3>
    __members__: typing.ClassVar[
        dict[str, CppDType]
    ]  # value = {'FLOAT32': <CppDType.FLOAT32: 0>, 'FLOAT64': <CppDType.FLOAT64: 1>, 'INT32': <CppDType.INT32: 2>, 'INT64': <CppDType.INT64: 3>, 'BOOL': <CppDType.BOOL: 4>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class CppDevice:
    def __init__(self, type: DeviceType, index: int = -1) -> None: ...

class CppDeviceType:
    """
    Members:

      CPU

      CUDA

      METAL
    """

    CPU: typing.ClassVar[CppDeviceType]  # value = <CppDeviceType.CPU: 0>
    CUDA: typing.ClassVar[CppDeviceType]  # value = <CppDeviceType.CUDA: 1>
    METAL: typing.ClassVar[CppDeviceType]  # value = <CppDeviceType.METAL: 2>
    __members__: typing.ClassVar[
        dict[str, CppDeviceType]
    ]  # value = {'CPU': <CppDeviceType.CPU: 0>, 'CUDA': <CppDeviceType.CUDA: 1>, 'METAL': <CppDeviceType.METAL: 2>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Grad:
    def __enter__(self) -> Grad: ...
    def __exit__(
        self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any
    ) -> bool: ...
    def __init__(self) -> None: ...

class Random:
    def __init__(self, seed: int = 1895392949) -> None: ...
    def uniform_cpp(
        self, shape: list[int], min: float, max: float, device: Device
    ) -> Tensor: ...

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
    def __init__(
        self, shape: list[int], dtype: DType, device: Device, requires_grad: bool
    ) -> None:
        """
        Construct a Tensor of given shape, zero-initialized. Optionally set requires_grad.
        """

    @typing.overload
    def __init__(
        self,
        shape: list[int],
        data: list[float],
        dtype: DType,
        device: Device,
        requires_grad: bool,
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
    def __neg__(self) -> ...: ...
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
    @typing.overload
    def backward(self) -> None: ...
    @typing.overload
    def backward(self) -> None: ...
    def exp(self) -> Tensor: ...
    def get_grad(self) -> Tensor | None: ...
    def log(self) -> Tensor: ...
    @typing.overload
    def maximum(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def maximum(self, arg0: float) -> Tensor: ...
    def mean(self) -> Tensor:
        """
        Return the global mean of the Tensor.
        """

    def rank(self) -> int: ...
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

    @property
    def dtype(self) -> numpy.dtype[typing.Any]:
        """
        NumPy dtype of the tensor.
        """

    @property
    def name(self) -> str:
        """
        Returns the name of the Tensor
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
