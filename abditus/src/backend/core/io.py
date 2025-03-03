from typing import TYPE_CHECKING

import numpy as np

from abditus.src.backend.dtypes import as_dtype

from ._tensor import Tensor
from .variable import Variable

if TYPE_CHECKING:
    from abditus.src.backend.dtypes import DType


def as_numpy_array(data: list | np.ndarray, dtype: "DType") -> np.ndarray:
    """Converts a list to a numpy array.

    Args:
        data (list): The list to convert to a numpy array.
        dtype (DType): The data type of the numpy array (see backend.dtypes module).

    Returns:
        np.ndarray: A numpy array.
    """

    dtype = as_dtype(dtype)

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, list):
        return np.array(data, dtype=dtype)

    raise ValueError(f"Invalid data type provided: {type(data)}")


def as_tensor(data: np.ndarray, requires_grad: bool = False, dtype="float32") -> Tensor:
    """Converts a numpy array to a tensor.

    Args:
        data (np.ndarray): The numpy array to convert to a tensor.
        requires_grad (bool, optional): Flag to indicate if finite_difference should be computed.
        dtype (np.dtype, optional): The data type of the tensor. Defaults to np.float32.

    Returns:
        Tensor: A tensor object.
    """
    dtype = as_dtype(dtype)
    array = np.array(data, dtype=dtype)
    return Tensor(data=array, requires_grad=requires_grad, dtype=dtype)


def as_variable(data: np.ndarray, dtype: np.dtype = "float32") -> Variable:
    """Converts a numpy array to a variable tensor.

    Args:
        data (np.ndarray): The numpy array to convert to a tensor.
        dtype (np.dtype, optional): The data type of the tensor. Defaults to np.float32.

    Returns:
        Variable: A variable tensor object.
    """
    dtype = as_dtype(dtype)
    array = np.array(data, dtype=dtype)
    return Variable(data=array, requires_grad=True, dtype=dtype)
