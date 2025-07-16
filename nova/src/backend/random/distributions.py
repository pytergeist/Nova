from typing import TYPE_CHECKING

from nova.src.backend.core import io
from nova.src.backend.core._C import random_methods

if TYPE_CHECKING:
    from nova.src.backend.core import Tensor


def uniform(
    shape: tuple[int, ...],
    min: float = 0.0,
    max: float = 1.0,
) -> "Tensor":
    """
    Generates a C++ tensor with values uniformly distributed between `min` and `max`.
    Converts C++ tensor to a Python tensor.

    Args:
        shape (tuple[int, ...]): The shape of the output tensor.
        min (float): The minimum value of the uniform distribution.
        max (float): The maximum value of the uniform distribution.
        dtype (str): The data type of the output tensor.

    Returns:
        Tensor: A tensor with values uniformly distributed between `min` and `max`.
    """

    data = random_methods.uniform(shape=shape, min=min, max=max).to_numpy()
    return io.as_tensor(data)
