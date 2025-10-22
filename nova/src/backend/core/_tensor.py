from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np

from nova.src.backend.autodiff import Engine, Node
from nova.src.backend.core import clib

if TYPE_CHECKING:
    from nova.src.backend.core.dtypes import DType


class Tensor(clib.Tensor):
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
        data: Union[Sequence, np.ndarray, float, int],
        requires_grad: bool = True,
        dtype: "DType" = np.float64,
        role: Optional[Literal["kernel", "bias"]] = None,
    ):
        arr = np.array(data, dtype=dtype)
        shape = list(arr.shape)
        if not shape:
            shape = [1]
        flat = arr.ravel().tolist()
        if not isinstance(flat, list):
            flat = [flat]
        super().__init__(shape, flat, requires_grad)

    @property
    def data(self) -> np.ndarray:
        """Data property getter. There is no corresponding setter as the data attribute
        is read-only.

        Returns:
            np.ndarray: The underlying data of the tensor.
        """
        return self.to_numpy()

    @property
    def grad(self) -> np.ndarray:
        """Grad (gradient) property getter.

        Returns:
            np.ndarray: The gradient of the tensor.
        """
        return self.get_grad()

    @staticmethod
    def standardise_dtype(
        dtype: "DType",
    ):  # TODO: This needs to be abstracted into a separate module
        if dtype is None:
            return np.float32

    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype)

    @property
    def shape(self):
        return super().shape

    @property
    def T(self):
        return self.swapaxes(-1, -2)


def get_cpp(t: Tensor) -> clib.Tensor:
    return t
