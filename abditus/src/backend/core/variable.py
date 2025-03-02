import numpy as np

from abditus.src.backend.core import Tensor


class Variable(Tensor):
    """A variable tensor used as a model parameter.

    This class represents a tensor that is used as a parameter in a model.
    It inherits from the Tensor class and is employed in the base classes for the
    high-level neural network API.

    Attributes:
        data (np.ndarray): The underlying data of the tensor.
        requires_grad (bool): Always True for Variable tensors, overridden from Tensor.
    """

    def __init__(self, data: np.ndarray, requires_grad: bool = True, **kwargs) -> None:
        """Initializes the Variable.

        Args:
            data (np.ndarray): A numpy array containing the tensor data.
            requires_grad (bool, optional): Flag to indicate if finite_difference should be computed.
                Always to True.
            **kwargs: Additional keyword arguments to be passed to the parent Tensor class.
        """
        super().__init__(data, requires_grad=requires_grad, **kwargs)

    def __repr__(self) -> str:
        """Returns a string representation of the Variable.

        Returns:
            str: A string showing the Variable's data.
        """
        return f"Variable(data={self.data}"
