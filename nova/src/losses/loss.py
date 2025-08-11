from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """
    Base class for loss functions.
    """

    @abstractmethod
    def call(self, *args, **kwargs): ...

    def __call__(self, *args, **kwargs):
        """
        Call the loss function with the provided arguments.
        """
        return self.call(*args, **kwargs)

    @staticmethod
    def reduce_loss(
        values: np.ndarray, reduction_method="mean"
    ):  # TODO: make this method more generic
        """
        Reduce the loss values based on the specified reduction method.

        Args:
            values: The loss values to be reduced.
            reduction_method: The method of reduction ('mean', 'sum', etc.).

        Returns:
            Reduced loss value.
        """
        if reduction_method == "mean":
            return (
                values / values.size if values.size > 0 else 0.0
            )  # TODO: implement mean in Fusion backend
        elif reduction_method == "sum":
            return values.sum()
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")

    def __repr__(self):
        """
        Return a string representation of the loss function.
        """
        return f"<LossFunction: {self.__class__.__name__}() at {hex(id(self))}>"
