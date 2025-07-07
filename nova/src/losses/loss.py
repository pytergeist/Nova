import numpy as np


class Loss:
    """
    Base class for loss functions.
    """

    def __call__(self, *args, **kwargs):
        """
        Call the loss function with the provided arguments.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def reduce_loss(values: np.ndarray, reduction_method="mean"):
        """
        Reduce the loss values based on the specified reduction method.

        Args:
            values: The loss values to be reduced.
            reduction_method: The method of reduction ('mean', 'sum', etc.).

        Returns:
            Reduced loss value.
        """
        if reduction_method == "mean":
            return values.mean()
        elif reduction_method == "sum":
            return values.sum()
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")

    def __repr__(self):
        """
        Return a string representation of the loss function.
        """
        return f"<LossFunction: {self.__class__.__name__}() at {hex(id(self))}>"
