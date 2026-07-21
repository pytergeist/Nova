from abc import ABC, abstractmethod

from nova.src.backend.core import Tensor


class Loss(ABC):
    """
    Base class for loss functions.
    Subclasses should define only the per-element loss.
    """

    def __init__(self, reduction_method: str = "mean"):
        self.reduction_method = reduction_method

    @abstractmethod
    def call(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        sample_weights: Tensor | None = None,
        **kwargs,
    ): ...

    def __call__(self, *args, **kwargs):
        """
        Call the loss function with the provided arguments, and reduces the loss to a single scalar.
        Allows subclasses to be called without explicit .call() method, while allowing additional operations to be chained, such as loss reduction.
        """
        values = self.call(*args, **kwargs)
        return self.reduce_loss(values)

    def reduce_loss(
        self, values: Tensor, sample_weights: Tensor | None = None
    ) -> Tensor:  # TODO: make this method more generic
        """
        Reduce the loss values based on the specified reduction method.
        Loss is being reduced from a vector of sample errors to a single scalar that the optimiser can minimise.

        Args:
            values: The loss values to be reduced.
            reduction_method: The method of reduction ('mean', 'sum', etc.).

        Returns:
            Reduced loss value scalar.
        """
        if self.reduction_method == "mean":
            loss = values.sum() / values.size
        elif self.reduction_method == "mean_with_sample_weight":
            if sample_weights is None:
                raise ValueError("sample weights empty")
            loss = values.sum() / sample_weights.sum()
        else:
            raise KeyError("invalid reduction method")
        return loss

    def __repr__(self):
        """
        Return a string representation of the loss function.
        """
        return f"<LossFunction: {self.__class__.__name__}() at {hex(id(self))}>"
