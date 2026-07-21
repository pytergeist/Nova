from abc import ABC, abstractmethod
from enum import StrEnum

from nova.src.backend.core import Tensor

_REDUCTION_FUNCTIONS: dict = {}


def _reduce_mean(values: Tensor, **kwargs) -> Tensor:
    return values.sum() / values.size


def _reduce_mean_with_sample_weight(
    values: Tensor, sample_weights: Tensor | None = None
):
    if sample_weights is None:
        raise ValueError("No sample weights provided")
    else:
        return values.sum() / sample_weights.sum()


class ReductionMethod(StrEnum):
    MEAN = "mean"
    MEAN_WITH_SAMPLE_WEIGHT = "mean_with_sample_weight"

    def apply(reduction_method, values: Tensor, sample_weights: Tensor | None = None):

        return _REDUCTION_FUNCTIONS[reduction_method](
            values, sample_weights=sample_weights
        )


_REDUCTION_FUNCTIONS.update({ReductionMethod.MEAN: _reduce_mean})
_REDUCTION_FUNCTIONS.update(
    {ReductionMethod.MEAN_WITH_SAMPLE_WEIGHT: _reduce_mean_with_sample_weight}
)


class Loss(ABC):
    """
    Base class for loss functions.
    Subclasses should define only the per-element loss.
    """

    def __init__(self, reduction_method: str = "mean"):
        self.reduction_method = reduction_method
        self._is_valid_reduction_method(self.reduction_method)

    @staticmethod
    def _is_valid_reduction_method(reduction_method):
        try:
            ReductionMethod(reduction_method)
            return
        except ValueError:
            raise ValueError(f"Invalid reduction method: {reduction_method}")

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

    def reduce_loss(self, values) -> Tensor:  # TODO: make this method more generic
        """
        Reduce the loss values based on the specified reduction method.
        Loss is being reduced from a vector of sample errors to a single scalar that the optimiser can minimise.

        Args:
            values: The loss values to be reduced.
            reduction_method: The method of reduction ('mean', 'sum', etc.).

        Returns:
            Reduced loss value scalar.
        """
        return ReductionMethod.apply(
            reduction_method=self.reduction_method, values=values
        )

    def __repr__(self):
        """
        Return a string representation of the loss function.
        """
        return f"<LossFunction: {self.__class__.__name__}() at {hex(id(self))}>"
