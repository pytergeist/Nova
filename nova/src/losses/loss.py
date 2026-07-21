from abc import ABC, abstractmethod
from enum import Enum

from nova.src.backend.core import Tensor


class ReductionMethods(str, Enum):
    MEAN = "mean"
    MEAN_WITH_SAMPLE_WEIGHT = "mean_with_sample_weight"
    SUM = "sum"

    def apply(self, values: Tensor, sample_weights: Tensor | None = None):
        match self:
            case ReductionMethods.MEAN:
                return values.sum() / values.size()
            case ReductionMethods.SUM:
                return values.sum()
            case ReductionMethods.MEAN_WITH_SAMPLE_WEIGHT:
                if sample_weights is None:
                    raise ValueError("sample weights must be provided.")
                return values.sum() / (sample_weights.sum() + 1e-8)


class Loss(ABC):
    """
    Base class for loss functions.
    """

    def __init__(self, reduction: ReductionMethods = ReductionMethods.MEAN):
        self.reduction = reduction

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
        Call the loss function with the provided arguments.
        """
        return self.call(*args, **kwargs)

    @staticmethod
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
        self.reduction.apply(values, sample_weights)

    def __repr__(self):
        """
        Return a string representation of the loss function.
        """
        return f"<LossFunction: {self.__class__.__name__}() at {hex(id(self))}>"
