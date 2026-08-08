from nova.src.backend.core import Tensor
from nova.src.losses import Loss


class MeanSquaredError(Loss):
    def __init__(self, reduction_method="mean"):
        super().__init__(reduction_method=reduction_method)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return (y_true - y_pred) ** 2


class MeanAbsoluteError(Loss):
    def __init__(self, reduction_method="mean"):
        super().__init__(reduction_method=reduction_method)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        # TODO: when abs function has been implemented, refactor this.
        return ((y_true - y_pred) ** 2).sqrt()


class BinaryCrossEntropy(Loss):
    def __init__(self, reduction_method="mean"):
        super().__init__(reduction_method=reduction_method)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        delta = 1e-7  # Delta added to account for numerical stability. Log 0 is giving a nan. Remove when implemented in C++.
        # Also refactor this equation when Tensor's don't have to be on the left.

        positive_term = y_true * (y_pred + delta).log()
        negative_term = (y_true * -1 + 1) * (y_pred * -1 + 1 + delta).log()

        return (positive_term + negative_term) * -1
