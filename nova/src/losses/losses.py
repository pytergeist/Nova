from nova.src.backend.core import Tensor
from nova.src.losses import Loss


class MeanSquaredError(Loss):
    def __init__(self, reduction_method="mean"):
        super().__init__(reduction_method=reduction_method)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return (y_true - y_pred) ** 2
