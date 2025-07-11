from typing import TYPE_CHECKING

from nova.src.losses import Loss

if TYPE_CHECKING:
    from nova.src.backend.core import Tensor


class MeanSquaredError(Loss):
    def __init__(self, reduction_method="mean"):
        self.reduction_method = reduction_method
        super().__init__()

    def call(self, y_true: "Tensor", y_pred: "Tensor") -> "Tensor":
        return self.reduce_loss((y_true - y_pred) ** 2)
