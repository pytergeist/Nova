from nova.src.backend.core import Tensor
from nova.src.losses import Loss


class MeanSquaredError(Loss):
    def __init__(self, reduction_method="mean"):
        super().__init__(reduction_method=reduction_method)

    def call(
        self, y_true: Tensor, y_pred: Tensor, sample_weights: Tensor | None = None
    ) -> Tensor:
        return self.reduce_loss(
            (y_true - y_pred) ** 2
        )  # can we make it so we don't have to call reduce here every time?
