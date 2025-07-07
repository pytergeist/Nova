import numpy as np

from nova.src.losses import Loss


class MeanSquaredError(Loss):
    def __init__(self, reduction_method="mean"):
        self.reduction_method = reduction_method
        super().__init__()

    def call(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.reduce_loss((y_true - y_pred) ** 2)


if __name__ == "__main__":
    import numpy as np

    mse = MeanSquaredError()
    print(str(mse))
    y_true = np.asarray([1.0, 2.0, 3.0])
    y_pred = np.asarray([1.5, 2.5, 3.5])
    loss_value = mse.call(y_true, y_pred)
    print(f"Mean Squared Error: {loss_value}")
