import pytest

from nova.src.backend.core import Tensor
from nova.src.losses import MeanSquaredError


@pytest.mark.parametrize(
    "reduction_method, y_true, y_pred, expected",
    [
        ("mean", Tensor([1.0, 3.0, 5.0]), Tensor([4.0, 3.0, 2.0]), 6),
        ("mean", Tensor([1.0, 3.0, 5.0]), Tensor([1.0, 3.0, 5.0]), 0),
    ],
)
def test_mean_squared_error(reduction_method, y_true, y_pred, expected):
    mse = MeanSquaredError(reduction_method=reduction_method)
    result = mse(y_true, y_pred)
    print(result)

    assert result.to_numpy() == expected
