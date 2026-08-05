import pytest

from nova.src.backend.core import Tensor
from nova.src.losses import (
    BinaryCrossEntropy,
    Loss,
    MeanAbsoluteError,
    MeanSquaredError,
)


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
    assert result.to_numpy() == expected


@pytest.mark.parametrize(
    "reduction_method, y_true, y_pred, expected",
    [
        ("mean", Tensor([1.0, 3.0, 5.0]), Tensor([4.0, 3.0, 2.0]), 2),
        ("mean", Tensor([1.0, 3.0, 5.0]), Tensor([1.0, 3.0, 5.0]), 0),
    ],
)
def test_mean_absolute_error(reduction_method, y_true, y_pred, expected):
    mae = MeanAbsoluteError(reduction_method=reduction_method)
    result = mae(y_true, y_pred)
    assert result.to_numpy() == expected


@pytest.mark.parametrize(
    "reduction_method, y_true, y_pred, expected",
    [
        ("mean", Tensor([1.0, 0.0, 1.0]), Tensor([1.0, 0.0, 0.6]), 0.17027505),
        ("mean", Tensor([1.0, 0.0, 1.0]), Tensor([0.553, 0.3253, 0.2352]), 0.8110675),
    ],
)
def test_binary_cross_entropy(reduction_method, y_true, y_pred, expected):
    bce = BinaryCrossEntropy(reduction_method=reduction_method)
    result = bce(y_true, y_pred)
    assert result.to_numpy() == expected


class DummyLoss(Loss):
    def call(self, y_true, y_pred, sample_weights=None, **kwargs):
        return


@pytest.mark.parametrize(
    "reduction_method, expected",
    [("mean", "mean"), ("mean_with_sample_weight", "mean_with_sample_weight")],
)
def test_is_valid_reduction_method(reduction_method, expected):
    loss = DummyLoss(reduction_method)
    assert loss.reduction_method == expected


@pytest.mark.parametrize("reduction_method", ["invalid", "average", "mode", None])
def test_invalid_reduction_method(reduction_method):
    with pytest.raises(
        ValueError, match=f"Invalid reduction method: {reduction_method}"
    ):
        DummyLoss(reduction_method)


@pytest.mark.parametrize("expected", ["DummyLoss"])
def test_repr(expected):
    assert repr(DummyLoss()) == f"<LossFunction: {expected}>"
