import numpy as np
import pytest

from neurothread.tensor import Tensor, add, subtract


@pytest.mark.parametrize(
    "data_a, data_b, expected_data, requires_grad",
    [
        ([1, 2, 3], [4, 5, 6], [5, 7, 9], False),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], False),
        ([1, -1, 1], [-1, 1, -1], [0, 0, 0], False),
    ],
)
def test_add(data_a, data_b, expected_data, requires_grad):
    a = Tensor(data_a)
    b = Tensor(data_b)
    data, grad = add(a, b)
    np.testing.assert_array_equal(data, expected_data)
    assert grad == requires_grad


@pytest.mark.parametrize(
    "data_a, data_b, expected_data, requires_grad",
    [
        ([1, 2, 3], [4, 5, 6], [-3, -3, -3], False),
        ([0, 0, 0], [1, 1, 1], [-1, -1, -1], False),
        ([1, -1, 1], [-1, 1, -1], [2, -2, 2], False),
    ],
)
def test_subtract(data_a, data_b, expected_data, requires_grad):
    a = Tensor(data_a)
    b = Tensor(data_b)
    data, grad = subtract(a, b)
    np.testing.assert_array_equal(data, expected_data)
    assert grad == requires_grad
