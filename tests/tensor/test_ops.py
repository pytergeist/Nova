import numpy as np
import pytest

from neurothread.operations.registry import (add_op, divide_op,
                                             right_subtract_op, subtract_op)
from neurothread.tensor import Tensor


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

    result_data = add_op.forward_func(a, b)
    requires_grad_result = a.requires_grad or b.requires_grad

    np.testing.assert_array_equal(result_data, expected_data)
    assert requires_grad_result == requires_grad


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

    result_data = subtract_op.forward_func(a, b)
    requires_grad_result = a.requires_grad or b.requires_grad

    np.testing.assert_array_equal(result_data, expected_data)
    assert requires_grad_result == requires_grad


@pytest.mark.parametrize(
    "data_a, data_b, expected_data, requires_grad",
    [
        ([1, 2, 3], [4, 5, 6], [3, 3, 3], False),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], False),
        ([1, -1, 1], [-1, 1, -1], [-2, 2, -2], False),
    ],
)
def test_right_subtract(data_a, data_b, expected_data, requires_grad):
    a = Tensor(data_a)
    b = Tensor(data_b)

    result_data = right_subtract_op.forward_func(b, a)
    requires_grad_result = a.requires_grad or b.requires_grad

    np.testing.assert_array_equal(result_data, expected_data)
    assert requires_grad_result == requires_grad


@pytest.mark.parametrize(
    "data_a, data_b, expected_data, requires_grad",
    [
        ([1, 2, 3], [4, 5, 6], [0.25, 0.4, 0.5], False),
        ([0, 0, 0], [1, 1, 1], [0.0, 0.0, 0.0], False),
        ([1, -2, 3], [-1, 2, -3], [-1.0, -1.0, -1.0], False),
    ],
)
def test_true_div(data_a, data_b, expected_data, requires_grad):
    a = Tensor(data_a)
    b = Tensor(data_b)

    result_data = divide_op.forward_func(a, b)
    requires_grad_result = a.requires_grad or b.requires_grad

    np.testing.assert_array_almost_equal(result_data, expected_data, decimal=5)
    assert requires_grad_result == requires_grad
