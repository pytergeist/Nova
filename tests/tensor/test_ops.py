import numpy as np
import pytest

from threads.operations.registry import (
    add_op,
    divide_op,
    matmul_op,
    right_subtract_op,
    subtract_op,
    sum_op,
)
from threads.tensor import Tensor


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

    result_data = right_subtract_op.forward_func(a, b)
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


@pytest.mark.parametrize(
    "input_data, expected_data, requires_grad",
    [
        ([1, 2, 3], 6.0, False),
        ([[1, 2], [3, 4]], 10.0, False),
        ([0, 0, 0], 0.0, False),
    ],
)
def test_sum(input_data, expected_data, requires_grad):
    """
    Tests sum_op's forward_func using arrays with various shapes.
    """
    a = Tensor(input_data)
    result_data = sum_op.forward_func(a)
    requires_grad_result = a.requires_grad

    np.testing.assert_almost_equal(result_data, expected_data, decimal=5)
    assert requires_grad_result == requires_grad


@pytest.mark.parametrize(
    "data_a, data_b, expected_data, requires_grad",
    [
        (
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[19.0, 22.0], [43.0, 50.0]],
                False,
        ),
        ([[1.0, 2.0, 3.0]], [[10.0], [20.0], [30.0]], [[140.0]], False),
    ],
)
def test_matmul(data_a, data_b, expected_data, requires_grad):
    """
    Tests matmul_op's forward_func for matrix multiplication.
    """
    a = Tensor(data_a)
    b = Tensor(data_b)

    result_data = matmul_op.forward_func(a, b)
    requires_grad_result = a.requires_grad or b.requires_grad

    np.testing.assert_array_almost_equal(result_data, expected_data, decimal=5)
    assert requires_grad_result == requires_grad


if __name__ == "__main__":
    pytest.main([__file__])
