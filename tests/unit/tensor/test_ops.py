# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import numpy as np
import pytest

from nova.src.backend.core import Tensor, autodiff

autodiff.enabled(True)


@pytest.mark.parametrize(
    "data_a, data_b, expected_data, requires_grad",
    [
        ([1, 2, 3], [4, 5, 6], [5, 7, 9], False),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], False),
        ([1, -1, 1], [-1, 1, -1], [0, 0, 0], False),
        ([1, 2, 3], [0], [1, 2, 3], False),
        ([0, 2, 3], [1], [1, 3, 4], False),
    ],
)
def test_add(data_a, data_b, expected_data, requires_grad):
    a = Tensor(data_a, requires_grad=False)
    b = Tensor(data_b, requires_grad=False)

    result_data = (a + b).to_numpy()
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
    a = Tensor(data_a, requires_grad=False)
    b = Tensor(data_b, requires_grad=False)

    result_data = (a - b).to_numpy()
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
    a = Tensor(data_a, requires_grad=False)
    b = Tensor(data_b, requires_grad=False)

    result_data = (b - a).to_numpy()
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
    a = Tensor(data_a, requires_grad=False)
    b = Tensor(data_b, requires_grad=False)

    result_data = (a / b).to_numpy()
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
    """Tests sum_op's forward_func using arrays with various shapes."""
    a = Tensor(input_data, requires_grad=False)
    result_data = a.sum().to_numpy()
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
def test_2d_matmul(data_a, data_b, expected_data, requires_grad):
    """Tests matmul_op's forward_func for matrix multiplication."""
    a = Tensor(data_a, requires_grad=False)
    b = Tensor(data_b, requires_grad=False)

    result_data = (a @ b).to_numpy()
    requires_grad_result = a.requires_grad or b.requires_grad

    np.testing.assert_array_almost_equal(result_data, expected_data, decimal=5)
    assert requires_grad_result == requires_grad


@pytest.mark.parametrize(
    "data_a, data_b, expected_data, requires_grad",
    [
        (
            [
                [[3, 1], [4, 4]],
                [[3, 1], [4, 4]],
            ],
            [
                [[2, 4], [1, 3]],
                [[4, 2], [4, 3]],
            ],
            [
                [[7, 15], [12, 28]],
                [[16, 9], [32, 20]],
            ],
            False,
        )
    ],
)
def test_3d_matmul(data_a, data_b, expected_data, requires_grad):
    """Tests matmul_op's forward_func for matrix multiplication."""
    a = Tensor(data_a, requires_grad=False)
    b = Tensor(data_b, requires_grad=False)

    result_data = (a @ b).to_numpy()
    requires_grad_result = a.requires_grad or b.requires_grad

    np.testing.assert_array_almost_equal(result_data, expected_data, decimal=5)
    assert requires_grad_result == requires_grad


if __name__ == "__main__":
    pytest.main([__file__])
