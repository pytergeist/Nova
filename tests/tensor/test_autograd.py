import numpy as np
import pytest

from threads.autodiff.autodiff import AutoDiff
from threads.tensor import Tensor


@pytest.mark.parametrize(
    "data_a, data_b, grad_output, expected_grad_a, expected_grad_b",
    [
        ([1, 2, 3], [4, 5, 6], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
    ],
)
def test_add_backward(data_a, data_b, grad_output, expected_grad_a, expected_grad_b):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    grad_output = np.array(grad_output)

    AutoDiff.apply("add", a, b, grad_output)

    np.testing.assert_array_equal(a.grad, expected_grad_a)
    np.testing.assert_array_equal(b.grad, expected_grad_b)


@pytest.mark.parametrize(
    "data_a, data_b, grad_output, expected_grad_a, expected_grad_b",
    [
        ([1, 2, 3], [4, 5, 6], [1, 1, 1], [1, 1, 1], [-1, -1, -1]),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [-1, -1, -1]),
    ],
)
def test_subtract_backward(
    data_a, data_b, grad_output, expected_grad_a, expected_grad_b
):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    grad_output = np.array(grad_output)

    AutoDiff.apply("subtract", a, b, grad_output)

    np.testing.assert_array_equal(a.grad, expected_grad_a)
    np.testing.assert_array_equal(b.grad, expected_grad_b)
