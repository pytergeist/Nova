import numpy as np
import pytest

from neurothread.tensor import Tensor


@pytest.mark.parametrize(
    "data_a, data_b, expected_data",
    [
        ([1, 2, 3], [4, 5, 6], [5, 7, 9]),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1]),
        ([1, -1, 1], [-1, 1, -1], [0, 0, 0]),
    ],
)
def test_tensor_addition(data_a, data_b, expected_data):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    c = a + b
    np.testing.assert_array_equal(c.data, expected_data)
    assert c.requires_grad is True


@pytest.mark.parametrize(
    "data_a, data_b, expected_data",
    [
        ([1, 2, 3], [4, 5, 6], [-3, -3, -3]),
        ([0, 0, 0], [1, 1, 1], [-1, -1, -1]),
        ([1, -1, 1], [-1, 1, -1], [2, -2, 2]),
    ],
)
def test_tensor_subtraction(data_a, data_b, expected_data):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    c = a - b
    np.testing.assert_array_equal(c.data, expected_data)
    assert c.requires_grad is True


@pytest.mark.parametrize(
    "data_a, data_b, expected_data",
    [
        ([1, 2, 3], [4, 5, 6], [3, 3, 3]),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1]),
        ([1, -1, 1], [-1, 1, -1], [-2, 2, -2]),
    ],
)
def test_tensor_right_subtraction(data_a, data_b, expected_data):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    c = b - a
    np.testing.assert_array_equal(c.data, expected_data)
    assert c.requires_grad is True


@pytest.mark.parametrize(

    "data_a, data_b, expected_grad_a, expected_grad_b",
    [
        ([1, 2, 3], [4, 5, 6], [1, 1, 1], [1, 1, 1]),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
        ([1, -1, 1], [-1, 1, -1], [1, 1, 1], [1, 1, 1]),
    ],
)
def test_tensor_backward_addition(data_a, data_b, expected_grad_a, expected_grad_b):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    c = a + b
    c.backward()
    np.testing.assert_array_equal(a.grad, expected_grad_a)
    np.testing.assert_array_equal(b.grad, expected_grad_b)


@pytest.mark.parametrize(
    "data_a, data_b, expected_grad_a, expected_grad_b",
    [
        ([1, 2, 3], [4, 5, 6], [1, 1, 1], [-1, -1, -1]),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], [-1, -1, -1]),
        ([1, -1, 1], [-1, 1, -1], [1, 1, 1], [-1, -1, -1]),
    ],
)
def test_tensor_backward_subtraction(data_a, data_b, expected_grad_a, expected_grad_b):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    c = a - b
    c.backward()
    np.testing.assert_array_equal(a.grad, expected_grad_a)
    np.testing.assert_array_equal(b.grad, expected_grad_b)


@pytest.mark.parametrize( # TODO: is this correct?
    "data_a, data_b, expected_grad_a, expected_grad_b",
    [
        ([1, 2, 3], [4, 5, 6], [1, 1, 1], [-1, -1, -1]),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], [-1, -1, -1]),
        ([1, -1, 1], [-1, 1, -1], [1, 1, 1], [-1, -1, -1]),
    ],
)
def test_tensor_backward_right_subtraction(data_a, data_b, expected_grad_a, expected_grad_b):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    c = a - b
    c.backward()
    np.testing.assert_array_equal(a.grad, expected_grad_a)
    np.testing.assert_array_equal(b.grad, expected_grad_b)
