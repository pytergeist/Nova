import numpy as np
import pytest

from abditus.src.tensor import Tensor


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


@pytest.mark.parametrize(  # TODO: is this correct?
    "data_a, data_b, expected_grad_a, expected_grad_b",
    [
        ([1, 2, 3], [4, 5, 6], [1, 1, 1], [-1, -1, -1]),
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], [-1, -1, -1]),
        ([1, -1, 1], [-1, 1, -1], [1, 1, 1], [-1, -1, -1]),
    ],
)
def test_tensor_backward_right_subtraction(
    data_a, data_b, expected_grad_a, expected_grad_b
):
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    c = a - b
    c.backward()
    np.testing.assert_array_equal(a.grad, expected_grad_a)
    np.testing.assert_array_equal(b.grad, expected_grad_b)


@pytest.mark.parametrize(
    "data_a, data_b, expected_data",
    [
        ([1, 2, 3], [4, 5, 6], [0.25, 0.40, 0.50]),
        ([0, 0, 0], [1, 1, 1], [0.0, 0.0, 0.0]),
        ([2, -4, 10], [-2, 2, 5], [-1.0, -2.0, 2.0]),
    ],
)
def test_tensor_division(data_a, data_b, expected_data):
    """Tests the forward pass of division (Tensor.__truediv__)."""
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)
    c = a / b

    np.testing.assert_array_almost_equal(c.data, expected_data, decimal=5)
    assert c.requires_grad is True


@pytest.mark.parametrize(
    "data_a, data_b, expected_grad_a, expected_grad_b",
    [
        (
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.25, 0.2, 1 / 6],
            [-0.0625, -0.08, -0.0833333],
        ),
        (
            [2.0, -4.0],
            [-1.0, 2.0],
            [-1.0, 0.5],
            [-2.0, 1.0],
        ),
    ],
)
def test_tensor_backward_division(data_a, data_b, expected_grad_a, expected_grad_b):
    """Tests the backward pass (gradient) of division (Tensor.__truediv__)."""
    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)

    c = a / b
    c.backward()

    np.testing.assert_array_almost_equal(a.grad, expected_grad_a, decimal=5)
    np.testing.assert_array_almost_equal(b.grad, expected_grad_b, decimal=5)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0],
        [[1.0, 2.0], [3.0, 4.0]],
        [[-1.0, -2.0], [3.0, 0.5]],
    ],
)
def test_tensor_sum_backward(data):
    """
    For s = a.sum(), we expect ds/da = 1 for every element in `a`.
    """
    a = Tensor(data, requires_grad=True)
    s = a.sum()

    s.backward()

    expected_grad = np.ones_like(a.data)
    np.testing.assert_array_almost_equal(a.grad, expected_grad, decimal=5)


def test_tensor_matmul_backward():
    """
    Checks gradient correctness for C = A @ B,
    using an example where A:(2,3) and B:(3,2).
    Then we sum up C into a scalar -> backward.
    """
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    B = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)

    a = Tensor(A, requires_grad=True)
    b = Tensor(B, requires_grad=True)

    c = a @ b

    s = c.sum()
    s.backward()

    expected_grad_a = np.ones_like(c.data) @ B.T
    np.testing.assert_array_almost_equal(a.grad, expected_grad_a, decimal=5)

    expected_grad_b = A.T @ np.ones_like(c.data)
    np.testing.assert_array_almost_equal(b.grad, expected_grad_b, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__])
