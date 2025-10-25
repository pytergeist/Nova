import numpy as np
import pytest

from nova.src.backend.core import Tensor
from tests.integration.gradient import set_grad_tape
from tests.integration.gradient.finite_difference import (
    Tolerance,
    finite_difference_jacobian,
)


@set_grad_tape
def compute_autodiff_gradient(x):
    x_tensor = Tensor(x, requires_grad=True)

    output = x_tensor @ x_tensor

    output.backward()

    return x_tensor.grad.to_numpy()


def test_matmul_2d_grad(
    request,
):  # TODO: Should there be a test for matmul or 1D arrays?
    x_test = np.random.rand(10, 10)
    fn = request.getfixturevalue("binary_fn_numpy")

    def partial_fn(x):
        return fn(x, fn_str="matmul")

    numerical_jacobian = finite_difference_jacobian(
        partial_fn, x_test, epsilon=Tolerance.EPSILON.value
    )
    numerical_jacobian = numerical_jacobian.squeeze()
    numerical_vector_grad = np.dot(
        np.ones(numerical_jacobian.shape[0]), numerical_jacobian
    )
    numerical_vector_grad = numerical_vector_grad.reshape(x_test.shape)
    analytical_grad = compute_autodiff_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_vector_grad,
        rtol=Tolerance.RTOL.value,
        atol=Tolerance.ATOL.value,
        err_msg="Autodiff gradient does not match numerical gradient for matrix multiplication.",
    )


def test_matmul_3d_grad(
    request,
):  # TODO: Should there be a test for matmul or 1D arrays?
    batch_size = 5
    x_test = np.random.rand(batch_size, 10, 10)
    fn = request.getfixturevalue("binary_fn_numpy")

    def partial_fn(x):
        return fn(x, fn_str="matmul")

    numerical_jacobian = finite_difference_jacobian(
        partial_fn, x_test, epsilon=Tolerance.EPSILON.value
    )
    numerical_jacobian = numerical_jacobian.squeeze()
    numerical_vector_grad = np.dot(
        np.ones(numerical_jacobian.shape[0]), numerical_jacobian
    )
    numerical_vector_grad = numerical_vector_grad.reshape(x_test.shape)
    analytical_grad = compute_autodiff_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_vector_grad,
        rtol=Tolerance.RTOL.value,
        atol=Tolerance.ATOL.value,
        err_msg="Autodiff gradient does not match numerical gradient for matrix multiplication.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
