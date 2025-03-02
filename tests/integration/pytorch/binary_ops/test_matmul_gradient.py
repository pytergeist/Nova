import numpy as np
import pytest
import torch

from tests.integration.gradient.finite_difference import finite_difference_jacobian


# TODO: Add parameterisation for multiple test cases


def fn_numpy(x):
    return x @ x


def compute_pytorch_gradient(x):
    x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)

    output = x_tensor @ x_tensor

    output.backward(torch.ones_like(x_tensor))

    return x_tensor.grad.detach().numpy()


def test_matmul_grad():  # TODO: Should there be a test for matmul or 1D arrays?
    x_test = np.random.rand(10, 10)

    numerical_jacobian = finite_difference_jacobian(fn_numpy, x_test, epsilon=1.5e-8)
    numerical_jacobian = numerical_jacobian.squeeze()
    numerical_vector_grad = np.dot(
        np.ones(numerical_jacobian.shape[0]), numerical_jacobian
    )
    numerical_vector_grad = numerical_vector_grad.reshape(x_test.shape)
    analytical_grad = compute_pytorch_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_vector_grad,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Autodiff gradient does not match numerical gradient for matrix multiplication.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
