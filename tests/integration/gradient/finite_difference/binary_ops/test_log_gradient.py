import numpy as np
import pytest

from nova.src.backend.core import Tensor
from tests.integration.gradient.finite_difference import finite_difference_jacobian

# TODO: Add parameterisation for multiple test cases


def fn_numpy(x):
    return np.log(x)


def compute_autodiff_gradient(x):
    x_tensor = Tensor(x, requires_grad=True)

    output = x_tensor.log()

    output.backward()

    return x_tensor.grad


def test_log_grad():
    x_test = np.random.rand(5)

    numerical_jacobian = finite_difference_jacobian(fn_numpy, x_test, epsilon=1.5e-8)
    numerical_vector_grad = np.dot(np.ones(x_test.shape), numerical_jacobian)
    analytical_grad = compute_autodiff_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_vector_grad,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Autodiff gradient does not match numerical gradient for natural logarithm.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
