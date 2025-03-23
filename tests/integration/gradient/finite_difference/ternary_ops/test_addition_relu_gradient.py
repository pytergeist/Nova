import numpy as np
import pytest

from nova.src.backend.core import Tensor
from nova.src.blocks.activations import ReLU
from tests.integration.gradient.finite_difference import finite_difference_jacobian

# TODO: Add parameterisation for multiple test cases
# TODO: Implement integration tests for larger computation graphs


def fn_numpy(x):
    y = x + x
    return np.maximum(0, y)


def compute_autodiff_gradient(x):
    x_tensor = Tensor(x, requires_grad=True)

    tensor_output = x_tensor + x_tensor

    output = ReLU()(tensor_output)

    output.backward()

    return x_tensor.grad


def test_addition_grad():
    x_test = np.random.uniform(-1, 1, 5)

    numerical_jacobian = finite_difference_jacobian(fn_numpy, x_test, epsilon=1.5e-8)
    numerical_vector_grad = np.dot(np.ones(x_test.shape), numerical_jacobian)
    analytical_grad = compute_autodiff_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_vector_grad,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Autodiff gradient does not match numerical gradient for addition.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
