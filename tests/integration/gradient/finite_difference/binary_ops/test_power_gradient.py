import numpy as np
import pytest

from nova.src.backend.core import Tensor
from tests.integration.gradient.finite_difference import finite_difference_jacobian

# TODO: Add parameterisation for multiple test cases


def compute_autodiff_gradient(x):
    x_tensor = Tensor(x, requires_grad=True)

    output = x_tensor**x_tensor

    output.backward()

    return x_tensor.grad


def test_power_grad(request):
    x_test = np.random.rand(5)

    fn = request.getfixturevalue("binary_fn_numpy")

    def partial_fn(x):
        return fn(x, fn_str='pow')

    numerical_jacobian = finite_difference_jacobian(partial_fn, x_test, epsilon=1.5e-8)
    numerical_vector_grad = np.dot(np.ones(x_test.shape), numerical_jacobian)
    analytical_grad = compute_autodiff_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_vector_grad,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Autodiff gradient does not match numerical gradient for power.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
