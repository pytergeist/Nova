import numpy as np
import pytest

from nova.src.backend.core import Tensor, autodiff
from tests.integration.gradient.finite_difference import (
    Tolerance,
    finite_difference_jacobian,
)

autodiff.enabled(True)


def compute_autodiff_gradient(x):
    x_tensor = Tensor(x, requires_grad=True)

    output = x_tensor * x_tensor

    output.backward()

    return x_tensor.grad.to_numpy()


def test_mul_grad(request):
    x_test = np.random.rand(5)

    fn = request.getfixturevalue("binary_fn_numpy")

    def partial_fn(x):
        return fn(x, fn_str="multiply")

    numerical_jacobian = finite_difference_jacobian(
        partial_fn, x_test, epsilon=Tolerance.EPSILON.value
    )
    numerical_vector_grad = np.dot(np.ones(x_test.shape), numerical_jacobian)
    analytical_grad = compute_autodiff_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_vector_grad,
        rtol=Tolerance.RTOL.value,
        atol=Tolerance.ATOL.value,
        err_msg="Autodiff gradient does not match numerical gradient for multiplication.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
