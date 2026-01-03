import numpy as np
import pytest
import torch

from nova.src.backend.core import Tensor
from tests.integration.gradient import set_grad
from tests.integration.gradient.finite_difference import Tolerance


@set_grad
def compute_autodiff_gradient(x):
    x_tensor = Tensor(x, requires_grad=True)
    output = x_tensor @ x_tensor
    output.backward()
    return x_tensor.grad.to_numpy()


def compute_pytorch_addition_gradient(x):
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    output = x_tensor @ x_tensor
    output.backward(torch.ones_like(x_tensor))
    return x_tensor.grad.detach().numpy()


def test_addition_grad():
    np.random.seed(42)
    torch.manual_seed(42)

    x_test = np.random.rand(5, 5)
    analytical_grad = compute_autodiff_gradient(x_test)
    pytorch_grad = compute_pytorch_addition_gradient(x_test)

    np.testing.assert_allclose(
        analytical_grad,
        pytorch_grad,
        rtol=Tolerance.RTOL.value,
        atol=Tolerance.ATOL.value,
        err_msg="Autodiff gradient does not match PyTorch gradient for addition.",
    )


if __name__ == "__main__":
    pytest.main([__file__])
