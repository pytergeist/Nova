import numpy as np
import torch
import pytest

from abditus.src.backend.core import Tensor  # your tensor class
from abditus.src.blocks.linear import Linear
from abditus.src.blocks.activations.activations import ReLU

from abditus.src.initialisers import Constant, Zeros, Ones, RandomNormal, RandomUniform


def compute_autodiff_network_grad(x, initializer):
    np.random.seed(42)
    layer1 = Linear(units=10, kernel_initialiser=initializer, bias=False)
    layer1.build(input_shape=x.shape)  # x.shape is (batch, 5)
    relu1 = ReLU()
    layer2 = Linear(units=7, kernel_initialiser=initializer, bias=False)
    layer2.build(input_shape=(None, 10))
    relu2 = ReLU()
    layer3 = Linear(units=1, kernel_initialiser=initializer, bias=False)
    layer3.build(input_shape=(None, 7))

    input_tensor = Tensor(data=x, requires_grad=True)
    out1 = relu1(layer1(input_tensor))
    out2 = relu2(layer2(out1))
    output = layer3(out2)
    # Sum the output (to get a scalar) then backpropagate.
    output_sum = output.sum()
    output_sum.backward()

    return {
        "layer1_kernel": layer1.kernel.grad,
        "layer2_kernel": layer2.kernel.grad,
        "layer3_kernel": layer3.kernel.grad,
    }


def compute_pytorch_network_grad(x, initializer):
    W1 = initializer((10, x.shape[1]), dtype=np.float32)
    W2 = initializer((7, 10), dtype=np.float32)
    W3 = initializer((1, 7), dtype=np.float32)

    x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    layer1 = torch.nn.Linear(x.shape[1], 10, bias=False).double()
    layer2 = torch.nn.Linear(10, 7, bias=False).double()
    layer3 = torch.nn.Linear(7, 1, bias=False).double()

    with torch.no_grad():
        layer1.weight.copy_(torch.tensor(W1, dtype=torch.float64))
        layer2.weight.copy_(torch.tensor(W2, dtype=torch.float64))
        layer3.weight.copy_(torch.tensor(W3, dtype=torch.float64))

    out1 = torch.relu(layer1(x_tensor))
    out2 = torch.relu(layer2(out1))
    output = layer3(out2)
    output_sum = output.sum()
    output_sum.backward()

    return {
        "layer1_kernel": layer1.weight.grad.detach().numpy(),
        "layer2_kernel": layer2.weight.grad.detach().numpy(),
        "layer3_kernel": layer3.weight.grad.detach().numpy(),
    }


# Parameterize the test over different initializers.
@pytest.mark.parametrize(
    "initializer",
    [
        Constant(0.5),  # constant value initializer
        Zeros(),
        Ones(),
        RandomNormal(mean=0.0, stddev=1.0, seed=42),
        RandomUniform(minval=-1.0, maxval=1.0, seed=42),
    ],
)
def test_network_grad(initializer):
    np.random.seed(42)
    torch.manual_seed(42)
    # Test input: batch of 5 samples with 5 features.
    x_test = np.random.rand(5, 5)

    autodiff_grads = compute_autodiff_network_grad(x_test, initializer)
    pytorch_grads = compute_pytorch_network_grad(x_test, initializer)

    # Compare each corresponding gradient.
    for key in autodiff_grads:
        np.testing.assert_allclose(
            autodiff_grads[key],
            pytorch_grads[key],
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Gradient for {key} does not match PyTorch gradient using {initializer.__class__.__name__}."
        )


if __name__ == "__main__":
    pytest.main([__file__])
