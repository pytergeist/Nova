import numpy as np
import pytest
import torch

from nova.src.backend.core import Tensor
from nova.src.blocks.activations.activations import ReLU
from nova.src.blocks.linear import Linear
from nova.src.initialisers import Constant, Ones, RandomNormal, RandomUniform, Zeros

network_configs = [
    # 2-layer network: input 5 -> hidden 10 -> output 1, activation after first layer only.
    {"layers": [5, 10, 1], "activations": [True, False]},
    # 3-layer network: input 5 -> hidden 10 -> hidden 7 -> output 1, ReLU after layer 1 and layer 2.
    {"layers": [5, 10, 7, 1], "activations": [True, True, False]},
    # 4-layer network: input 5 -> hidden 8 -> hidden 6 -> hidden 4 -> output 1,
    # ReLU after every hidden layer.
    {"layers": [5, 8, 6, 4, 1], "activations": [True, True, True, False]},
]


def compute_autodiff_network_grad(x, initializer, config):
    layer_sizes = config["layers"]
    activations = config["activations"]

    layers = []
    for i in range(1, len(layer_sizes)):
        layer = Linear(units=layer_sizes[i], kernel_initialiser=initializer, bias=False)
        if i == 1:
            layer.build(input_shape=x.shape)
        else:
            layer.build(input_shape=(None, layer_sizes[i - 1]))
        layers.append(layer)

    activations_list = [ReLU() if act else None for act in activations]

    input_tensor = Tensor(data=x, requires_grad=True)
    out = input_tensor
    for layer, act in zip(layers, activations_list):
        out = layer(out)
        if act is not None:
            out = act(out)
    if len(layers) > len(activations_list):
        out = layers[-1](out)

    output_sum = out.sum()
    output_sum.backward()

    grads = {}
    for i, layer in enumerate(layers, start=1):
        grads[f"layer{i}_kernel"] = layer.kernel.grad
    return grads


def compute_pytorch_network_grad(x, initializer, config):
    layer_sizes = config["layers"]
    activations = config["activations"]

    W = []
    for i in range(1, len(layer_sizes)):
        W.append(initializer((layer_sizes[i], layer_sizes[i - 1]), dtype=np.float32))

    layers = []
    x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    for i in range(1, len(layer_sizes)):
        in_features = layer_sizes[i - 1]
        out_features = layer_sizes[i]
        layer = torch.nn.Linear(in_features, out_features, bias=False).double()
        layers.append(layer)

    with torch.no_grad():
        for layer, weight in zip(layers, W):
            layer.weight.copy_(torch.tensor(weight, dtype=torch.float64))

    out = x_tensor
    for layer, act_flag in zip(layers, activations):
        out = torch.relu(layer(out)) if act_flag else layer(out)
    if len(layers) > len(activations):
        out = layers[-1](out)

    output_sum = out.sum()
    output_sum.backward()

    grads = {}
    for i, layer in enumerate(layers, start=1):
        grads[f"layer{i}_kernel"] = layer.weight.grad.detach().numpy()
    return grads


@pytest.mark.parametrize(
    "initializer",
    [
        Constant(0.5),
        Zeros(),
        Ones(),
        RandomNormal(mean=0.0, stddev=1.0, seed=42),
        RandomUniform(minval=-1.0, maxval=1.0, seed=42),
    ],
)
@pytest.mark.parametrize("config", network_configs)
def test_network_grad(initializer, config):
    np.random.seed(42)
    torch.manual_seed(42)
    x_test = np.random.rand(5, 5)

    autodiff_grads = compute_autodiff_network_grad(x_test, initializer, config)
    pytorch_grads = compute_pytorch_network_grad(x_test, initializer, config)

    for key in autodiff_grads:
        np.testing.assert_allclose(
            autodiff_grads[key],
            pytorch_grads[key].squeeze(),
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Gradient for {key} does not match PyTorch gradient using"
            f" {initializer.__class__.__name__} and config {config}.",
        )


if __name__ == "__main__":
    pytest.main([__file__])
