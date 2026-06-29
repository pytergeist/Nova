from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from nova.src.backend.core import Tensor
from nova.src.backend.topology.builder import Builder
from nova.src.blocks import Block, activations
from nova.src.blocks.activations.activations import LeakyReLU, ReLU


class MockActivation(Block):

    def get_config(self) -> Dict[str, Any]:
        return {}

    def build(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        pass


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ReLU", "relu"),
        ("LeakyReLU", "leakyrelu"),
    ],
)
def test_name_lower_case(name, expected):
    assert MockActivation.lower_case(name) == expected


def test_name_method_returns_snake_case_class_name():
    assert MockActivation.name() == "mock_activation"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("relu", ReLU),
        ("leaky_relu", LeakyReLU),
    ],
)
def test_activations_module_str_get_method(name, expected):
    with Builder():  # TODO: change | temporary test fix for builder context
        assert type(activations.get(name)) is type(expected())


def test_from_config_method_returns_instance_of_activation():
    with Builder():
        config = {}
        activation = MockActivation.from_config(config)
        assert isinstance(activation, MockActivation)


# TODO: Need to add super lock test for activations?


@pytest.mark.parametrize(
    "activation_fn, data, expected, negative_slope",
    [
        ("relu", Tensor([1.0, 1.0]), [1, 1], 0),
        ("relu", Tensor([-1.0, 1.0]), [0, 1], 0),
        ("leaky_relu", Tensor([1.0, 1.0]), [1, 1], 0.1),
        ("leaky_relu", Tensor([-1.0, 1.0]), [-0.1, 1], 0.1),
        ("leaky_relu", Tensor([-1.0, 1.0]), [-0.5, 1], 0.5),
    ],
)
def test_call_method(activation_fn, data, expected, negative_slope):
    with Builder():
        activation = activations.get(activation_fn)
        assert np.allclose(
            activation.call(data, alpha=negative_slope).to_numpy(),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__])
