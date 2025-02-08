import numpy as np
import pytest

from threads.initialisers.constant_initialisers import Constant


def test_constant_initialiser_init_value():
    constant = Constant(1.0)
    assert constant(1, "float32").value == 1.0


def test_constant_initialiser_get_config():
    constant = Constant(1.0)
    assert constant.get_config() == {"value": 1.0}


def test_constant_initialiser_from_config():
    config = {"value": 1.0}
    constant = Constant.from_config(config)
    assert isinstance(constant, Constant)
    assert constant.value == 1.0


@pytest.mark.parametrize(
    "shape, dtype, expected",
    [
        ((1, 1), "float32", [[1.0]]),
        ((2, 2), "float32", [[1.0, 1.0], [1.0, 1.0]]),
        ((1, 1), "int32", [[1.0]]),
        ((2, 2), "int32", [[1.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_constant_initialiser_call_method(shape, dtype, expected):
    constant = Constant(1.0)
    tensor = constant(shape, dtype)
    np.testing.assert_array_equal(tensor.data, expected)


if __name__ == "__main__":
    pytest.main([__file__])
