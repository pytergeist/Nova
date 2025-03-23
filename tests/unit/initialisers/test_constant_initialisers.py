import numpy as np
import pytest

from nova.src.initialisers import Constant, Ones, Zeros


def test_constant_initialiser_init_value():
    constant = Constant(1.0)
    assert constant(1, "float32") == 1.0


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
    output = constant(shape, dtype)
    np.testing.assert_array_equal(output, expected)


def test_zeros_initialiser_get_config():
    zeros = Zeros()
    assert zeros.get_config() == {}


def test_zeros_initialiser_from_config():
    config = {}
    zeros = Zeros.from_config(config)
    assert isinstance(zeros, Zeros)


def test_zeros_initialiser_call_method():
    zeros = Zeros()
    output = zeros((2, 2), "float32")
    np.testing.assert_array_equal(output, np.zeros((2, 2)))


def test_ones_initialiser_get_config():
    ones = Ones()
    assert ones.get_config() == {}


def test_ones_initialiser_from_config():
    config = {}
    ones = Ones.from_config(config)
    assert isinstance(ones, Ones)


def test_ones_initialiser_call_method():
    ones = Ones()
    output = ones((2, 2), "float32")
    np.testing.assert_array_equal(output, np.ones((2, 2)))


if __name__ == "__main__":
    pytest.main([__file__])
