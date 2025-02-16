from typing import Any, Dict

import pytest

from abditus.initialisers import Initialiser


class MockInitialiser(Initialiser):
    def get_config(self) -> Dict[str, Any]:
        return {}


def test_call_method_raises_not_implamented_error():
    initialiser = MockInitialiser()
    with pytest.raises(NotImplementedError):
        initialiser(1, "float32")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("RandomNormal", "random_normal"),
        ("RandomUniform", "random_uniform"),
        ("HeNormal", "he_normal"),
        ("HeUniform", "he_uniform"),
        ("XavierNormal", "xavier_normal"),
        ("XavierUniform", "xavier_uniform"),
    ],
)
def test_camel_to_snake_case(name, expected):
    assert Initialiser.camel_to_snake_case(name) == expected


def test_name_method_returns_snake_case_class_name():
    assert MockInitialiser.name() == "mock_initialiser"


def test_from_config_method_returns_instance_of_initialiser():
    config = {}
    initialiser = MockInitialiser.from_config(config)
    assert isinstance(initialiser, MockInitialiser)


if __name__ == "__main__":
    pytest.main([__file__])
