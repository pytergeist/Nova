from typing import TYPE_CHECKING

from .activations import LeakyReLU, ReLU, Sigmoid

if TYPE_CHECKING:
    from nova.src.blocks import Block

_OBJECTS = [ReLU, LeakyReLU, Sigmoid]
_ACTIVATIONS = {cls.name(): cls for cls in _OBJECTS}


def get(name: str) -> "Block":
    cls = _ACTIVATIONS.get(name)
    if cls is None:
        raise ValueError(f"Unknown activation: {name}")
    return cls()
