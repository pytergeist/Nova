from .activations import Activation, ReLU

_OBJECTS = [
    ReLU,
]

_ACTIVATIONS = {cls.name(): cls for cls in _OBJECTS}


def get(name: str) -> "Activation":
    cls = _ACTIVATIONS.get(name)
    if cls is None:
        raise ValueError(f"Unknown initialiser: {name}")
    return cls()
