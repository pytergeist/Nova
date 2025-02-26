from .constant_initialisers import Constant, Ones, Zeros
from .initialiser import Initialiser
from .random_initialisers import RandomNormal, RandomSeed, RandomUniform

_OBJECTS = [
    Constant,
    Zeros,
    Ones,
    RandomSeed,
    RandomNormal,
]

_INITIALSERS = {cls.name(): cls for cls in _OBJECTS}


def get(name: str) -> "Initialiser":
    cls = _INITIALSERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown initialiser: {name}")
    return cls()
