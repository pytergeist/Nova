import numpy as np

from nova.src.backend.factory import factory
from nova.src.initialisers.initialiser import Initialiser

# TODO: why are we using numpy here?


class Constant(Initialiser):
    def __init__(self, value: float) -> None:
        self.value = value
        super().__init__()

    def __call__(self, shape, dtype, **kwargs) -> np.ndarray:  # TODO: Add dtype here?
        return self.value * np.ones(shape)

    def get_config(self):
        return {"value": self.value}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Zeros(Initialiser):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return {}

    def __call__(self, shape, dtype, **kwargs) -> np.ndarray:
        return np.zeros(shape)


class Ones(Initialiser):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return {}

    def __call__(self, shape, dtype, **kwargs) -> np.ndarray:
        return np.ones(shape)
