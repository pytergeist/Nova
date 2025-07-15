import numpy as np

from nova.src.factory import factory
from nova.src.initialisers.initialiser import Initialiser


class Constant(Initialiser):
    def __init__(self, value: float) -> None:
        self.value = value
        super().__init__()

    def __call__(self, shape, dtype, **kwargs) -> np.ndarray:  # TODO: Add dtype here?
        return self.value * factory.ones(shape).to_numpy()

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
        return factory.zeros(shape).to_numpy()


class Ones(Initialiser):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return {}

    def __call__(self, shape, dtype, **kwargs) -> np.ndarray:
        return factory.ones(shape).to_numpy()
