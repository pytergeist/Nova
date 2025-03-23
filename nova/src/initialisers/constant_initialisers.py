import numpy as np

from nova.src.backend import io
from nova.src.initialisers.initialiser import Initialiser


class Constant(Initialiser):
    def __init__(self, value: float) -> None:
        self.value = value
        super().__init__()

    def __call__(self, shape, dtype, **kwargs) -> np.ndarray:  # TODO: Add dtype here?
        return io.as_numpy_array(
            self.value * np.ones(shape), dtype=dtype
        )  # TODO: change from np calls to generate core data

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
        return io.as_numpy_array(np.zeros(shape), dtype=dtype)


class Ones(Initialiser):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return {}

    def __call__(self, shape, dtype, **kwargs) -> np.ndarray:
        return io.as_numpy_array(np.ones(shape), dtype=dtype)
