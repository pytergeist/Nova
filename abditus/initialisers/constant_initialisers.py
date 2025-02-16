import numpy as np

from abditus.initialisers.initialiser import Initialiser
from abditus.tensor import Tensor


class Constant(Initialiser):
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, shape, dtype, **kwargs) -> Tensor:
        dtype = Tensor.standardise_dtype(dtype)
        return Tensor(
            self.value * np.ones(shape), dtype=dtype
        )  # TODO: change from np calls to generate tensor data

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

    def __call__(self, shape, dtype, **kwargs) -> Tensor:
        dtype = Tensor.standardise_dtype(dtype)
        return Tensor(np.zeros(shape), dtype=dtype)


class Ones(Initialiser):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return {}

    def __call__(self, shape, dtype, **kwargs) -> Tensor:
        dtype = Tensor.standardise_dtype(dtype)
        return Tensor(np.ones(shape), dtype=dtype)
