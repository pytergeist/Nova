import numpy as np

from threads.tensor import Tensor
from threads.initialisers.initialiser import Initialiser


class ConstantInitialiser(Initialiser):
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, shape, dtype, **kwargs) -> Tensor:
        dtype = Tensor.standardise_dtype(dtype)
        return Tensor(self.value * np.ones(shape), dtype=dtype)


