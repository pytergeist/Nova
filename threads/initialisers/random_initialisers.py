# TODO: Replace numpy functionality with backend operations - written in c++?

import os
import numpy as np
from typing import Dict, Any, Tuple

from threads.initialisers.initialiser import Initialiser
from threads.tensor import Tensor


class RandomSeed(Initialiser):  # TODO: helper functions should be moved to backend ops?
    def __init__(self, seed: int = None):
        self.seed = (
            seed if seed is not None else self._generate_seed_from_system_entropy()
        )
        self._ensure_seed_is_int()
        self._ensure_seed_is_32bit()

    def _ensure_seed_is_int(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError(
                f"Seed must be an integer, but got {type(self.seed).__name__}"
            )

    def _ensure_seed_is_32bit(self) -> None:
        if self.seed.bit_length() > 32:
            raise ValueError(
                f"Seed must be a 32-bit integer, but got {self.seed.bit_length()}-bit integer"
            )

    @staticmethod
    def _generate_seed_from_system_entropy() -> int:
        seed_bytes = os.urandom(4)
        return int.from_bytes(seed_bytes, byteorder="big")

    def get_config(self) -> Dict[str, Any]:
        return {"seed": self.seed}


class RandomNormal(RandomSeed):
    def __init__(self, mean: float = 0.0, stddev: float = 1.0, seed: int = None):
        self.mean = mean
        self.stddev = stddev
        super().__init__(seed=seed)

    def _generate_random_normal_data(self, shape: Tuple[int, ...]) -> Any:
        return np.random.normal(loc=self.mean, scale=self.stddev, size=shape)

    def __call__(self, shape: Tuple[int, ...], dtype, **kwargs: Any) -> "Tensor":
        dtype = Tensor.standardise_dtype(dtype)
        return Tensor(self._generate_random_normal_data(shape), dtype=dtype)

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "mean": self.mean, "stddev": self.stddev}


class TruncatedNormal(object):
    def __init__(self):
        raise NotImplementedError


class RandomUniform(object):
    def __init__(self):
        raise NotImplementedError
