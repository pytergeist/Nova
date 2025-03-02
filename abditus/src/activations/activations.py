from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Activation(ABC):
    def __call__(self, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError(
            "Activation subclasses must implement a __call__ method"
        )

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Activation":
        return cls(**config)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def __call__(self, dtype, **kwargs: Any) -> np.ndarray:
        return np.maximum(0, dtype)

    def get_config(self) -> Dict[str, Any]:
        return {}
