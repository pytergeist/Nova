from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod
from threads.tensor import Tensor


class Initialiser(ABC):  # TODO: create dtype class/types
    def __call__(self, shape: Tuple[int, ...], dtype, **kwargs: Any) -> "Tensor":
        raise NotImplementedError(
            "Initialiser subclasses must implement a __call__ method"
        )

    @abstractmethod
    @property
    def name(self) -> str:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Initialiser":
        return cls(**config)
