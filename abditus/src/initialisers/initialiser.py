import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from abditus.src.tensor import Tensor


class Initialiser(ABC):  # TODO: create dtype class/types
    def __call__(self, shape: Tuple[int, ...], dtype, **kwargs: Any) -> "Tensor":
        raise NotImplementedError(
            "Initialiser subclasses must implement a __call__ method"
        )

    @staticmethod
    def camel_to_snake_case(name: str) -> str:
        """
        Convert CamelCase class names (e.g., 'RandomNormal')
        into snake_case strings (e.g., 'random_normal').
        """
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()

    @classmethod
    def name(cls) -> str:
        """
        By default, convert the class name from CamelCase to snake_case.
        Subclasses can override this classmethod if they want a custom name.
        """
        return cls.camel_to_snake_case(cls.__name__)

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Initialiser":
        return cls(**config)
