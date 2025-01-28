from typing import Any, Dict, Tuple

from threads.tensor import Tensor


class Initialiser:  # TODO: create dtype class/types
    def __call__(self, shape: Tuple[int, ...], dtype, **kwargs: Any) -> "Tensor":
        raise NotImplementedError(
            "Initialiser subclasses must implement a __call__ method"
        )

    def get_config(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Initialiser":
        return cls(**config)
