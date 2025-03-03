from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from abditus.src import initialisers
from abditus.src.backend import io

if TYPE_CHECKING:
    from abditus.src.initialisers import Initialiser


class Block(ABC):
    def __init__(self):
        self._inheritance_lock = True

    def _check_super_called(self):  # TODO add inheritance_lock attr in child classes
        if getattr(self, "_inheritance_lock", True):
            raise RuntimeError(
                f"In layer {self.__class__.__name__},"
                "you forgot to call super.__init__()"
            )

    @staticmethod
    def lower_case(name: str) -> str:  # TODO: what about leaky_relu?
        """Convert class names (e.g., 'Linear, ReLU') into lower case strings
        (e.g., 'linear, relu')."""
        return name.lower()

    @classmethod
    def name(cls) -> str:
        """By default, convert the class name from CamelCase to snake_case.

        Subclasses can override this classmethod if they want a custom name.
        """
        return cls.lower_case(cls.__name__)

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Block":
        return cls(**config)

    @staticmethod
    def _check_valid_kernel_initialiser(kernel_initialiser: "Initialiser") -> None:
        if initialisers.get(kernel_initialiser) is None:
            raise ValueError(f"Unknown initialiser: {kernel_initialiser}")

    def add_weight(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        initialiser: Optional[Union[str, "Initialiser"]] = None,
        dtype=None,
        role=None,
    ):
        self._check_super_called()
        if isinstance(initialiser, str):
            self._check_valid_kernel_initialiser(initialiser)
            initialiser = initialisers.get(initialiser)
        return io.as_variable(data=initialiser(shape, dtype), role=role)

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)
