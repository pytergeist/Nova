from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from nova.src.backend.parameter import Parameter


class Optimiser(ABC):
    def __init__(self, parameters: List["Parameter"]) -> None:
        self._parameters = parameters

    def add_parameter(
        self, parameter_dict: Dict[str, Any]
    ):  # TODO: create better type hints for param dict
        self._parameters.append(Parameter(**parameter_dict))

    @property
    def parameters(self) -> List["Parameter"]:
        return self._parameters

    @abstractmethod
    def step(self):
        pass
