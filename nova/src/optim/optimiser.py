from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from nova.src.backend.parameter import Parameter


class Optimiser(ABC):
    def __init__(self, parameters: List["Parameter"]) -> None:
        self._parameters = parameters

    @abstractmethod
    def step(self):
        pass
