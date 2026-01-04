# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from nova.src.backend.parameter import Parameter


class Optimiser(ABC):
    def __init__(self, parameters: List["Parameter"]) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> List["Parameter"]:
        return self._parameters

    @abstractmethod
    def build(self): ...

    @abstractmethod
    def step(self): ...
