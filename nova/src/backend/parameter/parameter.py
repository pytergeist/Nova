# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from nova.src.backend.core import Tensor
from nova.src.backend.core.clib import factory_methods as fm


@dataclass
class Parameter:
    uuid: UUID
    name: str
    trainable: bool
    tensor: Tensor
    requires_grad: bool
    grad: Optional[Tensor] = None

    def __post_init__(self):
        if self.grad is None:
            self.grad = fm.zeros_like(self.tensor)

    def zero_grad(self):
        if self.grad is not None:
            self.grad = fm.zeros_like(self.tensor)
        else:
            raise ValueError("Gradient is not initialized.")
