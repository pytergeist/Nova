from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import numpy as np

from nova.src.backend.core import Tensor
from nova.src.blocks import Block


@dataclass
class Parameter:
    uuid: UUID
    operator: Block
    name: str
    trainable: bool
    tensor: Tensor
    requires_grad: bool
    grad: Optional[Tensor] = None

    def __post_init__(self):
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.tensor.data))

    def zero_grad(self):
        if self.grad is not None:
            self.grad = Tensor(
                np.zeros_like(self.tensor.data)
            )  # TODO: create tensor factory functions
        else:
            raise ValueError("Gradient is not initialized.")
