from typing import TYPE_CHECKING, List, Optional

import numpy as np

from nova.src.backend.core import Tensor
from nova.src.optim.optimiser import Optimiser

if TYPE_CHECKING:
    from nova.src.backend.parameter import Parameter


class SGD(
    Optimiser
):  # TODO: figure out more user friendly way to add velocity in op base class
    def __init__(self, parameters: List["Parameter"], lr, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        super().__init__(parameters=parameters)
        self.velocities: Optional = None

    def build(self):
        self.velocities = [
            Tensor(np.zeros_like(p.tensor.data)) for p in self.parameters
        ]

    def step(self):
        for idx, p in enumerate(self.parameters):
            g_t = p.tensor.grad
            if g_t is None:
                continue

            if self.momentum > 0.0:
                v_t = self.momentum * self.velocities[idx]
                g_t = (1 - self.momentum) * g_t

                p.tensor -= self.lr * (v_t + g_t)

            else:
                p.tensor -= self.lr * g_t

            p.zero_grad()
