import pdb
from typing import TYPE_CHECKING, List

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
        self.velocities = self._build_velocity_buffer() if momentum > 0.0 else None

    def _build_velocity_buffer(self):
        return [Tensor(np.zeros_like(p.tensor.data)) for p in self.parameters]

    def step(self):
        for idx, p in enumerate(self.parameters):
            grad = p.tensor.grad
            if grad is None:
                continue

            if self.momentum > 0.0:
                momentum_term = self.momentum * self.velocities[idx]
                grad_term = (1 - self.momentum) * grad

                p.tensor -= self.lr * (momentum_term + grad_term)

            else:
                pdb.set_trace()
                p.tensor -= self.lr * grad

            p.zero_grad()
