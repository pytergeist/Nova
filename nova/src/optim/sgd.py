from typing import TYPE_CHECKING, List, Optional

from nova.src.backend.core.clib import factory_methods as fm
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
        self.velocities = [fm.zeros_like(p.tensor) for p in self.parameters]

    def step(self):
        for idx, p in enumerate(self.parameters):
            g_t = p.tensor.get_grad()
            if g_t is None:
                continue
            if self.momentum > 0.0:
                v_t = self.velocities[idx] * self.momentum

                g_t = g_t * (1 - self.momentum)

                p.tensor -= (g_t + v_t) * self.lr

            else:
                p.tensor -= p.tensor - self.lr * g_t

            p.zero_grad()
