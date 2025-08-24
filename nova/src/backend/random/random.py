from typing import TYPE_CHECKING, Optional

from nova.src.backend.core import _C, io

if TYPE_CHECKING:
    from nova.src.backend.core import Tensor


class Random(_C.Random):
    def __init__(self, seed: Optional[int] = None):
        if seed:
            self.seed = seed
            super().__init__(seed)
        else:
            super().__init__()

    def uniform(
        self,
        shape: tuple[int, ...],
        min: float = 0.0,
        max: float = 1.0,
    ) -> "Tensor":
        return io.as_tensor(
            self.uniform_cpp(shape=shape, min=min, max=max).to_numpy()
        )  # TODO: we're passing data in and out of c-layer to numpy with this
