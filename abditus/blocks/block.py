from typing import Optional, Tuple

from abditus import initialisers
from abditus.tensor import Tensor


class Block:
    def __init__(self):
        self._inheritance_lock = True

    def _check_super_called(self):  # TODO add inheritance_lock attr in child classes
        if getattr(self, "_inheritance_lock", True):
            raise RuntimeError(
                f"In layer {self.__class__.__name__},"
                "you forgot to call super.__init__()"
            )

    @staticmethod
    def _check_valid_kernel_initialiser(kernel_initialiser: "Initialiser") -> None:
        if initialisers.get(kernel_initialiser) is None:
            raise ValueError(f"Unknown initialiser: {kernel_initialiser}")

    def add_weight(
        self, shape: Optional[Tuple[int, int]] = None, initialiser=None, dtype=None
    ):
        self._check_super_called()
        self._check_valid_kernel_initialiser(initialiser)
        initialiser = initialisers.get(initialiser)
        return Tensor(
            data=initialiser(shape, dtype).data,
            requires_grad=True,
        )

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)
