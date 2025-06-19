import numpy as np

from nova.src.backend.core import Tensor
from nova.src.blocks.block import Block


class InputBlock(Block):
    def __init__(self, input_shape, dtype=np.float32):
        super().__init__()
        self._inheritance_lock = False
        self.shape = input_shape
        self.dtype = dtype
        self.built = True

    def __call__(self):  # TODO: remove this once lazy execution is implamented
        shape = [dim or 1 for dim in self.shape]
        dummy = np.zeros(shape, dtype=self.dtype)
        t = Tensor(dummy, requires_grad=False)
        self.children = [t]
        return t

    def get_config(self):
        return {"input_shape": self.shape, "dtype": str(self.dtype)}
