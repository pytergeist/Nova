import numpy as np

from nova.src.backend.core import Tensor
from nova.src.backend.topology import Builder


class InputBlock:
    def __init__(self, input_shape, dtype=np.float32):
        self._inheritance_lock = False
        self.shape = input_shape
        self.dtype = dtype
        self.built = True
        self.builder = Builder()
        self._node = self.builder.build_leaf_model_node(
            self, parents=(), inbound_tensors=None, outbound_tensors=None
        )
        self.input_block = True

    @property
    def node(self):
        return self._node

    def __call__(self):  # TODO: remove this once lazy execution is implamented
        shape = [dim or 1 for dim in self.shape]
        dummy = np.zeros(shape, dtype=self.dtype)
        t = Tensor(dummy, requires_grad=False)
        self.children = [t]
        return t

    def get_config(self):
        return {"input_shape": self.shape, "dtype": str(self.dtype)}
