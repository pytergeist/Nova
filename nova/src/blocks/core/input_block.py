import uuid
from typing import Optional

import numpy as np

from nova.src.backend.core import Tensor
from nova.src.backend.topology import Builder


class InputBlock:
    def __init__(
        self, input_shape, dtype=np.float32, builder: Optional[Builder] = None
    ):
        self.trainable = False
        self._inheritance_lock = False
        self.input_shape = input_shape
        self.output_shape = None
        self.dtype = dtype
        self._built = False
        self.builder = builder or Builder.get_current()
        self._node = self.builder.build_leaf_model_node(
            self, parents=(), inbound_tensors=None, outbound_tensors=None
        )
        self.input_block = True
        self._uuid = uuid.uuid4()

    @property
    def uuid(self) -> uuid.UUID:
        return self._uuid

    @property
    def node(self):
        return self._node

    @property
    def built(self) -> bool:
        """Check if the block has been built."""
        return self._built

    @built.setter
    def built(self, value: bool) -> None:
        """Set the built property for the block."""
        if not isinstance(value, bool):
            raise ValueError("Built property must be a boolean value.")
        self._built = value

    def build_block(self, input_shape):
        self.output_shape = input_shape
        self.built = True

    def __call__(self):  # TODO: remove this once lazy execution is implamented
        shape = [dim or 1 for dim in self.input_shape]
        dummy = np.zeros(shape, dtype=self.dtype)
        t = Tensor(dummy, requires_grad=False)
        self.children = [t]
        return t

    def get_config(self):
        return {"input_shape": self.input_shape, "dtype": str(self.dtype)}
