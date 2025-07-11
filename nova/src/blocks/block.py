import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nova.src import initialisers
from nova.src.backend import io
from nova.src.backend.core import Tensor, Variable
from nova.src.backend.topology import Builder

if TYPE_CHECKING:
    from nova.src.backend.topology import ModelNode
    from nova.src.blocks.core.input_block import InputBlock
    from nova.src.initialisers import Initialiser


class Block(ABC):
    def __init__(
        self, builder: Optional[Builder] = None, trainable=True, *args, **kwargs
    ):
        self.builder = builder or Builder.get_current()
        self.trainable = trainable
        self._inheritance_lock = True
        self._built = False
        self.input_shape = None
        self.output_shape = None
        self.node = self.builder.build_model_node(
            self, inbound_tensors=[], outbound_tensors=[]
        )
        self._uuid = uuid.uuid4()
        self._kernel: Optional[Variable] = None
        self._bias_value: Optional[Variable] = None

    @property
    def kernel(self) -> Optional[Variable]:
        return self._kernel

    @kernel.setter
    def kernel(self, value: Optional[Variable]) -> None:
        if value is not None and not isinstance(
            value, Tensor
        ):  # TODO: this value migrates from Variable to Tensor during training
            raise ValueError(
                "Kernel must be a Variable (Tensor subclass where requires_grad = True)."
            )
        self._kernel = value

    @property
    def bias_value(self) -> Optional[Variable]:
        return self._bias_value

    @bias_value.setter
    def bias_value(self, value: Optional[Variable]) -> None:
        if value is not None and not isinstance(
            value, Tensor
        ):  # TODO: this value migrates from Variable to Tensor during training
            raise ValueError(
                "Bias must be a Variable (Tensor subclass where requires_grad = True)."
            )
        self._bias_value = value

    @property
    def uuid(self) -> uuid.UUID:
        return self._uuid

    @property
    def built(self) -> bool:
        """Built propert for block has been built."""
        return self._built

    @built.setter
    def built(self, value: bool) -> None:
        """Set the built property for the block."""
        if not isinstance(value, bool):
            raise ValueError("Built property must be a boolean value.")
        self._built = value

    def _check_super_called(self):  # TODO add inheritance_lock attr in child classes
        if getattr(self, "_inheritance_lock", True):
            raise RuntimeError(
                f"In layer {self.__class__.__name__},"
                "you forgot to call super.__init__()"
            )

    @staticmethod
    def lower_case(name: str) -> str:  # TODO: what about leaky_relu?
        """Convert class names (e.g., 'Linear, ReLU') into lower case strings
        (e.g., 'linear, relu')."""
        return name.lower()

    @classmethod
    def name(cls) -> str:
        """By default, convert the class name from CamelCase to snake_case.

        Subclasses can override this classmethod if they want a custom name.
        """
        return cls.lower_case(cls.__name__)

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Block":
        return cls(**config)

    @staticmethod
    def _check_valid_kernel_initialiser(kernel_initialiser: str) -> None:
        if initialisers.get(kernel_initialiser) is None:
            raise ValueError(f"Unknown initialiser: {kernel_initialiser}")

    def add_weight(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        initialiser: Optional[Union[str, "Initialiser"]] = None,
        dtype=None,
        role=None,
    ) -> "Variable":
        self._check_super_called()
        if isinstance(initialiser, str):
            self._check_valid_kernel_initialiser(initialiser)
            initialiser = initialisers.get(initialiser)
        return io.as_variable(
            data=initialiser(shape, dtype), role=role
        )  # TODO: asses whether bias should be variable or possibly untrainable tensor

    def forward(self, *inputs):
        raise NotImplementedError

    def call(self, *inputs):
        return self.forward(*inputs)

    def build_block(self, input_shape):
        """
        Sets input_shape, calls subclass’s _build(),
        checks that output_shape was set, and flips the built flag.
        """
        self.input_shape = input_shape

        self.build(input_shape)

        if self.output_shape is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.build() must set self.output_shape"
            )
        self.built = True

    # TODO: Make abstract method once relevant tests are refactored
    def build(self, input_shape):
        """
        Subclasses override this to:
          • create self.kernel / self.bias
          • set self.output_shape

        Currently, this is needed for activation functions and input blocks, the output shape
        is set in the override build method for layers, but not for activation functions.
        TODO: find a better way of handling output shape setting
        """
        self.output_shape = input_shape

    def _flatten_blocks(self) -> List["Block"]:
        blocks: list = []
        visited_blocks = set()
        stack = [self]
        while stack:
            current_block = stack.pop()
            if id(current_block) in visited_blocks:
                continue
            visited_blocks.add(id(current_block))

            if isinstance(current_block, Block):
                blocks.append(current_block)

            if hasattr(current_block, "parents"):
                for parent in current_block.parents:
                    stack.append(parent)

        return blocks

    def _set_parents(
        self, parents: Tuple[Union["ModelNode", "InputBlock"], ...]
    ) -> None:
        """Set the parents of the model node."""
        self.node.parents = tuple(
            p.node if hasattr(p, "input_block") else p for p in parents
        )

    def __call__(  # TODO: this currently only works for the first input, need to fix for multi input models
        self, *inputs: Union[Tensor, np.ndarray]
    ) -> "ModelNode":
        self._check_super_called()
        self._set_parents(inputs)
        self.node.set_children()
        builder_outputs = self.builder.created_model_nodes[-1].outbound_tensors
        self.outbound_tensors = builder_outputs
        return self.node
