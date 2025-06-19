from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from nova.src import initialisers
from nova.src.backend import io
from nova.src.backend.core import Tensor
from nova.src.backend.topology import Builder

if TYPE_CHECKING:
    from nova.src.initialisers import Initialiser

builder = Builder()


class Block(ABC):
    def __init__(self):
        self._inheritance_lock = True
        self._built = False
        self.parents = []
        self.children = []

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

    # @abstractmethod
    # def build(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
    #     """Build the block with the given input shape."""
    #     pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Block":
        return cls(**config)

    @staticmethod
    def _check_valid_kernel_initialiser(kernel_initialiser: "Initialiser") -> None:
        if initialisers.get(kernel_initialiser) is None:
            raise ValueError(f"Unknown initialiser: {kernel_initialiser}")

    def add_weight(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        initialiser: Optional[Union[str, "Initialiser"]] = None,
        dtype=None,
        role=None,
    ):
        self._check_super_called()
        if isinstance(initialiser, str):
            self._check_valid_kernel_initialiser(initialiser)
            initialiser = initialisers.get(initialiser)
        return io.as_variable(data=initialiser(shape, dtype), role=role)

    def forward(self, *inputs):
        raise NotImplementedError

    def call(self, *inputs):
        return self.forward(*inputs)

    def build(self):
        pass

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

    def __call__(
        self, *inputs: Union[Tensor, np.ndarray]
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        self._check_super_called()

        # TODO: migrate this method to builder class
        tensor_inputs = []
        for x in inputs:
            if isinstance(x, Tensor):
                tensor_inputs.append(x)
            else:
                tensor_inputs.append(Tensor(x))

        # TODO: migrate built functionality to the Builder class
        if not getattr(self, "built", False):
            input_shape = tensor_inputs[0].shape
            self.build(input_shape)
            self.built = True

        # TODO: implement lazy call execution
        raw_output = self.call(*tensor_inputs)

        if isinstance(raw_output, Tensor):
            outputs = (raw_output,)
        else:
            outputs = tuple(raw_output)

        builder.build_model_node(
            self, inbound_tensors=tensor_inputs, outbound_tensors=outputs
        )

        builder_outputs = builder.created_model_nodes[-1].outbound_tensors
        return (
            builder_outputs[0] if len(builder_outputs) == 1 else tuple(builder_outputs)
        )
