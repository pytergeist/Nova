from typing import TYPE_CHECKING, List

from nova.src.backend.topology import Builder
from nova.src.backend.trainers import Trainer

if TYPE_CHECKING:
    from nova.src.backend.topology.node import ModelNode
    from nova.src.blocks.core.input_block import InputBlock


class Model(Trainer):
    def __init__(self, inputs: List["InputBlock"], outputs: List["ModelNode"]) -> None:
        super().__init__()
        Builder.ensure_current()
        self.inputs = inputs
        self.outputs = outputs
        self.build()
        Builder.finalise_current()

    def blocks(self):
        """Get the model blocks."""
        return [node.operator for node in self.topology]

    def __call__(self, inputs, *args, **kwargs):  # TODO: this impl is dodgy
        if not self.topology:
            raise ValueError("Model topology has not been built yet.")
        for node in self.topology:
            if node.operator.trainable:
                outputs = node.operator.forward(inputs)
                inputs = outputs
        return inputs
