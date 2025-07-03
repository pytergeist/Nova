from typing import TYPE_CHECKING, List

from nova.src.backend.trainers import Trainer

if TYPE_CHECKING:
    from nova.src.backend.topology.node import ModelNode
    from nova.src.blocks.core.input_block import InputBlock


class Model(Trainer):
    def __init__(self, inputs: List["InputBlock"], outputs: List["ModelNode"]) -> None:
        self.inputs = inputs
        self.outputs = outputs
        super().__init__()
