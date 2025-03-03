from abditus.src.backend.core import Tensor
from abditus.src.blocks._block import Block


class ReLU(Block):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs: Tensor, **kwargs):
        return inputs.maximum(0)

    def get_config(self):
        return {}
