from abditus.src.backend.core import Tensor
from abditus.src.blocks._block import Block


class ReLU(Block):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs: Tensor, **kwargs):
        return inputs.maximum(0)

    def get_config(self):
        return {}


class GeLU(Block):
    raise NotImplementedError("GeLU activation function is not implemented.")


class Sigmoind(Block):
    raise NotImplementedError("Sigmoid activation function is not implemented.")


class Tanh(Block):
    raise NotImplementedError("Tanh activation function is not implemented.")


class LeakyReLU(Block):
    raise NotImplementedError("LeakyReLU activation function is not implemented.")


class Softmax(Block):
    raise NotImplementedError("Softmax activation function is not implemented.")


class Softplus(Block):
    raise NotImplementedError("Softplus activation function is not implemented.")
