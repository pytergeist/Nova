from nova.src.backend.core import Tensor
from nova.src.blocks.block import Block


class ReLU(Block):
    def __init__(self):
        super().__init__()

    def call(self, inputs: Tensor, **kwargs):
        return inputs.maximum(0)

    def get_config(self):
        return {}


class GeLU:
    def __init__(self):
        raise NotImplementedError("GeLU activation function is not implemented.")


class Sigmoid(Block):
    def __init__(self):
        super().__init__()

    def call(self, inputs: Tensor, **kwargs):
        return (
            (inputs * -1).exp() + 1
        ) ** -1  # TODO: refactor when exp is a free function as nova.exp(x)

    def get_config(self):
        return {}


class Tanh:
    def __init__(self):
        raise NotImplementedError("Tanh activation function is not implemented.")


class LeakyReLU:
    def __init__(self):
        raise NotImplementedError("LeakyReLU activation function is not implemented.")


class Softmax:
    def __init__(self):
        raise NotImplementedError("Softmax activation function is not implemented.")


class Softplus:
    def __init__(self):
        raise NotImplementedError("Softplus activation function is not implemented.")
