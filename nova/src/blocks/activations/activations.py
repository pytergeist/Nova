from nova.src.backend.core import Tensor
from nova.src.blocks.block import Block


class ReLU(Block):
    def __init__(self):
        super().__init__()

    def call(self, inputs: Tensor, **kwargs):
        return inputs.maximum(0)

    def get_config(self):
        return {}

    # Overwriting the name as ReLU with parent functionality would be re_l_u in snake case.
    @classmethod
    def name(cls) -> str:
        return cls.lower_case(cls.__name__)


class GeLU:
    def __init__(self):
        raise NotImplementedError("GeLU activation function is not implemented.")


class Sigmoind:
    def __init__(self):
        raise NotImplementedError("Sigmoid activation function is not implemented.")


class Tanh:
    def __init__(self):
        raise NotImplementedError("Tanh activation function is not implemented.")


class LeakyReLU(Block):
    def __init__(self):
        super().__init__()

    def call(self, inputs: Tensor, alpha: float = 0.1, **kwargs):
        return inputs.maximum(inputs * alpha)

    def get_config(self):
        return {}

    # Overwriting the name as LeakyReLU with parent functionality would be leaky_re_l_u in snake case.
    @classmethod
    def name(cls) -> str:
        return "leaky_relu"


class Softmax:
    def __init__(self):
        raise NotImplementedError("Softmax activation function is not implemented.")


class Softplus:
    def __init__(self):
        raise NotImplementedError("Softplus activation function is not implemented.")
