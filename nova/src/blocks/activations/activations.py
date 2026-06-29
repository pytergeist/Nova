from nova.src.backend.core import Tensor
from nova.src.blocks.block import Block


class ReLU(Block):
    def __init__(self):
        super().__init__()

    def call(self, inputs: Tensor, **kwargs):
        return inputs.maximum(0)

    def get_config(self):
        return {}

    # Overriding the default CamelCase-to-snake_case conversion, because ReLU would otherwise become "re_l_u"'
    @classmethod
    def name(cls) -> str:
        return cls.lower_case(cls.__name__)


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


class LeakyReLU(Block):
    def __init__(self):
        super().__init__()

    def call(self, inputs: Tensor, alpha: float = 0.1, **kwargs):
        return inputs.maximum(inputs * alpha)

    def get_config(self):
        return {}

    # Overriding the default CamelCase-to-snake_case conversion, because LeakyReLU would otherwise become "leaky_re_l_u"
    @classmethod
    def name(cls) -> str:
        return "leaky_relu"


class Softmax:
    def __init__(self):
        raise NotImplementedError("Softmax activation function is not implemented.")


class Softplus:
    def __init__(self):
        raise NotImplementedError("Softplus activation function is not implemented.")
