from nova.src.backend.core import Tensor
from nova.src.blocks.block import Block


class ReLU(Block):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs: Tensor):
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


class Sigmoind:
    def __init__(self):
        raise NotImplementedError("Sigmoid activation function is not implemented.")


class Tanh:
    def __init__(self):
        raise NotImplementedError("Tanh activation function is not implemented.")


class LeakyReLU(Block):
    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__()
        self.alpha = alpha

    def call(self, inputs: Tensor):
        return inputs.maximum(inputs * self.alpha)

    def get_config(self):
        return {"alpha": self.alpha}

    # Overriding the default CamelCase-to-snake_case conversion, because LeakyReLU would otherwise become "leaky_re_l_u"
    @classmethod
    def name(cls) -> str:
        return "leaky_relu"


class Softmax(Block):
    def __init__(self, **kwargs):
        super().__init__()

    def call(
        self, inputs: Tensor
    ):  # TODO: implement scaling down of inputs by max input. Requires max function
        exps = inputs.exp()
        return exps / exps.sum()

    def get_config(self):
        return {}


class Softplus(Block):
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__()
        self.beta = beta

    def call(self, inputs: Tensor):
        return ((inputs * self.beta).exp() + 1).log() * (1 / self.beta)

    def get_config(self):
        return {"beta": self.beta}
