from typing import Any, Dict

from nova.src.blocks.block import Block


class Linear(Block):
    def __init__(
        self,
        units: int,
        kernel_initialiser: str,
        bias: bool = True,
        bias_initialiser: str = "zeros",
    ) -> None:
        super().__init__()
        self.units = units
        self.kernel_initialiser = kernel_initialiser
        self.bias = bias
        self.bias_initialiser = bias_initialiser

    def build(self, input_shape):
        in_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(self.units, in_dim),
            initialiser=self.kernel_initialiser,
            role="kernel",
        )

        if self.bias:
            self.bias_value = self.add_weight(
                shape=(self.units,), initialiser=self.bias_initialiser, role="bias"
            )

    def get_config(self) -> Dict[str, Any]:
        return {
            "units": self.units,
            "kernel_initialiser": self.kernel_initialiser,
            "bias": self.bias,
            "bias_initialiser": self.bias_initialiser,
        }

    def call(self, inputs):
        # For an input of shape (batch, in_features) and a kernel of shape (units, in_features),
        # we compute the output as: output = inputs @ kernel.T + bias.
        output = inputs @ self.kernel.T
        if self.bias:
            output += self.bias
        return output
