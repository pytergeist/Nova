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
        self._inheritance_lock = (
            False  # TODO: override super in parent to set inheritance lock
        )
        self.units = units
        self.kernel_initialiser = kernel_initialiser
        self.bias = bias
        self.bias_initialiser = bias_initialiser

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(
                self.units,
                input_shape[-1],
            ),
            initialiser=self.kernel_initialiser,
            role="kernel",
        )
        if self.bias:
            self.bias = self.add_weight(
                shape=(
                    1,
                    self.units,
                ),
                initialiser=self.bias_initialiser,
                role="bias",
            )

    def get_config(self) -> Dict[str, Any]:
        return {
            "units": self.units,
            "kernel_initialiser": self.kernel_initialiser,
            "bias": self.bias,
            "bias_initialiser": self.bias_initialiser,
        }

    def forward(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        # For an input of shape (batch, in_features) and a kernel of shape (units, in_features),
        # we compute the output as: output = inputs @ kernel.T + bias.
        output = inputs @ self.kernel.T
        if self.bias:
            output += self.bias
        return output


if __name__ == "__main__":
    from nova.src.blocks.core import InputBlock

    inp = InputBlock((None, 5))  # a “symbolic” input
    x = inp()
    dense1 = Linear(10, "random_normal")
    dense2 = Linear(10, "random_normal")
    dense3 = Linear(1, "random_normal")
    y = dense1(x)
    z = dense2(y)
    out = dense3(z)

    print(out)
