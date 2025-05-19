from typing import Any, Dict

from nova.logging.log_config import logger
from nova.src.backend.core import Tensor
from nova.src.blocks._block import Block


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
        logger.debug(
            f"Initialising linear layer with units: {self.units}, bias: {self.bias}"
        )

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
                shape=(self.units,),
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
        logger.debug(
            f"Forward pass through linear layer with inputs: {inputs.data.shape}"
        )
        return self.call(inputs)

    def call(self, inputs):
        # For an input of shape (batch, in_features) and a kernel of shape (units, in_features),
        # we compute the output as: output = inputs @ kernel.T + bias.
        logger.debug(f"Calling linear layer with inputs: {inputs.data.shape}")
        output = inputs @ self.kernel.T
        logger.debug(f"Output after kernel multiplication: {output.data.shape}")
        if self.bias:
            output += self.bias
        return output


if __name__ == "__main__":
    from nova.src.backend.graph import print_graph
    from nova.src.blocks.activations.activations import ReLU

    layer = Linear(units=10, kernel_initialiser="random_normal", bias=True)
    layer.build(input_shape=(None, 5))
    ll = layer(Tensor(data=[[1, 2, 3, 4, 5]]))
    relu = ReLU()
    ll = relu(ll)
    # print(ll.data)
    # print(relu(ll))

    # print([node for node in ll.engine.created_nodes])
    print_graph(ll._node)
