from abditus.src.autodiff import Engine
from abditus.src.blocks.block import Block
from abditus.src.graph import print_graph
from abditus.src.tensor import Tensor


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
            shape=(input_shape[-1], self.units),
            initialiser=self.kernel_initialiser,
        )
        if self.bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initialiser=self.bias_initialiser,
            )

    def forward(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        output = inputs @ self.kernel
        if self.bias:
            output += self.bias
        return output


if __name__ == "__main__":
    layer = Linear(units=10, kernel_initialiser="random_normal", bias=True)
    layer.build(input_shape=(None, 5))
    ll = layer(Tensor(data=[[1, 2, 3, 4, 5]]))
    # print([node for node in ll.engine.created_nodes])
    print_graph(ll._node)
