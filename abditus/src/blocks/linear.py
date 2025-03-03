from abditus.src.backend.core import Tensor
from abditus.src.blocks._block import Block


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
    from abditus.src.backend.graph import print_graph
    from abditus.src.blocks.activations.activations import ReLU

    layer = Linear(units=10, kernel_initialiser="random_normal", bias=True)
    layer.build(input_shape=(None, 5))
    ll = layer(Tensor(data=[[1, 2, 3, 4, 5]]))
    relu = ReLU()
    ll = relu(ll)
    # print(ll.data)
    # print(relu(ll))

    # print([node for node in ll.engine.created_nodes])
    print_graph(ll._node)
