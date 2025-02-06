from threads import initialisers
from threads.blocks.block import Block


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
