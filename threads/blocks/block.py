from threads.tensor import Tensor


class Block:
    def __init__(self):
        self._subblocks = {}

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def add_weight(self, initialiser):
        pass