from typing import TYPE_CHECKING, List

from nova.src.optim.optimiser import Optimiser

if TYPE_CHECKING:
    from nova.src.backend.parameter import Parameter


class SGD(Optimiser):  # TODO: finish SGD implementation
    def __init__(self, parameters: List["Parameter"], lr):
        self.lr = lr
        super().__init__(parameters=parameters)

    def step(self):
        for p in self.parameters:
            grad = p.tensor.grad
            if grad is None:
                continue

            p.tensor -= self.lr * grad

            p.zero_grad()


if __name__ == "__main__":
    from nova.src.backend.core._tensor import Tensor
    from nova.src.backend.topology import Builder
    from nova.src.blocks.activations import ReLU
    from nova.src.blocks.core import InputBlock
    from nova.src.blocks.core.linear import Linear
    from nova.src.losses import MeanSquaredError

    with Builder() as builder:
        inp = InputBlock((None, 10))  # a “symbolic” input
        x = inp
        dense1 = Linear(10, "random_normal")
        relu1 = ReLU()
        dense3 = Linear(10, "random_normal")
        relu2 = ReLU()
        y = dense1(x)
        z = relu1(y)
        x1 = dense3(z)
        out = relu2(x1)

        from nova.src.models import Model

        model = Model(inputs=[inp], outputs=[out])
        parameters = model.parameters()

        # for idx, parameter in enumerate(model.parameters()):
        #     print(idx, parameter)

    sgd = SGD(parameters, lr=1e-3)
    import numpy as np

    #
    x = np.arange(10).reshape(1, -1)
    y = 2 * x + 1 + np.random.normal(scale=0.5, size=x.shape)
    print(y)
    x_tensor = Tensor(x)
    y_tensor = Tensor(y)
    loss_fn = MeanSquaredError()
    for epoch in range(50):
        y_pred = model(x_tensor)

        loss = loss_fn(y_pred, y_tensor)

        loss.backward()

        sgd.step()

        print(f"Epoch {epoch:2d}, MSE = {loss.data.sum()}")
# 86f15bb7-dc07-4798-80fa-23ae9f6dfd48
