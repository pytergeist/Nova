# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Tom Pope
#
# Nova — a high-performance hybrid physics and deep learning tensor engine.

import numpy as np
import matplotlib.pyplot as plt
from nova.src.backend.core import Tensor, autodiff, Grad
from nova.src.blocks.activations import ReLU
from nova.src.blocks.core import InputBlock
from nova.src.blocks.core.linear import Linear
from nova.src.losses import MeanSquaredError
from nova.src.models import Model
from nova.src.optim.sgd import SGD
from nova.src.blocks.regularisation import Dropout


from profiling import Profiler

prof = Profiler()

inp = InputBlock((None, 10))
x = inp
dense1 = Linear(100, "random_normal")
relu1 = ReLU()
drop1 = Dropout(0.5)
dense2 = Linear(10, "random_normal")
relu2 = ReLU()
y = dense1(x)
z = relu1(y)
z = drop1(z)
x1 = dense2(z)
out = relu2(x1)
model = Model(inputs=[inp], outputs=[out])


params = model.parameters()

N, D = (
    20487,
    10,
)  # TODO: Batch size currently failing at 40 step for N=1000: dynamic batch size fix
X = np.random.randn(N, D).astype(np.float32)
Y = 2 * X + 1 + 0.1 * np.random.randn(N, D).astype(np.float32)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

batch_size = 64
epochs = 100
lr = 1e-3
momentum = 0.9

optimizer = SGD(params, lr=lr, momentum=momentum)
optimizer.build()
loss_fn = MeanSquaredError()

epoch_mses = []


def ten_mean(x: Tensor):
    return x.mean()


for epoch in range(1, epochs + 1):
    perm = np.random.permutation(N)
    X_shuf, Y_shuf = X[perm], Y[perm]
    batch_losses = []

    for i in range(0, N, batch_size):
        xb = Tensor(X_shuf[i : i + batch_size])
        yb = Tensor(Y_shuf[i : i + batch_size])

        with Grad():
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()

        optimizer.step()
        batch_losses.append(loss.mean().to_numpy())

    epoch_mses.append(
        batch_losses[0].mean()
    )  # TODO: the data structure created here is horrible
    losses = np.min(batch_losses[0])
    print(
        f"Epoch {epoch:2d} | "
        f"Batch MSE range: {np.min(losses):.4f}–{np.max(losses):.4f} | "
        f"Epoch MSE: {epoch_mses[-1]:.4f}"
    )

plt.figure()
plt.plot(range(1, epochs + 1), epoch_mses)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.show()
