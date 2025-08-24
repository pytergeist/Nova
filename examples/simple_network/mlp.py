import numpy as np
import matplotlib.pyplot as plt
from nova.src.backend.core._tensor import Tensor
from nova.src.blocks.activations import ReLU
from nova.src.blocks.core import InputBlock
from nova.src.blocks.core.linear import Linear
from nova.src.losses import MeanSquaredError
from nova.src.models import Model
from nova.src.optim.sgd import SGD
from nova.src.blocks.regularisation import Dropout

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
model.build()


params = model.parameters()

N, D = 1000, 10
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

for epoch in range(1, epochs + 1):
    perm = np.random.permutation(N)
    X_shuf, Y_shuf = X[perm], Y[perm]
    batch_losses = []

    for i in range(0, N, batch_size):
        xb = Tensor(X_shuf[i : i + batch_size])
        yb = Tensor(Y_shuf[i : i + batch_size])

        y_pred = model(xb)
        loss = loss_fn(y_pred, yb)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.data.mean())

    epoch_mses.append(np.mean(batch_losses))
    print(
        f"Epoch {epoch:2d} | "
        f"Batch MSE range: {min(batch_losses):.4f}–{max(batch_losses):.4f} | "
        f"Epoch MSE: {epoch_mses[-1]:.4f}"
    )

plt.figure()
plt.plot(range(1, epochs + 1), epoch_mses)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.show()
