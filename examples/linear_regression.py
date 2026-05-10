"""Simple linear regression with TensorVault."""

import numpy as np
from tensorvault import Tensor, nn, optim

# Generate synthetic data
np.random.seed(42)
X_np = np.random.randn(100, 1).astype(np.float32)
w_true = 3.5
b_true = -1.2
y_np = w_true * X_np + b_true + np.random.randn(100, 1).astype(np.float32) * 0.1

# Model
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("Training linear regression...")
for epoch in range(100):
    x = Tensor(X_np)
    y = Tensor(y_np)

    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        w = model.weight.data.item()
        b = model.bias.data.item()
        print(f"Epoch {epoch+1}: loss={loss.item():.6f}  w={w:.4f}  b={b:.4f}")

print(f"\nLearned: y = {model.weight.data.item():.4f}x + {model.bias.data.item():.4f}")
print(f"True:    y = {w_true}x + {b_true}")
