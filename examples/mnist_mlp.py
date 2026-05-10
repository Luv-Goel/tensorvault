"""Train an MLP on MNIST to demonstrate TensorVault's autograd + nn modules."""

import numpy as np
from tensorvault import Tensor, nn, optim
from tensorvault.data import TensorDataset, DataLoader


def load_mnist_subset(size: int = 2000):
    """Load a subset of MNIST (or generate synthetic data)."""
    # Use synthetic data for the demo — real MNIST load would need sklearn/requests
    np.random.seed(42)
    X = np.random.randn(size, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=size).astype(np.int64)
    # Make it learnable by creating class-conditional patterns
    for i in range(size):
        X[i] += np.eye(10)[y[i]] * 2.0
    return X, y


def accuracy(pred: Tensor, target: np.ndarray) -> float:
    pred_class = np.argmax(pred.data, axis=1)
    return (pred_class == target).mean()


def main():
    print("=" * 50)
    print("TensorVault MNIST Demo")
    print("=" * 50)

    # Hyperparameters
    batch_size = 64
    epochs = 5
    lr = 0.001

    # Data
    X_train, y_train = load_mnist_subset(2000)
    X_test, y_test = load_mnist_subset(500)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTrain samples: {len(train_ds)}")
    print(f"Test samples:  {len(test_ds)}")
    print(f"Parameters:    {sum(p.data.size for p in model.parameters())}")
    print(f"Device:        CPU (CuPy optional)")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            x = Tensor(batch_x)
            y = Tensor(batch_y)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        # Evaluation
        model.eval()
        test_preds = []
        test_targets = []
        for batch_x, batch_y in test_loader:
            x = Tensor(batch_x)
            pred = model(x)
            test_preds.append(pred.data)
            test_targets.append(batch_y)

        all_preds = np.concatenate(test_preds)
        all_targets = np.concatenate(test_targets)
        acc = accuracy(Tensor(all_preds), all_targets)

        avg_loss = total_loss / batches
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.2%}")

    print(f"\nFinal test accuracy: {acc:.2%}")
    print("Demo complete! TensorVault is working.")


if __name__ == "__main__":
    main()
