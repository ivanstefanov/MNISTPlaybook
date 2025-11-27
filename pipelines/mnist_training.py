import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import json
from pathlib import Path


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_dataloaders(batch_size: int = 64):
    transform = ToTensor()

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def evaluate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    return test_loss, accuracy


def run_training(output_dir: str = "/tmp/mnist_artifacts", epochs: int = 1, batch_size: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss_val = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        test_loss, acc = evaluate(test_dataloader, model, loss_fn, device)
        print(f"Epoch {epoch+1}: loss={loss_val:.4f}, test_loss={test_loss:.4f}, acc={acc:.4f}")

    # ARTEFACTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Save the model
    model_path = out / "model.pt"
    torch.save(model.state_dict(), model_path)

    # 2) Save metrics
    metrics = {
        "epochs": epochs,
        "final_train_loss": float(loss_val),
        "test_loss": float(test_loss),
        "accuracy": float(acc),
    }
    metrics_path = out / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")

    return str(model_path), str(metrics_path)


if __name__ == "__main__":
    run_training()