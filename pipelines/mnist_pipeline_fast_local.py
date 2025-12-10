from kfp import dsl, compiler
from kfp.dsl import component, Input, Output, Artifact


# ---------------------------------------------------------------------
# COMPONENT 1: Data preprocessing (бърз, без интернет – синтетични данни)
# ---------------------------------------------------------------------
@component(
    base_image="python:3.11-slim",
    packages_to_install=["torch"],
)
def preprocess_mnist(
    train_data: Output[Artifact],
    test_data: Output[Artifact],
):
    """
    Генерира MNIST-подобни данни и ги записва като torch тензори.
    Формат:
      - images: (N, 1, 28, 28)
      - labels: (N,)
    """

    import torch

    num_train = 2000
    num_test = 500
    num_classes = 10

    # Случайни "картинки" и етикети – достатъчни за демонстрация на пайплайн
    train_images = torch.rand(num_train, 1, 28, 28)
    test_images = torch.rand(num_test, 1, 28, 28)

    train_labels = torch.randint(0, num_classes, (num_train,))
    test_labels = torch.randint(0, num_classes, (num_test,))

    torch.save({"images": train_images, "labels": train_labels}, train_data.path)
    torch.save({"images": test_images, "labels": test_labels}, test_data.path)

    print(f"Saved train data to: {train_data.path}")
    print(f"Saved test data to:  {test_data.path}")


# Ако искаш ИСТИНСКИ MNIST (и подовете ти имат интернет), можеш да замениш
# тялото на preprocess_mnist с нещо като:
#
#   from torchvision import datasets
#   from torchvision.transforms import ToTensor
#   transform = ToTensor()
#   training_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
#   test_data_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
#   train_images = torch.stack([img for img, _ in training_data])
#   train_labels = torch.tensor([label for _, label in training_data])
#   test_images = torch.stack([img for img, _ in test_data_ds])
#   test_labels = torch.tensor([label for _, label in test_data_ds])
#   torch.save({"images": train_images, "labels": train_labels}, train_data.path)
#   torch.save({"images": test_images, "labels": test_labels}, test_data.path)


# ---------------------------------------------------------------------
# COMPONENT 2: Model training
# ---------------------------------------------------------------------
@component(
    base_image="python:3.11-slim",
    packages_to_install=["torch"],
)
def train_mnist(
    train_data: Input[Artifact],
    model_out: Output[Artifact],
    train_metrics_out: Output[Artifact],
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """
    Тренира прост MLP върху данните от train_data и записва:
      - model_out: state_dict на модела (torch.save)
      - train_metrics_out: JSON с основни метрики
    """
    import json
    import os
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
        model.train()
        last_loss = None
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
        return last_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    data = torch.load(train_data.path)
    images = data["images"]
    labels = data["labels"]

    dataset = TensorDataset(images, labels)

    cpu_count = os.cpu_count() or 1
    num_workers = max(cpu_count - 1, 1)

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = None
    for epoch in range(epochs):
        last_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        print(f"[train] Epoch {epoch + 1}: loss={last_loss:.4f}")

    # Запис на модела
    torch.save(model.state_dict(), model_out.path)
    print(f"[train] Saved model to: {model_out.path}")

    # Запис на метрики
    train_metrics = {
        "epochs": epochs,
        "final_train_loss": float(last_loss),
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_samples": int(len(dataset)),
    }

    with open(train_metrics_out.path, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    print(f"[train] Saved train metrics to: {train_metrics_out.path}")


# ---------------------------------------------------------------------
# COMPONENT 3: Evaluation
# ---------------------------------------------------------------------
@component(
    base_image="python:3.11-slim",
    packages_to_install=["torch"],
)
def evaluate_mnist(
    test_data: Input[Artifact],
    model_in: Input[Artifact],
    eval_metrics_out: Output[Artifact],
    batch_size: int = 64,
):
    """
    Зарежда модела и тестовите данни, измерва loss и accuracy и записва JSON.
    """
    import json
    import os
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),o
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Using device: {device}")

    data = torch.load(test_data.path)
    images = data["images"]
    labels = data["labels"]

    dataset = TensorDataset(images, labels)

    cpu_count = os.cpu_count() or 1
    num_workers = max(cpu_count - 1, 1)

    test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = NeuralNetwork().to(device)
    state_dict = torch.load(model_in.path, map_location=device)
    model.load_state_dict(state_dict)

    loss_fn = nn.CrossEntropyLoss()

    test_loss, acc = evaluate(test_dataloader, model, loss_fn, device)

    print(f"[eval] Test: loss={test_loss:.4f}, acc={acc:.4f}")

    eval_metrics = {
        "test_loss": float(test_loss),
        "accuracy": float(acc),
        "batch_size": batch_size,
        "num_samples": int(len(dataset)),
    }

    with open(eval_metrics_out.path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"[eval] Saved eval metrics to: {eval_metrics_out.path}")


# ---------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------
@dsl.pipeline(
    name="mnist_pipeline_fast_local",
    description="MNIST-like preprocessing, training and evaluation (fast, local-friendly)",
)
def mnist_pipeline(
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    prep = preprocess_mnist()

    train = train_mnist(
        train_data=prep.outputs["train_data"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    evaluate_mnist(
        test_data=prep.outputs["test_data"],
        model_in=train.outputs["model_out"],
        batch_size=batch_size,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path="mnist_pipeline_fast_local.yaml",
    )
