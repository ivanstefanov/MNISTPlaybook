import kfp
from kfp import dsl
from kfp.components import create_component_from_func, InputPath, OutputPath


# =========================
# 1️⃣ PREPROCESS STEP
# =========================
def preprocess_op(
    x_train: OutputPath("npy"),
    y_train: OutputPath("npy"),
    x_val: OutputPath("npy"),
    y_val: OutputPath("npy"),
):
    import numpy as np
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)

    X = rng.random((2000, 10), dtype=np.float32)
    y = (X.sum(axis=1) > 5).astype(np.int64)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # IMPORTANT: write using file handle (avoid .npy auto suffix issue)
    with open(x_train, "wb") as f:
        np.save(f, Xtr)

    with open(y_train, "wb") as f:
        np.save(f, ytr)

    with open(x_val, "wb") as f:
        np.save(f, Xva)

    with open(y_val, "wb") as f:
        np.save(f, yva)


# =========================
# 2️⃣ TRAIN STEP
# =========================
def train_op(
    x_train: InputPath("npy"),
    y_train: InputPath("npy"),
    model: OutputPath("pt"),
    lr: float = 0.01,
    epochs: int = 30,
):
    import os
    import subprocess
    import sys

    # Install CPU torch inside container (works without DockerHub image)
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2+cpu",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu",
    ])

    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    X = np.load(x_train)
    y = np.load(y_train)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    net = nn.Sequential(
        nn.Linear(X.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = net(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), model)


# =========================
# 3️⃣ EVALUATE STEP
# =========================
def evaluate_op(
    model: InputPath("pt"),
    x_val: InputPath("npy"),
    y_val: InputPath("npy"),
):
    import os
    import subprocess
    import sys

    # Install CPU torch here as well
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2+cpu",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu",
    ])

    import json
    import numpy as np
    import torch
    import torch.nn as nn

    Xv = np.load(x_val)
    yv = np.load(y_val)

    Xv = torch.tensor(Xv, dtype=torch.float32)
    yv = torch.tensor(yv, dtype=torch.long)

    net = nn.Sequential(
        nn.Linear(Xv.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    net.load_state_dict(torch.load(model, map_location="cpu"))
    net.eval()

    with torch.no_grad():
        preds = torch.argmax(net(Xv), dim=1)

    acc = (preds == yv).float().mean().item()
    print("accuracy:", acc)

    with open("/mlpipeline-metrics.json", "w") as f:
        json.dump(
            {"metrics": [{"name": "accuracy", "numberValue": acc}]},
            f
        )


# =========================
# COMPONENT DEFINITIONS
# =========================
preprocess = create_component_from_func(
    preprocess_op,
    base_image="python:3.10",
    packages_to_install=[
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
    ],
)

train = create_component_from_func(
    train_op,
    base_image="python:3.10",
    packages_to_install=["numpy==1.26.4"],
)

evaluate = create_component_from_func(
    evaluate_op,
    base_image="python:3.10",
    packages_to_install=["numpy==1.26.4"],
)


# =========================
# PIPELINE
# =========================
@dsl.pipeline(
    name="Final MLOps Pipeline",
    description="Preprocess → Train → Evaluate"
)
def ml_pipeline(lr: float = 0.01, epochs: int = 30):
    p = preprocess()

    t = train(
        x_train=p.outputs["x_train"],
        y_train=p.outputs["y_train"],
        lr=lr,
        epochs=epochs,
    )

    evaluate(
        model=t.outputs["model"],
        x_val=p.outputs["x_val"],
        y_val=p.outputs["y_val"],
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        ml_pipeline,
        "ml_pipeline.yaml"
    )
