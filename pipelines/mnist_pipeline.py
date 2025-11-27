from kfp import dsl
from kfp.compiler import Compiler


OUTPUT_DIR = "/tmp/mnist_artifacts"


def train_op(epochs: int = 1, batch_size: int = 64):
    return dsl.ContainerOp(
        name="Train MNIST model (PyTorch)",
        image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",  # готов образ, без да правим Dockerfile
        command=["sh", "-c"],
        arguments=[
            f"""
            pip install torchvision --quiet && \
            python - << 'EOF'
from mnist_training import run_training
run_training(output_dir="{OUTPUT_DIR}", epochs={epochs}, batch_size={batch_size})
EOF
            """
        ],
        file_outputs={
            # казваме на Kubeflow къде да търси артефактите от стъпката
            "model_path": f"{OUTPUT_DIR}/model.pt",
            "metrics_path": f"{OUTPUT_DIR}/metrics.json",
        }
    )


def report_metrics_op(metrics_path: str):
    return dsl.ContainerOp(
        name="Report Metrics",
        image="python:3.10-slim",
        command=["sh", "-c"],
        arguments=[
            f"""
            pip install json5 >/dev/null 2>&1 || true
            python - << 'EOF'
import json

metrics_file = "{metrics_path}"
with open(metrics_file, "r") as f:
    metrics = json.load(f)

print("=== Training Metrics ===")
for k, v in metrics.items():
    print(f"{{k}}: {{v}}")
EOF
            """
        ]
    )


@dsl.pipeline(
    name="MNIST PyTorch Pipeline",
    description="Pipeline that trains and evaluates the MNIST model from mnist_training.py"
)
def mnist_pipeline(
    epochs: int = 1,
    batch_size: int = 64,
):
    train_step = train_op(epochs=epochs, batch_size=batch_size)

    report_step = report_metrics_op(
        metrics_path=train_step.outputs["metrics_path"]
    ).after(train_step)


if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path="mnist_pipeline.yaml",
    )
