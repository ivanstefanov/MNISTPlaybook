import os
import torch
import numpy as np
from kserve import Model, ModelServer
from typing import Dict

class WineModel(Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.preprocess = None
        self.load()

    def load(self):
        model_path = "/mnt/model/model.pt"
        preprocess_path = "/mnt/model/preprocess.pt"

        self.model = torch.jit.load(model_path)
        self.model.eval()

        self.preprocess = torch.jit.load(preprocess_path)

        self.ready = True

    def predict(self, payload: Dict, headers: Dict = None):
        inputs = payload["instances"]
        x = torch.tensor(inputs, dtype=torch.float32)

        x = self.preprocess(x)

        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

        return {"predictions": preds.tolist()}


if __name__ == "__main__":
    model = WineModel("wine-model")
    ModelServer().start([model])
