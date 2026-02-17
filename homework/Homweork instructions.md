
1. Choose a real‑world dataset (e.g., tabular, image, or NLP) and train a PyTorch model.
---

## 1) Отвори Kubeflow UI

1. В WSL пусни port-forward (ако не е пуснат):

   ```bash
   export KUBECONFIG=/tmp/kubeflow-config
   kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
   ```
2. В Windows браузър: `http://localhost:8080`
3. Влез с (ако си на default):

   * `user@example.com`
   * `12341234`

---

## 2) Създай Notebook Server (UI)

1. Ляво меню → **Notebooks** → **+ New Notebook**
2. Препоръчителни настройки (за CPU тренинг, напълно достатъчно):

   * Image: избери стандартен Jupyter образ (ако имаш **PyTorch**-образ — още по-добре; ако не, пак става)
   * CPU: 2
   * RAM: 4–8 Gi
   * Workspace Volume: 10–20 Gi (PVC)
3. **Create** и изчакай статусът да стане **Running**, после **Connect**.

---

## 3) Вътре в Jupyter: инсталирай зависимости

Отвори Terminal в Jupyter (или клетка в notebook) и изпълни:

```bash
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas scikit-learn numpy
```

---

## 4) Реален dataset + PyTorch модел (готов notebook код)

Най-практично през UI е **tabular dataset** (реални данни, бързо). Дай си нов Notebook: `adult_income.ipynb` и пусни следните клетки.

### Клетка 1 — зареждане на Adult Income (UCI)

```python
import numpy as np
import pandas as pd

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLUMNS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

train = pd.read_csv(TRAIN_URL, header=None, names=COLUMNS, skipinitialspace=True)
test  = pd.read_csv(TEST_URL, header=0, names=COLUMNS, skipinitialspace=True, comment="|")
test["income"] = test["income"].str.replace(".", "", regex=False)

df = pd.concat([train, test], ignore_index=True)

for col in df.columns:
    df = df[df[col] != "?"]

df["income"] = (df["income"] == ">50K").astype(np.int64)

df.shape, df["income"].mean()
```

### Клетка 2 — preprocessing (OneHot + StandardScaler)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

y = df["income"].values
Xraw = df.drop(columns=["income"])

num_cols = Xraw.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in Xraw.columns if c not in num_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    Xraw, y, test_size=0.2, random_state=42, stratify=y
)

X_train = pre.fit_transform(X_train_raw)
X_val = pre.transform(X_val_raw)

X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val

X_train.shape, X_val.shape
```

### Клетка 3 — PyTorch Dataset + MLP

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)

train_ds = AdultDataset(X_train, y_train)
val_ds = AdultDataset(X_val, y_val)

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(in_features=X_train.shape[1]).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

device
```

### Клетка 4 — training loop + метрики + запис на модел

```python
import os

best_acc = 0.0
os.makedirs("artifacts", exist_ok=True)

for epoch in range(1, 11):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(p)
            trues.append(yb.numpy())

    acc = accuracy_score(np.concatenate(trues), np.concatenate(preds))
    print(f"Epoch {epoch:02d} | val_acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "artifacts/model.pt")

print("Best val_acc =", round(best_acc, 4))
print("Saved artifacts/model.pt")
```

---

## 5) Как да “покажеш”, че задачата е изпълнена

В UI вече ще имаш:

* notebook с **реален dataset** (UCI Adult)
* обучен PyTorch модел (MLP)
* изведени метрики (val accuracy)
* артефакт `artifacts/model.pt`

---
# Pipeline: Build a Kubeflow Pipeline with preprocessing, training, tuning, serving, monitoring, and retraining.

1) Стандартизирай структурата в PVC

В терминала на notebook-а:

```sh
mkdir -p /home/jovyan/workspace/{data,processed,models,metrics,logs}
```

И занапред:
- данни → workspace/data
- обработени → workspace/processed
- модели → workspace/models
- метрики → workspace/metrics

Отлично — това е **Kubeflow Pipelines v1 UI** (класическият интерфейс).
Имаш бутон **Upload pipeline**, което означава:

> 🎯 Готов си да качим реален multi-step pipeline и да го управляваме изцяло през UI.

Сега ще ти дам **минимален, но пълен MLOps pipeline**, който покрива:

* preprocessing
* training
* evaluation
* tuning (Katib отделно)
* serving (KServe)
* monitoring
* retraining (Recurring Runs)

Ще го направим така, че да работи с твоя PVC (`/home/jovyan`).

---

# 🧠 Стъпка 1 — Инсталирай KFP SDK в Notebook

В notebook terminal:

```bash
pip install kfp==1.8.22
```

---

# 🧠 Стъпка 2 — Създай pipeline.py в Notebook

Виж  `train_pipeline.py` ....
pipeline-а работи за за KFP v1, който:
- не използва външни Docker images (само python:3.10)
- не разчита на Docker Hub за PyTorch image
- инсталира torch CPU вътре в стъпката
- използва правилно InputPath / OutputPath
- не чупи artifact paths
- не изисква PVC
- работи в multi-user Kubeflow
- компилира с kfp==1.8.x

---

# 🧠 Стъпка 3 — Компилирай YAML

В terminal:

```bash
python pipeline.py
```
(името на файла за папйлайн)
Ще се създаде:

```
ml_pipeline.yaml
```
3.1. След като се създаде ml_pipeline.yaml, го отвори в notebook-a и го download на компютъра за последващ ъплоуд.
---

# 🧠 Стъпка 4 — Качи го в UI

Pipelines → **Upload pipeline**

* Name: `mlops-adult`
* File: `ml_pipeline.yaml` //името на файла е записано в текста на пайплайна
* Upload

---

# Стъпка 4.1 - Създай експеримент

1. В лявото меню под Pipelines натисни Experiments (или след създаване на пайплайна)
2. Натисни + New experiment
3. Име, напр. `adult-income-exp`
4. Create
После:
5. Отиди пак на Pipelines → избери качения pipeline
6. Натисни Create run (или + Run)
7. В полето Experiment избери adult-income-exp
8. Create/Start

# 🧠 Стъпка 5 — Създай Run

След upload:

* Create Run
* Start

Ще видиш graph:

```
Preprocess → Train → Evaluate
```

И accuracy ще се визуализира като metric.

---
Отлично.
Сега вече сме на стабилна основа и можем да направим **Katib tuning през UI**, без да чупим нищо.

Ще използваме същата логика като в pipeline-а, но Katib ще пуска training job-и с различни `lr` и `epochs`.

---

# 🎯 Цел

Да оптимизираме:

* `lr`
* `epochs`

за метриката:

```
accuracy
```

---

# 🧠 Важно: Katib не стартира pipeline

Katib стартира **Kubernetes Job**, не KFP pipeline.

Затова ще създадем отделен training entrypoint за Katib, който:

* приема аргументи
* тренира
* отпечатва `accuracy=<value>`

---

# ✅ Стъпка 1 – Създай katib_train.py

В Notebook създай файл `katib_train.py` (текста е запазен в тази директория)


Провери локално:

```bash
python katib_train.py --lr 0.01 --epochs 30
```

Трябва да видиш:
```
accuracy=0.9...
```
---

# 🚀 Стъпка 2 – Създай Katib Experiment (UI)

UI → **Katib Experiments** → **New ExperimДолу натисни да редактираш YAML-a и paste съдържанието на `katlib_experiment.yaml`
## Как да го приложиш през UI

1. Katib Experiments → Create Experiment (или създай празен и после Edit YAML)
2. Отвори YAML editor за експеримента
3. Paste този YAML
4. Save / Apply

## Как да провериш, че работи

1. В Katib experiment-а трябва да се появят trials
2. В trials ще има Jobs в namespace kubeflow-user-example-com
3. След 1–2 trial-а трябва да видиш accuracy отчетено