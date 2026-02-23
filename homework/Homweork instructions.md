
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

# Katib tuning през UI

Katib стартира **Kubernetes Job**, не KFP pipeline.

Затова ще създадем отделен training entrypoint за Katib, който:

* приема аргументи
* тренира
* отпечатва `accuracy=<value>`

---

# ✅ Стъпка 1 – Създай katib_train.py

В Notebook създай файл `katlib experiment\katib_train.py` (текста е запазен в тази директория).
-Реалният training код е в: `katib-wine-train-configmap.yaml`. Вътре e  `data.train.py`
- Експериментът `katib-wine-experiment.yaml` само стартира този код чрез:
  - python3 `/opt/train/train.py`, като `train.py` идва от ConfigMap katib-train-script-wine (mount-нат в /opt/train).


Провери локално:

```bash
python katib_train.py --lr 0.01 --epochs 30
```
*Може да се наложи да инсталираш torch чрез*
```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Трябва да видиш: `accuracy=0.9...`
---

# 🚀 Стъпка 2 – Създай Katib Experiment (UI)

UI → **Katib Experiments** → **New ExperimДолу натисни да редактираш YAML-a и paste съдържанието на `katlib experiment\katib-wine-experiment.yaml`
## Как да го приложиш през UI

1. Katib Experiments → Create Experiment (или създай празен и после Edit YAML)
2. Отвори YAML editor за експеримента
3. Paste този YAML
4. Save / Apply

## Как да провериш, че работи

1. В Katib experiment-а трябва да се появят trials
2. В trials ще има Jobs в namespace kubeflow-user-example-com
3. След 1–2 trial-а трябва да видиш accuracy отчетено

# Алтернатива на стъпка 2 (при мен имаше проблеми с кода който генерираше ymml с конфигурация и питон код на трейнинга)
1. Нужни файлове:
- `katib-wine-train-configmap.yaml`
- `katib-wine-experiment.yaml`

2. Можеш да ги качиш през JupyterLab:
- Upload в File Browser
- или copy/paste съдържанието и Save As в .yaml файл
 Не е нужно „физическо“ копиране извън Jupyter, ако имаш terminal в notebook pod-а.

3. В Jupyter terminal (или друг shell с kubectl достъп) пусни:
```sh
kubectl apply -f katib-wine-train-configmap.yaml
kubectl apply -f katib-wine-experiment.yaml
```

4. Провери:
```sh
kubectl -n kubeflow-user-example-com get experiment mlops-katib-wine-pass-final -o wide
kubectl -n kubeflow-user-example-com get trials -l katib.kubeflow.org/experiment=mlops-katib-wine-pass-final -o wide
```

5. Ако namespace е различен:

- смени metadata.namespace и в двата файла с твоя namespace
- после пак kubectl apply -f ...

6. Ако вече има експеримент със същото име:
```sh
kubectl -n kubeflow-user-example-com delete experiment mlops-katib-wine-pass-final
kubectl apply -f katib-wine-experiment.yaml
```


## Команди проверяващи какви се случва и защо не работи:
Reason: Do you want me to inspect the live Katib experiment spec and status for the failing run?
```sh
kubectl -n kubeflow-user-example-com get experiment katib-wine-pytorch-realworld -o yaml
```

- Виж подовете
```sh
kubectl -n kubeflow-user-example-com get pods
```
- Виж лог за конкретен под
```sh
kubectl -n kubeflow-user-example-com logs <POD_NAME> -c training-container
```
- Get information about trials (може и без exp name)
```sh
kubectl -n kubeflow-user-example-com get trials | <exp name>
```
- статус на експеримент
```sh
kubectl -n kubeflow-user-example-com describe experiment mlops-katib-v8
```
- вземи имената на триаловете
```sh
kubectl -n kubeflow-user-example-com get trials -o name
```

вземи триал инфо
```sh
kubectl -n kubeflow-user-example-com describe trial <trial-name>
```

- Изтриване на експеримент
```sh
kubectl delete experiment <exp name> -n kubeflow-user-example-com`
```

1. За да активирам изпозлването на файлове за да чета входа/изхода
```sh
kubectl label namespace kubeflow-user-example-com \
  katib.kubeflow.org/metrics-collector-injection=enabled \
  --overwrite
```
2. Ако искаш първо да провериш името на namespace-а:
```sh
kubectl get ns | grep kubeflow
```

3. След това провери дали label-ът е приложен:
```sh
kubectl get ns kubeflow-user-example-com --show-labels
```
Трябва да видиш нещо подобно: `katib.kubeflow.org/metrics-collector-injection=enabled`

# KServe модела
## Модифицирай трейнинга си така че да пише модел

## Създай “стабилно място” (PVC) през UI
   1. Влез в Kubeflow UI и избери твоя namespace (напр. kubeflow-user-example-com).
   2. Отиди на Volumes (понякога е “PVCs” или “Persistent Volumes” според UI).
   3. Натисни New Volume / Create Volume:
      1. Name: wine-model-pvc
      2. Size: 1Gi (стига)
      3. Access mode: ако има избор, избери ReadWriteOnce
   4. Create.

Това PVC е “дискът”, който ще пази модела устойчиво.

## Създаване на експеримент, който чете от трейнинг модел и пише в PVC
Необходимите файлове се намират в serve директорията.
- Конфиг мапа е `/.serve/katib-wine-serve-experiment.yaml`.
- Трейнинга е `./serve/serve-model.py`

Ето точните стъпки, така че **твоят Experiment YAML** (който очаква `/opt/train/serve-model.py` от ConfigMap `katib-train-script-wine`) наистина да има достъп до `serve-model.py`.

## Стъпка 1: Сложи `serve-model.py` в Jupyter Notebook-а `adult-income`

В Kubeflow UI:
1. **Notebooks →** отвори `adult-income`.
2. В JupyterLab:
   * качи файла `serve-model.py` в root-а (например `/home/jovyan/serve-model.py`), или го създай като нов файл и постави съдържанието.

## Стъпка 2: Отвори Terminal в същия notebook
В JupyterLab:
1. **File → New → Terminal** (или Launcher → Terminal)

## Стъпка 3: Провери, че файлът е на място
В Terminal:
```bash
ls -la ./serve-model.py
```

Трябва да го виждаш в текущата директория. Ако не е там, отиди където е:
```bash
cd /home/jovyan
ls -la serve-model.py
```
(Пътят може да е различен, но най-често е `/home/jovyan`.)

## Стъпка 4: Създай/обнови ConfigMap-а, който Experiment-ът монтира
Твоят YAML казва:
```yaml
volumes:
  - name: train-script
    configMap:
      name: katib-train-script-wine
```

Значи **трябва** да има ConfigMap с име `katib-train-script-wine` в namespace `kubeflow-user-example-com`, съдържащ ключ `serve-model.py`.
В Terminal изпълни:

```bash
# 1) Изтрий старата версия (ако има)
kubectl -n kubeflow-user-example-com delete configmap katib-train-script-wine --ignore-not-found=true

# 2) Създай нова от файла serve-model.py
kubectl -n kubeflow-user-example-com create configmap katib-train-script-wine \
  --from-file=serve-model.py=./serve-model.py
```

Важно: `--from-file=serve-model.py=./serve-model.py` гарантира, че **ключът вътре в ConfigMap-а** ще се казва `serve-model.py` (точно както Trial-ът ще го вижда под `/opt/train/serve-model.py`).

## Стъпка 5: Провери, че ConfigMap-ът е правилен

```bash
kubectl -n kubeflow-user-example-com get configmap katib-train-script-wine
```

После провери дали ключът е вътре:

```bash
kubectl -n kubeflow-user-example-com get configmap katib-train-script-wine \
  -o jsonpath='{.data}' | head
```

Трябва да видиш нещо от типа `serve-model.py: "..."`.

## Стъпка 6: Стартирай Katib експеримента от UI

В Kubeflow UI:

1. **Katib → Experiments → Create**
2. Пейстни YAML-а (този, който изпрати) и Create.

## Стъпка 7: Какво да очакваш в Trial логовете

Когато Trial Pod-ът стартира, той ще има:

* `/opt/train/serve-model.py` (от ConfigMap)
* `/mnt/model` (от PVC `wine-model-pvc`)

В логовете трябва да видиш:

* `accuracy=...`
* `model_saved=/mnt/model/model.pt`
* `preprocess_saved=/mnt/model/preprocess.pt`

---

### Ако редактираш `serve-model.py` по-късно

Всеки път след промяна:

1. пак изпълняваш **само** Step 4 (delete + create configmap),
2. пускаш нов експеримент / нови trial-и.

Това е — и е напълно съвместимо с кода на експеримента, който даде (mount `/opt/train` от ConfigMap + команда `/opt/train/serve-model.py`).

### Проверка
Ако всичко е наред и trials в експеримента минавата успешно, можеш да провериш в Volumes -> wine-model-pvc -> иконката за директория в дясно дали се садържат model.pt и preprocess.pt в PVC

## KServe
Ще направим малък Python server, който:
- зарежда /mnt/model/model.pt
- зарежда /mnt/model/preprocess.pt
- имплементира predict()
- се пуска от KServe
Това е стандартният production подход.

### Стъпка 1 – Създай serving скрипт (server.py) - `./serve/server.py`
### Стъпка 2 – Dockerfile - `./serve/Dockerfile`
#### Стъпка A1: Отвори Terminal в notebook-а
1. JupyterLab → Launcher → Terminal
2. Отиди в папката:
```sh
cd /home/jovyan/wine-kserve
ls -la
```
#### Стъпка A2: Направи “build context” като tar.gz
Kaniko очаква контекст (папката) да е достъпна. Най-лесно е да направим архив:
```sh
tar -czf context.tar.gz Dockerfile server.py
ls -la context.tar.gz
```
#### Стъпка A3: Качи контекста в PVC (за да е видим за Kaniko Job)
Чудесно — това уточнение прави картината ясна: **MicroK8s (и Kubeflow/KServe) ти живеят вътре в Docker контейнера `kubeflow-control-plane`**. Значи KServe няма как да “видя” image-а, който си билднал на Docker Desktop, докато **не го внесеш вътре** в container runtime-а на MicroK8s в този контейнер.

1. Дръж `server.py` и `Dockerfile` локално (на твоя компютър)

В папката ти (примерно `C:\Projects\wine-kserve\` или WSL path), да имаш:

* `server.py`
* `Dockerfile`

(Това са serving файловете; *не* ти трябват `model.pt` локално — KServe ще ги чете от PVC.)

2. Build image в Docker Desktop

В терминал (PowerShell или WSL) от папката с Dockerfile:
```bash
docker build -t wine-kserve:latest .
```

Провери, че го имаш:
```bash
docker images | findstr wine-kserve
```

3. Save image като tar
```bash
docker save wine-kserve:latest -o wine-kserve.tar
```

Провери, че файлът съществува:
```powershell
dir .\wine-kserve.tar
```

4. Качи tar файла във контейнера `kubeflow-control-plane`

Първо провери името (вече го знаем, но да е сигурно):
```bash
docker ps --format "table {{.Names}}\t{{.Image}}"
```
После:
```bash
docker cp "C:\Projects\MNISTPlaybook\homework\serve\wine-kserve.tar" kubeflow-control-plane:/root/wine-kserve.tar
```
Провери дали се е копирало с
```sh
docker exec -it kubeflow-control-plane sh -lc "ls -la /root | head -n 50; stat /root/wine-kserve.tar && du -h /root/wine-kserve.tar | head -n 1"
```
Открий кой runtime имаш вътре и дали има ctr
```sh
docker exec -it kubeflow-control-plane sh -lc "which ctr || true; which nerdctl || true; which crictl || true; which kubectl || true; ls -la /run/containerd/containerd.sock 2>/dev/null || true; ls -la /var/run/containerd/containerd.sock 2>/dev/null || true"
```
-В 90% от подобни “control-plane” контейнери (kind/k3d/подобни) има ctr и socket /run/containerd/containerd.sock.

5. Импортни image-а в containerd (k8s.io)
```sh
docker exec -it kubeflow-control-plane sh -lc "ctr -n k8s.io images import /root/wine-kserve.tar"
```
След това провери, че е вътре:
```sh
docker exec -it kubeflow-control-plane sh -lc "ctr -n k8s.io images ls | grep -E 'wine-kserve|wine' || true"
```
Ако видиш wine-kserve:latest — готово.

#### Създай InferenceService YAML файл (в notebook-а или локално)
Предпоставка

Вече имаш:

* `wine-model-pvc` с `model.pt` и `preprocess.pt`
* serving image `wine-kserve:latest` импортнат в containerd **namespace `k8s.io`** (след `ctr -n k8s.io images import ...`)

---
1. Създай InferenceService YAML файл (локално на уиндоус машината) (`serve\wine-isvc.yaml`)
```sh
docker cp wine-isvc.yaml kubeflow-control-plane:/root/wine-isvc.yaml
docker exec -it kubeflow-control-plane sh -lc "kubectl apply -f /root/wine-isvc.yaml"
```
Стъпка 4: Провери статус
```sh
docker exec -it kubeflow-control-plane sh -lc "kubectl -n kubeflow-user-example-com get inferenceservice wine-model -o wide"
```
// TODO: Check why it is false and check hot do serve the response