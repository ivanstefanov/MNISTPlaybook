**Setup the project**
1. Create a folder for the project
2. Create the virtual environment for this project using ```python -m venv .venv``` (make sure the terminal is inside the projects folder)
3. Activate the environment ```.\.venv\Scripts\activate```.
4. You should see (venv) in the beginning of the prompt. If you cannot see such (because of oh-my-posh for example), try with this command ```python -c "import sys; print(sys.executable)"``` if the result is pointing to yourpoject\.venv\scripts\python.exe, then it is activated (like C:\Projects\Python\MyProject\.venv\Scripts\python.exe)

**Add dependencies**
1. While the .vinv is activated, install the following required packages:
1. Add kernel: ```pip install ipykernel```
2. Set the env variables:  ```pip install requests python-dotenv rich```
3. Pandas and NumPy: ```pip install numpy pandas matplotlib``` (if you need them)
4. PyTorch: ```pip install torch torchvision```

**Setup VS Code (if needed)**
1. If you want to have nicer python environment, add the following to VisualStudio Code (Ctrl+Shift+P -> Preferences: Open User Settings (JSON))
```json
    "editor.formatOnSave": true,

    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnType": true
    },

    "python.analysis.typeCheckingMode": "basic",
    "python.testing.pytestEnabled": true,

    "jupyter.notebookFileRoot": "${workspaceFolder}"
```
2. If you want python linter, install it using ```pip install ruff```
3.  ..and add support in VS Code Preferences (as in point 8)
```json
    "ruff.lint.enable": true,
    "ruff.format.enable": true,
```
4.  Set the notebook kernel with the python.exe from the current .venv by choosing it manually


# Run py files
1. Activate the virtual environment (if not already activated) by running ```.\.venv\Scripts\activate```
2. Run ```python <path_and_name_of_python_file>.py``` Example: `python pipelines/simple_pipeline.py`
3. If you have errors, most of the time, install whatever is needed into the virtual environment

# Ingress
In order to run the multiple modules, ingres should be enabled by `microk8s enable ingress`. Then in the root directory, the ingres configuration should be created. You may copy `/ingess/kubeflow-ingress.yaml` file. The configuration is applied by calling `microk8s kubectl apply -f kubeflow-ingress.yaml`
 - check if ingres is working
```
  microk8s kubectl get pods -n ingress
  microk8s kubectl get ingress -n kubeflow
```
 - `microk8s kubectl get svc -n kubeflow` - check the service names (in order to edit kubeflow-ingress.yaml if needed )
 - `ip addr show eth0 | grep 'inet '` - get the ip address to be called from windows browser in order to load kubeflow and ingress

# katib
- Get the url using the command from above and open it something like: http://172.18.123.127/katib/
- Can create a new experiment by directly uploading the yaml file
- Check for running trials: `microk8s kubectl get trials -n kubeflow`
- Wait for the trials to finish successfully in order to analyze the results. The more yellow lines are better.



# Usefull commands in WSL
 - Install ubunti in WSL - `wsl --install -d Ubuntu`
 - Start ubuntu - `wsl -d Ubuntu`
 - See pods
 - see pods in your pipeline `kubectl get pods -n participant-20` or `kubectl get pods -n kubeflow`
 - check pod logs `kubectl logs <pod-name> -n kubeflow`
 - `microk8s status --wait-ready` make sure MicroK8s works
 - `microk8s kubectl get ns` returns namespaces. should show them withour errors
 - `kubectl` - use AWS
 - `microk8s kubectl` - user local MicroK8s
 - `microk8s kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80` - start port forward, then open http://localhost:8080
