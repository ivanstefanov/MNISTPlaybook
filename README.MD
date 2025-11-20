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