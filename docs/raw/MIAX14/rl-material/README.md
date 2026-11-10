# RL MATERIAL

RL material for BME Institute (MIAX)

There are two ways to set up the environment. Choose **one**:

| | Option 1: Dev Container | Option 2: Local install with pip |
|---|---|---|
| You need | Docker + VSCode | Python 3.11 |
| Setup | Automatic (Docker image) | `pip install -e .` |
| Best for | Identical environment for everyone | Not using Docker / lighter setup |

---

## Option 1: Dev Container (Docker)

### Requisites

This Repo includes a DevContainer that is ready to use.

You will need **Docker**, **Visual Studio Code**,
and **Dev Containers** VSCode extension installed on your machine.

You can find instructions for installation here:
- Docker: <https://docs.docker.com/get-docker/>
- Visual Studio Code: <https://code.visualstudio.com/download>
- Dev Containers: <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>

### Open Workspace in Dev Container

Open workspace in a dev container with VSCode Dev Containers Extension

First, start VSCode.
Then, select **"File > Open Folder..."** to open project root.

On the bottom left side of the window,
you can see the **"Open a remote window"** button
(the blue rectangular button).
Click on it.

VSCode will open a menu on the top of the window.
Select **"Dev Containers: Reopen in Container"**.

It will take some time until docker images are built
& containers are started. After a few minutes VSCode will open your workspace in the dev container.

Finally, to render gym environments start jupyter-lab with the following command:
```bash
xvfb-run -a -s "-screen 0 1400x900x24" jupyter-lab
```

To stop container:
```bash
docker stop devcontainers_rl-material-miax_1
```

---

## Option 2: Local install with pip (`setup.py`)

### 1. Install Python 3.11

Download it from <https://www.python.org/downloads/> (on Windows and macOS, **3.11.9** is the
last 3.11 release with an official installer). It can live alongside any other Python version
you already have.

> **Windows:** prefer the python.org installer over the Microsoft Store version.

### 2. Create and activate a virtual environment

From the project root:

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Conda (any OS)**
```bash
conda create -n rl-material python=3.11
conda activate rl-material
```

### 3. Install the project

With the environment activated, from the project root:
```bash
python -m pip install --upgrade pip
pip install -e .
```

This installs all required packages and makes everything under `src/` importable from any
notebook (e.g. `from src.utils.plots import plot_training_results`). Thanks to `-e` (editable),
changes in `src/` are picked up without reinstalling (just restart the notebook kernel).

Check that everything works:
```bash
python -c "import gym, tensorflow as tf; env = gym.make('LunarLander-v2'); env.reset(); print('OK', gym.__version__, tf.__version__)"
```

### 4. Run the notebooks

- **JupyterLab:** run `jupyter lab` from the project root.
- **VSCode:** install the *Python* and *Jupyter* extensions, open a notebook and select the
  `.venv` (or `rl-material`) kernel.

No `xvfb` is needed: outside Docker, environments are rendered using your own display.

### Python version

| Version | Status |
|---|---|
| **3.11** | Recommended |
| 3.9, 3.10 | Supported |
| 3.12 or newer | Not supported: `pip install` stops with *"requires a different Python"* |
| 3.8 or older | Not supported |

Why not the latest Python? The notebooks rely on two pinned libraries:
- **TensorFlow 2.15**: last version with Keras 2 and it only has wheels for Python 3.9-3.11.
  Newer versions ship Keras 3, which breaks the dueling / rainbow networks.
- **gym 0.23.1**: the notebooks use its API (`env.reset()` returns the observation and
  `env.step()` returns 4 values). Newer gym / gymnasium versions changed it.

If your default Python is a different version you don't need to uninstall it: create the
environment with 3.11 explicitly as shown above (`py -3.11`, `python3.11` or conda).

### Troubleshooting

- **`pip install` fails with `SSL: CERTIFICATE_VERIFY_FAILED` (corporate network / proxy):** ask
  your IT department for the proxy root certificate, save it in the project root (e.g. `cacert.crt`,
  it is git-ignored) and, with the environment activated, tell pip to use it (only for this environment):
  ```bash
  pip config --site set global.cert <full path to cacert.crt>
  ```
  If the SSL error goes away but downloads are still denied, your company blocks PyPI: ask IT for
  access or install on a computer outside the corporate network.
- **An extra pygame window opens when rendering:** expected. `env.render(mode='rgb_array')` in
  gym 0.23.1 opens a pygame window although frames are shown in the notebook. It closes with
  `env.close()`. On Windows it may show *"Not responding"*: ignore it.
- **Linux / WSL without a display:** set `SDL_VIDEODRIVER=dummy` before starting Jupyter to use
  `render(mode='rgb_array')`. `render()` (human mode) needs a real display.
- **Windows: TensorFlow install fails with "No such file or directory":** path too long. Clone the
  repo in a short path (e.g. `C:\rl-material`) or enable Windows long paths.
- **Windows: `DLL load failed` when importing TensorFlow:** install the latest
  [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).
- **GPU:** not needed, everything runs on CPU (on native Windows TensorFlow is CPU-only).

---

Author: Jesús Sanz del Real
