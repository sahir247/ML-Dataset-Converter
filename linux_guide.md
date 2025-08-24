# Linux Guide — Dataset Converter

This guide covers installation on Linux (Ubuntu/Debian/RHEL), GPU setup for NVIDIA/AMD/Intel, running from source, and optional packaging with PyInstaller.

- App entry point: `app.py`
- Linux requirements (latest, unpinned): `linux_requirements.txt`
- Windows requirements (pinned): `requirements.txt`
- Build script (Windows): `build.ps1`
- Spec file (optional): `DatasetConverter.spec`
- Optional model weights: `yolo11n.pt`, `yolov8n.pt`, `yolov8n-cls.pt`

## 1) Prerequisites

- A recent Linux distro (Ubuntu 22.04/24.04, Debian 12, RHEL/CentOS Stream 9)
- Python 3.10 or 3.11 recommended
- Internet access to download Python wheels and model weights
- Optional: GPU (NVIDIA/AMD/Intel) if you want accelerated inference/training

Recommended system packages (many setups work without these, but they help):

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential
```

## 2) Quickstart: Run from Source (CPU by default)

```bash
# In the project root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r linux_requirements.txt
python app.py
```

Notes:

- `linux_requirements.txt` installs the latest versions of all packages (unpinned).
- We use `opencv-python-headless` in Linux to avoid GUI dependencies on servers.
- First-time neural runs will download model weights to your user cache; you can also place `.pt` files next to `app.py`.

## 3) GPU Setup Options

You can use the app entirely on CPU. If you want GPU acceleration, choose the path for your hardware.

### A. NVIDIA GPUs

1. Install the NVIDIA driver

   - Ubuntu (recommended):

     ```bash
     sudo ubuntu-drivers autoinstall
     sudo reboot
     # Verify after reboot
     nvidia-smi
     ```

2. Install PyTorch

   - CPU-only (works everywhere):

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```

   - CUDA 12.1 wheels:

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

   - CUDA 11.8 wheels:

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

   - Tip: PyTorch CUDA wheels are self-contained (CUDA toolkit not required), but the NVIDIA driver MUST be installed.
   - Find exact commands for your OS/Python/GPU: <https://pytorch.org/get-started/locally/>

3. TensorFlow (optional)

   - On Linux, TensorFlow GPU is supported. Follow official guidance for the current CUDA/cuDNN matrix:
     - Install doc: <https://www.tensorflow.org/install/pip>
     - GPU matrix: <https://www.tensorflow.org/install/source#gpu>
   - If you only need TFRecord export, CPU TensorFlow is fine.

### B. AMD GPUs (ROCm)

- ROCm is Linux-only and supported on a subset of AMD GPUs and distros.
- Install ROCm per AMD docs: <https://rocm.docs.amd.com/>
- Then install PyTorch ROCm wheels (example for ROCm 5.6, check the site for your version):

  ```bash
  pip install --index-url https://download.pytorch.org/whl/rocm5.6 torch torchvision torchaudio
  ```

- Verify with `python -c "import torch; print(torch.version, torch.cuda.is_available())"` (ROCm shows as CUDA interface).

### C. Intel GPUs (oneAPI)

- Default install runs on CPU.
- For Intel GPU acceleration with PyTorch, consider Intel Extension for PyTorch (advanced):
  - oneAPI Base Toolkit runtime + drivers (see Intel docs)
  - IPEX: <https://github.com/intel/intel-extension-for-pytorch>
  - Guidance evolves quickly—follow the official instructions for your distro/GPU.

## 4) Installing the Rest of the Requirements

After setting up your GPU wheel (if any):

- If you installed PyTorch CPU/ROCm/CUDA already:

  ```bash
  # Avoid re-resolving Torch deps; install the rest without deps
  pip install -r linux_requirements.txt --no-deps
  ```

- Otherwise (CPU-only default):

  ```bash
  pip install -r linux_requirements.txt
  ```

`linux_requirements.txt` includes:

- `flaml[automl]` for the ML Training tab AutoML
- `opencv-python-headless` for server-friendly OpenCV
- Popular ML libs: scikit-learn, xgboost, lightgbm, ultralytics, tensorflow (optional), etc.

## 5) Running the App

```bash
source .venv/bin/activate
python app.py
```

- Tabs:
  - Converter: pick CSV, output folder, format, optional split/stratify.
  - ML Training: classic ML, Keras, AutoML.
  - Environment: GPU detect, framework checks, installer.
- Tools menu:
  - Verify Environment: import checks
  - Setup Environment: guided installs (scikit-learn, XGBoost, LightGBM, TensorFlow, FLAML)

## 6) Model Weights

- YOLO presets used by the app: `yolo11n.pt`, `yolov8n.pt`, `yolov8n-cls.pt`.
- Download options:
  - In-app: Tools → Download Models.
  - Manual (example):

    ```bash
    # Make a models folder and download
    mkdir -p models && cd models
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O yolo11n.pt
    wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt -O yolov8n.pt
    wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n-cls.pt -O yolov8n-cls.pt
    ```

  - Place the `.pt` files next to `app.py` or run from a folder containing them.

## 7) Packaging on Linux (optional)

You can package a Linux build with PyInstaller similarly to Windows (no PowerShell script provided here):

```bash
python -m pip install --upgrade pip
python -m pip install pyinstaller
# Optional: include local .pt files alongside the build folder
python -m PyInstaller \
  --name "DatasetConverter" \
  --noconfirm \
  --windowed \
  --clean \
  --paths . \
  --add-data "linux_requirements.txt:." \
  app.py
```

Output will be in `dist/DatasetConverter/`. Test by running the executable in that folder.

## 8) Troubleshooting

- Import errors for Torch/TensorFlow: start with CPU wheels, then add GPU wheels.
- If frameworks were installed/updated while app is running, restart the app.
- Headless environments: we use `opencv-python-headless` to avoid GUI libs; that’s expected.
- Permissions: if you run into permission errors with system Python, always use a venv.

## 9) Download Links

- Python: <https://www.python.org/downloads/>
- PyTorch (get started): <https://pytorch.org/get-started/locally/>
- TensorFlow pip install: <https://www.tensorflow.org/install/pip>
- NVIDIA drivers: <https://www.nvidia.com/Download/index.aspx>
- AMD ROCm: <https://rocm.docs.amd.com/>
- Intel oneAPI: <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html>
