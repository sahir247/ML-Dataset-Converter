# Windows Guide — Dataset Converter

This guide covers installation, GPU setup, running from source, and building a portable EXE on Windows.

- App entry point: `app.py`
- Build script: `build.ps1`
- Spec file (optional): `DatasetConverter.spec`
- Requirements (Windows): `requirements.txt`
- Optional model weights (same folder as app/EXE): `yolo11n.pt`, `yolov8n.pt`, `yolov8n-cls.pt`

## 1) Prerequisites

- Windows 10/11, 64-bit
- Python 3.10 or 3.11 (64-bit)
- Disk space: ~5–10 GB if installing ML frameworks
- Optional: NVIDIA GPU (CUDA 11.8 or 12.1 supported by modern PyTorch), otherwise CPU-only

## 2) Quickstart: Run from Source

```powershell
# In the project root
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Notes:

- If you renamed the file to `requirments.txt` (typo), fix the name back to `requirements.txt` for clarity.
- First-time neural use will download model weights. You can pre-place `.pt` files next to `app.py`.

## 3) GPU Setup (Windows)

### A. NVIDIA (Recommended)

- Install the latest NVIDIA driver from GeForce/Quadro Studio Driver.
- Verify with:

  ```powershell
  nvidia-smi
  ```

- Choose your PyTorch build:
  - CPU-only (works on any machine):

    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

  - CUDA 11.8:

    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

  - CUDA 12.1:

    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

- Get the exact command for your OS/Python/CUDA on:
  - <https://pytorch.org/get-started/locally/>

### B. AMD/Intel GPUs on Windows

- PyTorch ROCm (AMD) is Linux-only. On Windows, prefer CPU wheels.
- For GPU acceleration on Windows with non-NVIDIA hardware, consider running the app inside WSL2 (Ubuntu) and follow the Linux guide.

### C. TensorFlow on Windows

- Official GPU support on native Windows ends at TF 2.10.x (CUDA 11.2 + cuDNN 8.1.1).
- For newer TF GPU, use WSL2 Ubuntu and follow the Linux guide.
- CPU TensorFlow works on Windows; if TF isn’t needed, you can omit it.

## 4) Building a Portable EXE (PyInstaller)

Two options: use our PowerShell script or build from the spec.

### Option 1 — Scripted build

```powershell
# Clean previous build and pack
./build.ps1 -Clean
# EXE output:
# dist/DatasetConverter/DatasetConverter.exe
```

`build.ps1` will:

- Ensure `.venv` exists
- Upgrade pip
- Install from `requirements.txt` (or fallback to `requirments.txt` if present)
- Include `.pt` weight files it finds in the project root
- Package with PyInstaller

### Option 2 — Spec build

```powershell
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install pyinstaller
.\.venv\Scripts\python -m PyInstaller DatasetConverter.spec --clean --noconfirm
```

- `DatasetConverter.spec` bundles `requirements.txt` plus any `.pt` files listed under `datas`.

## 5) Using the App

- Launch: `python app.py` or double-click the EXE.
- Tabs:
  - Converter: choose input CSV, output folder, format, and optional split/stratify.
  - ML Training: classic ML, Keras, and AutoML via `flaml[automl]`.
  - Environment: detect GPU, check frameworks, install/upgrade key libs.
- Tools menu:
  - Verify Environment: quick import checks
  - Setup Environment: guided installs (scikit-learn, XGBoost, LightGBM, TensorFlow, FLAML)

## 6) Model Weights

- YOLO presets used by the app: `yolo11n.pt`, `yolov8n.pt`, `yolov8n-cls.pt`.
- Place the `.pt` files next to `app.py` or the EXE, or use the in-app Download Models button.

## 7) Troubleshooting

- “ImportError” for Torch/TensorFlow: install CPU builds first, then add GPU builds if desired.
- If you installed/updated core libs (NumPy/Torch/TorchVision/Ultralytics) during a session, restart the app.
- SmartScreen: click “More info” → “Run anyway”.
- If TensorFlow fails on Windows, remove it from `requirements.txt`; app features not requiring TF will still work.

## 8) Uninstall/Cleanup

```powershell
# Remove virtual env and builds
Remove-Item -Recurse -Force .venv, build, dist
```

## 9) Download Links

- Python (Windows x86-64): <https://www.python.org/downloads/windows/>
- NVIDIA Drivers: <https://www.nvidia.com/Download/index.aspx>
- PyTorch: <https://pytorch.org/get-started/locally/>
- TensorFlow pip: <https://www.tensorflow.org/install/pip>
- WSL2: <https://learn.microsoft.com/windows/wsl/install>
