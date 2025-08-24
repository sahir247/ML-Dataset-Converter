# Raw CSV → Training Dataset Converter (Tkinter)

A simple desktop app to convert a raw CSV into user-selected training dataset formats with optional train/val/test splits and stratification.

## Features

- **Browse or paste** path to input CSV.
- **Choose output format** from many exporters (see below).
- **Select output folder** and base filename.
- **Optional column filtering** (keep only specified columns).
- **Optional train/val/test split** with optional stratification column.
- **Neural processing mode (optional)** with multi-engine support for auto-labeling:
  - Tasks: Detection or Classification
  - Engines: Ultralytics YOLO (Det/Cls), TorchVision (Classification)
  - Model path/name or preset (e.g., `yolo11n.pt`, `yolov8n.pt`, `resnet18`)
  - Confidence threshold (YOLO)
  - Overwrite existing labels toggle
  - Download Models button to fetch common weights offline with a progress bar
- **Logging panel** for progress and validation messages.
- **Preview** first 50 rows of the CSV.
- **Settings auto-save** to reload your last used paths/options.
- **About** menu with version.

### New in this version

- **Auto-install missing dependencies**: If Ultralytics/TorchVision are missing, the app offers to install them in your current Python environment.
- **Animated progress dialog** during long-running downloads/inference so the UI stays visibly active.
- **Safer installs**: Uses pip’s `--upgrade-strategy only-if-needed` and pins current NumPy when installing Ultralytics to avoid unnecessary downgrades.
- **Restart recommendation**: If core libs (e.g., NumPy, Torch, TorchVision, Ultralytics) change during install, the app recommends a restart to ensure a clean state.
- **More robust splitting**: Uses scikit-learn (`train_test_split`, `StratifiedShuffleSplit`) when available for random/stratified splits, with a numpy fallback.

## Requirements

- Python 3.10+ for Windows
- All dependencies are bundled in `requirements.txt` (core + neural + ML training + exporters).
  - Includes: `pandas`, `pyarrow`, `Pillow`, `scikit-learn`, `xgboost`, `lightgbm`, `flaml[automl]`, `tensorflow==2.10.*`, `torch`/`torchvision`/`torchaudio`, `ultralytics`, and more.

Optional extras (install only if you need the feature):

- `openpyxl` — Excel (XLSX) export
- `ultralytics` — Neural mode (YOLO detection/classification)
- `torch`, `torchvision` — TorchVision Classification engine (CPU wheels recommended)
- `tensorflow` — TFRecord export
- `scikit-learn` — Robust random/stratified splitting
- `flaml[automl]` — AutoML backend for the ML Training tab

Install dependencies (all-in-one):

```powershell
pip install -r requirements.txt
# Optional (CPU-only wheels for Torch/TorchVision via PyTorch index):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Run the app

```powershell
python app.py
```

### Fresh machine quickstart (Windows)

You can either run the portable EXE or run from source.

- Run the EXE (no install):
  - Build or download `dist/DatasetConverter/DatasetConverter.exe`
  - Double-click to launch. If SmartScreen warns: click “More info” → “Run anyway”.

- Run from source:

```powershell
# 1) Install Python 3.10 or 3.11 (64-bit)

# 2) Create a virtual environment
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip

# 3) Install all dependencies (all-in-one)
pip install -r requirements2.txt
# Optional: on CPU-only machines, smaller wheels via PyTorch CPU index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4) Run the app
python app.py
```

Notes:

- Model weights (Ultralytics/TorchVision) download on first use. If offline, use the app’s `Download Models` button later when online, or place the `.pt` files next to the EXE or in your working directory.
- TensorFlow is only required for TFRecord export; if it fails to install on your Python/Windows version, remove it from `requirements2.txt` and other features will still work.
- Some systems may prompt for Visual C++ runtime; follow the prompt once if needed.

### PATH warnings (Windows)

If pip prints warnings like “installed in `...\Python313\Scripts` which is not on PATH”, it only affects launching CLI tools (e.g., `yolo`, `ultralytics`) from a terminal. The app itself does not require those to be on PATH.

To add Scripts to your User PATH (optional):

- Path: `C:\Users\LENOVO\AppData\Roaming\Python\Python313\Scripts`
- Environment Variables → User `Path` → Edit → New → paste the path above → OK. Then restart your terminal.

## Supported output formats

- **Tabular**: CSV, JSONL, Parquet, Feather, Excel (XLSX), SQLite
- **Detection**: COCO (Detection), YOLO TXT (Detection), Pascal VOC (XML), YOLO Dataset (images+labels)
- **Classification**: ImageFolder (class-per-subdir)
- **Segmentation**: COCO (Segmentation), YOLO TXT (Segmentation)
- **ML/Hub friendly**: Hugging Face Dataset (JSONL), WebDataset (tar shards)
- **Other**: TFRecord (requires TensorFlow), Audio Manifest (JSONL), TimeSeries Windows (Parquet)

## Notes

- Parquet requires `pyarrow`.
- Excel export requires `openpyxl`.
- TFRecord export requires `tensorflow`.
- Neural mode requires `ultralytics` and/or `torch`+`torchvision`. See below.
- Stratified split uses scikit-learn when available; ensure enough samples per class. If scikit-learn is not installed or a class is too small, the app falls back to a random split.
- Column list should be comma-separated without quotes, e.g.: `feature1, feature2, label`.
- Your settings are stored at `%APPDATA%/DatasetConverter/settings.json` on Windows.

## Output naming

- Without split: `<output_folder>/<base>.{csv|jsonl|parquet}`
- With split: `<output_folder>/<base>_train.*`, `<base>_val.*`, `<base>_test.*`

## Example

1. Click "Browse..." to pick `data/raw.csv` (or "Paste Path").
2. Choose output folder, e.g., `data/processed/`.
3. Select format: `JSONL` (or any from the dropdown).
4. Set base filename: `dataset`.
5. (Optional) Keep columns: `text,label`.
6. Enable split: `Train=0.8, Val=0.1, Test=0.1`, Stratify: `label`.
7. Click **Convert**.

### Neural example (optional)

1. Set Processing Mode to `Neural`.
2. Choose Task `Detection` or `Classification`.
3. Select `Engine`: Ultralytics YOLO (Det/Cls) or TorchVision (Cls).
4. Pick a `Preset` or type a model/arch (e.g., `yolo11n.pt`, `yolov8n.pt`, `resnet18`).
5. Adjust `Confidence` and `Overwrite` as needed (YOLO only).
6. Click **Convert** — predictions are applied before any split/export.
7. During inference, a progress dialog shows per-image progress.

## Troubleshooting

- If CSV reading fails, ensure the file is not open/locked and is a valid CSV.
- If saving Parquet fails, install `pyarrow`.
- Large files: consider running from a 64-bit Python and enough RAM.
- If you used `Download Models` and the app installed/updated core libraries (NumPy/Torch/TorchVision/Ultralytics), **restart the app** if prompted. Running with changed libs in the same process can cause subtle import/ABI issues.

## Build a standalone Windows EXE

You can package this app as a single-folder Windows executable using PyInstaller.

### Quick build (PowerShell)

```powershell
./build.ps1
# or to clean previous builds
./build.ps1 -Clean
```

The EXE will be at `dist/DatasetConverter/DatasetConverter.exe`.

### Manual build

```powershell
# Create venv if needed
py -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements2.txt
.\.venv\Scripts\python -m pip install pyinstaller

# Build
.\.venv\Scripts\python -m PyInstaller \
  --name "DatasetConverter" \
  --noconfirm \
  --windowed \
  --clean \
  app.py
```

If Windows SmartScreen warns when running the EXE, click "More info" → "Run anyway" (you may sign the binary if distributing).

## Neural processing mode

Neural mode auto-annotates your data using Ultralytics YOLO or TorchVision (classification-only).

### Install

```powershell
# In your virtual environment (pick what you need)
# Ultralytics (YOLO engines)
pip install ultralytics

# TorchVision Classification (CPU wheels example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Usage

- Switch Processing Mode to `Neural`.
- Choose `Task` (Detection or Classification).
- Select `Engine`: Ultralytics YOLO (Det/Cls) or TorchVision (Cls).
- Pick a `Preset` or type a model/arch (e.g., `yolo11n.pt`, `yolov8n.pt`, `resnet18`).
- Adjust `Confidence` and `Overwrite` as needed (YOLO only).
- Click `Convert` — predictions are applied before any split/export.
- During inference, a progress dialog shows per-image progress.

### Offline models

- Use the `Download Models` button to prefetch: `yolo11n.pt`, `yolov8n.pt`, `yolov8n-cls.pt`, `resnet18`, `resnet50`, `mobilenet_v3_small`, `efficientnet_b0`.
- The downloader runs PowerShell and shows logs with a green progress bar and percentage.
- If you attempt inference without network and weights are missing, the app will prompt you to download common models.

## ML Training Page (Standard ML, Keras, AutoML)

The `ML Training` tab lets you train and evaluate models on tabular datasets.

### Modes

- **Standard ML**: classic models like LogisticRegression, RandomForest, SVM, XGBoost, LightGBM.
- **Neural (Keras)**: simple Keras MLP with configurable layers, epochs, batch size.
- **AutoML (FLAML)**: automatic model/parameter search using `flaml[automl]`.

### Workflow

1. Click `Browse` and `Load` to open a dataset (`.csv`, `.tsv`, `.json`, `.jsonl`, `.parquet`, `.feather`, `.xlsx`, `.xls`).
2. Pick the `Target` and select `Features`.
3. Choose `Mode`, `Level` (Beginner/Advanced), and `Model`.
4. (Optional) Toggle `Cross-Validate`, set `Folds`, `Seed`, and `Test Size`.
5. For Keras, set `Epochs`, `Batch`, and advanced `Layers` string.
6. Click `Start` to train, then metrics appear and you can export a report.

### Tools in this tab

- **Compare Models**: trains several classic models and picks the best.
- **Tune (Optuna)**: quick hyperparameter tuning for supported models.
- **Save Model**: exports `.pkl` (classic) or `.h5/.keras` (Keras).
- **Export Report**: writes HTML (and optional PDF) to `ml_reports/`.
- **GPU utilities**: Detect GPU, CUDA info, and a quick import/visibility test for TF/Torch.

### AutoML with FLAML

- Uses `flaml.automl.AutoML` (requires `flaml[automl]`). Install via Tools → Setup Environment or `pip install "flaml[automl]"`.
- Chooses task based on target (classification/regression) and uses sensible default metrics.
- Falls back to built-in classic-model comparison if FLAML is unavailable.

## Environment Checker tab

The `Environment` tab helps verify and set up your ML stack.

- **Detect GPU**: shows available GPUs and basic CUDA info.
- **Check Frameworks**: verifies imports for key libraries (Torch/TensorFlow/FLAML, etc.).
- **Install/Upgrade Frameworks**: installs scikit-learn, XGBoost, LightGBM, TensorFlow, FLAML; guided installs with logged progress.
- **Check CUDA/cuDNN**: runs `nvidia-smi` and related checks where available.
- **Calibrate GPU**: runs simple GPU workloads to measure baseline performance.
- **Refresh Summary**: prints a concise status summary to the panel.

Tip: From the `Tools` menu you can also run `Verify Environment` or open `ML Training Page` directly.

## Help: Windows (NVIDIA GPU)

### PyTorch (CUDA)

- **Current (PyTorch ≥ 2.1.0)**: Official binaries support CUDA 11.8 and CUDA 12.1.
  - Install examples:

    ```powershell
    # CUDA 11.8 wheels
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # CUDA 12.1 wheels
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

  - Get the exact command for your OS/Python/CUDA combo on the official page:
    - [PyTorch Get Started](https://pytorch.org/get-started/locally/)
    - [Previous versions](https://pytorch.org/get-started/previous-versions/) (legacy CUDA 8.0/9.0/10.0)
  - Legacy (historical support): CUDA 10.0, 9.0, 8.0

### TensorFlow

- **Native Windows (GPU) support ends at TF 2.10.1.** A known working combo is:
  - TensorFlow 2.10.x + CUDA 11.2 + cuDNN 8.1.1

- For TensorFlow versions > 2.10 on Windows, use **WSL2 (Ubuntu)** for GPU acceleration and follow the Linux install guide.

- Official guidance and references:
  - [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)
  - [GPU compatibility matrix (CUDA/cuDNN)](https://www.tensorflow.org/install/source#gpu)
  - [Linux pip install guide](https://www.tensorflow.org/install/pip#linux) — current GPU guidance uses CUDA Toolkit 12.3 + cuDNN 8.9.7
  - [Set up WSL2 on Windows](https://learn.microsoft.com/windows/wsl/install)
