# ML-Dataset-Converter
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

## Requirements
- Python 3.9+
- Core packages (installed via `requirements.txt`):
  - `pandas`
  - `pyarrow` (for Parquet)
  - `Pillow` (image IO for some exporters)

Optional extras (install only if you need the feature):
- `openpyxl` — Excel (XLSX) export
- `ultralytics` — Neural mode (YOLO detection/classification)
- `torch`, `torchvision` — TorchVision Classification engine (CPU wheels recommended)
- `tensorflow` — TFRecord export

Install dependencies:
```powershell
pip install -r requirements.txt

# Optional (CPU-only wheels for TorchVision via PyTorch index):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Run the app
```powershell
python app.py
```

## Supported output formats
- __Tabular__: CSV, JSONL, Parquet, Feather, Excel (XLSX), SQLite
- __Detection__: COCO (Detection), YOLO TXT (Detection), Pascal VOC (XML), YOLO Dataset (images+labels)
- __Classification__: ImageFolder (class-per-subdir)
- __Segmentation__: COCO (Segmentation), YOLO TXT (Segmentation)
- __ML/Hub friendly__: Hugging Face Dataset (JSONL), WebDataset (tar shards)
- __Other__: TFRecord (requires TensorFlow), Audio Manifest (JSONL), TimeSeries Windows (Parquet)

## Notes
- Parquet requires `pyarrow`.
- Excel export requires `openpyxl`.
- TFRecord export requires `tensorflow`.
- Neural mode requires `ultralytics` and/or `torch`+`torchvision`. See below.
- Stratified split requires enough samples per class; otherwise the app falls back to a random split.
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
.\.venv\Scripts\python -m pip install -r requirements.txt
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

