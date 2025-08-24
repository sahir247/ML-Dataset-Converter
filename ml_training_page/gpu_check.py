from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import subprocess
import shutil
import os


def _tf_gpu() -> list[str]:
    try:
        import tensorflow as tf  # type: ignore
        gpus = tf.config.list_physical_devices('GPU')
        return [str(g) for g in gpus] if gpus else []
    except Exception:
        return []


def _torch_gpu() -> list[str]:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return []
    except Exception:
        return []


def _xgb_gpu() -> bool:
    try:
        import xgboost as xgb  # type: ignore
        # XGBoost >= 2.0: use device='cuda' with tree_method='hist'
        try:
            xgb.XGBClassifier(tree_method='hist', device='cuda')
            return True
        except Exception:
            return False
    except Exception:
        return False


def _lgbm_gpu() -> bool:
    try:
        import lightgbm as lgb  # type: ignore
        # LightGBM supports GPU if compiled with OpenCL/CUDA; we try to instantiate with device='gpu'
        try:
            lgb.LGBMClassifier(device_type='gpu')  # type: ignore[arg-type]
            return True
        except Exception:
            return False
    except Exception:
        return False


@dataclass
class GPUSummary:
    tensorflow_gpus: list[str]
    torch_gpus: list[str]
    xgboost_gpu: bool
    lightgbm_gpu: bool

    @property
    def any_gpu(self) -> bool:
        return bool(self.tensorflow_gpus or self.torch_gpus or self.xgboost_gpu or self.lightgbm_gpu)


def detect_gpu() -> GPUSummary:
    return GPUSummary(
        tensorflow_gpus=_tf_gpu(),
        torch_gpus=_torch_gpu(),
        xgboost_gpu=_xgb_gpu(),
        lightgbm_gpu=_lgbm_gpu(),
    )


WINDOWS_GUIDE = """
Windows (PowerShell):

1) Verify NVIDIA driver
   nvidia-smi
   # If not found, install/update the NVIDIA driver via GeForce Experience or your GPU vendor.

2) (Optional) CUDA Toolkit (useful for XGBoost/LightGBM GPU)
   winget install Nvidia.CUDA

3) TensorFlow on Windows
   # Native Windows wheels of TF 2.11+ are CPU-only. For GPU, use WSL2 Ubuntu or Linux.
   pip install "tensorflow==2.12.*"  # CPU build

4) PyTorch with CUDA (Windows/Linux)
   # For CUDA 11.8 builds:
   pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

5) Verify in Python
   import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())
   import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
""".strip()


LINUX_GUIDE = """
Linux (Ubuntu/Debian):

1) Drivers & CUDA
   sudo apt update
   # Recommended: use the tested driver
   sudo ubuntu-drivers autoinstall
   # Or manually install a specific driver and CUDA toolkit
   # sudo apt install nvidia-driver-535 nvidia-cuda-toolkit -y

2) Install TensorFlow (Linux wheels provide GPU when CUDA/cuDNN match)
   pip install "tensorflow==2.12.*"
   # Ensure system CUDA/cuDNN versions are compatible with your TF version

3) Verify in Python
   python - <<'PY'
   import tensorflow as tf
   print('TF GPUs:', tf.config.list_physical_devices('GPU'))
   PY

4) (Optional) PyTorch with CUDA 11.8
   pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
   python - <<'PY'
   import torch
   print('Torch CUDA available:', torch.cuda.is_available())
   print('Device count:', torch.cuda.device_count())
   PY
""".strip()


def install_guide(os_name: str | None = None) -> str:
    osn = (os_name or '').lower()
    if 'win' in osn:
        return WINDOWS_GUIDE
    if 'linux' in osn or 'ubuntu' in osn or 'debian' in osn:
        return LINUX_GUIDE
    # default: show both
    return WINDOWS_GUIDE + "\n\n" + LINUX_GUIDE


def get_nvidia_smi_output() -> str:
    """Return the output of `nvidia-smi` if available, else empty string."""
    try:
        exe = shutil.which("nvidia-smi")
        if not exe:
            # Common Windows path fallback; may not exist
            fallback = r"C:\\Windows\\System32\\nvidia-smi.exe"
            exe = fallback if os.path.exists(fallback) else None  # type: ignore[name-defined]
        if not exe:
            return ""
        out = subprocess.check_output([exe], stderr=subprocess.STDOUT, text=True)
        return out
    except Exception:
        return ""
