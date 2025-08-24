"""ML Training Page package

Modular training components for integration with the Dataset Converter app.

Usage (Tkinter):
    from ml_training_page.ui import MLTrainingPage
    frame = MLTrainingPage(parent)
    frame.pack(fill='both', expand=True)

The UI uses the core modules:
- gpu_check: Detects CUDA/GPU and provides setup instructions
- dataset_validator: Loads and validates datasets, builds preprocessing pipeline
- train_model: Trains classic ML or Keras neural models with optional GPU
- evaluate_model: Computes metrics and plots; exports HTML report
- utils: Shared helpers (logging, subprocess install, JSON, filenames)
"""
from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
