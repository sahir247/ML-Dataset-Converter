from __future__ import annotations

import json
import os
import queue
import threading
import time
import sys
from dataclasses import asdict
from tkinter import BOTH, BOTTOM, DISABLED, END, LEFT, N, RIGHT, TOP, VERTICAL, HORIZONTAL, X, Y
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Avoid heavy imports at module load time to keep UI responsive on tab open.
# Use TYPE_CHECKING for type hints only; these will not import at runtime.
if TYPE_CHECKING:
    import pandas as pd  # type: ignore
    from .dataset_validator import PreparedData  # type: ignore
    from .evaluate_model import EvaluationResult  # type: ignore
    from .utils import TrainResult  # type: ignore


MODEL_CHOICES = [
    "LogisticRegression",
    "RandomForest",
    "SVM",
    "XGBoost",
    "LightGBM",
    "Keras",
]


class MLTrainingPage(tk.Frame):
    def __init__(self, parent: tk.Misc, default_export_dir: Optional[str] = None):
        super().__init__(parent)
        self.parent = parent
        self.default_export_dir = default_export_dir or os.getcwd()

        # State
        self.df: Optional["pd.DataFrame"] = None
        self.prepared: Optional["PreparedData"] = None
        self.train_result: Optional["TrainResult"] = None
        self.train_model_type: Optional[str] = None
        self.stop_flag = [False]
        self.log_q: queue.Queue[str] = queue.Queue()

        # UI variables
        self.path_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.features_listbox: Optional[tk.Listbox] = None
        self.mode_var = tk.StringVar(value="Standard ML")  # Standard ML | Neural (Keras) | AutoML
        self.level_var = tk.StringVar(value="Beginner")  # Beginner | Advanced
        self.model_var = tk.StringVar(value=MODEL_CHOICES[1])
        self.use_gpu_var = tk.BooleanVar(value=True)
        self.cv_var = tk.BooleanVar(value=False)
        self.cv_folds_var = tk.IntVar(value=5)
        self.seed_var = tk.IntVar(value=42)
        self.epochs_var = tk.IntVar(value=15)
        self.batch_var = tk.IntVar(value=32)
        self.layers_var = tk.StringVar(value="Dense:128 relu | Dropout:0.5 | Dense:64 relu")
        self.test_size_var = tk.DoubleVar(value=0.2)
        self.gpu_status_var = tk.StringVar(value="GPU: detecting…")

        self._build_ui()
        # Initial GPU status
        self._update_gpu_status()
        self._start_log_pump()

    # ------------------------------ UI ------------------------------
    def _build_ui(self):
        # Top: dataset selection
        dataset_fr = ttk.LabelFrame(self, text="Dataset")
        dataset_fr.pack(fill=X, padx=8, pady=6)

        ttk.Label(dataset_fr, text="Path").pack(side=LEFT, padx=4)
        ttk.Entry(dataset_fr, textvariable=self.path_var, width=60).pack(side=LEFT, padx=4, fill=X, expand=True)
        ttk.Button(dataset_fr, text="Browse", command=self._browse_file).pack(side=LEFT, padx=4)
        ttk.Button(dataset_fr, text="Load", command=self._load_dataset).pack(side=LEFT, padx=4)

        # Columns selection
        cols_fr = ttk.LabelFrame(self, text="Columns")
        cols_fr.pack(fill=X, padx=8, pady=6)
        self.target_cb = ttk.Combobox(cols_fr, textvariable=self.target_var, state="readonly", values=[])
        ttk.Label(cols_fr, text="Target").pack(side=LEFT, padx=4)
        self.target_cb.pack(side=LEFT, padx=4)
        ttk.Button(cols_fr, text="Refresh Columns", command=self._refresh_columns).pack(side=LEFT, padx=4)

        # features listbox
        feat_fr = ttk.Frame(cols_fr)
        feat_fr.pack(side=LEFT, padx=10, fill=X, expand=True)
        ttk.Label(feat_fr, text="Features").pack(anchor="w")
        self.features_listbox = tk.Listbox(feat_fr, selectmode=tk.MULTIPLE, height=6)
        self.features_listbox.pack(fill=X, expand=True)

        # Preview
        prev_fr = ttk.LabelFrame(self, text="Preview")
        prev_fr.pack(fill=X, padx=8, pady=6)
        self.preview_text = tk.Text(prev_fr, height=10, wrap="none")
        self.preview_text.pack(fill=X, padx=4, pady=4)
        # Make preview read-only by default
        self.preview_text.configure(state=DISABLED)

        # Mode & model
        mode_fr = ttk.LabelFrame(self, text="Mode & Model")
        mode_fr.pack(fill=X, padx=8, pady=6)
        ttk.Label(mode_fr, text="Mode:").pack(side=LEFT, padx=4)
        ttk.Combobox(mode_fr, textvariable=self.mode_var, state="readonly", values=["Standard ML", "Neural (Keras)", "AutoML"]).pack(side=LEFT, padx=4)
        ttk.Label(mode_fr, text="Level:").pack(side=LEFT, padx=10)
        ttk.Combobox(mode_fr, textvariable=self.level_var, state="readonly", values=["Beginner", "Advanced"]).pack(side=LEFT, padx=4)
        ttk.Label(mode_fr, text="Model:").pack(side=LEFT, padx=10)
        self.model_cb = ttk.Combobox(mode_fr, textvariable=self.model_var, state="readonly", values=MODEL_CHOICES)
        self.model_cb.pack(side=LEFT, padx=4)

        # Advanced settings
        adv_fr = ttk.LabelFrame(self, text="Advanced Settings")
        adv_fr.pack(fill=X, padx=8, pady=6)
        ttk.Checkbutton(adv_fr, text="Cross-Validate", variable=self.cv_var).pack(side=LEFT, padx=4)
        ttk.Label(adv_fr, text="Folds").pack(side=LEFT)
        ttk.Entry(adv_fr, textvariable=self.cv_folds_var, width=5).pack(side=LEFT, padx=4)
        ttk.Label(adv_fr, text="Seed").pack(side=LEFT)
        ttk.Entry(adv_fr, textvariable=self.seed_var, width=7).pack(side=LEFT, padx=4)
        ttk.Label(adv_fr, text="Test Size").pack(side=LEFT)
        ttk.Entry(adv_fr, textvariable=self.test_size_var, width=6).pack(side=LEFT, padx=4)
        ttk.Checkbutton(adv_fr, text="Use GPU (if available)", variable=self.use_gpu_var).pack(side=LEFT, padx=8)
        # GPU status line
        gpu_fr = ttk.Frame(self)
        gpu_fr.pack(fill=X, padx=8, pady=0)
        ttk.Label(gpu_fr, textvariable=self.gpu_status_var, foreground="#0a6").pack(side=LEFT)

        # Keras settings
        keras_fr = ttk.LabelFrame(self, text="Keras Settings")
        keras_fr.pack(fill=X, padx=8, pady=6)
        ttk.Label(keras_fr, text="Epochs").pack(side=LEFT)
        ttk.Entry(keras_fr, textvariable=self.epochs_var, width=6).pack(side=LEFT, padx=4)
        ttk.Label(keras_fr, text="Batch").pack(side=LEFT)
        ttk.Entry(keras_fr, textvariable=self.batch_var, width=6).pack(side=LEFT, padx=4)
        ttk.Label(keras_fr, text="Layers").pack(side=LEFT)
        ttk.Entry(keras_fr, textvariable=self.layers_var, width=50).pack(side=LEFT, padx=4, fill=X, expand=True)

        # Actions
        act_fr = ttk.LabelFrame(self, text="Actions")
        act_fr.pack(fill=X, padx=8, pady=6)
        ttk.Button(act_fr, text="Check GPU", command=self._check_gpu).pack(side=LEFT, padx=4)
        ttk.Button(act_fr, text="Detect GPU", command=self._update_gpu_status).pack(side=LEFT, padx=4)
        ttk.Button(act_fr, text="CUDA Info", command=self._cuda_info).pack(side=LEFT, padx=4)
        ttk.Button(act_fr, text="GPU Quick Test", command=self._gpu_quick_test).pack(side=LEFT, padx=4)
        ttk.Button(act_fr, text="Install Dependencies", command=self._install_deps).pack(side=LEFT, padx=4)
        ttk.Button(act_fr, text="Start", command=self._start).pack(side=LEFT, padx=8)
        ttk.Button(act_fr, text="Stop", command=self._stop).pack(side=LEFT)
        ttk.Button(act_fr, text="Compare Models", command=self._compare_models).pack(side=LEFT, padx=8)
        ttk.Button(act_fr, text="Tune (Optuna)", command=self._tune).pack(side=LEFT)
        ttk.Button(act_fr, text="Save Config", command=self._save_config).pack(side=LEFT, padx=8)
        ttk.Button(act_fr, text="Load Config", command=self._load_config).pack(side=LEFT)
        ttk.Button(act_fr, text="Export Report", command=self._export_report).pack(side=RIGHT, padx=4)
        ttk.Button(act_fr, text="Save Model", command=self._save_model).pack(side=RIGHT)

        # Progress + log
        prog_fr = ttk.LabelFrame(self, text="Progress & Logs")
        prog_fr.pack(fill=BOTH, padx=8, pady=6, expand=True)
        self.progress = ttk.Progressbar(prog_fr, orient=HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=X, padx=4, pady=4)
        self.log_text = tk.Text(prog_fr, height=12, wrap="word")
        self.log_text.pack(fill=BOTH, expand=True, padx=4, pady=4)

    # --------------------------- Utilities ---------------------------
    def _log(self, msg: str):
        self.log_q.put(msg)

    def _pump_logs(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                self.log_text.insert(END, line + "\n")
                self.log_text.see(END)
        except queue.Empty:
            pass
        self.after(100, self._pump_logs)

    def _start_log_pump(self):
        self.after(100, self._pump_logs)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select dataset",
            filetypes=[
                ("Data files", "*.csv *.tsv *.json *.jsonl *.parquet *.feather *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
            initialdir=self.default_export_dir,
        )
        if path:
            self.path_var.set(path)

    def _load_dataset(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Dataset", "Please choose a dataset file")
            return
        try:
            # Lazy import to avoid heavy pandas/IO imports during tab open
            from .dataset_validator import load_dataframe, preview_info  # type: ignore
            import pandas as pd  # type: ignore

            self.df = load_dataframe(path)
            info = preview_info(self.df)
            # Temporarily enable, write, then disable to keep it read-only
            self.preview_text.configure(state="normal")
            self.preview_text.delete("1.0", END)
            self.preview_text.insert(END, f"Shape: {info['shape']}\n")
            self.preview_text.insert(END, f"Dtypes: {info['dtypes']}\n\n")
            self.preview_text.insert(END, "Head (10 rows):\n")
            head_df = pd.DataFrame(info['head'])
            self.preview_text.insert(END, head_df.to_string(index=False))
            self.preview_text.configure(state=DISABLED)
            self._refresh_columns()
            self._log("Dataset loaded.")
        except Exception as e:
            messagebox.showerror("Load Dataset", str(e))

    def _refresh_columns(self):
        if self.df is None:
            return
        cols = list(self.df.columns)
        self.target_cb["values"] = cols
        if cols:
            self.target_var.set(cols[-1])
        self.features_listbox.delete(0, END)
        for c in cols:
            if c != self.target_var.get():
                self.features_listbox.insert(END, c)
        # select all by default
        self.features_listbox.select_set(0, END)

    def _selected_features(self) -> List[str]:
        if not self.features_listbox:
            return []
        sel = [self.features_listbox.get(i) for i in self.features_listbox.curselection()]
        return sel

    def _check_gpu(self):
        from platform import system as os_system
        # Lazy import to avoid importing TF/Torch inadvertently
        from .gpu_check import detect_gpu, install_guide  # type: ignore
        g = detect_gpu()
        msg = [
            f"TensorFlow GPUs: {g.tensorflow_gpus or 'None'}",
            f"PyTorch GPUs: {g.torch_gpus or 'None'}",
            f"XGBoost GPU support: {'Yes' if g.xgboost_gpu else 'No'}",
            f"LightGBM GPU support: {'Yes' if g.lightgbm_gpu else 'No'}",
        ]
        if not g.any_gpu:
            msg.append("\nNo GPU detected — training will run on CPU.\n\n" + install_guide(os_system()))
        else:
            msg.append("\nGPUs detected. You can run 'CUDA Info' and 'GPU Quick Test' for validation.")
        # Also log details
        self._log("GPU Check:\n" + "\n".join(msg))
        messagebox.showinfo("GPU Check", "\n".join(msg))
        # Refresh status line
        self._update_gpu_status()

    def _install_deps(self):
        pkgs = [
            "pandas",
            "numpy",
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "tensorflow==2.12.*",
            "matplotlib",
            "seaborn",
            "optuna",
            "joblib",
            "flaml",
            "xhtml2pdf",
        ]
        # Ask user for optional steps on main thread
        upgrade_tools = messagebox.askyesno(
            "Install Dependencies",
            "Upgrade pip/setuptools/wheel first? Recommended.",
        )
        install_torch_cuda = messagebox.askyesno(
            "Optional: PyTorch",
            "Also install PyTorch with CUDA 11.8 support (Windows/Linux with NVIDIA GPU)?",
        )
        def run():
            self._log("Starting dependency installation...")
            if upgrade_tools:
                self._log("Upgrading pip/setuptools/wheel...")
                from .utils import log_stream_process  # type: ignore
                log_stream_process([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], self._log)
            from .utils import install_packages  # type: ignore
            ok = install_packages(pkgs, self._log)
            if ok:
                self._log("Core dependencies installed.")
            # Optional: install PyTorch CUDA build
            if install_torch_cuda:
                try:
                    import platform
                    osn = platform.system().lower()
                    if "windows" in osn:
                        index_url = "https://download.pytorch.org/whl/cu118"
                    elif "linux" in osn:
                        index_url = "https://download.pytorch.org/whl/cu118"
                    else:
                        index_url = "https://download.pytorch.org/whl/cu118"
                    self._log(f"Installing PyTorch (CUDA 11.8) from {index_url} ...")
                    from .utils import log_stream_process  # type: ignore
                    rc = log_stream_process([sys.executable, "-m", "pip", "install", "--index-url", index_url, "torch", "torchvision", "torchaudio"], self._log)
                    if rc == 0:
                        self._log("PyTorch (CUDA) installed.")
                    else:
                        self._log("PyTorch install failed; you can continue without it or try CPU builds from pytorch.org.")
                except Exception as e:
                    self._log(f"PyTorch install step error: {e}")
            self._log("Dependency installation finished.")
        threading.Thread(target=run, daemon=True).start()

    def _cuda_info(self):
        """Show NVIDIA driver/CUDA info via nvidia-smi and log details."""
        from .gpu_check import get_nvidia_smi_output  # type: ignore
        out = get_nvidia_smi_output()
        if not out.strip():
            out = "nvidia-smi not found. Ensure NVIDIA driver is installed. See GPU Check > guide."
        self._log("nvidia-smi output:\n" + out)
        messagebox.showinfo("CUDA Info", out if len(out) < 4000 else out[:4000] + "\n... (truncated in dialog, see logs)")

    def _gpu_quick_test(self):
        """Import TF and Torch and report GPU visibility."""
        def run():
            self._log("Running GPU quick test...")
            # TensorFlow
            try:
                import tensorflow as tf  # type: ignore
                gpus = tf.config.list_physical_devices('GPU')
                self._log(f"TensorFlow GPUs: {gpus if gpus else 'None'}")
            except Exception as e:
                self._log(f"TensorFlow import failed: {e}")
            # PyTorch
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    self._log(f"PyTorch CUDA available. Device count: {torch.cuda.device_count()}")
                    self._log(f"Current device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
                else:
                    self._log("PyTorch CUDA not available.")
            except Exception as e:
                self._log(f"PyTorch import failed: {e}")
            self._log("GPU quick test finished.")
            # Update status on UI thread
            self.after(0, self._update_gpu_status)
        threading.Thread(target=run, daemon=True).start()

    def _update_gpu_status(self):
        """Update the GPU status label asynchronously to avoid blocking the UI."""
        self.gpu_status_var.set("GPU: detecting…")

        def compute():
            status = "GPU: None detected"
            # Try PyTorch first (can be a heavy import)
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    cnt = torch.cuda.device_count()
                    names = []
                    try:
                        names = [torch.cuda.get_device_name(i) for i in range(cnt)]
                    except Exception:
                        names = [f"cuda:{i}" for i in range(cnt)]
                    cuda_ver = getattr(torch.version, 'cuda', None)
                    if names:
                        status = f"GPU: {', '.join(names)} (CUDA {cuda_ver or '?'})"
                    else:
                        status = f"GPU: {cnt} device(s) (CUDA {cuda_ver or '?'})"
                else:
                    status = "GPU: PyTorch CUDA not available"
            except Exception:
                # No torch; try nvidia-smi presence
                try:
                    from .gpu_check import get_nvidia_smi_output  # type: ignore
                    out = get_nvidia_smi_output()
                    if out.strip():
                        # Attempt to parse a GPU name line
                        name = None
                        for line in out.splitlines():
                            if "GPU" in line and "NVIDIA" in line and "Driver Version" not in line:
                                name = line.strip()
                                break
                        status = f"GPU: NVIDIA driver present{f' ({name})' if name else ''}"
                    else:
                        status = "GPU: No NVIDIA driver detected"
                except Exception:
                    status = "GPU: status unknown"
            # Post result back to the Tk thread
            self.after(0, lambda: self.gpu_status_var.set(status))

        threading.Thread(target=compute, daemon=True).start()

    # --------------------------- Training ----------------------------
    def _start(self):
        if self.df is None:
            messagebox.showwarning("Start", "Load a dataset first.")
            return
        target = self.target_var.get().strip()
        feats = self._selected_features()
        if not target or not feats:
            messagebox.showwarning("Columns", "Select target and features.")
            return

        try:
            # Lazy import to avoid sklearn at module import
            from .dataset_validator import DatasetSpec, validate_and_prepare  # type: ignore
            spec = DatasetSpec(
                path=self.path_var.get().strip(),
                target_col=target,
                feature_cols=feats,
                test_size=float(self.test_size_var.get()),
                random_state=int(self.seed_var.get()),
                dropna=True,
            )
            prepared = validate_and_prepare(self.df, spec)
            self.prepared = prepared
        except Exception as e:
            messagebox.showerror("Validation", str(e))
            return

        mode = self.mode_var.get()
        model_type = self.model_var.get()
        self.train_model_type = model_type
        self.stop_flag[0] = False
        self.progress['value'] = 0
        self.log_text.delete("1.0", END)

        def run():
            try:
                self._log(f"Mode: {mode}, Model: {model_type}")
                # Beginner -> sensible defaults
                # Lazy import training functions to avoid heavy libs until needed
                from .train_model import MLConfig, train_keras, train_standard_ml  # type: ignore
                from .evaluate_model import evaluate  # type: ignore

                cfg = MLConfig(
                    model_type=model_type,
                    cross_validate=bool(self.cv_var.get()) if self.level_var.get()=="Advanced" else False,
                    cv_folds=int(self.cv_folds_var.get()),
                    random_state=int(self.seed_var.get()),
                    use_gpu=bool(self.use_gpu_var.get()),
                    epochs=int(self.epochs_var.get()),
                    batch_size=int(self.batch_var.get()),
                )
                if self.level_var.get() == "Advanced" and model_type.lower() == "keras":
                    cfg.layers = self._parse_layers(self.layers_var.get())

                # Train
                if mode == "Neural (Keras)" or model_type.lower() == "keras":
                    res = train_keras(self.prepared, cfg, log=self._log, progress=self._progress, stop_flag=self.stop_flag)
                elif mode == "AutoML":
                    res = self._train_automl(self.prepared, cfg)
                else:
                    res = train_standard_ml(self.prepared, cfg, log=self._log, progress=self._progress, stop_flag=self.stop_flag)

                self.train_result = res
                if self.stop_flag[0]:
                    self._log("Training was stopped.")
                    return

                # Evaluate
                ev = evaluate(self.prepared, res)
                self._show_metrics(ev)
                self._last_eval = ev
                self.progress['value'] = 100
                self._log("Done.")
            except Exception as e:
                self._log(f"Error: {e}")

        threading.Thread(target=run, daemon=True).start()

    def _progress(self, p: float, msg: str):
        p = max(0.0, min(1.0, float(p))) * 100
        self.progress['value'] = p
        self._log(msg)

    def _stop(self):
        self.stop_flag[0] = True
        self._log("Stop requested. For classical models, stop takes effect between phases. For Keras, end of epoch.")

    def _parse_layers(self, s: str) -> List[tuple]:
        # Format: "Dense:128 relu | Dropout:0.5 | Dense:64 relu"
        layers: List[tuple] = []
        for part in s.split('|'):
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith('dense'):
                rest = part.split(':', 1)[1] if ':' in part else ''
                toks = rest.strip().split()
                units = int(toks[0]) if toks else 64
                act = toks[1] if len(toks) > 1 else 'relu'
                layers.append(("Dense", units, act))
            elif part.lower().startswith('dropout'):
                rest = part.split(':', 1)[1] if ':' in part else '0.5'
                try:
                    rate = float(rest.strip())
                except Exception:
                    rate = 0.5
                layers.append(("Dropout", rate))
            elif part.lower().startswith('batchnorm'):
                layers.append(("BatchNorm",))
        return layers

    def _train_automl(self, prepared: PreparedData, cfg: MLConfig) -> TrainResult:
        # Prefer FLAML if available; fallback to PyCaret if installed
        try:
            from flaml import AutoML  # type: ignore
            self._log("Using FLAML AutoML...")
            import pandas as pd
            X = pd.DataFrame(prepared.X_train, columns=prepared.feature_names)
            y = prepared.y_train
            automl = AutoML()
            task = 'classification' if prepared.task_type == 'classification' else 'regression'
            metric = 'f1' if task == 'classification' else 'rmse'
            time_budget = int(cfg.params.get('time_budget', 60))
            automl.fit(X_train=X, y_train=y, task=task, time_budget=time_budget, metric=metric, verbose=1)
            return TrainResult(model=automl, task_type=prepared.task_type, feature_names=prepared.feature_names)
        except Exception as e:
            self._log(f"FLAML not available or failed: {e}")

        try:
            if prepared.task_type == 'classification':
                from pycaret.classification import setup, compare_models, finalize_model, pull  # type: ignore
            else:
                from pycaret.regression import setup, compare_models, finalize_model, pull  # type: ignore
            self._log("Using PyCaret AutoML...")
            import pandas as pd
            X = pd.DataFrame(prepared.X_train, columns=prepared.feature_names)
            y = pd.Series(prepared.y_train)
            df = pd.concat([X, y.rename("target")], axis=1)
            s = setup(df, target="target", silent=True, verbose=False)
            best = compare_models()
            best_final = finalize_model(best)
            return TrainResult(model=best_final, task_type=prepared.task_type, feature_names=prepared.feature_names)
        except Exception as e:
            self._log(f"PyCaret not available or failed: {e}")

        # Built-in fallback: try a few classic models using our standard trainer and pick the best
        self._log("Using built-in AutoML fallback (scikit-learn pipeline)...")
        candidates = [
            "XGBoost",
            "LightGBM",
            "RandomForest",
            "LogisticRegression",
        ]
        best_tuple: Optional[tuple[str, TrainResult, float]] = None
        for mt in candidates:
            try:
                # Reuse our standard trainer to keep behavior consistent
                from .train_model import MLConfig, train_standard_ml  # type: ignore
                from .evaluate_model import evaluate  # type: ignore
                c = MLConfig(
                    model_type=mt,
                    cross_validate=False,
                    random_state=cfg.random_state,
                    use_gpu=cfg.use_gpu,
                )
                tr = train_standard_ml(prepared, c, log=self._log)
                ev = evaluate(prepared, tr)
                score = self._score_for_compare(ev)
                self._log(f"[AutoML fallback] {mt}: score={score:.6f}")
                if best_tuple is None or score > best_tuple[2]:
                    best_tuple = (mt, tr, score)
            except Exception as ex:
                self._log(f"[AutoML fallback] {mt} failed: {ex}")

        if best_tuple is None:
            raise RuntimeError("AutoML fallback failed: no candidate models could be trained.")
        self._log(f"AutoML fallback selected: {best_tuple[0]} (score={best_tuple[2]:.6f})")
        return best_tuple[1]

    # ------------------------- Post-training -------------------------
    def _show_metrics(self, ev: EvaluationResult):
        self._log("Evaluation metrics:")
        for k, v in ev.metrics.items():
            self._log(f"  {k}: {v:.6f}")
        self._log("You can Export Report to view visuals (HTML).")

    def _save_model(self):
        if not self.train_result or not self.train_model_type:
            messagebox.showwarning("Save Model", "Train a model first.")
            return
        if self.train_model_type.lower() == "keras":
            defaultext = ".h5"
            filetypes = [("Keras Model", "*.h5 *.keras"), ("All files", "*.*")]
        else:
            defaultext = ".pkl"
            filetypes = [("Pickle", "*.pkl"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(defaultextension=defaultext, filetypes=filetypes)
        if not path:
            return
        try:
            from .train_model import save_trained_model  # type: ignore
            save_trained_model(self.train_result, path, self.train_model_type, self._log)
            messagebox.showinfo("Save Model", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Model", str(e))

    def _export_report(self):
        if not hasattr(self, "_last_eval") or self._last_eval is None:
            messagebox.showwarning("Export Report", "Evaluate a model first (train finishes with evaluation).")
            return
        out_dir = os.path.join(os.getcwd(), "ml_reports")
        os.makedirs(out_dir, exist_ok=True)
        from .utils import unique_path, write_html_report  # type: ignore
        from .evaluate_model import build_html_sections  # type: ignore
        html_path = unique_path(out_dir, "training_report", "html")
        sections = build_html_sections(self._last_eval)
        write_html_report(html_path, "Training Report", sections)
        self._log(f"HTML report written: {html_path}")
        try:
            from xhtml2pdf import pisa  # type: ignore
            pdf_path = html_path[:-5] + ".pdf"
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            with open(pdf_path, "wb") as out:
                pisa.CreatePDF(html, dest=out)
            self._log(f"PDF report written: {pdf_path}")
        except Exception as e:
            self._log(f"PDF export optional (install xhtml2pdf). Skipped: {e}")
        messagebox.showinfo("Export Report", f"Report saved to:\n{html_path}")

    # --------------------------- Compare ----------------------------
    def _compare_models(self):
        if self.df is None:
            messagebox.showwarning("Compare", "Load a dataset first.")
            return
        target = self.target_var.get().strip()
        feats = self._selected_features()
        if not target or not feats:
            messagebox.showwarning("Columns", "Select target and features.")
            return
        try:
            from .dataset_validator import DatasetSpec, validate_and_prepare  # type: ignore
            spec = DatasetSpec(
                path=self.path_var.get().strip(),
                target_col=target,
                feature_cols=feats,
                test_size=float(self.test_size_var.get()),
                random_state=int(self.seed_var.get()),
                dropna=True,
            )
            prepared = validate_and_prepare(self.df, spec)
        except Exception as e:
            messagebox.showerror("Validation", str(e))
            return

        models_to_try = [m for m in MODEL_CHOICES if m != "Keras"]  # classic models
        results: List[tuple[str, TrainResult, EvaluationResult]] = []

        def run():
            from .train_model import MLConfig, train_standard_ml  # type: ignore
            from .evaluate_model import evaluate  # type: ignore
            self.log_text.delete("1.0", END)
            self._log("Comparing models...")
            best_score = -1e18
            best_tuple = None
            for i, mt in enumerate(models_to_try, start=1):
                if self.stop_flag[0]:
                    break
                cfg = MLConfig(model_type=mt, cross_validate=False, random_state=int(self.seed_var.get()), use_gpu=bool(self.use_gpu_var.get()))
                tr = train_standard_ml(prepared, cfg, log=self._log)
                ev = evaluate(prepared, tr)
                score = self._score_for_compare(ev)
                self._log(f"[{i}/{len(models_to_try)}] {mt}: score={score:.6f}")
                results.append((mt, tr, ev))
                if score > best_score:
                    best_score = score
                    best_tuple = (mt, tr, ev)
                self.progress['value'] = 100 * i / len(models_to_try)
            if best_tuple:
                self.train_model_type, self.train_result, self._last_eval = best_tuple
                self._log(f"Best model: {self.train_model_type} (score={best_score:.6f})")
            else:
                self._log("No models trained.")

        threading.Thread(target=run, daemon=True).start()

    def _score_for_compare(self, ev: EvaluationResult) -> float:
        if ev.task_type == 'classification':
            return float(ev.metrics.get('f1_macro') or ev.metrics.get('accuracy') or 0.0)
        else:
            # lower rmse is better — convert to negative
            rmse = ev.metrics.get('rmse')
            return -float(rmse) if rmse is not None else float(ev.metrics.get('r2') or 0.0)

    # ---------------------------- Tuning -----------------------------
    def _tune(self):
        if self.df is None:
            messagebox.showwarning("Tune", "Load a dataset first.")
            return
        target = self.target_var.get().strip()
        feats = self._selected_features()
        if not target or not feats:
            messagebox.showwarning("Columns", "Select target and features.")
            return
        try:
            from .dataset_validator import DatasetSpec, validate_and_prepare  # type: ignore
            spec = DatasetSpec(
                path=self.path_var.get().strip(),
                target_col=target,
                feature_cols=feats,
                test_size=float(self.test_size_var.get()),
                random_state=int(self.seed_var.get()),
                dropna=True,
            )
            prepared = validate_and_prepare(self.df, spec)
        except Exception as e:
            messagebox.showerror("Validation", str(e))
            return

        mt = self.model_var.get()
        if mt not in ("RandomForest", "XGBoost", "LightGBM"):
            messagebox.showinfo("Tune", "Tuning supported for RandomForest, XGBoost, and LightGBM.")
            return

        def run():
            try:
                from .train_model import tune_hyperparams  # type: ignore
                best_params, best_score = tune_hyperparams(prepared, mt, n_trials=15, log=self._log)
                self._log(f"Best params for {mt}: {best_params}, score={best_score:.4f}")
            except Exception as e:
                self._log(f"Tuning failed: {e}")
        threading.Thread(target=run, daemon=True).start()

    # -------------------------- Config I/O ---------------------------
    def _save_config(self):
        cfg = {
            "path": self.path_var.get(),
            "target": self.target_var.get(),
            "features": self._selected_features(),
            "mode": self.mode_var.get(),
            "level": self.level_var.get(),
            "model": self.model_var.get(),
            "use_gpu": bool(self.use_gpu_var.get()),
            "cv": bool(self.cv_var.get()),
            "cv_folds": int(self.cv_folds_var.get()),
            "seed": int(self.seed_var.get()),
            "test_size": float(self.test_size_var.get()),
            "epochs": int(self.epochs_var.get()),
            "batch": int(self.batch_var.get()),
            "layers": self.layers_var.get(),
        }
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self._log(f"Config saved: {path}")

    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.path_var.set(cfg.get("path", ""))
            self.mode_var.set(cfg.get("mode", "Standard ML"))
            self.level_var.set(cfg.get("level", "Beginner"))
            self.model_var.set(cfg.get("model", MODEL_CHOICES[1]))
            self.use_gpu_var.set(bool(cfg.get("use_gpu", True)))
            self.cv_var.set(bool(cfg.get("cv", False)))
            self.cv_folds_var.set(int(cfg.get("cv_folds", 5)))
            self.seed_var.set(int(cfg.get("seed", 42)))
            self.test_size_var.set(float(cfg.get("test_size", 0.2)))
            self.epochs_var.set(int(cfg.get("epochs", 15)))
            self.batch_var.set(int(cfg.get("batch", 32)))
            self.layers_var.set(cfg.get("layers", self.layers_var.get()))
            if self.df is not None and cfg.get("features"):
                # Rebuild columns UI
                self._refresh_columns()
                feats = cfg.get("features")
                # Select listed features
                names = [self.features_listbox.get(i) for i in range(self.features_listbox.size())]
                idxs = [i for i, n in enumerate(names) if n in feats]
                self.features_listbox.selection_clear(0, END)
                for i in idxs:
                    self.features_listbox.selection_set(i)
            self.target_var.set(cfg.get("target", self.target_var.get()))
            self._log(f"Config loaded: {path}")
        except Exception as e:
            messagebox.showerror("Load Config", str(e))
