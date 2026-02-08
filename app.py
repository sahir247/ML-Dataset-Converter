 
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import json
import shutil
from xml.etree import ElementTree as ET
import tarfile
from pathlib import Path
from tkinter import simpledialog
import io
import subprocess
import tempfile
import re
import webbrowser
# ML Training Page is imported lazily inside open_ml_training_page() to avoid
# import-time dependency errors if optional ML packages are not installed yet.

APP_NAME = "Dataset Converter"
APP_VERSION = "1.0.0"

# Optional dependency for Parquet; pandas will error if missing when used
try:
    import pyarrow  # noqa: F401
except Exception:
    pyarrow = None

# Optional dependency for image IO (COCO/YOLO exporters)
try:
    from PIL import Image  # noqa: F401
except Exception:
    Image = None


class DataProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("860x620")
        self.minsize(820, 600)

        # Preprocessing defaults (backend; UI can bind to these)
        try:
            self.do_preprocess_var = tk.BooleanVar(value=False)
            self.missing_strategy_var = tk.StringVar(value="None")  # None | Drop rows | Fill numeric mean | Fill numeric median | Fill with 0/'' | FFill | BFill
            self.scaling_var = tk.StringVar(value="None")            # None | Standardize (Z-score) | Min-Max [0,1]
            self.encoding_var = tk.StringVar(value="None")           # None | One-Hot | Label/Ordinal
            # Common paths/IDs not to touch by default
            self.preproc_exclude_cols_var = tk.StringVar(value="image_path,audio_path,file_path,label")
            # Advanced controls
            self.preproc_include_num_cols_var = tk.StringVar(value="")
            self.preproc_include_cat_cols_var = tk.StringVar(value="")
            self.onehot_include_nan_var = tk.BooleanVar(value=False)
            self.drop_nonfinite_after_scale_var = tk.BooleanVar(value=False)
        except Exception:
            # If Tk variables cannot be created for some reason, fall back to plain attributes
            self.do_preprocess_var = type("_", (), {"get": lambda *_: False})()
            self.missing_strategy_var = type("_", (), {"get": lambda *_: "None"})()
            self.scaling_var = type("_", (), {"get": lambda *_: "None"})()
            self.encoding_var = type("_", (), {"get": lambda *_: "None"})()
            self.preproc_exclude_cols_var = type("_", (), {"get": lambda *_: "image_path,audio_path,file_path,label"})()
            self.preproc_include_num_cols_var = type("_", (), {"get": lambda *_: ""})()
            self.preproc_include_cat_cols_var = type("_", (), {"get": lambda *_: ""})()
            self.onehot_include_nan_var = type("_", (), {"get": lambda *_: False})()
            self.drop_nonfinite_after_scale_var = type("_", (), {"get": lambda *_: False})()

        self._build_menu()
        self._build_ui()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # Menus
    def _build_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Verify Environment", command=self.verify_environment)
        tools_menu.add_command(label="Setup Environment", command=self.setup_environment_ui)
        tools_menu.add_command(label="ML Training Page", command=self.open_ml_training_page)
        tools_menu.add_command(label="Environment Checker", command=self.open_env_checker_page)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _build_ui(self):
        # Notebook with three tabs: main converter, ML training, and Environment
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        main_tab = ttk.Frame(self.notebook)
        self.ml_tab = ttk.Frame(self.notebook)
        self.env_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Converter")
        self.notebook.add(self.ml_tab, text="ML Training")
        self.notebook.add(self.env_tab, text="Environment")

        # Lazy init flag for ML tab
        self._ml_inited = False
        # Lazy init flag for Environment tab
        self._env_inited = False

        def _init_ml_tab():
            if self._ml_inited:
                return
            try:
                # Create scrollable container in ML tab
                ml_scroll_root = ttk.Frame(self.ml_tab)
                ml_scroll_root.pack(fill=tk.BOTH, expand=True)
                ml_canvas = tk.Canvas(ml_scroll_root, borderwidth=0, highlightthickness=0)
                ml_vscroll = ttk.Scrollbar(ml_scroll_root, orient="vertical", command=ml_canvas.yview)
                ml_canvas.configure(yscrollcommand=ml_vscroll.set)

                ml_canvas.grid(row=0, column=0, sticky=tk.NSEW)
                ml_vscroll.grid(row=0, column=1, sticky=tk.NS)
                ml_scroll_root.grid_rowconfigure(0, weight=1)
                ml_scroll_root.grid_columnconfigure(0, weight=1)

                ml_inner = ttk.Frame(ml_canvas)
                ml_window = ml_canvas.create_window((0, 0), window=ml_inner, anchor="nw")

                # Instantiate MLTrainingPage inside scrollable area
                try:
                    default_dir = (self.output_dir_var.get() or "").strip()
                except Exception:
                    default_dir = ""
                from ml_training_page.ui import MLTrainingPage  # lazy import
                ml_page = MLTrainingPage(ml_inner, default_export_dir=default_dir or os.getcwd())
                ml_page.pack(fill=tk.BOTH, expand=True)

                def _on_ml_frame_configure(event=None):
                    try:
                        ml_canvas.configure(scrollregion=ml_canvas.bbox("all"))
                    except Exception:
                        pass
                ml_inner.bind("<Configure>", _on_ml_frame_configure)

                def _on_ml_canvas_configure(event):
                    try:
                        ml_canvas.itemconfigure(ml_window, width=event.width)
                    except Exception:
                        pass
                ml_canvas.bind("<Configure>", _on_ml_canvas_configure)

                def _on_ml_mousewheel(event):
                    try:
                        delta = int(event.delta / 120)
                    except Exception:
                        delta = 0
                    if delta:
                        ml_canvas.yview_scroll(-delta, "units")

                def _on_shift_mousewheel(event):
                    try:
                        delta = int(event.delta / 120)
                    except Exception:
                        delta = 0
                    if delta:
                        ml_canvas.xview_scroll(-delta, "units")

                # Bind/unbind mouse wheel only when pointer is over the canvas
                def _bind_ml_wheel(event=None):
                    try:
                        ml_canvas.bind_all("<MouseWheel>", _on_ml_mousewheel)
                        ml_canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
                    except Exception:
                        pass
                def _unbind_ml_wheel(event=None):
                    try:
                        ml_canvas.unbind_all("<MouseWheel>")
                        ml_canvas.unbind_all("<Shift-MouseWheel>")
                    except Exception:
                        pass
                ml_canvas.bind("<Enter>", _bind_ml_wheel)
                ml_canvas.bind("<Leave>", _unbind_ml_wheel)

                self._ml_inited = True
                self._ml_canvas = ml_canvas
            except Exception as e:
                try:
                    self.log(f"Failed to initialize ML tab: {e}")
                    messagebox.showerror("ML Training Page", f"Failed to initialize ML tab: {e}")
                except Exception:
                    pass

        # Expose initializer for external callers
        self._init_ml_tab = _init_ml_tab

        def _init_env_tab():
            if self._env_inited:
                return
            try:
                container = ttk.Frame(self.env_tab, padding=12)
                container.pack(fill=tk.BOTH, expand=True)

                # Buttons row
                btn_row = ttk.Frame(container)
                btn_row.pack(fill=tk.X, pady=(0, 8))
                ttk.Button(btn_row, text="Detect GPU", command=self.env_detect_gpu).pack(side=tk.LEFT)
                ttk.Button(btn_row, text="Check Frameworks", command=self.env_check_frameworks).pack(side=tk.LEFT, padx=(6,0))
                ttk.Button(btn_row, text="Install Frameworks", command=self.env_install_frameworks).pack(side=tk.LEFT, padx=(6,0))
                ttk.Button(btn_row, text="Upgrade Frameworks (Linux)", command=self.env_upgrade_frameworks).pack(side=tk.LEFT, padx=(6,0))
                ttk.Button(btn_row, text="Check CUDA/cuDNN", command=self.env_check_cuda_cudnn).pack(side=tk.LEFT, padx=(6,0))
                ttk.Button(btn_row, text="Calibrate GPU", command=self.env_calibrate_gpu).pack(side=tk.LEFT, padx=(6,0))
                ttk.Button(btn_row, text="Refresh Summary", command=self.env_refresh_summary).pack(side=tk.LEFT, padx=(6,0))

                # Output log
                out_frame = ttk.LabelFrame(container, text="Environment Checker Output", padding=8)
                out_frame.pack(fill=tk.BOTH, expand=True)
                of_body = ttk.Frame(out_frame)
                of_body.pack(fill=tk.BOTH, expand=True)
                self._env_out_text = tk.Text(of_body, height=10, wrap=tk.WORD, state=tk.DISABLED)
                of_scroll = ttk.Scrollbar(of_body, orient="vertical", command=self._env_out_text.yview)
                self._env_out_text.configure(yscrollcommand=of_scroll.set)
                self._env_out_text.grid(row=0, column=0, sticky=tk.NSEW)
                of_scroll.grid(row=0, column=1, sticky=tk.NS)
                of_body.columnconfigure(0, weight=1)
                of_body.rowconfigure(0, weight=1)

                # Summary panel
                sum_frame = ttk.LabelFrame(container, text="Final Status Summary", padding=8)
                sum_frame.pack(fill=tk.BOTH, expand=False, pady=(8,0))
                self._env_summary_text = tk.Text(sum_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
                self._env_summary_text.pack(fill=tk.BOTH, expand=True)

                # Status store
                self._env_status = {}
                self._env_inited = True
            except Exception as e:
                try:
                    self.log(f"Failed to initialize Environment tab: {e}")
                    messagebox.showerror("Environment Checker", f"Failed to initialize Environment tab: {e}")
                except Exception:
                    pass

        # Expose initializer
        self._init_env_tab = _init_env_tab

        def _on_tab_changed(event=None):
            try:
                idx = self.notebook.index("current")
                ml_idx = self.notebook.index(self.ml_tab)
                env_idx = self.notebook.index(self.env_tab)
                if idx == ml_idx:
                    _init_ml_tab()
                if idx == env_idx:
                    _init_env_tab()
            except Exception:
                pass
        self.notebook.bind("<<NotebookTabChanged>>", _on_tab_changed)

        # Root scrollable area (Canvas + both scrollbars) for main tab
        scroll_root = ttk.Frame(main_tab)
        scroll_root.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(scroll_root, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(scroll_root, orient="vertical", command=canvas.yview)
        hscroll = ttk.Scrollbar(scroll_root, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        # Layout scroll components
        canvas.grid(row=0, column=0, sticky=tk.NSEW)
        vscroll.grid(row=0, column=1, sticky=tk.NS)
        hscroll.grid(row=1, column=0, sticky=tk.EW)
        scroll_root.columnconfigure(0, weight=1)
        scroll_root.rowconfigure(0, weight=1)

        # Content frame inside canvas
        container = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")

        # Update scrollregion when content changes size
        def _on_frame_configure(event=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass
        container.bind("<Configure>", _on_frame_configure)

        # Keep inner frame width in sync with canvas width (reduces unnecessary horizontal scroll)
        def _on_canvas_configure(event):
            try:
                canvas.itemconfigure(canvas_window, width=event.width)
            except Exception:
                pass
        canvas.bind("<Configure>", _on_canvas_configure)

        # Optional: mouse wheel scrolling (Windows/macOS). Shift+Wheel for horizontal
        def _on_mousewheel(event):
            try:
                delta = int(event.delta / 120)
            except Exception:
                delta = 0
            if delta:
                canvas.yview_scroll(-delta, "units")

        def _on_shift_mousewheel(event):
            try:
                delta = int(event.delta / 120)
            except Exception:
                delta = 0
            if delta:
                canvas.xview_scroll(-delta, "units")

        # Bind/unbind mouse wheel only when pointer is over the canvas
        def _bind_wheel(event=None):
            try:
                canvas.bind_all("<MouseWheel>", _on_mousewheel)
                canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
            except Exception:
                pass
        def _unbind_wheel(event=None):
            try:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Shift-MouseWheel>")
            except Exception:
                pass
        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        # Input file
        in_frame = ttk.LabelFrame(container, text="Input Raw CSV", padding=10)
        in_frame.pack(fill=tk.X, expand=False, pady=(0, 8))

        self.input_path_var = tk.StringVar()
        ttk.Label(in_frame, text="CSV Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.input_entry = ttk.Entry(in_frame, textvariable=self.input_path_var, width=80)
        self.input_entry.grid(row=0, column=1, sticky=tk.W+tk.E, pady=4)
        in_frame.columnconfigure(1, weight=1)

        browse_btn = ttk.Button(in_frame, text="Browse...", command=self.browse_input)
        browse_btn.grid(row=0, column=2, padx=4)
        paste_btn = ttk.Button(in_frame, text="Paste Path", command=self.paste_input)
        paste_btn.grid(row=0, column=3, padx=4)
        preview_btn = ttk.Button(in_frame, text="Preview", command=self.preview_input)
        preview_btn.grid(row=0, column=4, padx=4)

        # Output settings
        out_frame = ttk.LabelFrame(container, text="Output Settings", padding=10)
        out_frame.pack(fill=tk.X, expand=False, pady=(0, 8))

        # Output directory
        self.output_dir_var = tk.StringVar()
        ttk.Label(out_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.output_entry = ttk.Entry(out_frame, textvariable=self.output_dir_var, width=60)
        self.output_entry.grid(row=0, column=1, sticky=tk.W+tk.E, pady=4)
        out_frame.columnconfigure(1, weight=1)
        out_browse_btn = ttk.Button(out_frame, text="Browse...", command=self.browse_output_dir)
        out_browse_btn.grid(row=0, column=2, padx=4)

        # Base filename
        self.base_name_var = tk.StringVar(value="dataset")
        ttk.Label(out_frame, text="Base Filename:").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        ttk.Entry(out_frame, textvariable=self.base_name_var, width=30).grid(row=1, column=1, sticky=tk.W, pady=4)

        # Format selection
        self.format_var = tk.StringVar(value="CSV")
        ttk.Label(out_frame, text="Output Format:").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        format_combo = ttk.Combobox(
            out_frame,
            textvariable=self.format_var,
            values=[
                "CSV",
                "TSV",
                "JSONL",
                "JSON",
                "Parquet",
                "Feather",
                "Excel (XLSX)",
                "SQLite",
                "COCO (Detection)",
                "YOLO TXT (Detection)",
                "ImageFolder (Classification)",
                "Pascal VOC (Detection)",
                "YOLO Dataset (images+labels)",
                "COCO (Segmentation)",
                "YOLO TXT (Segmentation)",
                "HF Dataset (JSONL)",
                "WebDataset (tar shards)",
                "TFRecord (optional)",
                "Audio Manifest (JSONL)",
                "TimeSeries Windows (Parquet)",
            ],
            state="readonly",
            width=27,
        )
        format_combo.grid(row=2, column=1, sticky=tk.W, pady=4)

        # Processing Mode
        ttk.Label(out_frame, text="Processing Mode:").grid(row=3, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        self.mode_var = tk.StringVar(value="Standard")
        self.mode_combo = ttk.Combobox(out_frame, textvariable=self.mode_var, values=["Standard", "Neural"], state="readonly", width=27)
        self.mode_combo.grid(row=3, column=1, sticky=tk.W, pady=4)

        # Neural options (inline)
        self.task_var = tk.StringVar(value="Detection")
        ttk.Label(out_frame, text="Task:").grid(row=4, column=0, sticky=tk.E)
        self.task_combo = ttk.Combobox(out_frame, textvariable=self.task_var, values=["Detection", "Classification"], state="readonly", width=20)
        self.task_combo.grid(row=4, column=1, sticky=tk.W, pady=2)

        # Engine selector
        self.engine_var = tk.StringVar(value="Ultralytics YOLO")
        ttk.Label(out_frame, text="Engine:").grid(row=5, column=0, sticky=tk.E)
        self.engine_combo = ttk.Combobox(out_frame, textvariable=self.engine_var, values=["Ultralytics YOLO", "TorchVision (Cls)"], state="readonly", width=20)
        self.engine_combo.grid(row=5, column=1, sticky=tk.W, pady=2)

        # Model preset selector
        self.preset_model_var = tk.StringVar(value="yolov8n.pt")
        ttk.Label(out_frame, text="Preset:").grid(row=5, column=2, sticky=tk.E)
        self.preset_combo = ttk.Combobox(out_frame, textvariable=self.preset_model_var, state="readonly", width=22)
        self.preset_combo.grid(row=5, column=3, sticky=tk.W, padx=(4,0))

        # Download models button (offline fetch) + Verify Env
        btns = ttk.Frame(out_frame)
        btns.grid(row=5, column=4, sticky=tk.W, padx=(8,0))
        self.download_btn = ttk.Button(btns, text="Download Models", command=self.download_models_ui)
        self.download_btn.pack(side=tk.LEFT)
        self.verify_btn = ttk.Button(btns, text="Verify Env", command=self.verify_environment)
        self.verify_btn.pack(side=tk.LEFT, padx=(6,0))

        # Freeform model path/name
        self.model_var = tk.StringVar(value="yolov8n.pt")
        ttk.Label(out_frame, text="Model:").grid(row=4, column=2, sticky=tk.E)
        self.model_entry = ttk.Entry(out_frame, textvariable=self.model_var, width=22)
        self.model_entry.grid(row=4, column=3, sticky=tk.W, padx=(4,0))

        self.conf_var = tk.StringVar(value="0.25")
        ttk.Label(out_frame, text="Conf:").grid(row=4, column=4, sticky=tk.E)
        self.conf_entry = ttk.Entry(out_frame, textvariable=self.conf_var, width=6)
        self.conf_entry.grid(row=4, column=5, sticky=tk.W)

        self.overwrite_var = tk.BooleanVar(value=False)
        self.overwrite_chk = ttk.Checkbutton(out_frame, text="Overwrite", variable=self.overwrite_var)
        self.overwrite_chk.grid(row=4, column=6, sticky=tk.W, padx=(6,0))

        def _refresh_presets(*_):
            """Update preset list based on engine and task, and optionally set model field."""
            eng = self.engine_var.get()
            task = self.task_var.get()
            if eng == "Ultralytics YOLO":
                if task == "Detection":
                    options = ["yolo11n.pt", "yolov8n.pt", "yolov8s.pt", "yolov5s.pt", "custom..."]
                else:
                    options = ["yolov8n-cls.pt", "yolov8s-cls.pt", "custom..."]
            else:  # TorchVision
                # Classification only
                options = ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0", "custom..."]
            self.preset_combo.configure(values=options)
            if options:
                self.preset_model_var.set(options[0])
                if options[0] != "custom...":
                    self.model_var.set(options[0])

        def _on_preset_change(event=None):
            val = self.preset_model_var.get()
            if val and val != "custom...":
                self.model_var.set(val)

        _refresh_presets()
        self.engine_combo.bind("<<ComboboxSelected>>", lambda e: _refresh_presets())
        self.task_combo.bind("<<ComboboxSelected>>", lambda e: _refresh_presets())
        self.preset_combo.bind("<<ComboboxSelected>>", _on_preset_change)

        def _toggle_neural(*_):
            enable = (self.mode_var.get().lower() == "neural")
            state = "normal" if enable else "disabled"
            for w in [self.task_combo, self.engine_combo, self.preset_combo, self.model_entry, self.conf_entry, self.overwrite_chk]:
                w.configure(state=state)
        _toggle_neural()
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: _toggle_neural())

        # Columns
        col_frame = ttk.LabelFrame(container, text="Column Selection (optional)", padding=10)
        col_frame.pack(fill=tk.X, expand=False, pady=(0, 8))
        self.cols_var = tk.StringVar()
        ttk.Label(col_frame, text="Keep Columns (comma-separated):").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        ttk.Entry(col_frame, textvariable=self.cols_var, width=80).grid(row=0, column=1, sticky=tk.W+tk.E, pady=4)
        col_frame.columnconfigure(1, weight=1)

        # Auto-create image_path helpers (used by Neural mode)
        self.auto_imgpath_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(col_frame, text="Auto-create image_path if missing", variable=self.auto_imgpath_var).grid(row=1, column=0, sticky=tk.W, pady=(6,2))

        ttk.Label(col_frame, text="Filename column:").grid(row=2, column=0, sticky=tk.W, padx=(0,6))
        self.image_filename_col_var = tk.StringVar(value="filename")
        ttk.Entry(col_frame, textvariable=self.image_filename_col_var, width=30).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(col_frame, text="Image base folder:").grid(row=3, column=0, sticky=tk.W, padx=(0,6), pady=(2,4))
        self.image_base_dir_var = tk.StringVar()
        base_frame = ttk.Frame(col_frame)
        base_frame.grid(row=3, column=1, sticky=tk.W+tk.E, pady=(2,4))
        ttk.Entry(base_frame, textvariable=self.image_base_dir_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(base_frame, text="Browse...", command=self.browse_image_base_dir).pack(side=tk.LEFT, padx=(6,0))

        # Preprocessing
        pre_frame = ttk.LabelFrame(container, text="Preprocessing (optional)", padding=10)
        pre_frame.pack(fill=tk.X, expand=False, pady=(0, 8))

        ttk.Checkbutton(pre_frame, text="Enable preprocessing", variable=self.do_preprocess_var).grid(row=0, column=0, sticky=tk.W, pady=4)

        ttk.Label(pre_frame, text="Missing strategy:").grid(row=1, column=0, sticky=tk.E)
        self._missing_options = [
            "None",
            "Drop rows",
            "Fill numeric mean",
            "Fill numeric median",
            "Fill num mean + cat mode",
            "Fill num median + cat mode",
            "Fill with 0/''",
            "FFill",
            "BFill",
        ]
        self.missing_combo = ttk.Combobox(pre_frame, textvariable=self.missing_strategy_var, values=self._missing_options, state="readonly", width=28)
        self.missing_combo.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(pre_frame, text="Scaling:").grid(row=1, column=2, sticky=tk.E)
        self._scaling_options = ["None", "Standardize (Z-score)", "Min-Max [0,1]"]
        self.scaling_combo = ttk.Combobox(pre_frame, textvariable=self.scaling_var, values=self._scaling_options, state="readonly", width=22)
        self.scaling_combo.grid(row=1, column=3, sticky=tk.W, pady=2)

        ttk.Label(pre_frame, text="Encoding:").grid(row=1, column=4, sticky=tk.E)
        self._encoding_options = ["None", "One-Hot", "Label/Ordinal"]
        self.encoding_combo = ttk.Combobox(pre_frame, textvariable=self.encoding_var, values=self._encoding_options, state="readonly", width=18)
        self.encoding_combo.grid(row=1, column=5, sticky=tk.W, pady=2)

        self.onehot_include_nan_chk = ttk.Checkbutton(pre_frame, text="One-hot: include NaN", variable=self.onehot_include_nan_var)
        self.onehot_include_nan_chk.grid(row=1, column=6, sticky=tk.W, padx=(8,0))

        # Exclude/include columns
        ttk.Label(pre_frame, text="Exclude columns:").grid(row=2, column=0, sticky=tk.E, pady=(6,2))
        ttk.Entry(pre_frame, textvariable=self.preproc_exclude_cols_var, width=70).grid(row=2, column=1, columnspan=3, sticky=tk.W)

        ttk.Label(pre_frame, text="Include numeric cols:").grid(row=3, column=0, sticky=tk.E)
        ttk.Entry(pre_frame, textvariable=self.preproc_include_num_cols_var, width=70).grid(row=3, column=1, columnspan=3, sticky=tk.W)

        ttk.Label(pre_frame, text="Include categorical cols:").grid(row=4, column=0, sticky=tk.E)
        ttk.Entry(pre_frame, textvariable=self.preproc_include_cat_cols_var, width=70).grid(row=4, column=1, columnspan=3, sticky=tk.W)

        self.drop_nonfinite_chk = ttk.Checkbutton(pre_frame, text="Drop rows with NaN/Inf after transforms", variable=self.drop_nonfinite_after_scale_var)
        self.drop_nonfinite_chk.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(6,2))

        def _toggle_preproc_state(*_):
            on = bool(self.do_preprocess_var.get())
            state = "normal" if on else "disabled"
            for w in [self.missing_combo, self.scaling_combo, self.encoding_combo, self.onehot_include_nan_chk,
                      self.drop_nonfinite_chk]:
                try:
                    w.configure(state=state)
                except Exception:
                    pass
            # text entries use configure(state=)
            for child in [
                (pre_frame.grid_slaves(row=2, column=1)[0] if pre_frame.grid_slaves(row=2, column=1) else None),
                (pre_frame.grid_slaves(row=3, column=1)[0] if pre_frame.grid_slaves(row=3, column=1) else None),
                (pre_frame.grid_slaves(row=4, column=1)[0] if pre_frame.grid_slaves(row=4, column=1) else None),
            ]:
                if child is not None:
                    try:
                        child.configure(state=state)
                    except Exception:
                        pass
            # One-hot dependent
            onehot_state = state if (on and self.encoding_var.get().lower().startswith("one-hot")) else "disabled"
            try:
                self.onehot_include_nan_chk.configure(state=onehot_state)
            except Exception:
                pass

        def _toggle_onehot(*_):
            _toggle_preproc_state()

        _toggle_preproc_state()
        self.do_preprocess_var.trace_add('write', lambda *_: _toggle_preproc_state())
        self.encoding_combo.bind("<<ComboboxSelected>>", lambda e: _toggle_onehot())

        # Split settings
        split_frame = ttk.LabelFrame(container, text="Train/Val/Test Split (optional)", padding=10)
        split_frame.pack(fill=tk.X, expand=False, pady=(0, 8))

        self.do_split_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(split_frame, text="Perform split", variable=self.do_split_var).grid(row=0, column=0, sticky=tk.W, pady=4)

        ttk.Label(split_frame, text="Train:").grid(row=0, column=1, sticky=tk.E)
        self.train_ratio_var = tk.StringVar(value="0.8")
        ttk.Entry(split_frame, textvariable=self.train_ratio_var, width=6).grid(row=0, column=2, padx=(4, 10))

        ttk.Label(split_frame, text="Val:").grid(row=0, column=3, sticky=tk.E)
        self.val_ratio_var = tk.StringVar(value="0.1")
        ttk.Entry(split_frame, textvariable=self.val_ratio_var, width=6).grid(row=0, column=4, padx=(4, 10))

        ttk.Label(split_frame, text="Test:").grid(row=0, column=5, sticky=tk.E)
        self.test_ratio_var = tk.StringVar(value="0.1")
        ttk.Entry(split_frame, textvariable=self.test_ratio_var, width=6).grid(row=0, column=6, padx=(4, 10))

        ttk.Label(split_frame, text="Stratify by column (optional):").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.stratify_col_var = tk.StringVar()
        ttk.Entry(split_frame, textvariable=self.stratify_col_var, width=30).grid(row=1, column=1, columnspan=2, sticky=tk.W)

        # Actions
        act_frame = ttk.Frame(container)
        act_frame.pack(fill=tk.X, pady=(4, 8))
        process_btn = ttk.Button(act_frame, text="Convert", command=self.process)
        process_btn.pack(side=tk.RIGHT)

        # Log output
        log_frame = ttk.LabelFrame(container, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        # Wrap Text and Scrollbar in a frame to use grid for proper resizing
        log_body = ttk.Frame(log_frame)
        log_body.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_body, height=12, wrap=tk.WORD)
        yscroll = ttk.Scrollbar(log_body, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=yscroll.set)
        # Layout with grid
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)
        yscroll.grid(row=0, column=1, sticky=tk.NS)
        log_body.columnconfigure(0, weight=1)
        log_body.rowconfigure(0, weight=1)

    # UI helpers
    def browse_input(self):
        path = filedialog.askopenfilename(title="Select raw CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.input_path_var.set(path)

    def paste_input(self):
        try:
            data = self.clipboard_get()
            if data:
                self.input_path_var.set(data.strip())
        except tk.TclError:
            messagebox.showwarning("Clipboard", "Clipboard is empty or not accessible.")

    def browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    def browse_image_base_dir(self):
        path = filedialog.askdirectory(title="Select image base folder")
        if path:
            self.image_base_dir_var.set(path)

    def log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def _ask_radio_choice(self, title: str, message: str, options: list[str], default_index: int = 0) -> str | None:
        """Modal dialog with radio buttons for a non-typable choice.

        Returns the selected option string, or None if cancelled.
        Must be called on the Tk main thread.
        """
        if not options:
            return None
        try:
            dlg = tk.Toplevel(self)
            dlg.title(title)
            dlg.transient(self)
            dlg.resizable(False, False)
            dlg.grab_set()

            result: dict[str, str | None] = {"value": None}
            var = tk.StringVar(value=options[min(max(default_index, 0), len(options) - 1)])

            body = ttk.Frame(dlg, padding=12)
            body.pack(fill=tk.BOTH, expand=True)
            ttk.Label(body, text=message).pack(anchor=tk.W, pady=(0,6))
            for opt in options:
                ttk.Radiobutton(body, text=opt, variable=var, value=opt).pack(anchor=tk.W)

            btns = ttk.Frame(body)
            btns.pack(fill=tk.X, pady=(10,0))

            def on_ok():
                result["value"] = var.get()
                dlg.destroy()

            def on_cancel():
                result["value"] = None
                dlg.destroy()

            ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.RIGHT)
            ttk.Button(btns, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=(0,6))

            def on_close():
                on_cancel()
            dlg.protocol("WM_DELETE_WINDOW", on_close)

            # Center dialog over parent
            dlg.update_idletasks()
            try:
                px = self.winfo_rootx()
                py = self.winfo_rooty()
                pw = self.winfo_width()
                ph = self.winfo_height()
                dw = dlg.winfo_reqwidth()
                dh = dlg.winfo_reqheight()
                x = px + (pw - dw) // 2
                y = py + (ph - dh) // 2
                dlg.geometry(f"{dw}x{dh}+{max(x,0)}+{max(y,0)}")
            except Exception:
                pass

            dlg.wait_window()
            return result["value"]
        except Exception:
            return None

    def _config_dir(self) -> str:
        # Use AppData/Roaming on Windows, otherwise home dir
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        path = os.path.join(base, APP_NAME.replace(" ", ""))
        os.makedirs(path, exist_ok=True)
        return path

    def _config_path(self) -> str:
        return os.path.join(self._config_dir(), "settings.json")

    def _load_settings(self):
        try:
            with open(self._config_path(), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.input_path_var.set(cfg.get("input_path", ""))
            self.output_dir_var.set(cfg.get("output_dir", ""))
            self.base_name_var.set(cfg.get("base_name", "dataset"))
            self.format_var.set(cfg.get("format", "CSV"))
            self.cols_var.set(cfg.get("keep_cols", ""))
            # Preprocessing
            self.do_preprocess_var.set(bool(cfg.get("do_preprocess", False)))
            self.missing_strategy_var.set(cfg.get("missing_strategy", "None"))
            self.scaling_var.set(cfg.get("scaling", "None"))
            self.encoding_var.set(cfg.get("encoding", "None"))
            self.preproc_exclude_cols_var.set(cfg.get("preproc_exclude_cols", "image_path,audio_path,file_path,label"))
            self.preproc_include_num_cols_var.set(cfg.get("preproc_include_num_cols", ""))
            self.preproc_include_cat_cols_var.set(cfg.get("preproc_include_cat_cols", ""))
            self.onehot_include_nan_var.set(bool(cfg.get("onehot_include_nan", False)))
            self.drop_nonfinite_after_scale_var.set(bool(cfg.get("drop_nonfinite_after_scale", False)))
            # Neural
            self.mode_var.set(cfg.get("processing_mode", "Standard"))
            self.task_var.set(cfg.get("neural_task", "Detection"))
            self.engine_var.set(cfg.get("neural_engine", "Ultralytics YOLO"))
            self.preset_model_var.set(cfg.get("neural_preset", "yolov8n.pt"))
            self.conf_var.set(str(cfg.get("neural_conf", 0.25)))
            self.overwrite_var.set(bool(cfg.get("neural_overwrite", False)))
            # Image path helpers
            self.auto_imgpath_var.set(bool(cfg.get("auto_imgpath", False)))
            self.image_filename_col_var.set(cfg.get("image_filename_col", "filename"))
            self.image_base_dir_var.set(cfg.get("image_base_dir", ""))
            self.do_split_var.set(cfg.get("do_split", True))
            self.train_ratio_var.set(str(cfg.get("train_ratio", 0.8)))
            self.val_ratio_var.set(str(cfg.get("val_ratio", 0.1)))
            self.test_ratio_var.set(str(cfg.get("test_ratio", 0.1)))
            self.stratify_col_var.set(cfg.get("stratify_col", ""))
        except Exception:
            # Ignore if missing or invalid
            pass

    def _save_settings(self):
        cfg = {
            "input_path": self.input_path_var.get().strip(),
            "output_dir": self.output_dir_var.get().strip(),
            "base_name": self.base_name_var.get().strip() or "dataset",
            "format": self.format_var.get(),
            "keep_cols": self.cols_var.get().strip(),
            # Preprocessing
            "do_preprocess": bool(self.do_preprocess_var.get()),
            "missing_strategy": self.missing_strategy_var.get(),
            "scaling": self.scaling_var.get(),
            "encoding": self.encoding_var.get(),
            "preproc_exclude_cols": self.preproc_exclude_cols_var.get().strip(),
            "preproc_include_num_cols": self.preproc_include_num_cols_var.get().strip(),
            "preproc_include_cat_cols": self.preproc_include_cat_cols_var.get().strip(),
            "onehot_include_nan": bool(self.onehot_include_nan_var.get()),
            "drop_nonfinite_after_scale": bool(self.drop_nonfinite_after_scale_var.get()),
            "do_split": bool(self.do_split_var.get()),
            "train_ratio": float(self.train_ratio_var.get() or 0),
            "val_ratio": float(self.val_ratio_var.get() or 0),
            "test_ratio": float(self.test_ratio_var.get() or 0),
            "stratify_col": self.stratify_col_var.get().strip(),
            # Neural
            "processing_mode": self.mode_var.get(),
            "neural_task": self.task_var.get(),
            "neural_engine": self.engine_var.get(),
            "neural_preset": self.preset_model_var.get(),
            "neural_model": self.model_var.get(),
            "neural_conf": float(self.conf_var.get() or 0.25),
            "neural_overwrite": bool(self.overwrite_var.get()),
            # Image path helpers
            "auto_imgpath": bool(self.auto_imgpath_var.get()),
            "image_filename_col": self.image_filename_col_var.get().strip(),
            "image_base_dir": self.image_base_dir_var.get().strip(),
        }
        try:
            with open(self._config_path(), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def _pkg_version(self, name: str) -> str | None:
        """Return installed package version or None if not available."""
        try:
            from importlib.metadata import version, PackageNotFoundError  # py3.8+
            try:
                return version(name)
            except PackageNotFoundError:
                return None
        except Exception:
            return None

    def _collect_versions(self) -> dict:
        """Collect versions of core libs that may change during installs."""
        return {
            "numpy": self._pkg_version("numpy"),
            "torch": self._pkg_version("torch"),
            "torchvision": self._pkg_version("torchvision"),
            "ultralytics": self._pkg_version("ultralytics"),
            "pillow": self._pkg_version("pillow"),
            "opencv-python": self._pkg_version("opencv-python"),
            "matplotlib": self._pkg_version("matplotlib"),
        }

    def _on_close(self):
        self._save_settings()
        self.destroy()

    def _show_about(self):
        msg = (
            f"{APP_NAME} v{APP_VERSION}\n"
            "Convert raw CSV to ML-ready datasets.\n"
            "© 2025\n\n"
            "See Help for GPU guidance and links."
        )
        messagebox.showinfo("About", msg)

    def _show_help(self):
        win = tk.Toplevel(self)
        win.title("Help")
        win.geometry("780x560")

        container = ttk.Frame(win)
        container.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(container, wrap="word", padx=10, pady=10)
        scroll = ttk.Scrollbar(container, command=text.yview)
        # Default to arrow cursor for general help text
        text.configure(yscrollcommand=scroll.set, cursor="arrow")
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Styling for headings
        text.tag_configure("h1", font=(None, 12, "bold"))
        text.tag_configure("h2", font=(None, 10, "bold"))
        text.tag_configure("mono", font=("Consolas", 9))
        # Show I-beam only over monospace snippets
        text.tag_bind("mono", "<Enter>", lambda _e: text.configure(cursor="xterm"))
        text.tag_bind("mono", "<Leave>", lambda _e: text.configure(cursor="arrow"))

        link_count = {"n": 0}
        def add_link(label: str, url: str):
            tag = f"link{link_count['n']}"
            link_count["n"] += 1
            pos = text.index("end-1c")
            text.insert("end", label + "\n")
            text.tag_add(tag, pos, f"{pos}+{len(label)}c")
            text.tag_config(tag, foreground="#1a73e8", underline=1)
            text.tag_bind(tag, "<Button-1>", lambda _e, u=url: webbrowser.open(u))
            # Hand cursor on hover over links
            text.tag_bind(tag, "<Enter>", lambda _e: text.configure(cursor="hand2"))
            text.tag_bind(tag, "<Leave>", lambda _e: text.configure(cursor="arrow"))

        # Content (exact user text)
        text.insert("end", "Windows – NVIDIA GPU\n", ("h1",))
        text.insert("end", "\n")

        text.insert("end", "PyTorch (CUDA)\n", ("h2",))
        text.insert("end", "As of PyTorch 2.1.0 (latest), binaries support CUDA 11.8 and CUDA 12.1. Done—no need to overthink it.\n\n")
        text.insert("end", "If you're after legacy options, older PyTorch versions on both Linux and Windows supported:\n\n")
        text.insert("end", "CUDA 8.0, 9.0, 10.0\n\n")
        text.insert("end", "Summary: Supported CUDA versions for Windows (NVIDIA + PyTorch):\n\n")
        text.insert("end", "Current: CUDA 11.8, CUDA 12.1\n\n")
        text.insert("end", "Legacy: CUDA 10.0, 9.0, 8.0\n\n")

        text.insert("end", "Links (PyTorch):\n")
        add_link("  • PyTorch Get Started", "https://pytorch.org/get-started/locally/")
        add_link("  • Previous versions", "https://pytorch.org/get-started/previous-versions/")
        text.insert("end", "\n\n")
        text.insert("end", "Install (examples):\n")
        text.insert("end", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n", ("mono",))
        text.insert("end", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n", ("mono",))
        text.insert("end", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # CPU only\n", ("mono",))
        text.insert("end", "\n")

        text.insert("end", "—\n\n")

        text.insert("end", "TensorFlow\n", ("h2",))
        text.insert("end", "- Windows native GPU support ends at TF 2.10.1. Known working combo: TF 2.10.x + CUDA 11.2 + cuDNN 8.1.1.\n")
        text.insert("end", "- For TF > 2.10 on Windows: use WSL2 (Ubuntu) and follow Linux GPU guidance (e.g., CUDA 12.3 + cuDNN 8.9.7).\n")
        text.insert("end", "Links (TensorFlow):\n")
        add_link("  • Install TensorFlow with pip", "https://www.tensorflow.org/install/pip")
        add_link("  • GPU compatibility matrix (CUDA/cuDNN)", "https://www.tensorflow.org/install/source#gpu")
        add_link("  • Linux pip install guide", "https://www.tensorflow.org/install/pip#linux")
        add_link("  • Set up WSL2 on Windows", "https://learn.microsoft.com/windows/wsl/install")
        text.insert("end", "\nInstall (Windows, GPU up to TF 2.10):\n")
        text.insert("end", "pip install \"tensorflow==2.10.*\"\n", ("mono",))
        text.insert("end", "# Then install CUDA 11.2 + cuDNN 8.1.1 from NVIDIA (see links above)\n", ("mono",))
        text.insert("end", "Install (Windows, CPU):\n")
        text.insert("end", "pip install tensorflow\n", ("mono",))

        # Additional platforms
        text.insert("end", "\n\nWindows – AMD/Intel (DirectML)\n", ("h2",))
        text.insert("end", "Use Microsoft's DirectML builds for GPU acceleration on AMD/Intel GPUs on Windows.\n")
        text.insert("end", "CUDA is NVIDIA-only; ROCm is not supported on Windows.\n")
        text.insert("end", "Links:\n")
        add_link("  • PyTorch on DirectML (Windows)", "https://learn.microsoft.com/windows/ai/directml/gpu-pytorch-windows")
        add_link("  • TensorFlow on DirectML (Windows)", "https://learn.microsoft.com/windows/ai/directml/gpu-tensorflow-windows")
        add_link("  • DirectML overview", "https://learn.microsoft.com/windows/ai/directml/dml-intro")
        add_link("  • Hardware requirements (DirectML)", "https://learn.microsoft.com/windows/ai/directml/gpu-pytorch-windows#hardware-requirements")
        text.insert("end", "\nInstall (examples):\n")
        text.insert("end", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n", ("mono",))
        text.insert("end", "pip install torch-directml\n", ("mono",))
        text.insert("end", "pip install tensorflow-directml\n", ("mono",))

        text.insert("end", "\nLinux – NVIDIA\n", ("h2",))
        text.insert("end", "PyTorch: use official wheels for CUDA 11.8 or 12.1.\n")
        text.insert("end", "TensorFlow: follow Linux GPU guide (e.g., CUDA 12.3 + cuDNN 8.9.7).\n")
        text.insert("end", "Links:\n")
        add_link("  • PyTorch Get Started (Linux)", "https://pytorch.org/get-started/locally/")
        add_link("  • TensorFlow Linux pip install", "https://www.tensorflow.org/install/pip#linux")
        add_link("  • NVIDIA CUDA Toolkit downloads", "https://developer.nvidia.com/cuda-downloads")
        add_link("  • NVIDIA cuDNN downloads", "https://developer.nvidia.com/cudnn")
        add_link("  • CUDA driver/toolkit compatibility", "https://docs.nvidia.com/deploy/cuda-compatibility/")
        text.insert("end", "\nInstall (PyTorch CUDA wheels):\n")
        text.insert("end", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n", ("mono",))
        text.insert("end", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n", ("mono",))
        text.insert("end", "Install (TensorFlow):\n")
        text.insert("end", "pip install tensorflow\n", ("mono",))

        text.insert("end", "\nLinux – AMD (ROCm)\n", ("h2",))
        text.insert("end", "Use ROCm-enabled builds. Check GPU support list and driver/ROCm version compatibility.\n")
        text.insert("end", "Links:\n")
        add_link("  • PyTorch ROCm (select ROCm in Get Started)", "https://pytorch.org/get-started/locally/")
        add_link("  • AMD ROCm docs (install on Linux)", "https://rocm.docs.amd.com/")
        add_link("  • TensorFlow on ROCm (AMD guide)", "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/tensorflow-install.html")
        add_link("  • ROCm GPU support matrix", "https://rocm.docs.amd.com/en/latest/release/gpu_support.html")
        text.insert("end", "\nInstall (PyTorch ROCm):\n")
        text.insert("end", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1\n", ("mono",))
        text.insert("end", "Install (TensorFlow ROCm):\n")
        text.insert("end", "pip install tensorflow-rocm\n", ("mono",))

        text.insert("end", "\nWindows + WSL2 (NVIDIA)\n", ("h2",))
        text.insert("end", "For TensorFlow > 2.10 on Windows, use WSL2 for native NVIDIA CUDA support.\n")
        text.insert("end", "Links:\n")
        add_link("  • CUDA on WSL2 (NVIDIA guide)", "https://docs.nvidia.com/cuda/wsl-user-guide/index.html")
        add_link("  • Install WSL", "https://learn.microsoft.com/windows/wsl/install")
        text.insert("end", "\nInstall inside Ubuntu (examples):\n")
        text.insert("end", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n", ("mono",))
        text.insert("end", "pip install tensorflow\n", ("mono",))

        text.configure(state="disabled")

    def _has_nvidia_gpu(self) -> bool:
        """Best-effort check for an NVIDIA GPU using nvidia-smi."""
        try:
            p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
            return p.returncode == 0 and ("GPU" in (p.stdout or ""))
        except Exception:
            return False

    def verify_environment(self):
        """Spawn a subprocess to import core libs and print versions; show results in log."""
        try:
            python_exe = sys.executable
            script = (
                "import importlib, json, sys\n"
                "mods=['python','numpy','pandas','pyarrow','PIL','torch','torchvision','ultralytics','tensorflow','opencv-python','matplotlib','sklearn','xgboost','lightgbm','flaml']\n"
                "vers={}\n"
                "vers['python']=sys.version.split()[0]\n"
                "def ver(name):\n"
                "  try:\n"
                "    if name=='PIL':\n"
                "      import PIL; return getattr(PIL, '__version__', 'unknown')\n"
                "    if name=='opencv-python':\n"
                "      import cv2; return getattr(cv2, '__version__', 'unknown')\n"
                "    if name=='python': return vers['python']\n"
                "    m=importlib.import_module(name)\n"
                "    return getattr(m,'__version__', 'unknown')\n"
                "  except Exception as e:\n"
                "    return None\n"
                "for m in mods:\n"
                "  vers[m]=ver(m)\n"
                "flaml_info={'has_automl': None, 'error': None}\n"
                "try:\n"
                "  from flaml.automl import AutoML\n"
                "  flaml_info['has_automl']=True\n"
                "except Exception as e:\n"
                "  flaml_info['has_automl']=False\n"
                "  try:\n"
                "    flaml_info['error']=str(e)\n"
                "  except Exception:\n"
                "    flaml_info['error']=repr(e)\n"
                "cuda={'available': None, 'version': None, 'device_count': 0, 'devices': [], 'nvidia_smi': None}\n"
                "try:\n"
                "  import torch\n"
                "  cuda['available']=bool(torch.cuda.is_available())\n"
                "  cuda['version']=getattr(torch.version,'cuda', None)\n"
                "  try:\n"
                "    cnt=torch.cuda.device_count()\n"
                "  except Exception:\n"
                "    cnt=0\n"
                "  cuda['device_count']=cnt\n"
                "  try:\n"
                "    cuda['devices']=[torch.cuda.get_device_name(i) for i in range(cnt)]\n"
                "  except Exception:\n"
                "    pass\n"
                "except Exception:\n"
                "  pass\n"
                "try:\n"
                "  import subprocess\n"
                "  smi=subprocess.run(['nvidia-smi','-L'], capture_output=True, text=True)\n"
                "  cuda['nvidia_smi']=(smi.returncode==0)\n"
                "except Exception:\n"
                "  cuda['nvidia_smi']=None\n"
                "print(json.dumps({'versions':vers,'cuda':cuda,'flaml':flaml_info}, indent=2))\n"
            )
            cmd = [python_exe, "-c", script]
            self._start_progress("Verifying environment", style="Green.Horizontal.TProgressbar")
            def run():
                self._run_ps_step(cmd, "Verify Environment")
                self.after(0, self._end_progress)
            import threading
            threading.Thread(target=run, daemon=True).start()
        except Exception as e:
            self.log(f"Verify environment failed: {e}")

    def setup_environment_ui(self):
        """Install core/optional dependencies in current Python. Asks for confirmation."""
        if not messagebox.askyesno(
            "Setup Environment",
            "This will install or update recommended packages in the CURRENT Python environment:\n\n"
            "- Core: pandas, pyarrow, Pillow, openpyxl\n"
            "- ML (optional): torch, torchvision, torchaudio (CUDA if NVIDIA GPU detected, else CPU), ultralytics, scikit-learn\n"
            "- Optional: tensorflow (for TFRecord)\n\nProceed?"
        ):
            return
        python_exe = sys.executable
        steps: list[tuple[list[str], str]] = []
        # Core
        steps.append(([python_exe, "-m", "pip", "install", "--upgrade", "pip"], "Upgrade pip"))
        steps.append(([python_exe, "-m", "pip", "install", "pandas", "pyarrow", "Pillow", "openpyxl"], "Install core packages"))
        # Torch/TorchVision: prefer CUDA wheels if NVIDIA GPU is present
        try:
            prefer_cuda = self._has_nvidia_gpu()
        except Exception:
            prefer_cuda = False
        torch_index = "https://download.pytorch.org/whl/cu124" if prefer_cuda else "https://download.pytorch.org/whl/cpu"
        torch_label = "Install Torch/TorchVision (CUDA cu124)" if prefer_cuda else "Install Torch/TorchVision (CPU)"
        steps.append(([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", torch_index], torch_label))
        # scikit-learn for robust splitting and metrics
        steps.append(([python_exe, "-m", "pip", "install", "scikit-learn"], "Install scikit-learn"))
        # FLAML with AutoML extra
        steps.append(([python_exe, "-m", "pip", "install", "flaml[automl]"], "Install FLAML AutoML"))
        # Ultralytics with only-if-needed and pin current numpy
        cur_numpy = self._pkg_version("numpy")
        ul = [python_exe, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "ultralytics"]
        if cur_numpy:
            ul += [f"numpy=={cur_numpy}"]
        steps.append((ul, "Install Ultralytics"))
        # Optional TensorFlow step (best-effort)
        steps.append(([python_exe, "-m", "pip", "install", "tensorflow"], "Install TensorFlow (optional)"))

        self._start_progress("Setting up environment", style="Green.Horizontal.TProgressbar")
        self.log("Starting environment setup...")
        def worker():
            for cmd, label in steps:
                self._run_ps_step(cmd, label)
            self.after(0, self._end_progress)
            # Show versions after setup
            try:
                self.after(0, self.verify_environment)
            except Exception:
                pass
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def open_ml_training_page(self):
        """Show the ML Training Page as an in-window tab and initialize it."""
        try:
            if hasattr(self, "notebook") and hasattr(self, "ml_tab"):
                try:
                    self.notebook.select(self.ml_tab)
                except Exception:
                    pass
                # Initialize ML tab lazily
                try:
                    init = getattr(self, "_init_ml_tab", None)
                    if callable(init):
                        init()
                except Exception:
                    pass
            else:
                messagebox.showerror("ML Training Page", "Main notebook is not available yet.")
        except Exception as e:
            messagebox.showerror("ML Training Page", f"Failed to open ML Training Page tab: {e}")

    def open_env_checker_page(self):
        """Show the Environment Checker tab and initialize it."""
        try:
            if hasattr(self, "notebook") and hasattr(self, "env_tab"):
                try:
                    self.notebook.select(self.env_tab)
                except Exception:
                    pass
                try:
                    init = getattr(self, "_init_env_tab", None)
                    if callable(init):
                        init()
                except Exception:
                    pass
            else:
                messagebox.showerror("Environment Checker", "Main notebook is not available yet.")
        except Exception as e:
            messagebox.showerror("Environment Checker", f"Failed to open Environment Checker tab: {e}")

    def _env_log(self, msg: str):
        """Append a line to the Environment Checker output box."""
        try:
            txt = getattr(self, "_env_out_text", None)
            if txt is None:
                self.log(msg)
                return
            def _append():
                try:
                    txt.configure(state=tk.NORMAL)
                    txt.insert(tk.END, msg + "\n")
                    txt.see(tk.END)
                    txt.configure(state=tk.DISABLED)
                except Exception:
                    pass
            self.after(0, _append)
        except Exception:
            pass

    def _env_set_summary(self, text: str):
        try:
            box = getattr(self, "_env_summary_text", None)
            if box is None:
                return
            def _set():
                try:
                    box.configure(state=tk.NORMAL)
                    box.delete("1.0", tk.END)
                    box.insert(tk.END, text)
                    box.configure(state=tk.DISABLED)
                except Exception:
                    pass
            self.after(0, _set)
        except Exception:
            pass

    def env_detect_gpu(self):
        """Detect NVIDIA GPU via nvidia-smi and framework visibility (TF/Torch)."""
        def worker():
            status = self._env_status if hasattr(self, "_env_status") else {}
            gpu = {"present": False, "name": None, "count": 0}
            try:
                p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
                if p.returncode == 0 and p.stdout:
                    lines = [ln for ln in (p.stdout or "").strip().splitlines() if "GPU" in ln]
                    gpu["present"] = len(lines) > 0
                    gpu["count"] = len(lines)
                    gpu["name"] = lines[0].split("(")[0].split(":",1)[-1].strip() if lines else None
                    self._env_log(f"nvidia-smi: {gpu['count']} GPU(s) detected: {gpu['name']}")
                else:
                    self._env_log("nvidia-smi not found or no NVIDIA GPU detected.")
            except Exception as e:
                self._env_log(f"nvidia-smi check failed: {e}")

            # TensorFlow visibility
            tf_gpu = None
            try:
                import tensorflow as tf  # type: ignore
                gpus = tf.config.list_physical_devices('GPU')
                tf_gpu = len(gpus) > 0
                self._env_log(f"TensorFlow GPUs: {len(gpus)}")
            except Exception as e:
                self._env_log(f"TensorFlow GPU check skipped: {e}")

            # PyTorch visibility
            torch_gpu = None
            try:
                import torch  # type: ignore
                torch_gpu = bool(torch.cuda.is_available())
                self._env_log(f"PyTorch CUDA available: {torch_gpu}")
            except Exception as e:
                self._env_log(f"PyTorch GPU check skipped: {e}")

            status["gpu"] = {"ok": bool(gpu["present"]), "detail": gpu, "tf_visible": tf_gpu, "torch_visible": torch_gpu}
            self._env_status = status
            self.env_refresh_summary()
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def env_check_frameworks(self):
        """Check frameworks presence, versions, and do small sanity tests (CPU/GPU where lightweight)."""
        def worker():
            status = self._env_status if hasattr(self, "_env_status") else {}

            # TensorFlow
            try:
                import tensorflow as tf  # type: ignore
                ver = getattr(tf, "__version__", "?")
                try:
                    _ = (tf.constant(1) + tf.constant(2)).numpy()
                    simple_ok = True
                except Exception:
                    simple_ok = False
                gpus = []
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                except Exception:
                    pass
                tf_ok = simple_ok
                self._env_log(f"TensorFlow {ver}: basic ops={'ok' if simple_ok else 'fail'}, GPUs={len(gpus)}")
                status["tensorflow"] = {"ok": tf_ok, "version": ver, "gpu_ready": len(gpus) > 0}
            except Exception as e:
                self._env_log(f"TensorFlow import failed: {e}")
                status["tensorflow"] = {"ok": False, "error": str(e)}

            # scikit-learn
            try:
                import sklearn  # type: ignore
                from sklearn.linear_model import LogisticRegression  # type: ignore
                ver = getattr(sklearn, "__version__", "?")
                import numpy as _np
                X = _np.random.randn(50, 4)
                y = (_np.sum(X[:, :2], axis=1) > 0).astype(int)
                LogisticRegression(max_iter=200).fit(X, y)
                self._env_log(f"scikit-learn {ver}: LogisticRegression fit ok")
                status["sklearn"] = {"ok": True, "version": ver}
            except Exception as e:
                self._env_log(f"scikit-learn check failed: {e}")
                status["sklearn"] = {"ok": False, "error": str(e)}

            # XGBoost
            try:
                import xgboost as xgb  # type: ignore
                ver = getattr(xgb, "__version__", "?")
                import numpy as _np
                X = _np.random.randn(64, 4)
                y = (_np.sum(X[:, :2], axis=1) > 0).astype(int)
                d = xgb.DMatrix(X, label=y)
                gpu_ready = False
                try:
                    xgb.train({"tree_method": "gpu_hist", "max_depth": 2, "verbosity": 0, "nthread": 1}, d, num_boost_round=1)
                    gpu_ready = True
                    self._env_log(f"XGBoost {ver}: gpu_hist ok")
                except Exception as ge:
                    self._env_log(f"XGBoost {ver}: gpu_hist not usable ({ge})")
                    # CPU sanity
                    try:
                        xgb.train({"tree_method": "hist", "max_depth": 2, "verbosity": 0, "nthread": 1}, d, num_boost_round=1)
                        self._env_log(f"XGBoost {ver}: CPU hist ok")
                    except Exception as ce:
                        self._env_log(f"XGBoost CPU hist failed: {ce}")
                status["xgboost"] = {"ok": True, "version": ver, "gpu_ready": gpu_ready}
            except Exception as e:
                self._env_log(f"XGBoost check failed: {e}")
                status["xgboost"] = {"ok": False, "error": str(e)}

            # LightGBM
            try:
                import lightgbm as lgb  # type: ignore
                ver = getattr(lgb, "__version__", "?")
                import numpy as _np
                X = _np.random.randn(64, 4)
                y = (_np.sum(X[:, :2], axis=1) > 0).astype(int)
                train = lgb.Dataset(X, label=y, free_raw_data=True)
                gpu_ready = False
                try:
                    params = {"objective": "binary", "num_leaves": 15, "min_data_in_leaf": 5, "verbose": -1, "device": "gpu"}
                    lgb.train(params, train, num_boost_round=5)
                    gpu_ready = True
                    self._env_log(f"LightGBM {ver}: GPU training ok")
                except Exception as ge:
                    self._env_log(f"LightGBM {ver}: GPU not usable ({ge})")
                    try:
                        params = {"objective": "binary", "num_leaves": 15, "min_data_in_leaf": 5, "verbose": -1}
                        lgb.train(params, train, num_boost_round=5)
                        self._env_log(f"LightGBM {ver}: CPU training ok")
                    except Exception as ce:
                        self._env_log(f"LightGBM CPU training failed: {ce}")
                status["lightgbm"] = {"ok": True, "version": ver, "gpu_ready": gpu_ready}
            except Exception as e:
                self._env_log(f"LightGBM check failed: {e}")
                status["lightgbm"] = {"ok": False, "error": str(e)}

            # FLAML
            try:
                from flaml.automl import AutoML  # type: ignore
                import flaml  # type: ignore
                ver = getattr(flaml, "__version__", "?")
                _ = AutoML  # symbol access
                self._env_log(f"FLAML {ver}: import ok")
                status["flaml"] = {"ok": True, "version": ver}
            except Exception as e:
                self._env_log(f"FLAML import failed: {e}")
                status["flaml"] = {"ok": False, "error": str(e)}

            self._env_status = status
            self.env_refresh_summary()
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def env_install_upgrade_frameworks(self):
        """Install/upgrade key ML frameworks in the current environment."""
        if not messagebox.askyesno(
            "Install/Upgrade",
            "Install/upgrade key ML frameworks in the CURRENT Python environment?\n\nPackages: scikit-learn, xgboost, lightgbm, tensorflow, flaml[automl]"
        ):
            return
        python_exe = sys.executable
        steps: list[tuple[list[str], str]] = []
        steps.append(([python_exe, "-m", "pip", "install", "-U", "scikit-learn"], "Install/Upgrade scikit-learn"))
        steps.append(([python_exe, "-m", "pip", "install", "-U", "xgboost"], "Install/Upgrade XGBoost"))
        steps.append(([python_exe, "-m", "pip", "install", "-U", "lightgbm"], "Install/Upgrade LightGBM"))
        steps.append(([python_exe, "-m", "pip", "install", "-U", "tensorflow"], "Install/Upgrade TensorFlow"))
        steps.append(([python_exe, "-m", "pip", "install", "-U", "flaml[automl]"], "Install/Upgrade FLAML"))

        self._start_progress("Installing frameworks", style="Green.Horizontal.TProgressbar")
        self._env_log("Starting framework installations...")
        def worker():
            for cmd, label in steps:
                self._run_ps_step(cmd, label)
            try:
                self.after(0, self._end_progress)
            except Exception:
                pass
            self._env_log("Install/upgrade completed. Re-run checks.")
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def env_install_frameworks(self):
        """Install exact pinned versions from requirements2.txt excluding TensorFlow and PyTorch.
        Shows progress and environment logs, and prompts users to install TF/PT manually after.
        """
        try:
            if not messagebox.askyesno(
                "Install Frameworks",
                "Install exact pinned versions required for running the app (excluding TensorFlow and PyTorch) in the CURRENT Python environment?"
            ):
                return
        except Exception:
            # In headless contexts, proceed without prompt
            pass

        python_exe = sys.executable
        req_path = os.path.join(os.path.dirname(__file__), "requirements2.txt")
        if not os.path.isfile(req_path):
            try:
                messagebox.showerror("Install Frameworks", f"requirements2.txt not found at: {req_path}")
            except Exception:
                self._env_log(f"requirements2.txt not found at: {req_path}")
            return

        # Read and filter requirements, excluding TF/PT families
        exclude = {"tensorflow", "tensorflow-rocm", "tensorflow-directml", "torch", "torchvision", "torchaudio"}
        def base_name(line: str) -> str:
            # Extract package name before any version specifiers or extras
            s = line.strip()
            if not s or s.startswith("#"):
                return ""
            # Stop at first comparator or whitespace
            m = re.match(r"^([A-Za-z0-9_.\-]+)", s)
            return (m.group(1) if m else s).strip()

        try:
            with open(req_path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip() for ln in f.readlines()]
            filtered = []
            for ln in lines:
                name = base_name(ln).lower()
                if not name:
                    continue
                # exclude families
                if name in exclude or name.startswith("tensorflow") or name.startswith("torch"):
                    continue
                filtered.append(ln)
        except Exception as e:
            try:
                messagebox.showerror("Install Frameworks", f"Failed to read requirements2.txt: {e}")
            except Exception:
                pass
            self._env_log(f"Failed to read requirements2.txt: {e}")
            return

        if not filtered:
            self._env_log("No packages to install after excluding TensorFlow/PyTorch.")
            return

        # Write to a temporary requirements file so pip installs exact pins
        tmp_req = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tf:
                tf.write("\n".join(filtered) + "\n")
                tmp_req = tf.name
        except Exception as e:
            self._env_log(f"Failed to create temp requirements file: {e}")
            return

        self._start_progress("Installing pinned frameworks", style="Green.Horizontal.TProgressbar")
        self._env_log("Installing pinned packages (excluding TensorFlow/PyTorch)...")

        def worker():
            try:
                cmd = [
                    python_exe, "-m", "pip", "install",
                    "--no-input", "--disable-pip-version-check", "--no-color",
                    "-r", tmp_req,
                ]
                self._run_ps_step(cmd, "Install pinned frameworks (excluding TF/PT)")
            finally:
                try:
                    if tmp_req and os.path.isfile(tmp_req):
                        os.unlink(tmp_req)
                except Exception:
                    pass
                try:
                    self.after(0, self._end_progress)
                except Exception:
                    pass
                self._env_log("Install completed. TensorFlow and PyTorch are excluded; install them manually via Help after this step.")
                try:
                    self.after(0, lambda: messagebox.showinfo(
                        "Manual Step Required",
                        "TensorFlow and PyTorch are excluded from this step. Go to Help for instructions to install them manually."
                    ))
                except Exception:
                    pass
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def env_upgrade_frameworks(self):
        """Upgrade frameworks on Linux, excluding TensorFlow and PyTorch families.
        Builds the package list from requirements2.txt by stripping version pins.
        """
        if not sys.platform.startswith("linux"):
            try:
                messagebox.showinfo("Upgrade Frameworks", "This upgrade action is available on Linux only.")
            except Exception:
                pass
            return
        try:
            if not messagebox.askyesno(
                "Upgrade Frameworks (Linux)",
                "Upgrade to latest versions on Linux for frameworks listed in requirements2.txt (excluding TensorFlow and PyTorch)?"
            ):
                return
        except Exception:
            pass

        req_path = os.path.join(os.path.dirname(__file__), "requirements2.txt")
        if not os.path.isfile(req_path):
            try:
                messagebox.showerror("Upgrade Frameworks", f"requirements2.txt not found at: {req_path}")
            except Exception:
                self._env_log(f"requirements2.txt not found at: {req_path}")
            return

        exclude = {"tensorflow", "tensorflow-rocm", "tensorflow-directml", "torch", "torchvision", "torchaudio"}
        def base_name(line: str) -> str:
            s = line.strip()
            if not s or s.startswith("#"):
                return ""
            m = re.match(r"^([A-Za-z0-9_.\-]+)", s)
            return (m.group(1) if m else s).strip()

        names: list[str] = []
        seen = set()
        try:
            with open(req_path, "r", encoding="utf-8") as f:
                for ln in f:
                    s = (ln or "").strip()
                    if not s or s.startswith("#"):
                        continue
                    # Skip options and non-package requirements
                    if s.startswith("-") or s.startswith("git+") or s.startswith("http://") or s.startswith("https://"):
                        continue
                    name = base_name(s).lower()
                    if not name or name in exclude or name.startswith("tensorflow") or name.startswith("torch"):
                        continue
                    if name not in seen:
                        names.append(name)
                        seen.add(name)
        except Exception as e:
            self._env_log(f"Failed to read requirements2.txt: {e}")
            return

        if not names:
            self._env_log("No packages to upgrade after excluding TensorFlow/PyTorch.")
            return

        # Chunk installs to keep command length reasonable
        def chunks(lst: list[str], n: int):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        python_exe = sys.executable
        pkgs = names
        batches = list(chunks(pkgs, 20))
        self._start_progress("Upgrading frameworks (Linux)", maximum=len(batches), style="Green.Horizontal.TProgressbar")
        self._env_log(f"Upgrading {len(pkgs)} packages in {len(batches)} batch(es) (excluding TensorFlow/PyTorch)...")

        def worker():
            done = 0
            for i, batch in enumerate(batches):
                label = f"Upgrade batch {i+1}/{len(batches)}"
                cmd = [python_exe, "-m", "pip", "install", "-U", "--no-input", "--disable-pip-version-check", "--no-color"] + batch
                self._run_ps_step(cmd, label)
                done += 1
                try:
                    self.after(0, lambda d=done: self._update_progress(d, f"{d}/{len(batches)}"))
                except Exception:
                    pass
            try:
                self.after(0, self._end_progress)
            except Exception:
                pass
            self._env_log("Upgrade completed. TensorFlow and PyTorch are excluded; install/upgrade them manually as needed.")
            try:
                self.after(0, lambda: messagebox.showinfo(
                    "Manual Step Recommended",
                    "TensorFlow and PyTorch were excluded from upgrade. Manage them manually per your GPU/OS guidance in Help."
                ))
            except Exception:
                pass

        import threading
        threading.Thread(target=worker, daemon=True).start()

    def env_calibrate_gpu(self):
        """Interactive GPU calibration: prompt for OS and GPU vendor on the main thread,
        then run installation steps asynchronously with progress and logging.

        Flows:
        - Windows + NVIDIA: PyTorch CUDA 12.4 wheels; TensorFlow 2.10.1 (CPU pip)
        - Windows + AMD/Intel: PyTorch CPU + torch-directml + tensorflow-directml
        - Linux + NVIDIA: prompt CUDA series (cu124/cu121/cu118) then install matching PyTorch; TensorFlow (pip)
        - Linux + AMD: prompt ROCm series (e.g., rocm6.1), then PyTorch ROCm + tensorflow-rocm
        """
        try:
            python_exe = sys.executable
            # Prompts MUST be on the Tk main thread
            default_os = "Windows" if sys.platform == "win32" else "Linux"
            os_choice = self._ask_radio_choice(
                title="Calibrate GPU",
                message="Select Operating System:",
                options=["Windows", "Linux"],
                default_index=(0 if default_os == "Windows" else 1),
            )
            if not os_choice:
                self._env_log("Calibration cancelled: OS not provided.")
                return
            os_key = os_choice.strip().lower()
            if os_key not in ("windows", "linux"):
                self._env_log(f"Invalid OS '{os_choice}'. Expected 'Windows' or 'Linux'.")
                return

            gpu_choice = self._ask_radio_choice(
                title="Calibrate GPU",
                message="Select GPU Vendor:",
                options=["NVIDIA", "AMD/Intel"],
                default_index=0,
            )
            if not gpu_choice:
                self._env_log("Calibration cancelled: GPU vendor not provided.")
                return
            gpu_key = gpu_choice.strip().lower()

            steps: list[tuple[list[str], str]] = []
            if os_key == "windows":
                if "nvidia" in gpu_key:
                    # Let user choose CUDA series on Windows as well
                    cuda_options = ["cu124", "cu121", "cu118"]
                    cu_choice = self._ask_radio_choice(
                        title="Calibrate GPU",
                        message="Select CUDA series for PyTorch:",
                        options=cuda_options,
                        default_index=0,
                    )
                    if not cu_choice:
                        self._env_log("Calibration cancelled: CUDA series not selected.")
                        return
                    cu_key = cu_choice.strip().lower()
                    self._env_log(f"Selected: Windows + NVIDIA (PyTorch {cu_key}, TensorFlow CPU 2.10.1).")
                    torch_index = f"https://download.pytorch.org/whl/{cu_key}"
                    steps.append(([python_exe, "-m", "pip", "install", "-U", "pip"], "Upgrade pip"))
                    steps.append(([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", torch_index], f"Install PyTorch ({cu_key})"))
                    steps.append(([python_exe, "-m", "pip", "install", "tensorflow==2.10.1"], "Install TensorFlow 2.10.1 (CPU on Windows)"))
                else:
                    self._env_log("Selected: Windows + AMD/Intel (DirectML).")
                    steps.append(([python_exe, "-m", "pip", "install", "-U", "pip"], "Upgrade pip"))
                    steps.append(([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"], "Install PyTorch (CPU)"))
                    steps.append(([python_exe, "-m", "pip", "install", "torch-directml"], "Install torch-directml"))
                    steps.append(([python_exe, "-m", "pip", "install", "tensorflow-directml"], "Install tensorflow-directml"))
            else:  # Linux
                if "nvidia" in gpu_key:
                    cuda_options = ["cu124", "cu121", "cu118"]
                    cu_choice = self._ask_radio_choice(
                        title="Calibrate GPU",
                        message="Select CUDA series for PyTorch:",
                        options=cuda_options,
                        default_index=2,
                    )
                    if not cu_choice:
                        self._env_log("Calibration cancelled: CUDA series not selected.")
                        return
                    cu_key = cu_choice.strip().lower()
                    steps.append(([python_exe, "-m", "pip", "install", "-U", "pip"], "Upgrade pip"))
                    steps.append(([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/{cu_key}"], f"Install PyTorch ({cu_key})"))
                    steps.append(([python_exe, "-m", "pip", "install", "tensorflow"], "Install TensorFlow"))
                else:
                    rocm_options = ["rocm6.1", "rocm6.0", "rocm5.7"]
                    rocm_choice = self._ask_radio_choice(
                        title="Calibrate GPU",
                        message="Select ROCm series for PyTorch:",
                        options=rocm_options,
                        default_index=0,
                    )
                    if not rocm_choice:
                        self._env_log("Calibration cancelled: ROCm series not selected.")
                        return
                    rocm_key = rocm_choice.strip().lower()
                    steps.append(([python_exe, "-m", "pip", "install", "-U", "pip"], "Upgrade pip"))
                    steps.append(([python_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/{rocm_key}"], f"Install PyTorch ({rocm_key})"))
                    steps.append(([python_exe, "-m", "pip", "install", "tensorflow-rocm"], "Install tensorflow-rocm"))

            if not steps:
                self._env_log("No steps generated; nothing to do.")
                return

            # Start progress UI before launching worker
            self._start_progress("Calibrating GPU", style="Green.Horizontal.TProgressbar")

            def worker():
                try:
                    self._env_log("Starting GPU calibration...")
                    for cmd, label in steps:
                        self._run_ps_step(cmd, label)
                    self._env_log("GPU calibration completed. Running environment checks...")
                    try:
                        self.env_detect_gpu()
                        self.env_check_frameworks()
                        self.env_check_cuda_cudnn()
                    except Exception:
                        pass
                except Exception as e:
                    self._env_log(f"Calibration error: {e}")
                finally:
                    try:
                        self.after(0, self._end_progress)
                    except Exception:
                        pass

            import threading
            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            self._env_log(f"Calibration error: {e}")

    def env_check_cuda_cudnn(self):
        """Best-effort CUDA/cuDNN info and TensorFlow build metadata."""
        def parse_nvcc_version(text: str) -> str | None:
            for ln in (text or "").splitlines():
                if "release" in ln and "Cuda compilation tools" in ln:
                    # e.g., Cuda compilation tools, release 11.8, V11.8.89
                    parts = ln.split("release")
                    if len(parts) > 1:
                        return parts[1].split(",")[0].strip()
            return None

        def worker():
            status = self._env_status if hasattr(self, "_env_status") else {}
            info = {"nvidia_smi_cuda": None, "nvcc_cuda": None, "tf_cuda": None, "tf_cudnn": None, "notes": []}
            # nvidia-smi reported CUDA
            try:
                p = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                if p.returncode == 0 and p.stdout:
                    for ln in p.stdout.splitlines():
                        if "CUDA Version:" in ln:
                            try:
                                info["nvidia_smi_cuda"] = ln.split("CUDA Version:")[-1].strip().split(" ")[0]
                            except Exception:
                                pass
                            break
            except Exception as e:
                self._env_log(f"nvidia-smi CUDA parse failed: {e}")

            # nvcc --version
            try:
                p = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
                if p.returncode == 0 and p.stdout:
                    info["nvcc_cuda"] = parse_nvcc_version(p.stdout)
            except Exception as e:
                self._env_log(f"nvcc check failed: {e}")

            # TensorFlow build info
            try:
                import tensorflow as tf  # type: ignore
                bi = {}
                try:
                    bi = tf.sysconfig.get_build_info() or {}
                except Exception:
                    pass
                cuda_v = bi.get("cuda_version") if isinstance(bi, dict) else None
                cudnn_v = bi.get("cudnn_version") if isinstance(bi, dict) else None
                info["tf_cuda"] = cuda_v
                info["tf_cudnn"] = cudnn_v
                self._env_log(f"TensorFlow build: CUDA={cuda_v}, cuDNN={cudnn_v}")
                # Windows note for TF>=2.11
                try:
                    from packaging.version import Version
                    tfv = Version(getattr(tf, "__version__", "0"))
                    if sys.platform == "win32" and tfv >= Version("2.11"):
                        info["notes"].append("On Windows, official TF>=2.11 pip builds do not include CUDA GPU support.")
                except Exception:
                    pass
            except Exception as e:
                self._env_log(f"TensorFlow build info unavailable: {e}")

            status["cuda_cudnn"] = {"ok": True, "detail": info}
            self._env_status = status
            self.env_refresh_summary()
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def env_refresh_summary(self):
        """Summarize statuses into the summary panel."""
        status = getattr(self, "_env_status", {}) or {}
        lines: list[str] = []
        def flag(ok: bool | None) -> str:
            if ok is True:
                return "✅"
            if ok is False:
                return "❌"
            return "⚠️"
        # GPU
        gpu = status.get("gpu")
        if gpu:
            present = bool(gpu.get("ok"))
            name = (gpu.get("detail") or {}).get("name")
            lines.append(f"{flag(present)} GPU: {'present' if present else 'not detected'}{(' - ' + str(name)) if name else ''}")
        # TensorFlow
        tf = status.get("tensorflow")
        if tf:
            lines.append(f"{flag(tf.get('ok'))} TensorFlow: v{tf.get('version','?')} (GPU ready: {bool(tf.get('gpu_ready'))})")
        # scikit-learn
        sk = status.get("sklearn")
        if sk:
            lines.append(f"{flag(sk.get('ok'))} scikit-learn: v{sk.get('version','?')}")
        # XGBoost
        xgb = status.get("xgboost")
        if xgb:
            lines.append(f"{flag(xgb.get('ok'))} XGBoost: v{xgb.get('version','?')} (GPU ready: {bool(xgb.get('gpu_ready'))})")
        # LightGBM
        lgb = status.get("lightgbm")
        if lgb:
            lines.append(f"{flag(lgb.get('ok'))} LightGBM: v{lgb.get('version','?')} (GPU ready: {bool(lgb.get('gpu_ready'))})")
        # FLAML
        fl = status.get("flaml")
        if fl:
            lines.append(f"{flag(fl.get('ok'))} FLAML: v{fl.get('version','?')}")
        # CUDA/cuDNN
        cc = status.get("cuda_cudnn")
        if cc:
            det = cc.get("detail") or {}
            lines.append(
                f"{flag(True)} CUDA/cuDNN: nvidia-smi CUDA={det.get('nvidia_smi_cuda')}, nvcc CUDA={det.get('nvcc_cuda')}, TF CUDA={det.get('tf_cuda')}, TF cuDNN={det.get('tf_cudnn')}"
            )
            notes = det.get("notes") or []
            for n in notes:
                lines.append(f"ℹ️ {n}")
        text = "\n".join(lines) if lines else "Run checks to populate summary."
        self._env_set_summary(text)

    # -------- Progress dialog helpers --------
    def _start_progress(self, title: str, maximum: int | None = None, style: str | None = None):
        try:
            if hasattr(self, "_prog_win") and self._prog_win is not None:
                try:
                    self._prog_win.destroy()
                except Exception:
                    pass
            win = tk.Toplevel(self)
            win.title(title)
            win.geometry("380x100")
            win.transient(self)
            win.grab_set()
            lbl = ttk.Label(win, text=title)
            lbl.pack(padx=12, pady=(12, 6), anchor=tk.W)
            # Optional green style
            try:
                style_obj = ttk.Style()
                style_name = style or "Horizontal.TProgressbar"
                if style == "Green.Horizontal.TProgressbar":
                    style_obj.configure(style, troughcolor="#f0f0f0", background="#2ecc71")
            except Exception:
                style_name = "Horizontal.TProgressbar"
            bar = ttk.Progressbar(win, orient="horizontal", mode=("determinate" if maximum else "indeterminate"), style=style_name)
            bar.pack(fill=tk.X, padx=12, pady=6)
            if maximum:
                bar.configure(maximum=maximum, value=0)
            else:
                bar.start(10)
            self._prog_win = win
            self._prog_bar = bar
            self._prog_lbl = lbl
            self.update_idletasks()
        except Exception:
            # Fail silently if UI cannot be created
            self._prog_win = None
            self._prog_bar = None
            self._prog_lbl = None

    def _update_progress(self, value: int | None = None, text: str | None = None):
        try:
            if getattr(self, "_prog_bar", None) is not None and value is not None:
                self._prog_bar.configure(value=value)
            if getattr(self, "_prog_lbl", None) is not None and text:
                self._prog_lbl.configure(text=text)
            self.update_idletasks()
        except Exception:
            pass

    def _progress_set_indeterminate(self, on: bool, text: str | None = None):
        """Temporarily toggle the progress bar to indeterminate (marquee) mode.

        Useful while a single long-running step is executing so users can see activity.
        When turned off, restores determinate mode using the existing value/maximum.
        """
        try:
            bar = getattr(self, "_prog_bar", None)
            lbl = getattr(self, "_prog_lbl", None)
            if bar is None:
                return
            if on:
                # Remember that we're in a temporary indeterminate state
                # We do not alter the current determinate value; just switch modes.
                bar.configure(mode="indeterminate")
                if text and lbl is not None:
                    lbl.configure(text=text)
                bar.start(10)
            else:
                # Switch back to determinate; keep prior value/maximum as already set
                try:
                    bar.stop()
                except Exception:
                    pass
                bar.configure(mode="determinate")
            self.update_idletasks()
        except Exception:
            pass

    def _end_progress(self):
        try:
            if getattr(self, "_prog_win", None) is not None:
                self._prog_win.destroy()
        except Exception:
            pass
        finally:
            self._prog_win = None
            self._prog_bar = None
            self._prog_lbl = None

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing-value handling, scaling, and encoding to a DataFrame.

        Controlled by Tk variables:
        - do_preprocess_var
        - missing_strategy_var
        - scaling_var
        - encoding_var
        - preproc_exclude_cols_var
        Uses scikit-learn if available; falls back to pandas/numpy implementations.
        """
        try:
            use_sklearn = True
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder  # type: ignore
        except Exception:
            use_sklearn = False
            StandardScaler = MinMaxScaler = OrdinalEncoder = None  # type: ignore

        df_proc = df.copy()
        # Determine excluded columns
        excl_text = ""
        try:
            excl_text = (self.preproc_exclude_cols_var.get() or "").strip()
        except Exception:
            pass
        excluded = {c.strip() for c in excl_text.split(',') if c.strip()}
        # Identify column types
        numeric_cols_auto = [c for c in df_proc.select_dtypes(include=[np.number]).columns if c not in excluded]
        cat_cols_auto = [c for c in df_proc.columns if c not in excluded and c not in numeric_cols_auto]
        # Optional include lists
        def _parse_list(val: str) -> list[str]:
            return [c.strip() for c in (val or "").split(',') if c.strip()]
        include_num = _parse_list(getattr(self.preproc_include_num_cols_var, 'get', lambda: "")())
        include_cat = _parse_list(getattr(self.preproc_include_cat_cols_var, 'get', lambda: "")())
        numeric_cols = [c for c in (include_num or numeric_cols_auto) if c in df_proc.columns and c not in excluded and pd.api.types.is_numeric_dtype(df_proc[c])]
        cat_cols = [c for c in (include_cat or cat_cols_auto) if c in df_proc.columns and c not in excluded and not pd.api.types.is_numeric_dtype(df_proc[c])]

        # 1) Missing values
        strat = (self.missing_strategy_var.get() if hasattr(self, 'missing_strategy_var') else 'None')
        if strat and strat.lower() != "none":
            self.log(f"Preprocess: missing strategy = {strat}")
            if strat.lower() == "drop rows":
                before = len(df_proc)
                df_proc = df_proc.dropna().reset_index(drop=True)
                self.log(f" - dropped {before - len(df_proc)} rows with any NaN")
            elif strat.lower() == "fill numeric mean":
                for c in numeric_cols:
                    try:
                        m = df_proc[c].astype(float).mean()
                        df_proc[c] = df_proc[c].fillna(m)
                    except Exception:
                        pass
            elif strat.lower() == "fill numeric median":
                for c in numeric_cols:
                    try:
                        m = df_proc[c].astype(float).median()
                        df_proc[c] = df_proc[c].fillna(m)
                    except Exception:
                        pass
            elif strat.lower() == "fill num mean + cat mode":
                for c in numeric_cols:
                    try:
                        m = df_proc[c].astype(float).mean()
                        df_proc[c] = df_proc[c].fillna(m)
                    except Exception:
                        pass
                for c in cat_cols:
                    try:
                        mode_val = df_proc[c].mode(dropna=True)
                        if not mode_val.empty:
                            df_proc[c] = df_proc[c].fillna(mode_val.iloc[0])
                    except Exception:
                        pass
            elif strat.lower() == "fill num median + cat mode":
                for c in numeric_cols:
                    try:
                        m = df_proc[c].astype(float).median()
                        df_proc[c] = df_proc[c].fillna(m)
                    except Exception:
                        pass
                for c in cat_cols:
                    try:
                        mode_val = df_proc[c].mode(dropna=True)
                        if not mode_val.empty:
                            df_proc[c] = df_proc[c].fillna(mode_val.iloc[0])
                    except Exception:
                        pass
            elif strat.lower() == "fill with 0/''":
                for c in numeric_cols:
                    df_proc[c] = df_proc[c].fillna(0)
                for c in cat_cols:
                    df_proc[c] = df_proc[c].fillna("")
            elif strat.lower() == "ffill":
                df_proc = df_proc.ffill()
            elif strat.lower() == "bfill":
                df_proc = df_proc.bfill()

        # 2) Encoding
        enc = (self.encoding_var.get() if hasattr(self, 'encoding_var') else 'None')
        if enc and enc.lower() != "none" and cat_cols:
            self.log(f"Preprocess: encoding = {enc} on {len(cat_cols)} column(s)")
            if enc.lower().startswith("one-hot"):
                try:
                    dummy_na = bool(getattr(self.onehot_include_nan_var, 'get', lambda: False)())
                    df_proc = pd.get_dummies(df_proc, columns=cat_cols, dummy_na=dummy_na)
                except Exception:
                    # Best effort fallback: operate per column
                    for c in cat_cols:
                        try:
                            dummy_na = bool(getattr(self.onehot_include_nan_var, 'get', lambda: False)())
                            d = pd.get_dummies(df_proc[c], prefix=c, dummy_na=dummy_na)
                            df_proc = pd.concat([df_proc.drop(columns=[c]), d], axis=1)
                        except Exception:
                            pass
                # After get_dummies, categorical columns updated; reset lists
                numeric_cols = [c for c in df_proc.select_dtypes(include=[np.number]).columns if c not in excluded]
            else:  # Label/Ordinal
                if use_sklearn and OrdinalEncoder is not None:
                    try:
                        enc_obj = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        df_proc[cat_cols] = enc_obj.fit_transform(df_proc[cat_cols].astype(str))
                    except Exception:
                        # fallback per-column
                        for c in cat_cols:
                            codes, _ = pd.factorize(df_proc[c].astype(str), sort=True)
                            df_proc[c] = codes
                else:
                    for c in cat_cols:
                        try:
                            codes, _ = pd.factorize(df_proc[c].astype(str), sort=True)
                            df_proc[c] = codes
                        except Exception:
                            pass
                # Encoded categories are now numeric
                numeric_cols = [c for c in df_proc.select_dtypes(include=[np.number]).columns if c not in excluded]

        # 3) Scaling
        scale = (self.scaling_var.get() if hasattr(self, 'scaling_var') else 'None')
        if scale and scale.lower() != "none" and numeric_cols:
            self.log(f"Preprocess: scaling = {scale} on {len(numeric_cols)} numeric column(s)")
            X = df_proc[numeric_cols].astype(float)
            if scale.lower().startswith("standardize"):
                if use_sklearn and StandardScaler is not None:
                    try:
                        scaler = StandardScaler()
                        df_proc[numeric_cols] = scaler.fit_transform(X)
                    except Exception:
                        # manual
                        mu = X.mean(axis=0)
                        sd = X.std(axis=0).replace(0, 1.0)
                        df_proc[numeric_cols] = (X - mu) / sd
                else:
                    mu = X.mean(axis=0)
                    sd = X.std(axis=0).replace(0, 1.0)
                    df_proc[numeric_cols] = (X - mu) / sd
            else:  # Min-Max [0,1]
                if use_sklearn and MinMaxScaler is not None:
                    try:
                        scaler = MinMaxScaler()
                        df_proc[numeric_cols] = scaler.fit_transform(X)
                    except Exception:
                        mn = X.min(axis=0)
                        mx = X.max(axis=0)
                        rng = (mx - mn).replace(0, 1.0)
                        df_proc[numeric_cols] = (X - mn) / rng
                else:
                    mn = X.min(axis=0)
                    mx = X.max(axis=0)
                    rng = (mx - mn).replace(0, 1.0)
                    df_proc[numeric_cols] = (X - mn) / rng

        # Optional: drop rows with non-finite after scaling/encoding
        if bool(getattr(self.drop_nonfinite_after_scale_var, 'get', lambda: False)()):
            before = len(df_proc)
            df_proc = df_proc.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            self.log(f"Preprocess: dropped {before - len(df_proc)} rows with non-finite/NaN after transforms")

        return df_proc

    # Core processing
    def process(self):
        try:
            input_path = self.input_path_var.get().strip()
            output_dir = self.output_dir_var.get().strip()
            base_name = self.base_name_var.get().strip() or "dataset"
            out_format = self.format_var.get()

            if not input_path or not os.path.isfile(input_path):
                messagebox.showerror("Input", "Please provide a valid path to a CSV file.")
                return
            if not input_path.lower().endswith(".csv"):
                if not messagebox.askyesno("Confirm", "Selected file does not end with .csv. Continue?"):
                    return

            if not output_dir:
                messagebox.showerror("Output", "Please select an output folder.")
                return
            os.makedirs(output_dir, exist_ok=True)

            self.log(f"Reading CSV: {input_path}")
            try:
                df = pd.read_csv(input_path)
            except Exception as e:
                messagebox.showerror("Read CSV", f"Failed to read CSV: {e}")
                return

            # Column filtering
            cols_text = self.cols_var.get().strip()
            if cols_text:
                keep_cols = [c.strip() for c in cols_text.split(',') if c.strip()]
                missing = [c for c in keep_cols if c not in df.columns]
                if missing:
                    messagebox.showerror("Columns", f"These columns are not in the CSV: {missing}\nAvailable: {list(df.columns)}")
                    return
                df = df[keep_cols]
                self.log(f"Keeping columns: {keep_cols}")

            # Preprocessing (optional)
            try:
                if bool(self.do_preprocess_var.get()):
                    self.log("Applying preprocessing...")
                    df = self._preprocess_dataframe(df)
                    self.log("Preprocessing complete.")
            except Exception as e:
                messagebox.showerror("Preprocess", f"Preprocessing failed: {e}")
                return

            # If user selected a detection export format but we're missing required columns and not in Neural mode,
            # offer to switch to Neural to auto-label.
            if self._is_detection_format(out_format):
                required = ["image_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
                missing_det = [c for c in required if c not in df.columns]
                if missing_det and self.mode_var.get().lower() != "neural":
                    self.log(f"Detection format selected but missing columns: {missing_det}")
                    if messagebox.askyesno(
                        "Detection format requires labels",
                        "This format requires detection columns (image_path, bbox_x, bbox_y, bbox_w, bbox_h).\n\n"
                        "Switch to Neural mode to auto-label now?"
                    ):
                        self.mode_var.set("Neural")
                        # Gentle reminder about image paths
                        if "image_path" not in df.columns and not bool(self.auto_imgpath_var.get()):
                            messagebox.showinfo(
                                "Neural setup",
                                "Since 'image_path' is missing, enable 'Auto-create image_path' and provide the filename column and image base folder."
                            )
                    else:
                        messagebox.showinfo(
                            "Choose another format",
                            "Please select a tabular export (CSV/JSONL/Parquet/Feather/Excel/SQLite) or enable Neural mode."
                        )
                        return

            # Neural enrichment (optional)
            if self.mode_var.get().lower() == "neural":
                # Build image_path if missing and user requested auto-creation
                if "image_path" not in df.columns and bool(self.auto_imgpath_var.get()):
                    fn_col = self.image_filename_col_var.get().strip()
                    base_dir = self.image_base_dir_var.get().strip()
                    if not fn_col or fn_col not in df.columns:
                        messagebox.showerror("Neural", f"Auto-create image_path requires a valid filename column. '{fn_col}' not found in data.")
                        return
                    if not base_dir:
                        messagebox.showerror("Neural", "Please set 'Image base folder' to build image_path.")
                        return
                    self.log(f"Auto-creating image_path from column '{fn_col}' with base '{base_dir}'")
                    try:
                        df = df.copy()
                        df["image_path"] = df[fn_col].astype(str).apply(lambda p: os.path.join(base_dir, p))
                    except Exception as e:
                        messagebox.showerror("Neural", f"Failed to build image_path: {e}")
                        return
                try:
                    df = self._apply_neural(df)
                    self.log("Neural processing applied.")
                except Exception as e:
                    messagebox.showerror("Neural", f"Neural processing failed: {e}")
                    return

            # Split ratios
            if self.do_split_var.get():
                try:
                    tr = float(self.train_ratio_var.get())
                    vr = float(self.val_ratio_var.get())
                    te = float(self.test_ratio_var.get())
                except ValueError:
                    messagebox.showerror("Split", "Ratios must be numeric.")
                    return
                total = tr + vr + te
                if abs(total - 1.0) > 1e-6:
                    messagebox.showerror("Split", f"Ratios must sum to 1.0 (got {total:.3f}).")
                    return

                strat_col = self.stratify_col_var.get().strip()
                if strat_col and strat_col not in df.columns:
                    messagebox.showerror("Stratify", f"Stratify column '{strat_col}' not found in data.")
                    return

                self.log("Performing train/val/test split...")
                try:
                    df_train, df_val, df_test = self._split_dataframe(df, tr, vr, te, strat_col if strat_col else None, random_state=42)
                except Exception as e:
                    messagebox.showerror("Split", f"Failed to split dataset: {e}")
                    return

                files = self._save_outputs_split(df_train, df_val, df_test, output_dir, base_name, out_format)
                self.log("Saved files:\n - " + "\n - ".join(files))
                messagebox.showinfo("Done", "Conversion completed successfully.")
            else:
                filepath = self._save_single(df, output_dir, base_name, out_format)
                self.log(f"Saved file: {filepath}")
                messagebox.showinfo("Done", "Conversion completed successfully.")

            # Save settings after successful run
            self._save_settings()

        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            self.log(f"Error: {e}")

    def _is_detection_format(self, fmt: str) -> bool:
        f = (fmt or "").strip().lower()
        return any(k in f for k in [
            "yolo txt (detection)",
            "pascal voc",
            "coco (detection)",
            "yolo dataset (images+labels)"
        ])

    def _apply_neural(self, df: pd.DataFrame) -> pd.DataFrame:
        task = (self.task_var.get() if hasattr(self, 'task_var') else 'Detection').lower()
        engine = (self.engine_var.get() if hasattr(self, 'engine_var') else 'Ultralytics YOLO')
        model_path = (self.model_var.get() if hasattr(self, 'model_var') else 'yolov8n.pt').strip() or "yolov8n.pt"
        try:
            conf = float(self.conf_var.get()) if hasattr(self, 'conf_var') and self.conf_var.get() else 0.25
        except Exception:
            conf = 0.25
        overwrite = bool(self.overwrite_var.get()) if hasattr(self, 'overwrite_var') else False
        if engine == "Ultralytics YOLO":
            try:
                if task == "detection":
                    return self._neural_detection(df, model_path, conf, overwrite)
                elif task == "classification":
                    return self._neural_classification(df, model_path, conf, overwrite)
            except Exception as e:
                # If likely due to missing weights/network, prompt to download
                if messagebox.askyesno("Models missing", f"Ultralytics model/weights unavailable or network error. Download common models now?\n\nDetails: {e}"):
                    self.download_models_ui()
                raise
        elif engine.startswith("TorchVision"):
            if task == "classification":
                arch = model_path or "resnet18"
                try:
                    return self._tv_classification(df, arch, overwrite)
                except Exception as e:
                    if messagebox.askyesno("Models missing", f"TorchVision weights unavailable or network error. Download common models now?\n\nDetails: {e}"):
                        self.download_models_ui()
                    raise
            else:
                raise ValueError("TorchVision backend supports Classification task only")
        raise ValueError(f"Unsupported neural task/engine: {engine} / {getattr(self, 'task_var', None) and self.task_var.get()}")

    def _tv_classification(self, df: pd.DataFrame, arch: str, overwrite: bool) -> pd.DataFrame:
        if "image_path" not in df.columns:
            raise ValueError("TorchVision Classification requires 'image_path' column")
        try:
            import torch  # type: ignore
            import torchvision  # type: ignore
            from torchvision import transforms  # type: ignore
        except Exception:
            raise RuntimeError("TorchVision is required. Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

        # Resolve model by name
        try:
            # Prefer new API if available
            if hasattr(torchvision.models, 'get_model'):
                model = torchvision.models.get_model(arch, weights='DEFAULT')
                weights = getattr(model, 'weights', None)
                categories = getattr(weights, 'meta', {}).get('categories') if weights else None
            else:
                # Fallback for common arch names
                fn = getattr(torchvision.models, arch)
                weights = getattr(torchvision.models, f"{arch}_Weights", None)
                weights = getattr(weights, 'DEFAULT', None) if weights is not None else None
                model = fn(weights=weights)
                categories = getattr(weights, 'meta', {}).get('categories') if weights else None
        except Exception as e:
            raise RuntimeError(f"Failed to create torchvision model '{arch}': {e}")

        model.eval()
        device = torch.device('cpu')
        model.to(device)
        preprocess = None
        try:
            if weights is not None and hasattr(weights, 'transforms'):
                preprocess = weights.transforms()
        except Exception:
            preprocess = None
        if preprocess is None:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        img_paths = [str(p) for p in df["image_path"].tolist()]
        labels: list[tuple[str, str | None]] = []
        self._start_progress("Neural inference (TorchVision Cls)", maximum=len(img_paths))
        try:
            for i, p in enumerate(img_paths):
                if not overwrite and ("label" in df.columns and not pd.isna(df.loc[df["image_path"] == p].iloc[0].get("label"))):
                    self._update_progress(i + 1, f"{i+1}/{len(img_paths)} (skip)")
                    continue
                try:
                    from PIL import Image as PILImage  # type: ignore
                    with PILImage.open(p).convert('RGB') as im:
                        x = preprocess(im).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0]
                        idx = int(torch.argmax(probs).item())
                        if categories and 0 <= idx < len(categories):
                            label = str(categories[idx])
                        else:
                            label = str(idx)
                except Exception as e:
                    self.log(f"TorchVision inference failed for {p}: {e}")
                    label = None
                labels.append((p, label))
                self._update_progress(i + 1, f"{i+1}/{len(img_paths)}")
        finally:
            self._end_progress()

        pred_map = {p: l for p, l in labels if l is not None}
        if not pred_map:
            self.log("TorchVision classification produced no predictions")
            return df

    # -------- Offline models downloader (PowerShell) --------
    def download_models_ui(self):
        # Debounce multiple clicks
        if not hasattr(self, "_downloading_models"):
            self._downloading_models = False
        if self._downloading_models:
            try:
                messagebox.showinfo("Download", "Downloads already in progress...")
            except Exception:
                pass
            return
        self._downloading_models = True
        try:
            if hasattr(self, "download_btn"):
                try:
                    self.download_btn.configure(state="disabled")
                except Exception:
                    pass
        except Exception:
            pass
        models_tv = ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]
        models_yolo = ["yolov8n.pt", "yolo11n.pt", "yolov8n-cls.pt"]
        total = len(models_tv) + len(models_yolo)
        self._start_progress("Downloading models (offline)", maximum=total, style="Green.Horizontal.TProgressbar")
        self.log("Starting offline model downloads via PowerShell...")

        def worker():
            done = 0
            # Snapshot versions before any installs
            prev_versions = self._collect_versions()
            python_exe = sys.executable
            # Detect availability of backends to avoid noisy failures
            try:
                import torchvision as _tv  # type: ignore
                tv_available = True
            except Exception:
                tv_available = False
                self.after(0, lambda: self.log("TorchVision not installed; attempting automatic install (CPU wheels)..."))
                # Attempt auto-install of Torch/TorchVision CPU build
                tv_install_cmd = [
                    python_exe, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu",
                ]
                self._run_ps_step(tv_install_cmd, "Install Torch/TorchVision (CPU)")
                # Re-check
                try:
                    import torchvision as _tv2  # type: ignore
                    tv_available = True
                    self.after(0, lambda: self.log("TorchVision installation succeeded."))
                except Exception as e:
                    tv_available = False
                    self.after(0, lambda: self.log(f"TorchVision installation failed: {e}"))
            try:
                import ultralytics as _ul  # type: ignore
                yolo_available = True
            except Exception:
                yolo_available = False
                self.after(0, lambda: self.log("Ultralytics not installed; attempting automatic install..."))
                # Try to avoid downgrading core deps like numpy during install
                cur_numpy = self._pkg_version("numpy")
                ul_install_cmd = [
                    python_exe, "-m", "pip", "install",
                    "--upgrade-strategy", "only-if-needed",
                    "ultralytics",
                ]
                if cur_numpy:
                    ul_install_cmd += [f"numpy=={cur_numpy}"]
                self._run_ps_step(ul_install_cmd, "Install Ultralytics")
                # Re-check
                try:
                    import ultralytics as _ul2  # type: ignore
                    yolo_available = True
                    self.after(0, lambda: self.log("Ultralytics installation succeeded."))
                except Exception as e:
                    yolo_available = False
                    self.after(0, lambda: self.log(f"Ultralytics installation failed: {e}"))

            # Compute total steps based on available backends
            try:
                total = (len(models_tv) if tv_available else 0) + (len(models_yolo) if yolo_available else 0)
            except Exception:
                total = 0
            if total == 0:
                self.after(0, lambda: self.log("No backends available for download after installation attempts."))
            # Detect if any core package versions changed during installs
            new_versions = self._collect_versions()
            changed = []
            try:
                for k, v in prev_versions.items():
                    if new_versions.get(k) != v:
                        changed.append((k, v, new_versions.get(k)))
            except Exception:
                pass
            # TorchVision
            if tv_available:
                for arch in models_tv:
                    cmd = [
                        python_exe, "-c",
                        f"import torchvision as tv; tv.models.get_model('{arch}', weights='DEFAULT')"
                    ]
                    self._run_ps_step(cmd, f"TorchVision: {arch}")
                    done += 1
                    if total:
                        self.after(0, lambda d=done: self._update_progress(d, f"{d}/{total} ({int(d/total*100)}%)"))
            # Ultralytics
            if yolo_available:
                for name in models_yolo:
                    cmd = [
                        python_exe, "-c",
                        f"from ultralytics import YOLO; YOLO('{name}')"
                    ]
                    self._run_ps_step(cmd, f"Ultralytics: {name}")
                    done += 1
                    if total:
                        self.after(0, lambda d=done: self._update_progress(d, f"{d}/{total} ({int(d/total*100)}%)"))
            self.after(0, self._end_progress)
            try:
                self.after(0, lambda: messagebox.showinfo("Download", "Model downloads completed."))
            except Exception:
                pass
            # If core libs changed, recommend restart to avoid in-process ABI/version mismatches
            if changed:
                msg_lines = ["Some core libraries changed during installation:"]
                for k, old, new in changed:
                    msg_lines.append(f" - {k}: {old or 'not installed'} -> {new or 'not installed'}")
                msg_lines.append("")
                msg_lines.append("It's recommended to restart the application to ensure a clean state.")
                try:
                    self.after(0, lambda: messagebox.showwarning("Restart Recommended", "\n".join(msg_lines)))
                except Exception:
                    # Fallback to log only
                    self.after(0, lambda: self.log("Restart Recommended due to library changes:\n" + "\n".join(msg_lines)))
            # Re-enable button and clear flag
            def _finish_reset():
                try:
                    if hasattr(self, "download_btn"):
                        self.download_btn.configure(state="normal")
                except Exception:
                    pass
                self._downloading_models = False
            self.after(0, _finish_reset)

        import threading
        threading.Thread(target=worker, daemon=True).start()

    def _run_ps_step(self, cmd: list[str], label: str):
        try:
            self.after(0, lambda: self.log(f"Running command: {label}"))
            # While this step runs, show indeterminate animation so UI looks active
            self.after(0, lambda: self._progress_set_indeterminate(True, f"{label} (working...)"))
            import subprocess, threading, os as _os
            # Environment tweaks to improve streaming and avoid interactive prompts
            _env = _os.environ.copy()
            _env.setdefault("PYTHONUNBUFFERED", "1")
            _env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
            _env.setdefault("PIP_NO_COLOR", "1")
            # Start process with text output, closed stdin to avoid waiting for input
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=_env,
            )

            # Reader thread to stream output without blocking main thread
            def _reader():
                try:
                    if proc.stdout is not None:
                        for line in iter(proc.stdout.readline, ''):
                            if not line:
                                break
                            self.after(0, lambda s=line.rstrip(): self.log(s))
                except Exception as re:
                    self.after(0, lambda: self.log(f"[reader] {label}: {re}"))

            t = threading.Thread(target=_reader, daemon=True)
            t.start()

            # Enforce a per-step timeout to avoid indefinite hangs
            STEP_TIMEOUT_S = 1800  # 30 minutes per step max
            try:
                proc.wait(timeout=STEP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.after(0, lambda: self.log(f"Timeout after {STEP_TIMEOUT_S}s: {label}. Terminating process..."))
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=15)
                except Exception:
                    pass

            rc = proc.returncode if proc.returncode is not None else -1
            if rc != 0:
                self.after(0, lambda: self.log(f"Failed: {label} (exit {rc})"))
            else:
                self.after(0, lambda: self.log(f"Done: {label}"))
        except Exception as e:
            self.after(0, lambda: self.log(f"Error running PowerShell for {label}: {e}"))
        finally:
            # Restore determinate mode so step counting continues
            self.after(0, lambda: self._progress_set_indeterminate(False))
        # Note: this helper only logs subprocess output; it doesn't modify DataFrames.

    def _neural_detection(self, df: pd.DataFrame, model_path: str, conf: float, overwrite: bool) -> pd.DataFrame:
        if "image_path" not in df.columns:
            raise ValueError("Neural Detection requires 'image_path' column")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            raise RuntimeError("Ultralytics is required. Install with: pip install ultralytics")
        model = YOLO(model_path)
        img_paths = [str(p) for p in df["image_path"].tolist()]

        pred_rows = []
        self._start_progress("Neural inference (Detection)", maximum=len(img_paths))
        try:
            for i, img_path in enumerate(img_paths):
                if not overwrite and ("bbox_x" in df.columns and not df.loc[df["image_path"] == img_path].get("bbox_x").isna().all()):
                    self._update_progress(i + 1, f"{i+1}/{len(img_paths)} (skip)")
                    continue
                try:
                    res = model(img_path, conf=conf, verbose=False)
                    res0 = res[0] if isinstance(res, (list, tuple)) else res
                except Exception as e:
                    self.log(f"Inference failed for {img_path}: {e}")
                    self._update_progress(i + 1, f"{i+1}/{len(img_paths)} (failed)")
                    continue

                w = h = 0
                try:
                    w, h = self._infer_image_size(img_path)
                except Exception:
                    pass
                names = getattr(res0, 'names', {})
                boxes_obj = getattr(res0, 'boxes', None)
                if boxes_obj is None:
                    self._update_progress(i + 1, f"{i+1}/{len(img_paths)} (no boxes)")
                    continue
                for box in boxes_obj:  # type: ignore[attr-defined]
                    try:
                        cls_id = int(box.cls.item())
                        score = float(box.conf.item())
                        cx, cy, bw, bh = box.xywh[0].tolist()
                        x = cx - bw / 2.0
                        y = cy - bh / 2.0
                        label = names.get(cls_id, str(cls_id)) if hasattr(names, 'get') else str(cls_id)
                        row = {
                            "image_path": img_path,
                            "label": label,
                            "bbox_x": x,
                            "bbox_y": y,
                            "bbox_w": bw,
                            "bbox_h": bh,
                            "confidence": score,
                        }
                        if w and h:
                            row.update({"width": w, "height": h})
                        pred_rows.append(row)
                    except Exception:
                        continue
                self._update_progress(i + 1, f"{i+1}/{len(img_paths)}")
        finally:
            self._end_progress()
        if not pred_rows:
            self.log("Neural detection produced no predictions (check confidence threshold/model)")
            return df
        pred_df = pd.DataFrame(pred_rows)
        if overwrite:
            imgs = set(pred_df["image_path"].unique())
            base_df = df[~df["image_path"].isin(imgs)].copy()
            out = pd.concat([base_df, pred_df], ignore_index=True)
        else:
            imgs_with_boxes = set(df.loc[df.get("bbox_x").notna() if "bbox_x" in df.columns else []]["image_path"].unique()) if "bbox_x" in df.columns else set()
            pred_df = pred_df[~pred_df["image_path"].isin(imgs_with_boxes)]
            out = pd.concat([df, pred_df], ignore_index=True)
        return out

    def _neural_classification(self, df: pd.DataFrame, model_path: str, conf: float, overwrite: bool) -> pd.DataFrame:
        if "image_path" not in df.columns:
            raise ValueError("Neural Classification requires 'image_path' column")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            raise RuntimeError("Ultralytics is required. Install with: pip install ultralytics")
        model = YOLO(model_path)
        img_paths = [str(p) for p in df["image_path"].tolist()]
        results = model(img_paths, conf=conf, verbose=False)
        labels = []
        for img_path, res in zip(img_paths, results):
            try:
                probs = getattr(res, 'probs', None)
                if probs is None:
                    if getattr(res, 'boxes', None) is not None and len(res.boxes) > 0:
                        cls_id = int(res.boxes[0].cls.item())
                        label = res.names.get(cls_id, str(cls_id))
                    else:
                        label = None
                else:
                    cls_id = int(probs.top1)
                    label = res.names.get(cls_id, str(cls_id))
            except Exception:
                label = None
            labels.append((img_path, label))
        pred_map = {p: l for p, l in labels if l is not None}
        if not pred_map:
            self.log("Neural classification produced no predictions")
            return df
        df = df.copy()
        if "label" not in df.columns:
            df["label"] = np.nan
        for i, row in df.iterrows():
            p = str(row.get("image_path"))
            if p in pred_map:
                if overwrite or pd.isna(row.get("label")) or row.get("label") in (None, ""):
                    df.at[i, "label"] = pred_map[p]
        return df

    def _save_outputs_split(self, df_train, df_val, df_test, out_dir, base, fmt):
        paths = []
        for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
            p = self._save_df(d, out_dir, f"{base}_{name}", fmt)
            paths.append(p)
        return paths

    def _save_single(self, df, out_dir, base, fmt):
        return self._save_df(df, out_dir, base, fmt)

    def _save_df(self, df: pd.DataFrame, out_dir: str, base: str, fmt: str) -> str:
        # Ensure we don't overwrite existing outputs; prompt to rename base if needed
        fmt_key = fmt.lower()
        new_base = self._ensure_unique_base(out_dir, base, fmt_key)
        if new_base is None:
            # user cancelled
            raise RuntimeError("Export cancelled: output already exists and no new name provided")
        base = new_base

        if fmt_key == "csv":
            path = os.path.join(out_dir, f"{base}.csv")
            df.to_csv(path, index=False)
            return path
        elif fmt_key == "tsv":
            path = os.path.join(out_dir, f"{base}.tsv")
            df.to_csv(path, index=False, sep='\t')
            return path
        elif fmt_key == "jsonl":
            path = os.path.join(out_dir, f"{base}.jsonl")
            df.to_json(path, orient='records', lines=True, force_ascii=False)
            return path
        elif fmt_key == "json":
            path = os.path.join(out_dir, f"{base}.json")
            df.to_json(path, orient='records', force_ascii=False, indent=2)
            return path
        elif fmt_key == "parquet":
            if pyarrow is None:
                raise RuntimeError("pyarrow is required for Parquet. Please install it (pip install pyarrow).")
            path = os.path.join(out_dir, f"{base}.parquet")
            df.to_parquet(path, index=False)
            return path
        elif fmt_key == "feather":
            if pyarrow is None:
                raise RuntimeError("pyarrow is required for Feather. Please install it (pip install pyarrow).")
            path = os.path.join(out_dir, f"{base}.feather")
            df.to_feather(path)
            return path
        elif fmt_key == "excel (xlsx)":
            path = os.path.join(out_dir, f"{base}.xlsx")
            try:
                df.to_excel(path, index=False)
            except Exception as e:
                raise RuntimeError(f"Excel export requires openpyxl. Install it with: pip install openpyxl. Error: {e}")
            return path
        elif fmt_key == "sqlite":
            import sqlite3
            path = os.path.join(out_dir, f"{base}.sqlite")
            table_name = base
            with sqlite3.connect(path) as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            return path
        elif fmt_key == "coco (detection)":
            return self._export_coco_detection(df, out_dir, base)
        elif fmt_key == "yolo txt (detection)":
            return self._export_yolo_detection(df, out_dir, base)
        elif fmt_key == "imagefolder (classification)":
            return self._export_imagefolder(df, out_dir, base)
        elif fmt_key == "pascal voc (detection)":
            return self._export_pascal_voc(df, out_dir, base)
        elif fmt_key == "yolo dataset (images+labels)":
            return self._export_yolo_dataset(df, out_dir, base)
        elif fmt_key == "coco (segmentation)":
            return self._export_coco_segmentation(df, out_dir, base)
        elif fmt_key == "yolo txt (segmentation)":
            return self._export_yolo_segmentation(df, out_dir, base)
        elif fmt_key == "hf dataset (jsonl)":
            return self._export_hf_dataset(df, out_dir, base)
        elif fmt_key == "webdataset (tar shards)":
            return self._export_webdataset(df, out_dir, base)
        elif fmt_key == "tfrecord (optional)":
            return self._export_tfrecord(df, out_dir, base)
        elif fmt_key == "audio manifest (jsonl)":
            return self._export_audio_manifest(df, out_dir, base)
        elif fmt_key == "timeseries windows (parquet)":
            return self._export_timeseries_windows(df, out_dir, base)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def _existing_outputs(self, out_dir: str, base: str, fmt_key: str) -> list[str]:
        """Return a list of output file/dir paths that would be created for this base+format."""
        f = fmt_key
        out: list[str] = []
        # Simple, single-file formats
        if f == "csv":
            out = [os.path.join(out_dir, f"{base}.csv")]
        elif f == "tsv":
            out = [os.path.join(out_dir, f"{base}.tsv")]
        elif f == "jsonl":
            out = [os.path.join(out_dir, f"{base}.jsonl")]
        elif f == "json":
            out = [os.path.join(out_dir, f"{base}.json")]
        elif f == "parquet":
            out = [os.path.join(out_dir, f"{base}.parquet")]
        elif f == "feather":
            out = [os.path.join(out_dir, f"{base}.feather")]
        elif f == "excel (xlsx)":
            out = [os.path.join(out_dir, f"{base}.xlsx")]
        elif f == "sqlite":
            out = [os.path.join(out_dir, f"{base}.sqlite")]
        elif f == "coco (detection)":
            out = [os.path.join(out_dir, f"{base}_coco.json")]
        elif f == "yolo txt (detection)":
            out = [os.path.join(out_dir, f"{base}_yolo")]
        elif f == "pascal voc":
            out = [os.path.join(out_dir, f"{base}_voc")]
        elif f == "yolo dataset (images+labels)":
            out = [os.path.join(out_dir, f"{base}_yolo_ds")]
        elif f == "coco (segmentation)":
            out = [os.path.join(out_dir, f"{base}_coco_seg.json")]
        elif f == "yolo txt (segmentation)":
            out = [os.path.join(out_dir, f"{base}_yolo_seg")]
        elif f == "imagefolder (classification)":
            out = [os.path.join(out_dir, f"{base}_imagefolder")]
        elif f == "hf dataset (jsonl)":
            out = [os.path.join(out_dir, f"{base}_hf.jsonl"), os.path.join(out_dir, f"{base}_README.md")]
        elif f == "webdataset (tar shards)":
            out = [os.path.join(out_dir, f"{base}.tar")]
        elif f == "tfrecord (optional)":
            out = [os.path.join(out_dir, f"{base}.tfrecord")]
        elif f == "audio manifest (jsonl)":
            out = [os.path.join(out_dir, f"{base}_audio_manifest.jsonl")]
        elif f == "timeseries windows (parquet)":
            out = [os.path.join(out_dir, f"{base}_ts_windows.parquet")]
        else:
            # Fallback: check common suffixes to be safe
            out = [
                os.path.join(out_dir, f"{base}.csv"),
                os.path.join(out_dir, f"{base}.json"),
                os.path.join(out_dir, f"{base}.jsonl"),
                os.path.join(out_dir, f"{base}.parquet"),
                os.path.join(out_dir, f"{base}.feather"),
                os.path.join(out_dir, f"{base}.xlsx"),
                os.path.join(out_dir, f"{base}.sqlite"),
                os.path.join(out_dir, f"{base}.tar"),
            ]
        return out

    def _ensure_unique_base(self, out_dir: str, base: str, fmt_key: str) -> str | None:
        """If any output for base+format exists, prompt user to rename. Returns new base or None if cancelled."""
        while True:
            targets = self._existing_outputs(out_dir, base, fmt_key)
            exists = [p for p in targets if (os.path.exists(p))]
            if not exists:
                return base
            # Show error-style prompt with a blank input to rename
            suggestion = self._next_available_base(out_dir, base, fmt_key)
            new_base = simpledialog.askstring(
                "Rename output",
                "Output already exists. Enter a new base filename:",
                initialvalue=suggestion or "",
                parent=self,
            )
            if not new_base:
                messagebox.showerror("Already exists", "The output name already exists. Please provide a new name.")
                return None
            base = new_base.strip()
            if not base:
                messagebox.showerror("Invalid name", "Base filename cannot be blank.")
                return None

    def _next_available_base(self, out_dir: str, base: str, fmt_key: str) -> str:
        """Return a suggested new base like base_v2, base_v3 that doesn't conflict."""
        # if base without suffix works when incremented, suggest the first free
        for i in range(2, 1000):
            candidate = f"{base}_v{i}"
            targets = self._existing_outputs(out_dir, candidate, fmt_key)
            if not any(os.path.exists(p) for p in targets):
                return candidate
        return f"{base}_new"

    def _require_columns(self, df: pd.DataFrame, cols: list[str]):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    def _infer_image_size(self, path: str) -> tuple[int, int]:
        if Image is None:
            raise RuntimeError("Pillow is required to infer image sizes. Install with: pip install Pillow")
        with Image.open(path) as im:
            w, h = im.size
        return int(w), int(h)

    def _export_coco_detection(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Expected columns: image_path,label,bbox_x,bbox_y,bbox_w,bbox_h and optional width,height,image_id
        req = ["image_path", "label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        self._require_columns(df, req)

        # Normalize types
        df = df.copy()
        for c in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        if "image_id" not in df.columns:
            # assign stable IDs by image path order
            img_ids = {p: i+1 for i, p in enumerate(pd.unique(df["image_path"]))}
            df["image_id"] = df["image_path"].map(img_ids)
        else:
            df["image_id"] = pd.to_numeric(df["image_id"], errors='coerce').fillna(0).astype(int)

        # Category mapping
        labels = sorted(map(str, pd.unique(df["label"])))
        cat_id = {name: i+1 for i, name in enumerate(labels)}

        images = []
        annotations = []
        seen_images = set()
        ann_id = 1
        for _, row in df.iterrows():
            img_path = str(row["image_path"])
            iid = int(row["image_id"])
            if iid not in seen_images:
                # width/height provided?
                if "width" in df.columns and "height" in df.columns and not pd.isna(row.get("width")) and not pd.isna(row.get("height")):
                    w = int(row["width"])
                    h = int(row["height"])
                else:
                    try:
                        w, h = self._infer_image_size(img_path)
                    except Exception:
                        w, h = 0, 0
                images.append({
                    "id": iid,
                    "file_name": os.path.basename(img_path),
                    "width": w,
                    "height": h,
                })
                seen_images.add(iid)

            bbox = [float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_w"]), float(row["bbox_h"])]
            area = float(max(0.0, bbox[2] * bbox[3]))
            annotations.append({
                "id": ann_id,
                "image_id": iid,
                "category_id": cat_id[str(row["label"])],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        categories = [{"id": i, "name": name} for name, i in cat_id.items()]
        coco = {"images": images, "annotations": annotations, "categories": categories}
        path = os.path.join(out_dir, f"{base}_coco.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
        return path

    def _export_yolo_detection(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Expected columns: image_path,label,bbox_x,bbox_y,bbox_w,bbox_h and optional width,height
        req = ["image_path", "label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        self._require_columns(df, req)

        df = df.copy()
        # build class mapping
        classes = sorted(map(str, pd.unique(df["label"])))
        class_to_id = {c: i for i, c in enumerate(classes)}

        # label folder
        yolo_root = os.path.join(out_dir, f"{base}_yolo")
        labels_dir = os.path.join(yolo_root, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        # Determine image sizes if not present; YOLO expects normalized cx,cy,w,h in [0,1]
        size_cache: dict[str, tuple[int, int]] = {}
        def get_size(p: str) -> tuple[int, int]:
            if p in size_cache:
                return size_cache[p]
            w = h = 0
            if "width" in df.columns and "height" in df.columns:
                sub = df.loc[df["image_path"] == p]
                try:
                    w = int(sub.iloc[0].get("width", 0))
                    h = int(sub.iloc[0].get("height", 0))
                except Exception:
                    w = h = 0
            if (w <= 0 or h <= 0) and Image is not None and os.path.exists(p):
                try:
                    w, h = self._infer_image_size(p)
                except Exception:
                    w = h = 0
            size_cache[p] = (w, h)
            return w, h

        # Write one .txt per image
        for img_path, grp in df.groupby("image_path", sort=False):
            w, h = get_size(str(img_path))
            lines = []
            for _, r in grp.iterrows():
                cls_id = class_to_id[str(r["label"])]
                x, y, bw, bh = float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])
                if w > 0 and h > 0:
                    # convert xywh top-left to YOLO cxcywh normalized
                    cx = (x + bw / 2.0) / w
                    cy = (y + bh / 2.0) / h
                    nw = bw / w
                    nh = bh / h
                else:
                    # if unknown size, leave as 0 to signal invalid normalization
                    cx = cy = nw = nh = 0.0
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            # label file name mirrors image file name with .txt
            fname = os.path.splitext(os.path.basename(str(img_path)))[0] + ".txt"
            with open(os.path.join(labels_dir, fname), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        # classes.txt
        with open(os.path.join(yolo_root, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(classes))

        return yolo_root

    def _export_imagefolder(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Expected columns: image_path,label
        req = ["image_path", "label"]
        self._require_columns(df, req)

        root = os.path.join(out_dir, f"{base}_imagefolder")
        os.makedirs(root, exist_ok=True)
        copied = 0
        errors = 0
        for lbl, grp in df.groupby("label", sort=False):
            cls_dir = os.path.join(root, str(lbl))
            os.makedirs(cls_dir, exist_ok=True)
            for _, r in grp.iterrows():
                src = str(r["image_path"])
                if not os.path.isfile(src):
                    errors += 1
                    continue
                dst = os.path.join(cls_dir, os.path.basename(src))
                try:
                    if os.path.abspath(src) != os.path.abspath(dst):
                        shutil.copy2(src, dst)
                    copied += 1
                except Exception:
                    errors += 1
        self.log(f"ImageFolder export: copied={copied}, errors={errors}")
        return root

    def _export_yolo_dataset(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Uses YOLO TXT (Detection) labels and copies images into images/; labels into labels/
        req = ["image_path", "label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        self._require_columns(df, req)
        root = os.path.join(out_dir, f"{base}_yolo_ds")
        images_dir = os.path.join(root, "images")
        labels_dir = os.path.join(root, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Build YOLO labels (normalized)
        classes = sorted(map(str, pd.unique(df["label"])))
        class_to_id = {c: i for i, c in enumerate(classes)}
        # Copy images and write labels
        for img_path, grp in df.groupby("image_path", sort=False):
            img_path = str(img_path)
            img_name = os.path.basename(img_path)
            # Copy
            if os.path.isfile(img_path):
                shutil.copy2(img_path, os.path.join(images_dir, img_name))
            # Size
            w = h = 0
            if "width" in df.columns and "height" in df.columns:
                try:
                    w = int(grp.iloc[0].get("width", 0))
                    h = int(grp.iloc[0].get("height", 0))
                except Exception:
                    w = h = 0
            if (w <= 0 or h <= 0) and Image is not None and os.path.exists(img_path):
                try:
                    w, h = self._infer_image_size(img_path)
                except Exception:
                    w = h = 0
            lines = []
            for _, r in grp.iterrows():
                cid = class_to_id[str(r["label"])]
                x, y, bw, bh = float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])
                if w > 0 and h > 0:
                    cx = (x + bw/2)/w
                    cy = (y + bh/2)/h
                    nw = bw/w
                    nh = bh/h
                else:
                    cx = cy = nw = nh = 0.0
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            with open(os.path.join(labels_dir, Path(img_name).stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        # dataset.yaml helper
        yaml_path = os.path.join(root, "dataset.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("names:\n")
            for i, c in enumerate(classes):
                f.write(f"  {i}: {c}\n")
            f.write("train: images\nval: images\n# update paths as needed\n")
        return root

    def _export_coco_segmentation(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Expected: image_path,label,segmentation (JSON list of polygon lists), optional width,height,image_id
        req = ["image_path", "label", "segmentation"]
        self._require_columns(df, req)
        df = df.copy()
        if "image_id" not in df.columns:
            img_ids = {p: i+1 for i, p in enumerate(pd.unique(df["image_path"]))}
            df["image_id"] = df["image_path"].map(img_ids)
        labels = sorted(map(str, pd.unique(df["label"])))
        cat_id = {name: i+1 for i, name in enumerate(labels)}
        images = []
        annotations = []
        seen = set()
        ann_id = 1
        for _, r in df.iterrows():
            img_path = str(r["image_path"])
            iid = int(r["image_id"])
            if iid not in seen:
                if "width" in df.columns and "height" in df.columns and not pd.isna(r.get("width")) and not pd.isna(r.get("height")):
                    w = int(r["width"])
                    h = int(r["height"])
                else:
                    try:
                        w, h = self._infer_image_size(img_path)
                    except Exception:
                        w, h = 0, 0
                images.append({"id": iid, "file_name": os.path.basename(img_path), "width": w, "height": h})
                seen.add(iid)
            try:
                seg = r["segmentation"]
                if isinstance(seg, str):
                    seg = json.loads(seg)
                # Expect list of lists (each polygon is flat [x1,y1,...])
                segmentation = seg
            except Exception as e:
                raise ValueError(f"Invalid segmentation JSON for {img_path}: {e}")
            annotations.append({
                "id": ann_id,
                "image_id": iid,
                "category_id": cat_id[str(r["label"])],
                "segmentation": segmentation,
                "iscrowd": 0,
                "bbox": [0, 0, 0, 0],
                "area": 0,
            })
            ann_id += 1
        categories = [{"id": i, "name": n} for n, i in cat_id.items()]
        coco = {"images": images, "annotations": annotations, "categories": categories}
        path = os.path.join(out_dir, f"{base}_coco_seg.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
        return path

    def _export_yolo_segmentation(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Expected: image_path,label,segmentation (JSON list of polygons [[x1,y1,...], ...]) and optional width,height
        req = ["image_path", "label", "segmentation"]
        self._require_columns(df, req)
        classes = sorted(map(str, pd.unique(df["label"])))
        class_to_id = {c: i for i, c in enumerate(classes)}
        root = os.path.join(out_dir, f"{base}_yolo_seg")
        labels_dir = os.path.join(root, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        size_cache: dict[str, tuple[int, int]] = {}
        def get_size(p: str) -> tuple[int, int]:
            if p in size_cache:
                return size_cache[p]
            w = h = 0
            if "width" in df.columns and "height" in df.columns:
                sub = df.loc[df["image_path"] == p]
                try:
                    w = int(sub.iloc[0].get("width", 0))
                    h = int(sub.iloc[0].get("height", 0))
                except Exception:
                    w = h = 0
            if (w <= 0 or h <= 0) and Image is not None and os.path.exists(p):
                try:
                    w, h = self._infer_image_size(p)
                except Exception:
                    w = h = 0
            size_cache[p] = (w, h)
            return w, h
        for img_path, grp in df.groupby("image_path", sort=False):
            w, h = get_size(str(img_path))
            lines = []
            for _, r in grp.iterrows():
                cid = class_to_id[str(r["label"])]
                seg = r["segmentation"]
                if isinstance(seg, str):
                    seg = json.loads(seg)
                # Flatten and normalize
                polys = []
                for poly in seg:
                    flat = []
                    for i in range(0, len(poly), 2):
                        x = float(poly[i]); y = float(poly[i+1])
                        if w > 0 and h > 0:
                            flat += [x / w, y / h]
                        else:
                            flat += [0.0, 0.0]
                    polys.append(" ".join(f"{v:.6f}" for v in flat))
                line = f"{cid} " + " ".join(polys)
                lines.append(line)
            with open(os.path.join(labels_dir, Path(str(img_path)).stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        with open(os.path.join(root, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(classes))
        return root

    def _export_hf_dataset(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Writes JSONL file suitable for load_dataset("json", data_files=...)
        path = os.path.join(out_dir, f"{base}_hf.jsonl")
        df.to_json(path, orient='records', lines=True, force_ascii=False)
        # Also write a minimal dataset card
        with open(os.path.join(out_dir, f"{base}_README.md"), "w", encoding="utf-8") as f:
            f.write(f"# {base} dataset\n\nGenerated by {APP_NAME} v{APP_VERSION}.\n")
        return path

    def _export_webdataset(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Simple tar sharding: write one tar containing .json lines and raw files if image_path/audio_path exists
        shard_path = os.path.join(out_dir, f"{base}.tar")
        with tarfile.open(shard_path, "w") as tar:
            for i, row in df.reset_index(drop=True).iterrows():
                rec = row.to_dict()
                # Write JSON metadata
                info = tarfile.TarInfo(name=f"{i}.json")
                data = json.dumps(rec, ensure_ascii=False).encode("utf-8")
                info.size = len(data)
                tar.addfile(info, fileobj=io.BytesIO(data))
                # Optionally include binary file if present
                for key in ["image_path", "audio_path", "file_path"]:
                    p = rec.get(key)
                    if isinstance(p, str) and os.path.isfile(p):
                        tar.add(p, arcname=f"{i}.{os.path.splitext(p)[1].lstrip('.')}")
        return shard_path

    def _export_tfrecord(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Minimal TFRecord writer; requires tensorflow installed
        try:
            import tensorflow as tf  # type: ignore
        except Exception:
            raise RuntimeError("TensorFlow is required for TFRecord export. Install with: pip install tensorflow")
        # Expect classification: columns image_path,label
        if not ("image_path" in df.columns and "label" in df.columns):
            raise ValueError("TFRecord export currently expects columns: image_path,label")
        tfrecord_path = os.path.join(out_dir, f"{base}.tfrecord")
        with tf.io.TFRecordWriter(tfrecord_path) as w:
            for _, r in df.iterrows():
                p = str(r["image_path"])
                label = str(r["label"])
                if not os.path.isfile(p):
                    continue
                with open(p, "rb") as f:
                    img_bytes = f.read()
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
                    "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.basename(p).encode('utf-8')])),
                }))
                w.write(ex.SerializeToString())
        return tfrecord_path

    def _export_audio_manifest(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Expected columns: audio_path,label and optional duration,sample_rate
        req = ["audio_path", "label"]
        self._require_columns(df, req)
        man_path = os.path.join(out_dir, f"{base}_audio_manifest.jsonl")
        with open(man_path, "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                rec = {
                    "audio": str(r["audio_path"]),
                    "label": r["label"],
                }
                for k in ["duration", "sample_rate"]:
                    if k in df.columns:
                        rec[k] = r[k]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return man_path

    def _export_timeseries_windows(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Prompt for window and stride, expect a time-sorted CSV with numeric columns; outputs a Parquet of windows
        try:
            window = int(simpledialog.askstring("TimeSeries", "Window length (rows)", initialvalue="50", parent=self) or "50")
            stride = int(simpledialog.askstring("TimeSeries", "Stride (rows)", initialvalue="50", parent=self) or "50")
        except Exception:
            raise ValueError("Invalid window/stride")
        if window <= 0 or stride <= 0:
            raise ValueError("Window and stride must be positive")
        values = df.select_dtypes(include=[np.number]).to_numpy()
        n, d = values.shape
        windows = []
        for start in range(0, max(0, n - window + 1), stride):
            windows.append(values[start:start+window])
        if not windows:
            raise ValueError("No windows produced; check window/stride vs data length")
        arr = np.stack(windows)  # [num_windows, window, features]
        # Save as Parquet by flattening second dimension into columns
        cols = [f"t{t}_{j}" for t in range(window) for j in range(d)]
        flat = arr.reshape(arr.shape[0], -1)
        out_df = pd.DataFrame(flat, columns=cols)
        path = os.path.join(out_dir, f"{base}_ts_windows.parquet")
        if pyarrow is None:
            raise RuntimeError("pyarrow is required for Parquet. Please install it (pip install pyarrow).")
        out_df.to_parquet(path, index=False)
        return path

    def _export_pascal_voc(self, df: pd.DataFrame, out_dir: str, base: str) -> str:
        # Expected columns: image_path,label,bbox_x,bbox_y,bbox_w,bbox_h and optional width,height
        req = ["image_path", "label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        self._require_columns(df, req)

        root = os.path.join(out_dir, f"{base}_voc")
        ann_dir = os.path.join(root, "Annotations")
        os.makedirs(ann_dir, exist_ok=True)

        # Helper to write one XML per image with possibly multiple objects
        for img_path, grp in df.groupby("image_path", sort=False):
            img_path = str(img_path)
            filename = os.path.basename(img_path)
            # infer size if not provided
            w = h = 0
            if "width" in df.columns and "height" in df.columns:
                try:
                    w = int(grp.iloc[0].get("width", 0))
                    h = int(grp.iloc[0].get("height", 0))
                except Exception:
                    w = h = 0
            if (w <= 0 or h <= 0) and Image is not None and os.path.exists(img_path):
                try:
                    w, h = self._infer_image_size(img_path)
                except Exception:
                    w = h = 0

            ann = ET.Element("annotation")
            ET.SubElement(ann, "folder").text = os.path.basename(root)
            ET.SubElement(ann, "filename").text = filename
            ET.SubElement(ann, "path").text = img_path
            src = ET.SubElement(ann, "source")
            ET.SubElement(src, "database").text = "Unknown"
            size = ET.SubElement(ann, "size")
            ET.SubElement(size, "width").text = str(max(0, w))
            ET.SubElement(size, "height").text = str(max(0, h))
            ET.SubElement(size, "depth").text = "3"
            ET.SubElement(ann, "segmented").text = "0"

            for _, r in grp.iterrows():
                name = str(r["label"]) 
                x, y, bw, bh = float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])
                obj = ET.SubElement(ann, "object")
                ET.SubElement(obj, "name").text = name
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                bnd = ET.SubElement(obj, "bndbox")
                ET.SubElement(bnd, "xmin").text = str(int(round(x)))
                ET.SubElement(bnd, "ymin").text = str(int(round(y)))
                ET.SubElement(bnd, "xmax").text = str(int(round(x + bw)))
                ET.SubElement(bnd, "ymax").text = str(int(round(y + bh)))

            tree = ET.ElementTree(ann)
            xml_name = os.path.splitext(filename)[0] + ".xml"
            tree.write(os.path.join(ann_dir, xml_name), encoding="utf-8", xml_declaration=True)

        return root

    def preview_input(self):
        path = self.input_path_var.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Preview", "Please select a valid CSV file first.")
            return
        try:
            df = pd.read_csv(path, nrows=50)
        except Exception as e:
            messagebox.showerror("Preview", f"Failed to read CSV: {e}")
            return
        top = tk.Toplevel(self)
        top.title("Preview (first 50 rows)")
        top.geometry("800x400")
        # Add scrollable text area for better UX
        xscroll = tk.Scrollbar(top, orient=tk.HORIZONTAL)
        yscroll = tk.Scrollbar(top, orient=tk.VERTICAL)
        txt = tk.Text(top, wrap=tk.NONE, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        yscroll.config(command=txt.yview)
        xscroll.config(command=txt.xview)
        # Layout
        txt.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        top.grid_rowconfigure(0, weight=1)
        txt.insert(tk.END, df.to_string(max_rows=50, max_cols=30))
        txt.configure(state=tk.DISABLED)

    def _split_dataframe(self, df: pd.DataFrame, tr: float, vr: float, te: float, strat_col: str | None, random_state: int = 42):
        """Split a DataFrame into train/val/test. If strat_col is provided, split within each class.

        Rounds counts per class to nearest integer and adjusts the test split to absorb rounding remainder.
        """
        rng = np.random.default_rng(random_state)

        # Try to use scikit-learn if available for robust splitting
        try:
            from sklearn.model_selection import StratifiedShuffleSplit, train_test_split  # type: ignore
            use_sklearn = True
        except Exception:
            use_sklearn = False

        if not strat_col:
            # No stratification
            if use_sklearn:
                # Do two-step split to get train, then split remaining into val/test with proper ratios
                df_temp, df_test = train_test_split(df, test_size=te, random_state=random_state)
                remaining = 1.0 - te
                val_ratio_in_remaining = vr / max(remaining, 1e-9)
                df_train, df_val = train_test_split(df_temp, test_size=val_ratio_in_remaining, random_state=random_state)
                return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
            else:
                # Fallback to numpy-based split
                n = len(df)
                idx = np.arange(n)
                rng.shuffle(idx)
                n_train = int(round(n * tr))
                n_val = int(round(n * vr))
                n_train = min(n_train, n)
                n_val = min(n_val, max(0, n - n_train))
                tr_i = idx[:n_train]
                va_i = idx[n_train:n_train+n_val]
                te_i = idx[n_train+n_val:]
                return df.iloc[tr_i].reset_index(drop=True), df.iloc[va_i].reset_index(drop=True), df.iloc[te_i].reset_index(drop=True)

        # With stratification
        if strat_col not in df.columns:
            raise ValueError(f"Stratify column '{strat_col}' not found in DataFrame")
        y = df[strat_col]
        if use_sklearn:
            # First split off test with stratification
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=te, random_state=random_state)
            idx_all = np.arange(len(df))
            for train_val_idx, test_idx in sss1.split(idx_all, y):
                df_train_val = df.iloc[train_val_idx]
                df_test = df.iloc[test_idx]
            # Now split train_val into train and val with stratification and adjusted ratio
            remaining = 1.0 - te
            val_ratio_in_remaining = vr / max(remaining, 1e-9)
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_in_remaining, random_state=random_state)
            y_tv = df_train_val[strat_col]
            idx_tv = np.arange(len(df_train_val))
            for train_idx, val_idx in sss2.split(idx_tv, y_tv):
                df_train = df_train_val.iloc[train_idx]
                df_val = df_train_val.iloc[val_idx]
            return (
                df_train.reset_index(drop=True),
                df_val.reset_index(drop=True),
                df_test.reset_index(drop=True),
            )
        else:
            # Fallback: per-class shuffle and slice
            parts_train = []
            parts_val = []
            parts_test = []
            for cls, group in df.groupby(strat_col, sort=False):
                n = len(group)
                if n == 0:
                    continue
                idx = np.arange(n)
                rng.shuffle(idx)
                n_train = int(round(n * tr))
                n_val = int(round(n * vr))
                n_train = min(n_train, n)
                n_val = min(n_val, max(0, n - n_train))
                tr_i = idx[:n_train]
                va_i = idx[n_train:n_train+n_val]
                te_i = idx[n_train+n_val:]
                parts_train.append(group.iloc[tr_i])
                parts_val.append(group.iloc[va_i])
                parts_test.append(group.iloc[te_i])
            df_train = pd.concat(parts_train, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True) if parts_train else df.iloc[0:0]
            df_val = pd.concat(parts_val, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True) if parts_val else df.iloc[0:0]
            df_test = pd.concat(parts_test, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True) if parts_test else df.iloc[0:0]
            return df_train, df_val, df_test


def main():
    app = DataProcessorApp()
    # Apply some ttk styling for a modern look
    try:
        style = ttk.Style()
        if sys.platform == 'win32':
            style.theme_use('vista')
        else:
            # Prefer a modern cross-platform theme if available
            if 'clam' in style.theme_names():
                style.theme_use('clam')
            else:
                # leave default theme
                pass
    except Exception:
        pass
    app.mainloop()


if __name__ == "__main__":
 main()
