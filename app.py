 
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

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _build_ui(self):
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

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

        # Download models button (offline fetch)
        self.download_btn = ttk.Button(out_frame, text="Download Models", command=self.download_models_ui)
        self.download_btn.grid(row=5, column=4, sticky=tk.W, padx=(8,0))

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
        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

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
            # Neural
            self.mode_var.set(cfg.get("processing_mode", "Standard"))
            self.task_var.set(cfg.get("neural_task", "Detection"))
            self.engine_var.set(cfg.get("neural_engine", "Ultralytics YOLO"))
            self.model_var.set(cfg.get("neural_model", "yolov8n.pt"))
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
            "do_split": bool(self.do_split_var.get()),
            "train_ratio": float(self.train_ratio_var.get() or 0),
            "val_ratio": float(self.val_ratio_var.get() or 0),
            "test_ratio": float(self.test_ratio_var.get() or 0),
            "stratify_col": self.stratify_col_var.get().strip(),
            # Neural
            "processing_mode": self.mode_var.get(),
            "neural_task": self.task_var.get(),
            "neural_engine": getattr(self, 'engine_var', tk.StringVar(value="Ultralytics YOLO")).get(),
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

    def _on_close(self):
        self._save_settings()
        self.destroy()

    def _show_about(self):
        messagebox.showinfo("About", f"{APP_NAME} v{APP_VERSION}\nConvert raw CSV to ML-ready datasets.\n© 2025")

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
            python_exe = sys.executable
            # TorchVision
            for arch in models_tv:
                cmd = [
                    "powershell", "-NoProfile", "-Command",
                    f"& '{python_exe}' -c \"import torchvision as tv; tv.models.get_model('{arch}', weights='DEFAULT')\""
                ]
                self._run_ps_step(cmd, f"TorchVision: {arch}")
                done += 1
                self.after(0, lambda d=done: self._update_progress(d, f"{d}/{total} ({int(d/total*100)}%)"))
            # Ultralytics
            for name in models_yolo:
                cmd = [
                    "powershell", "-NoProfile", "-Command",
                    f"& '{python_exe}' -c \"from ultralytics import YOLO; YOLO('{name}')\""
                ]
                self._run_ps_step(cmd, f"Ultralytics: {name}")
                done += 1
                self.after(0, lambda d=done: self._update_progress(d, f"{d}/{total} ({int(d/total*100)}%)"))
            self.after(0, self._end_progress)
            try:
                self.after(0, lambda: messagebox.showinfo("Download", "Model downloads completed."))
            except Exception:
                pass
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
            self.after(0, lambda: self.log(f"Running PowerShell: {label}"))
            import subprocess
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                self.after(0, lambda s=line.strip(): self.log(s))
            proc.wait()
            if proc.returncode != 0:
                self.after(0, lambda: self.log(f"Failed: {label} (exit {proc.returncode})"))
            else:
                self.after(0, lambda: self.log(f"Done: {label}"))
        except Exception as e:
            self.after(0, lambda: self.log(f"Error running PowerShell for {label}: {e}"))
        df = df.copy()
        if "label" not in df.columns:
            df["label"] = np.nan
        for i, row in df.iterrows():
            p = str(row.get("image_path"))
            if p in pred_map:
                if overwrite or pd.isna(row.get("label")) or row.get("label") in (None, ""):
                    df.at[i, "label"] = pred_map[p]
        return df

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
        txt = tk.Text(top, wrap=tk.NONE)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, df.to_string(max_rows=50, max_cols=30))
        txt.configure(state=tk.DISABLED)

    def _split_dataframe(self, df: pd.DataFrame, tr: float, vr: float, te: float, strat_col: str | None, random_state: int = 42):
        """Split a DataFrame into train/val/test. If strat_col is provided, split within each class.

        Rounds counts per class to nearest integer and adjusts the test split to absorb rounding remainder.
        """
        rng = np.random.default_rng(random_state)

        def split_indices(n):
            n_train = int(round(n * tr))
            n_val = int(round(n * vr))
            # ensure we don't exceed n due to rounding
            n_train = min(n_train, n)
            n_val = min(n_val, max(0, n - n_train))
            n_test = max(0, n - n_train - n_val)
            return n_train, n_val, n_test

        if strat_col is None:
            idx = np.arange(len(df))
            rng.shuffle(idx)
            n_train, n_val, n_test = split_indices(len(idx))
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val: n_train + n_val + n_test]
            return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

        # Stratified: process each class separately
        if strat_col not in df.columns:
            raise ValueError(f"Stratify column '{strat_col}' not found")

        parts_train = []
        parts_val = []
        parts_test = []
        for cls, group in df.groupby(strat_col, sort=False):
            n = len(group)
            if n == 0:
                continue
            # If class is too small and any split ratio > 0, still attempt best-effort split
            local_idx = np.arange(n)
            rng.shuffle(local_idx)
            n_train, n_val, n_test = split_indices(n)
            tr_i = local_idx[:n_train]
            va_i = local_idx[n_train:n_train + n_val]
            te_i = local_idx[n_train + n_val:n_train + n_val + n_test]
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
            style.theme_use(style.theme_use())
    except Exception:
        pass
    app.mainloop()


if __name__ == "__main__":
    main()
