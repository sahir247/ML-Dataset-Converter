from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

import pandas as pd


def log_stream_process(cmd: list[str], logger: Callable[[str], None]) -> int:
    """Run a subprocess and stream stdout to logger. Returns exit code."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            logger(line.rstrip())
        proc.wait()
        return int(proc.returncode or 0)
    except Exception as e:
        logger(f"Error running command: {e}")
        return 1


def install_packages(packages: Iterable[str], logger: Callable[[str], None]) -> bool:
    """Install given packages with pip; stream logs."""
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    logger(f"Running: {' '.join(cmd)}")
    rc = log_stream_process(cmd, logger)
    if rc == 0:
        logger("Packages installed successfully.")
        return True
    logger(f"pip exited with code {rc}")
    return False


def read_dataset(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if p.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if p.endswith(".json"):
        return pd.read_json(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".feather"):
        return pd.read_feather(path)
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def df_preview(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": df.shape,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": df.head(10).to_dict(orient="records"),
    }


def unique_path(base_dir: str, base_name: str, ext: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{base_name}_{ts}.{ext.lstrip('.')}"
    return os.path.join(base_dir, name)


def to_base64_png(fig) -> str:
    import matplotlib.pyplot as plt  # lazy import
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def write_html_report(path: str, title: str, sections: list[tuple[str, str]]) -> str:
    """Write a simple HTML report.
    sections: list of (section_title, html_content)
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>{}</title>".format(title),
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;} h1,h2{color:#333} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
    ]
    for sec_title, content in sections:
        html_parts.append(f"<h2>{sec_title}</h2>")
        html_parts.append(content)
    html_parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return path


@dataclass
class TrainResult:
    model: Any
    history: Optional[dict[str, list[float]]] = None
    task_type: str = "classification"  # or "regression"
    feature_names: Optional[list[str]] = None


def safe_import(module: str) -> tuple[bool, Optional[Any]]:
    try:
        mod = __import__(module, fromlist=["*"])
        return True, mod
    except Exception:
        return False, None


def is_classification_series(y: pd.Series) -> bool:
    # Heuristic: non-numeric or few unique values => classification
    try:
        if pd.api.types.is_numeric_dtype(y):
            nunique = int(y.nunique(dropna=True))
            return nunique <= max(20, int(len(y) * 0.05))
        return True
    except Exception:
        return True
