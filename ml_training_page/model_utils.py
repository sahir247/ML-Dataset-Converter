from __future__ import annotations

from typing import Optional, Dict, Any, Callable

LogFn = Callable[[str], None]


def _parse_version(ver: str) -> tuple[int, int, int]:
    try:
        parts = ver.split("+")[0].split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return major, minor, patch
    except Exception:
        return (0, 0, 0)


def auto_xgb_params(use_gpu: Optional[bool], log: LogFn | None = None) -> Dict[str, Any]:
    """Return version-safe XGBoost device params with graceful fallback.

    Strategy:
    - If use_gpu is True, try to enable GPU. If not available, log and fall back to CPU.
    - If use_gpu is None, auto-detect and prefer GPU when available.
    - If use_gpu is False, force CPU.
    - XGBoost >= 2.0: prefer device="cuda"/"cpu" with tree_method="hist".
    - XGBoost < 2.0: use tree_method="gpu_hist" + predictor="gpu_predictor" for GPU; otherwise tree_method="hist".
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        # If xgboost isn't available, return CPU defaults (hist)
        return {"tree_method": "hist"}

    is_v2 = _parse_version(getattr(xgb, "__version__", "0.0.0")) >= (2, 0, 0)

    gpu_available = False
    # Prefer explicit API if available
    try:
        cuda_mod = getattr(xgb, "cuda", None)
        if cuda_mod is not None:
            get_cnt = getattr(cuda_mod, "get_gpu_count", None)
            if callable(get_cnt):
                gpu_available = get_cnt() > 0
    except Exception:
        pass

    # Fallback capability probe
    if not gpu_available:
        try:
            if is_v2:
                xgb.XGBClassifier(tree_method="hist", device="cuda")
            else:
                xgb.XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor")
            gpu_available = True
        except Exception:
            gpu_available = False

    want_gpu = (use_gpu is True) or (use_gpu is None and gpu_available)

    params: Dict[str, Any] = {}
    if want_gpu and gpu_available:
        if is_v2:
            params.update(tree_method="hist", device="cuda")
        else:
            params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        if is_v2:
            params.update(tree_method="hist", device="cpu")
        else:
            params.update(tree_method="hist")
        if use_gpu is True and log:
            try:
                log("XGBoost GPU requested but not available; falling back to CPU.")
            except Exception:
                pass

    return params


def auto_lgbm_params(use_gpu: Optional[bool], log: LogFn | None = None) -> Dict[str, Any]:
    """Return LightGBM device params with graceful fallback across versions.

    Strategy:
    - If use_gpu is True, try device_type='gpu' then device='gpu'. If both fail, log and return {}.
    - If use_gpu is None, auto-detect support by probing instantiation with GPU.
    - If use_gpu is False, return {} (CPU default).
    """
    try:
        import lightgbm as lgb  # type: ignore
    except Exception:
        return {}

    def _supports_gpu() -> bool:
        try:
            lgb.LGBMClassifier(device_type="gpu")  # type: ignore[arg-type]
            return True
        except Exception:
            try:
                lgb.LGBMClassifier(device="gpu")  # type: ignore[arg-type]
                return True
            except Exception:
                return False

    gpu_ok = _supports_gpu()

    if use_gpu is False:
        return {}

    if use_gpu is None and not gpu_ok:
        return {}

    if use_gpu is True and not gpu_ok:
        if log:
            try:
                log("LightGBM GPU requested but not available; falling back to CPU.")
            except Exception:
                pass
        return {}

    # Try device_type first, else device
    return {"device_type": "gpu"} if gpu_ok else {}
