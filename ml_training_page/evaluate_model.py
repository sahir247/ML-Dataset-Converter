from __future__ import annotations

# Force a non-interactive backend to avoid GUI usage in worker threads
import matplotlib
matplotlib.use("Agg")

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset_validator import PreparedData
from .utils import TrainResult, to_base64_png


def _ensure_1d(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()
    return y


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_true = _ensure_1d(y_true)
    y_pred = _ensure_1d(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    # If the number of classes is extremely large, computing macro metrics may allocate
    # (n_samples x n_classes) indicator matrices internally. Skip in that case.
    try:
        n_classes = int(np.unique(np.concatenate([y_true, y_pred])).shape[0])
    except Exception:
        n_classes = int(np.unique(y_true).shape[0])
    if n_classes > 1000:
        return {
            "accuracy": acc,
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "f1_macro": float("nan"),
        }
    return {
        "accuracy": acc,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    y_true = _ensure_1d(y_true)
    y_pred = _ensure_1d(y_pred)
    # Limit confusion matrix size for very high-cardinality problems
    try:
        all_vals = np.concatenate([y_true, y_pred])
        vals, counts = np.unique(all_vals, return_counts=True)
        n_classes = int(vals.shape[0])
    except Exception:
        vals, counts = np.unique(y_true, return_counts=True)
        n_classes = int(vals.shape[0])

    labels = None
    title_suffix = ""
    if n_classes > 50:
        topk = 50
        idx = np.argsort(counts)[::-1][:topk]
        labels = list(vals[idx])
        title_suffix = f" (top {topk} classes)"

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=(cm.size <= 400), fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + title_suffix)
    fig.tight_layout()
    return fig


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = _ensure_1d(y_true)
    y_pred = _ensure_1d(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, s=12, alpha=0.7)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs True")
    fig.tight_layout()
    return fig


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred, residuals, s=12, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--', lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    return fig


def _feature_importance_fig(model: Any, feature_names: List[str], X: Optional[Any] = None, y: Optional[np.ndarray] = None):
    import numpy as np
    import matplotlib.pyplot as plt
    orig_model = model
    # Unwrap sklearn Pipeline to its final estimator if present
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]
    except Exception:
        pass

    scores = None
    title = "Feature Importance"

    try:
        if hasattr(model, "feature_importances_"):
            scores = np.asarray(getattr(model, "feature_importances_"))
        elif hasattr(model, "coef_"):
            coef = np.asarray(getattr(model, "coef_"))
            if coef.ndim == 2:
                coef = np.mean(np.abs(coef), axis=0)
            scores = np.abs(coef)
            title = "Coefficient Magnitudes"
    except Exception:
        scores = None

    # Fallback: permutation importance using available test data (fast subsample)
    if scores is None and X is not None and y is not None:
        try:
            from sklearn.inspection import permutation_importance
            # Subsample to speed up for large test sets
            try:
                n = len(y)
            except Exception:
                n = getattr(X, "shape", [0])[0]
            max_n = 200
            if n and n > max_n:
                X_s = X[:max_n]
                y_s = _ensure_1d(y)[:max_n]
            else:
                X_s = X
                y_s = _ensure_1d(y)
            result = permutation_importance(orig_model, X_s, y_s, n_repeats=5, random_state=42, n_jobs=1)
            scores = np.abs(result.importances_mean)
            title = "Permutation Importance"
        except Exception:
            scores = None

    # Fallback: SHAP importance (unified API) — best-effort and subsampled
    if scores is None and X is not None:
        try:
            import shap  # heavy, import lazily
            # Ensure numpy inputs and modest sample size
            X_arr = np.asarray(X)
            n = int(getattr(X_arr, "shape", [0])[0]) if hasattr(X_arr, "shape") else 0
            if n == 0:
                raise RuntimeError("Empty X for SHAP")
            max_bg = 100
            bg = X_arr[: min(max_bg, n)]
            max_eval = 100
            X_eval = X_arr[: min(max_eval, n)]
            # Use the unwrapped estimator when possible
            est = model
            # shap.Explainer auto-selects the best explainer
            explainer = shap.Explainer(est, bg)
            exp = explainer(X_eval)
            # SHAP returns Explanation with .values (n_samples, n_features) or list for multiclass
            shap_vals = getattr(exp, "values", exp)
            if isinstance(shap_vals, list):
                # multiclass: aggregate across classes
                agg = np.mean([np.abs(v) for v in shap_vals], axis=0)
                scores = np.mean(np.abs(agg), axis=0)
            else:
                vals = np.asarray(shap_vals)
                # If 3D (multiclass), aggregate appropriately
                if vals.ndim == 3:
                    vals = np.mean(np.abs(vals), axis=0)
                scores = np.mean(np.abs(vals), axis=0)
            title = "SHAP Importance"
        except Exception:
            scores = None

    if scores is None:
        return None

    idx = np.argsort(scores)[::-1]
    topk = idx[: min(20, len(idx))]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([feature_names[i] for i in topk][::-1], scores[topk][::-1])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _keras_curves_figs(history: Optional[Dict[str, List[float]]]):
    if not history:
        return []
    import matplotlib.pyplot as plt

    figs = []
    if "loss" in history:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(history.get("loss", []), label="loss")
        if "val_loss" in history:
            ax.plot(history.get("val_loss", []), label="val_loss")
        ax.set_title("Loss vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        figs.append(fig)
    # accuracy
    if "accuracy" in history or "val_accuracy" in history:
        fig, ax = plt.subplots(figsize=(5, 4))
        if "accuracy" in history:
            ax.plot(history.get("accuracy", []), label="accuracy")
        if "val_accuracy" in history:
            ax.plot(history.get("val_accuracy", []), label="val_accuracy")
        ax.set_title("Accuracy vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()
        figs.append(fig)
    return figs


@dataclass
class EvaluationResult:
    task_type: str
    metrics: Dict[str, float]
    images_b64: Dict[str, str]  # name -> base64 png


def evaluate(prepared: PreparedData, res: TrainResult) -> EvaluationResult:
    if res.model is None:
        raise ValueError("No trained model to evaluate")

    X_te, y_te = prepared.X_test, prepared.y_test
    task = prepared.task_type

    images: Dict[str, str] = {}

    # Attempt to handle models that require DataFrame (e.g., PyCaret pipeline)
    X_te_input: Any = X_te
    try:
        from pandas import DataFrame
        X_te_input = DataFrame(X_te, columns=prepared.feature_names)
    except Exception:
        pass

    # Predictions
    try:
        y_pred = res.model.predict(X_te_input)
    except Exception:
        # Some Keras models output probabilities; try argmax or >0.5
        try:
            y_prob = res.model.predict(X_te)
            if task == "classification":
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    y_pred = np.argmax(y_prob, axis=1)
                else:
                    y_pred = (y_prob.ravel() >= 0.5).astype(int)
            else:
                y_pred = y_prob.ravel()
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")

    # Guardrails: ensure equal lengths to avoid implicit broadcasting creating huge NxN arrays
    y_te = _ensure_1d(y_te)
    y_pred = _ensure_1d(y_pred)
    if y_te.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Prediction length mismatch: y_true={y_te.shape} vs y_pred={y_pred.shape}. "
            "Ensure predictions align with the test set and no reshuffling/reindexing occurred."
        )

    # Evaluation
    if task == "classification":
        metrics = _classification_metrics(y_te, y_pred)
        # Try to compute ROC-AUC if probabilities or decision scores are available
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            y_score = None
            # Prefer predict_proba
            try:
                proba = getattr(res.model, "predict_proba", None)
                if callable(proba):
                    y_prob = proba(X_te_input)
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        y_score = y_prob[:, 1]
                    else:
                        y_score = y_prob  # multiclass
            except Exception:
                y_score = None
            # Fallback to decision_function
            if y_score is None:
                decf = getattr(res.model, "decision_function", None)
                if callable(decf):
                    y_score = decf(X_te_input)
            # Compute AUC
            if y_score is not None:
                classes = np.unique(y_te)
                n_classes = int(classes.shape[0])
                if n_classes == 2 and y_score.ndim == 1:
                    auc = float(roc_auc_score(y_te, y_score))
                    metrics["roc_auc"] = auc
                    # Plot ROC curve
                    fpr, tpr, _ = roc_curve(y_te, y_score)
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
                    ax.plot([0,1],[0,1], 'k--', lw=1)
                    ax.set_xlabel("FPR")
                    ax.set_ylabel("TPR")
                    ax.set_title("ROC Curve (binary)")
                    ax.legend(loc="lower right")
                    fig.tight_layout()
                    images["roc_curve"] = to_base64_png(fig)
                elif y_score.ndim == 2 and n_classes == y_score.shape[1] and n_classes <= 10:
                    try:
                        auc = float(roc_auc_score(y_te, y_score, multi_class="ovr", average="macro"))
                        metrics["roc_auc_ovr_macro"] = auc
                    except Exception:
                        pass
        except Exception:
            pass
        fig_cm = _plot_confusion(y_te, y_pred)
        images["confusion_matrix"] = to_base64_png(fig_cm)
        fig_fi = _feature_importance_fig(res.model, res.feature_names or prepared.feature_names, X_te_input, y_te)
        if fig_fi is not None:
            images["feature_importance"] = to_base64_png(fig_fi)
    else:
        metrics = _regression_metrics(y_te, y_pred)
        fig_scatter = _plot_regression_scatter(y_te, y_pred)
        images["pred_vs_true"] = to_base64_png(fig_scatter)
        fig_res = _plot_residuals(y_te, y_pred)
        images["residuals"] = to_base64_png(fig_res)

    # Keras history plots
    for i, fig in enumerate(_keras_curves_figs(res.history)):
        images[f"keras_curve_{i}"] = to_base64_png(fig)

    return EvaluationResult(task_type=task, metrics=metrics, images_b64=images)


def build_html_sections(ev: EvaluationResult) -> list[tuple[str, str]]:
    # Metrics table
    metr_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in ev.metrics.items()
    )
    metrics_html = f"<table><tr><th>Metric</th><th>Value</th></tr>{metr_rows}</table>"

    # Images
    img_html_parts = []
    for name, b64 in ev.images_b64.items():
        img_html_parts.append(f"<div><h3>{name}</h3><img alt='{name}' src='data:image/png;base64,{b64}' style='max-width:100%'/></div>")
    images_html = "\n".join(img_html_parts)

    return [
        ("Summary Metrics", metrics_html),
        ("Visualizations", images_html),
    ]
