from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetSpec:
    path: str
    target_col: str
    feature_cols: list[str]
    task_type: str | None = None  # 'classification' | 'regression' | None (auto)
    test_size: float = 0.2
    random_state: int = 42
    dropna: bool = True


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str
    task_type: str
    label_encoder: Optional[LabelEncoder] = None


def load_dataframe(path: str) -> pd.DataFrame:
    from .utils import read_dataset
    df = read_dataset(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Loaded data is not a pandas DataFrame")
    return df


def preview_info(df: pd.DataFrame) -> dict:
    from .utils import df_preview
    return df_preview(df)


def validate_and_prepare(df: pd.DataFrame, spec: DatasetSpec) -> PreparedData:
    # Validate columns
    if spec.target_col not in df.columns:
        raise ValueError(f"Target column not found: {spec.target_col}")
    missing = [c for c in spec.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")

    data = df[spec.feature_cols + [spec.target_col]].copy()
    if spec.dropna:
        data = data.dropna(axis=0)

    # Separate X/y (defensive copies)
    X = data[spec.feature_cols].copy()
    y = data[spec.target_col].copy()

    # If y came as a DataFrame (e.g., multiple one-hot columns selected upstream),
    # collapse to a single 1D target or fail with a clear message.
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            vals = y.values
            try:
                uniq = set(np.unique(vals))
                row_sums = vals.sum(axis=1)
                # Treat as one-hot if values are {0,1} and each row sums to 1 (or 0 for unknown)
                if uniq.issubset({0, 1}) and np.all((row_sums == 1) | (row_sums == 0)):
                    y = np.argmax(vals, axis=1)
                    y = pd.Series(y, name=spec.target_col)
                else:
                    raise ValueError("Target appears to be multi-output (multiple columns). Please select a single target column.")
            except Exception:
                raise ValueError("Target appears to be multi-output (multiple columns). Please select a single target column.")

    # Determine task type if None
    from .utils import is_classification_series
    y_for_detect = y if isinstance(y, pd.Series) else pd.Series(y, name=spec.target_col)
    task = spec.task_type or ("classification" if is_classification_series(y_for_detect) else "regression")

    # Encode target for classification if non-numeric
    le: Optional[LabelEncoder] = None
    if task == "classification":
        if not pd.api.types.is_integer_dtype(y) and not pd.api.types.is_float_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            # Ensure dense 1D array
            y = y.astype(float).values
    else:
        y = pd.to_numeric(y, errors="coerce").values

    # Train/test split BEFORE encoding to avoid leakage of test categories
    if task == "classification":
        X_train_df, X_test_df, y_tr, y_te = train_test_split(
            X, y, test_size=spec.test_size, random_state=spec.random_state, stratify=y
        )
    else:
        X_train_df, X_test_df, y_tr, y_te = train_test_split(
            X, y, test_size=spec.test_size, random_state=spec.random_state
        )

    # One-hot encode categorical features
    # Guardrail: prevent exploding one-hot matrices for very high-cardinality columns
    # Fit mapping on TRAIN only, apply to TEST
    try:
        MAX_CARD = 1000  # adjustable heuristic
        for col in list(X_train_df.columns):
            try:
                if not pd.api.types.is_numeric_dtype(X_train_df[col]):
                    nunique = int(X_train_df[col].nunique(dropna=True))
                    if nunique > MAX_CARD:
                        freq = X_train_df[col].value_counts(dropna=False)
                        top = set(freq.head(MAX_CARD).index)
                        X_train_df.loc[:, col] = X_train_df[col].where(X_train_df[col].isin(top), other="__OTHER__")
                        X_test_df.loc[:, col] = X_test_df[col].where(X_test_df[col].isin(top), other="__OTHER__")
            except Exception:
                # best-effort; if inspection fails, leave column as-is
                pass
    except Exception:
        pass

    X_tr_enc = pd.get_dummies(X_train_df, drop_first=False)
    feature_names = list(X_tr_enc.columns)
    X_te_enc = pd.get_dummies(X_test_df, drop_first=False).reindex(columns=feature_names, fill_value=0)

    X_tr = X_tr_enc.values
    X_te = X_te_enc.values

    return PreparedData(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        feature_names=feature_names,
        target_name=spec.target_col,
        task_type=task,
        label_encoder=le,
    )
