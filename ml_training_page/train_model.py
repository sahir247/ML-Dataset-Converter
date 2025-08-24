from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .dataset_validator import PreparedData
from .utils import TrainResult
from .model_utils import auto_xgb_params, auto_lgbm_params


LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]  # progress [0..1], text


@dataclass
class MLConfig:
    model_type: str  # 'LogisticRegression','RandomForest','SVM','XGBoost','LightGBM','Keras'
    params: Dict[str, Any] = field(default_factory=dict)
    cross_validate: bool = False
    cv_folds: int = 5
    random_state: int = 42
    use_gpu: Optional[bool] = None
    early_stopping: bool = True
    # Keras specific
    epochs: int = 15
    batch_size: int = 32
    verbose: int = 0
    # Layers spec for Keras in advanced mode (list of tuples)
    # Example: [("Dense", 128, "relu"), ("Dropout", 0.5)]
    layers: Optional[list[tuple]] = None


def _get_sklearn_model(task: str, model_type: str, params: Dict[str, Any], use_gpu: Optional[bool], log: LogFn) -> Any:
    import sklearn
    if model_type.lower() == "logisticregression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        defaults = dict(max_iter=1000, n_jobs=None, solver="lbfgs")
        defaults.update(params)
        # Scale features to improve conditioning and generalization
        model = LogisticRegression(**defaults)
        return make_pipeline(StandardScaler(with_mean=True), model)

    if model_type.lower() == "randomforest":
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier
            defaults = dict(n_estimators=200, random_state=42, n_jobs=-1)
            defaults.update(params)
            return RandomForestClassifier(**defaults)
        else:
            from sklearn.ensemble import RandomForestRegressor
            defaults = dict(n_estimators=200, random_state=42, n_jobs=-1)
            defaults.update(params)
            return RandomForestRegressor(**defaults)

    if model_type.lower() == "svm":
        # Use scaling to improve generalization and numerical stability
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if task == "classification":
            from sklearn.svm import SVC
            defaults = dict(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=None)
            defaults.update(params)
            return make_pipeline(StandardScaler(with_mean=True), SVC(**defaults))
        else:
            from sklearn.svm import SVR
            defaults = dict(C=1.0, kernel="rbf", gamma="scale")
            defaults.update(params)
            return make_pipeline(StandardScaler(with_mean=True), SVR(**defaults))

    if model_type.lower() == "xgboost":
        import xgboost as xgb
        if task == "classification":
            defaults = dict(
                n_estimators=300,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=6,
                random_state=42,
                eval_metric="logloss",
            )
            # Centralized, version-safe device params
            defaults.update(auto_xgb_params(use_gpu, log))
            defaults.update(params)
            # Backward compatibility: retry removing unsupported keys
            try:
                model = xgb.XGBClassifier(**defaults)
            except TypeError:
                # Remove keys progressively and retry
                tmp = dict(defaults)
                tmp.pop("device", None)
                try:
                    model = xgb.XGBClassifier(**tmp)
                except TypeError:
                    tmp.pop("predictor", None)
                    model = xgb.XGBClassifier(**tmp)
            return model
        else:
            defaults = dict(
                n_estimators=300,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=6,
                random_state=42,
                eval_metric="rmse",
            )
            defaults.update(auto_xgb_params(use_gpu, log))
            defaults.update(params)
            try:
                model = xgb.XGBRegressor(**defaults)
            except TypeError:
                tmp = dict(defaults)
                tmp.pop("device", None)
                try:
                    model = xgb.XGBRegressor(**tmp)
                except TypeError:
                    tmp.pop("predictor", None)
                    model = xgb.XGBRegressor(**tmp)
            return model

    if model_type.lower() == "lightgbm":
        import lightgbm as lgb
        if task == "classification":
            defaults = dict(n_estimators=300, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)
            # Centralized GPU/CPU params with graceful fallback
            defaults.update(auto_lgbm_params(use_gpu, log))
            defaults.update(params)
            return lgb.LGBMClassifier(**defaults)
        else:
            defaults = dict(n_estimators=300, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)
            defaults.update(auto_lgbm_params(use_gpu, log))
            defaults.update(params)
            return lgb.LGBMRegressor(**defaults)

    raise ValueError(f"Unsupported model_type: {model_type}")


def train_standard_ml(prepared: PreparedData, cfg: MLConfig, log: LogFn, progress: ProgressFn | None = None, stop_flag: Optional[list[bool]] = None) -> TrainResult:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    # Reproducibility seeds
    try:
        import random
        random.seed(cfg.random_state)
        np.random.seed(cfg.random_state)
    except Exception:
        pass

    X_tr, y_tr = prepared.X_train, prepared.y_train
    # Ensure 1D target
    y_tr = np.asarray(y_tr).ravel()
    task = prepared.task_type

    log(f"Training {cfg.model_type} on {X_tr.shape[0]} samples, {X_tr.shape[1]} features ({task})")

    # Auto class_weight for imbalanced classification
    params_for_model = dict(cfg.params)
    if task == "classification":
        try:
            classes, counts = np.unique(y_tr, return_counts=True)
            if classes.shape[0] >= 2:
                ratio = float(np.max(counts)) / float(np.min(counts))
                needs_balance = ratio >= 2.0
            else:
                needs_balance = False
        except Exception:
            needs_balance = False
        mt = cfg.model_type.lower()
        supports_cw = mt in {"logisticregression", "svm", "randomforest"}
        if needs_balance and supports_cw and ("class_weight" not in params_for_model):
            params_for_model["class_weight"] = "balanced"
            log("Detected class imbalance; setting class_weight='balanced' for the classifier.")

    model = _get_sklearn_model(task, cfg.model_type, params_for_model, cfg.use_gpu, log)
    # Log inferred device settings for GPU-capable models
    try:
        getp = getattr(model, "get_params", None)
        if callable(getp):
            p = getp()
            dev = p.get("device") or p.get("device_type") or p.get("predictor")
            tm = p.get("tree_method")
            if dev or tm:
                log(f"Model device hints: device={dev}, tree_method={tm}")
    except Exception:
        pass

    # Cross-validation (optional)
    if cfg.cross_validate:
        log("Running cross-validation...")
        if task == "classification":
            cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
            scoring = "f1_macro"
        else:
            cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
            scoring = "neg_root_mean_squared_error"
        scores = []
        n = cfg.cv_folds
        for i, (train_idx, val_idx) in enumerate(cv.split(X_tr, y_tr), start=1):
            if stop_flag and stop_flag[0]:
                log("Training stopped by user during CV.")
                break
            Xtr, Xval = X_tr[train_idx], X_tr[val_idx]
            ytr, yval = y_tr[train_idx], y_tr[val_idx]
            ytr = np.asarray(ytr).ravel()
            yval = np.asarray(yval).ravel()
            m = _get_sklearn_model(task, cfg.model_type, cfg.params, cfg.use_gpu, log)
            t0 = time.time()
            try:
                m.fit(Xtr, ytr)
            except Exception as e:
                log(f"Fit failed in CV fold {i}: {type(e).__name__}: {e}")
                raise
            dt = time.time() - t0
            from sklearn.metrics import f1_score, mean_squared_error
            if task == "classification":
                ypred = m.predict(Xval)
                s = float(f1_score(yval, ypred, average="macro"))
                log(f"Fold {i}/{n}: F1-macro={s:.4f} (time {dt:.2f}s)")
            else:
                ypred = m.predict(Xval)
                rmse = math.sqrt(mean_squared_error(yval, ypred))
                s = -rmse
                log(f"Fold {i}/{n}: RMSE={-s:.4f} (time {dt:.2f}s)")
            scores.append(s)
            if progress:
                progress(i / n, f"CV fold {i}/{n}")
        if scores:
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            if task == "classification":
                log(f"CV mean F1-macro={mean:.4f} ± {std:.4f}")
            else:
                log(f"CV mean RMSE={-mean:.4f} ± {std:.4f}")

    if stop_flag and stop_flag[0]:
        return TrainResult(model=None, task_type=task, feature_names=prepared.feature_names)

    # Final fit with early stopping if supported
    t0 = time.time()
    fitted = False
    if cfg.early_stopping:
        try:
            from sklearn.model_selection import train_test_split
            is_classif = task == "classification"
            Xf, Xv, yf, yv = train_test_split(
                X_tr,
                y_tr,
                test_size=0.1,
                random_state=cfg.random_state,
                stratify=y_tr if is_classif else None,
            )
            name = type(model).__name__.lower()
            # XGBoost wrappers
            if "xgb" in name or "xgboost" in str(type(model)).lower():
                try:
                    model.fit(
                        Xf,
                        yf,
                        eval_set=[(Xv, yv)],
                        verbose=False,
                        early_stopping_rounds=20,
                    )
                    fitted = True
                except TypeError:
                    # older/newer API differences
                    try:
                        model.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
                        fitted = True
                    except Exception:
                        fitted = False
            # LightGBM wrappers
            if not fitted and ("lgbm" in name or "lightgbm" in str(type(model)).lower()):
                try:
                    import lightgbm as lgb
                    cb = [lgb.early_stopping(20, verbose=False)]
                    model.fit(
                        Xf,
                        yf,
                        eval_set=[(Xv, yv)],
                        callbacks=cb,
                    )
                    fitted = True
                except Exception:
                    fitted = False
        except Exception:
            fitted = False

    if not fitted:
        try:
            model.fit(X_tr, y_tr)
            fitted = True
        except Exception as e:
            log(f"Fit failed: {type(e).__name__}: {e}")
            raise
    dt = time.time() - t0
    log(f"Fit complete in {dt:.2f}s")

    return TrainResult(model=model, task_type=task, feature_names=prepared.feature_names)


def _build_keras_model(input_dim: int, out_dim: int, task: str, cfg: MLConfig):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    if cfg.layers:
        for layer in cfg.layers:
            if not layer:
                continue
            name = str(layer[0]).lower()
            if name == "dense":
                units = int(layer[1]) if len(layer) > 1 else 64
                activation = str(layer[2]) if len(layer) > 2 else "relu"
                model.add(layers.Dense(units, activation=activation))
            elif name == "dropout":
                rate = float(layer[1]) if len(layer) > 1 else 0.5
                model.add(layers.Dropout(rate))
            elif name == "batchnorm":
                model.add(layers.BatchNormalization())
            else:
                # ignore unknown layer names
                pass
    else:
        # Sensible defaults
        from tensorflow.keras import regularizers
        model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))

    if task == "classification" and out_dim > 2:
        model.add(layers.Dense(out_dim, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    elif task == "classification":
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        # Regression head: linear activation
        model.add(layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = ["mae"]

    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model


class _KerasStopCallback:
    def __init__(self, stop_flag: Optional[list[bool]], log: LogFn):
        self.stop_flag = stop_flag
        self.log = log

    def as_callbacks(self):
        from tensorflow.keras.callbacks import Callback, EarlyStopping

        stop_cb = self

        class _Stopper(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if stop_cb.stop_flag and stop_cb.stop_flag[0]:
                    stop_cb.log("Stop requested — terminating training after current epoch.")
                    self.model.stop_training = True

        cbs = [_Stopper()]
        # Early stopping
        cbs.append(EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))
        return cbs


def train_keras(prepared: PreparedData, cfg: MLConfig, log: LogFn, progress: ProgressFn | None = None, stop_flag: Optional[list[bool]] = None) -> TrainResult:
    import tensorflow as tf
    # Reproducibility seeds for non-TF RNGs
    try:
        import random
        random.seed(cfg.random_state)
        np.random.seed(cfg.random_state)
    except Exception:
        pass
    X_tr, y_tr = prepared.X_train, prepared.y_train
    X_te, y_te = prepared.X_test, prepared.y_test
    # Ensure 1D targets for Keras too
    y_tr = np.asarray(y_tr).ravel()
    y_te = np.asarray(y_te).ravel()

    # Determine number of classes for classification
    out_dim = 1
    if prepared.task_type == "classification":
        out_dim = int(np.unique(y_tr).shape[0])
        if out_dim < 2:
            out_dim = 2

    model = _build_keras_model(X_tr.shape[1], out_dim, prepared.task_type, cfg)

    log("Starting Keras training...")
    # Log device info
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            names = [getattr(g, 'name', 'GPU') for g in gpus]
            log(f"TensorFlow devices: GPU detected -> {names}")
            # Enable memory growth to reduce OOM risk
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                log("Enabled TF GPU memory growth")
            except Exception:
                pass
        else:
            log("TensorFlow devices: no GPU detected; using CPU")
    except Exception:
        pass
    # Ensure reproducibility
    try:
        tf.random.set_seed(cfg.random_state)
    except Exception:
        pass
    # Internal validation split from TRAIN to avoid test leakage
    try:
        from sklearn.model_selection import train_test_split
        is_classif = prepared.task_type == "classification"
        Xf, Xv, yf, yv = train_test_split(
            X_tr,
            y_tr,
            test_size=0.1,
            random_state=cfg.random_state,
            stratify=y_tr if is_classif else None,
        )
    except Exception:
        # Fallback: no split
        Xf, yf = X_tr, y_tr
        Xv, yv = X_te, y_te  # last resort
    # Callbacks
    callbacks = _KerasStopCallback(stop_flag, log).as_callbacks()
    validation_data = (Xv, yv)

    # Fit with one retry on OOM by reducing batch size
    try:
        history = model.fit(
            Xf,
            yf,
            validation_data=validation_data,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
    except Exception as e:
        if "ResourceExhaustedError" in type(e).__name__ or "OOM" in str(e).upper():
            new_bs = max(8, int(cfg.batch_size // 2) or 8)
            log(f"Keras OOM detected; retrying with smaller batch_size={new_bs}")
            history = model.fit(
                Xf,
                yf,
                validation_data=validation_data,
                epochs=cfg.epochs,
                batch_size=new_bs,
                verbose=0,
                callbacks=callbacks,
            )
        else:
            raise

    hist = {k: list(map(float, v)) for k, v in history.history.items()}
    log(f"Training finished. Last val metrics: { {k: v[-1] for k, v in hist.items() if v} }")

    return TrainResult(model=model, history=hist, task_type=prepared.task_type, feature_names=prepared.feature_names)


def save_trained_model(res: TrainResult, path: str, model_type: str, log: LogFn) -> str:
    if res.model is None:
        raise ValueError("No trained model to save")
    mt = model_type.lower()
    if mt == "keras" or "tensorflow" in str(type(res.model)).lower():
        # Save Keras model
        try:
            res.model.save(path)  # .h5 or .keras
        except Exception:
            # try h5 format
            if not path.lower().endswith(".h5"):
                path = path + ".h5"
            res.model.save(path)
        log(f"Saved Keras model: {path}")
        return path
    else:
        import joblib
        joblib.dump(res.model, path)
        log(f"Saved model: {path}")
        return path


# Optional: basic Optuna tuning for RF/XGB

def tune_hyperparams(
    prepared: PreparedData,
    model_type: str,
    n_trials: int,
    log: LogFn,
    progress: ProgressFn | None = None,
    stop_flag: Optional[list[bool]] = None,
    per_trial_timeout_sec: Optional[int] = None,
) -> tuple[dict, float]:
    import optuna
    from sklearn.metrics import f1_score, mean_squared_error
    from sklearn.model_selection import train_test_split

    X, y = prepared.X_train, prepared.y_train
    # Ensure 1D target for splitting/metrics
    y = np.asarray(y).ravel()
    task = prepared.task_type

    def objective(trial: optuna.Trial) -> float:
        # Stop if user requested
        if stop_flag and stop_flag[0]:
            raise optuna.exceptions.TrialPruned("Stopped by user")
        t_start = time.time()
        Xtr, Xval, ytr, yval = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=prepared.label_encoder is not None and 42 or 42,
            stratify=y if task == "classification" else None,
        )
        if model_type.lower() == "randomforest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            n_estimators = trial.suggest_int("n_estimators", 100, 600)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            if task == "classification":
                m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
                m.fit(Xtr, ytr)
                pred = m.predict(Xval)
                if per_trial_timeout_sec is not None and (time.time() - t_start) > per_trial_timeout_sec:
                    raise optuna.exceptions.TrialPruned("Trial timed out")
                return float(f1_score(yval, pred, average="macro"))
            else:
                m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
                m.fit(Xtr, ytr)
                pred = m.predict(Xval)
                rmse = math.sqrt(mean_squared_error(yval, pred))
                if per_trial_timeout_sec is not None and (time.time() - t_start) > per_trial_timeout_sec:
                    raise optuna.exceptions.TrialPruned("Trial timed out")
                return -float(rmse)
        elif model_type.lower() == "xgboost":
            import xgboost as xgb
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
            }
            if task == "classification":
                # Auto device: GPU if present else CPU (version-safe)
                extra = auto_xgb_params(None, log)
                try:
                    m = xgb.XGBClassifier(eval_metric="logloss", **extra, **params)
                except TypeError:
                    # Remove unsupported keys progressively
                    tmp = dict(extra)
                    tmp.pop("device", None)
                    try:
                        m = xgb.XGBClassifier(eval_metric="logloss", **tmp, **params)
                    except TypeError:
                        tmp.pop("predictor", None)
                        m = xgb.XGBClassifier(eval_metric="logloss", **tmp, **params)
                # Early stopping on validation set when possible
                try:
                    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False, early_stopping_rounds=20)
                except TypeError:
                    try:
                        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
                    except Exception:
                        m.fit(Xtr, ytr)
                pred = m.predict(Xval)
                if per_trial_timeout_sec is not None and (time.time() - t_start) > per_trial_timeout_sec:
                    raise optuna.exceptions.TrialPruned("Trial timed out")
                return float(f1_score(yval, pred, average="macro"))
            else:
                extra = auto_xgb_params(None, log)
                try:
                    m = xgb.XGBRegressor(eval_metric="rmse", **extra, **params)
                except TypeError:
                    tmp = dict(extra)
                    tmp.pop("device", None)
                    try:
                        m = xgb.XGBRegressor(eval_metric="rmse", **tmp, **params)
                    except TypeError:
                        tmp.pop("predictor", None)
                        m = xgb.XGBRegressor(eval_metric="rmse", **tmp, **params)
                # Early stopping on validation set when possible
                try:
                    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False, early_stopping_rounds=20)
                except TypeError:
                    try:
                        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
                    except Exception:
                        m.fit(Xtr, ytr)
                pred = m.predict(Xval)
                rmse = math.sqrt(mean_squared_error(yval, pred))
                if per_trial_timeout_sec is not None and (time.time() - t_start) > per_trial_timeout_sec:
                    raise optuna.exceptions.TrialPruned("Trial timed out")
                return -float(rmse)
        elif model_type.lower() == "lightgbm":
            import lightgbm as lgb
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
            }
            if task == "classification":
                extra = auto_lgbm_params(None, log)
                m = lgb.LGBMClassifier(**{**params, **extra})
                try:
                    cb = [lgb.early_stopping(20, verbose=False)]
                    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=cb)
                except Exception:
                    m.fit(Xtr, ytr)
                pred = m.predict(Xval)
                if per_trial_timeout_sec is not None and (time.time() - t_start) > per_trial_timeout_sec:
                    raise optuna.exceptions.TrialPruned("Trial timed out")
                return float(f1_score(yval, pred, average="macro"))
            else:
                extra = auto_lgbm_params(None, log)
                m = lgb.LGBMRegressor(**{**params, **extra})
                try:
                    cb = [lgb.early_stopping(20, verbose=False)]
                    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=cb)
                except Exception:
                    m.fit(Xtr, ytr)
                pred = m.predict(Xval)
                rmse = math.sqrt(mean_squared_error(yval, pred))
                if per_trial_timeout_sec is not None and (time.time() - t_start) > per_trial_timeout_sec:
                    raise optuna.exceptions.TrialPruned("Trial timed out")
                return -float(rmse)
        else:
            raise optuna.exceptions.TrialPruned("Unsupported model for tuning")

        # Per-trial timeout (checked post-fit; best effort)
        if per_trial_timeout_sec is not None and (time.time() - t_start) > per_trial_timeout_sec:
            raise optuna.exceptions.TrialPruned("Trial timed out")

    direction = "maximize" if prepared.task_type == "classification" else "minimize"
    if direction == "minimize":
        # our objective returns negative RMSE to maximize
        direction = "maximize"

    from optuna.samplers import TPESampler
    study = optuna.create_study(direction=direction, sampler=TPESampler(seed=42))
    log(f"Starting Optuna tuning for {model_type} ({n_trials} trials)...")

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # Report progress
        if progress:
            p = min((trial.number + 1) / max(n_trials, 1), 1.0)
            try:
                best_val = float(study.best_value)
            except Exception:
                best_val = float('nan')
            progress(p, f"Tuning {model_type}: trial {trial.number + 1}/{n_trials}, best={best_val:.4f}")
        # Stop on user request
        if stop_flag and stop_flag[0]:
            log("Tuning stop requested — stopping study.")
            study.stop()

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])

    best_params = study.best_params
    best_value = float(study.best_value)
    log(f"Best params: {best_params}, best score: {best_value:.4f}")
    return best_params, best_value
