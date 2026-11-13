"""Training/evaluation boundary for CaRSEL.

    Source data -> outer heldout TEST (untouched)
                -> DEV -> FIT / SELECT
                         FIT: candidate OOF selection and residual construction
                         SELECT: confidence threshold selection only
    TEST scores are reported once after the complete model is frozen.

No in-sample TP/TN transfer is used as a heldout accuracy estimate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .model import CaRSELClassifier, Stage, _logit
from .schema import CATEGORICAL, FEATURES, NUMERIC


@dataclass
class TrainingSettings:
    target: str = "cardio"
    seed: int = 42
    test_fraction: float = 0.20
    selection_fraction: float = 0.25  # of the development subset
    max_layers: int = 3
    min_residual: int = 30
    max_exit_error: float = 0.12  # empirical dev selection error, not a clinical guarantee

    def validate(self):
        if not 0.1 <= self.test_fraction <= 0.35:
            raise ValueError("test_fraction must be between 0.10 and 0.35")
        if not 0.1 <= self.selection_fraction <= 0.4:
            raise ValueError("selection_fraction must be between 0.10 and 0.40")
        if not 1 <= self.max_layers <= 5:
            raise ValueError("max_layers must be between 1 and 5")
        if not 12 <= self.min_residual <= 5000:
            raise ValueError("min_residual must be between 12 and 5000")
        if not 0 <= self.max_exit_error <= 0.5:
            raise ValueError("max_exit_error must be between 0 and 0.5")


def _preprocessor() -> ColumnTransformer:
    # All learned preprocessing is inside each estimator pipeline, and hence
    # fitted separately in every cross-validation fold.
    continuous = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    discrete = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("continuous", continuous, NUMERIC),
        ("discrete", discrete, CATEGORICAL),
    ], remainder="drop", sparse_threshold=0)


def _pipeline(name: str, seed: int, n_neighbors: int = 7) -> Pipeline:
    if name == "RF":
        clf = RandomForestClassifier(n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1)
    elif name == "NB":
        clf = GaussianNB(var_smoothing=1e-7)
    elif name == "GB":
        clf = GradientBoostingClassifier(n_estimators=65, learning_rate=0.08, max_depth=2,
                                         random_state=seed)
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=1)
    else:
        raise ValueError(f"Unknown learner '{name}'")
    return Pipeline([("preprocessing", _preprocessor()), ("classifier", clf)])


def _metrics(y: np.ndarray, positive_probability: np.ndarray) -> dict:
    predictions = (positive_probability >= 0.5).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, predictions, labels=[0, 1]).ravel())
    values = {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall_sensitivity": recall_score(y, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": f1_score(y, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y, predictions),
        "roc_auc": roc_auc_score(y, positive_probability) if len(set(y)) == 2 else None,
        "average_precision": average_precision_score(y, positive_probability) if len(set(y)) == 2 else None,
        "brier_score": brier_score_loss(y, positive_probability),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    return {key: (round(float(value), 5) if isinstance(value, (float, np.floating)) else value)
            for key, value in values.items()}


def _choose_gate_threshold(y: np.ndarray, p: np.ndarray, error_cap: float) -> tuple[float, int, float | None]:
    """Select high-confidence exit with error constraint on DEV selection only.

    Constraint is descriptive for this selection sample; no finite-sample or
    clinical-risk guarantee is implied. If impossible, route to next layer.
    """
    if not len(y):
        return 1.0, 0, None
    confidence = np.maximum(p, 1 - p)
    label_hat = (p >= 0.5).astype(int)
    min_exits = min(len(y), max(8, math.ceil(0.08 * len(y))))
    best = (1.0, 0, None)
    for threshold in np.unique(np.r_[np.arange(0.50, 1.0, 0.025), 0.999999]):
        exit_mask = confidence >= threshold
        num_exits = int(exit_mask.sum())
        if num_exits < min_exits:
            continue
        exit_error = float(np.mean(label_hat[exit_mask] != y[exit_mask]))
        if exit_error <= error_cap:
            if num_exits > best[1]:
                best = (float(threshold), num_exits, exit_error)
    return best


def train_carsel(df: pd.DataFrame, settings: TrainingSettings | None = None) -> tuple[CaRSELClassifier, dict]:
    """Return a frozen model and its *independent* holdout evaluation report."""
    started_at = time.perf_counter()
    config = settings or TrainingSettings()
    config.validate()
    if config.target not in df:
        raise ValueError(f"Missing target {config.target}")
    X = df[FEATURES].reset_index(drop=True)
    y = df[config.target].astype(int).to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Expected two target classes coded 0 and 1")
    ids = np.arange(len(df))
    dev_ids, test_ids = train_test_split(
        ids, test_size=config.test_fraction, random_state=config.seed, stratify=y
    )
    fit_ids, selection_ids = train_test_split(
        dev_ids, test_size=config.selection_fraction, random_state=config.seed + 1,
        stratify=y[dev_ids],
    )
    fit_X = X.iloc[fit_ids].reset_index(drop=True)
    fit_y = y[fit_ids]
    select_X = X.iloc[selection_ids].reset_index(drop=True)
    select_y = y[selection_ids]
    current = np.arange(len(fit_X))
    stages: list[Stage] = []
    last_error_ids = current
    stage_metadata = []

    for layer in range(config.max_layers):
        current_y = fit_y[current]
        values, counts = np.unique(current_y, return_counts=True)
        if len(current) < (config.min_residual if layer else 12) or len(values) != 2 or counts.min() < 3:
            break
        n_folds = min(3, int(counts.min()))
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=config.seed + layer * 101)
        chosen_name, chosen_oof, best_score = None, None, -1.0
        comparisons = {}
        for name in ("RF", "NB", "GB"):
            candidate = _pipeline(name, seed=config.seed + layer)
            oof = cross_val_predict(candidate, fit_X.iloc[current], current_y,
                                    cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            score = balanced_accuracy_score(current_y, (oof >= 0.5).astype(int))
            comparisons[name] = round(float(score), 5)
            if score > best_score:
                chosen_name, chosen_oof, best_score = name, oof, float(score)
        assert chosen_name is not None and chosen_oof is not None
        estimator = _pipeline(chosen_name, seed=config.seed + layer)
        estimator.fit(fit_X.iloc[current], current_y)
        # Platt/logistic calibration uses out-of-fold model predictions only.
        calibrator = LogisticRegression(solver="liblinear", random_state=config.seed)
        calibrator.fit(_logit(chosen_oof), current_y)
        errors = (chosen_oof >= 0.5).astype(int) != current_y
        last_error_ids = current[errors]
        stage = Stage(
            name=chosen_name, estimator=estimator, calibrator=calibrator,
            train_samples=int(len(current)), oof_errors=int(errors.sum()),
            oof_balanced_accuracy=best_score,
        )
        stages.append(stage)
        stage_metadata.append({"layer": layer + 1, "candidates_oof_balanced_accuracy": comparisons})
        current = last_error_ids

    if not stages:
        raise ValueError("Unable to train a valid stage; check class support")
    fallback_ids = last_error_ids
    fallback_source = "last_layer_oof_errors"
    classes, counts = np.unique(fit_y[fallback_ids], return_counts=True)
    if len(fallback_ids) < 12 or len(classes) != 2 or counts.min() < 3:
        fallback_ids = np.arange(len(fit_X))
        fallback_source = "full_fit_fallback_due_to_insufficient_residual_support"
    n_neighbors = min(7, len(fallback_ids))
    fallback = _pipeline("KNN", seed=config.seed, n_neighbors=n_neighbors)
    fallback.fit(fit_X.iloc[fallback_ids], fit_y[fallback_ids])
    model = CaRSELClassifier(
        stages=stages, fallback=fallback, fallback_train_samples=int(len(fallback_ids)),
        fallback_source=fallback_source,
    )

    # Gate thresholds are selected using development SELECT, never heldout TEST.
    active = np.ones(len(select_X), dtype=bool)
    for stage in stages:
        active_ids = np.flatnonzero(active)
        if not len(active_ids):
            break
        p = stage.predict_positive(select_X.iloc[active_ids])
        tau, exit_count, error = _choose_gate_threshold(select_y[active_ids], p,
                                                        config.max_exit_error)
        stage.threshold = tau
        stage.selection_exits = exit_count
        stage.selection_error_rate = round(error, 5) if error is not None else None
        exit_mask = np.maximum(p, 1 - p) >= tau
        active[active_ids[exit_mask]] = False

    # Heldout labels are accessed *only after* the entire cascade is frozen.
    inference_started_at = time.perf_counter()
    test_predictions = model.predict_with_trace(X.iloc[test_ids])
    inference_seconds = time.perf_counter() - inference_started_at
    test_proba = np.asarray([t["positive_probability"] for t in test_predictions])
    test_y = y[test_ids]
    routing_counts = pd.Series([t["decision_layer"] for t in test_predictions]).value_counts().to_dict()
    # Aggregated, non-record-level diagnostic curves. Calculated only on the
    # untouched outer holdout after the fitted routing policy is frozen.
    fpr, tpr, _ = roc_curve(test_y, test_proba)
    pr_precision, pr_recall, _ = precision_recall_curve(test_y, test_proba)
    def thin(x, y, count=100):
        ix = np.unique(np.linspace(0, len(x) - 1, min(count, len(x)), dtype=int))
        return [[round(float(x[i]), 5), round(float(y[i]), 5)] for i in ix]
    bins = np.linspace(0, 1, 11)
    bin_ids = np.clip(np.digitize(test_proba, bins) - 1, 0, 9)
    calibration = []
    for b in range(10):
        mask = bin_ids == b
        if mask.any():
            calibration.append([round(float(test_proba[mask].mean()), 5),
                                round(float(test_y[mask].mean()), 5), int(mask.sum())])
    report = {
        "visualization": {
            "roc": thin(fpr, tpr),
            "precision_recall": thin(pr_recall[::-1], pr_precision[::-1]),
            "calibration": calibration,
            "test_positive_rate": round(float(test_y.mean()), 5),
        },
        "software": "CaRSEL-CVD",
        "version": "0.2.0",
        "protocol": "stratified outer holdout; OOF candidate/residual training; development-only gate tuning",
        "settings": asdict(config),
        "dataset": {
            "total_rows": int(len(X)),
            "positive_class_n": int(y.sum()),
            "negative_class_n": int(len(y) - y.sum()),
            "fit_rows": int(len(fit_ids)),
            "selection_rows": int(len(selection_ids)),
            "untouched_test_rows": int(len(test_ids)),
        },
        "holdout_metrics": _metrics(test_y, test_proba),
        "holdout_routing_counts": {str(k): int(v) for k, v in routing_counts.items()},
        "candidate_oof_diagnostics": stage_metadata,
        "model": model.model_summary(),
        "runtime": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "end_to_end_train_and_evaluation_seconds": round(time.perf_counter() - started_at, 4),
            "holdout_inference_seconds": round(inference_seconds, 4),
            "holdout_inference_ms_per_record": round(inference_seconds * 1000 / len(test_ids), 4),
        },
        "limitations": [
            "This software is for research only, not medical diagnosis or clinical triage.",
            "Reported metrics apply only to the uploaded dataset and the stated split.",
            "Holdout is a single split; external and multi-seed validation are still needed.",
            "Gate threshold error constraints on development data are not clinical guarantees.",
            "Class 1 is assumed to mean the positive outcome; check the data dictionary.",
        ],
    }
    return model, report
