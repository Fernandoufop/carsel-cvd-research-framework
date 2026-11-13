"""Label-free inference through a frozen cascade; no inference path reads y."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .schema import FEATURES


def _logit(raw_positive: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(raw_positive, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class Stage:
    name: str
    estimator: Pipeline
    calibrator: LogisticRegression | None
    train_samples: int
    oof_errors: int
    oof_balanced_accuracy: float
    threshold: float = 1.0
    selection_exits: int = 0
    selection_error_rate: float | None = None

    def predict_positive(self, frame: pd.DataFrame) -> np.ndarray:
        p = self.estimator.predict_proba(frame[FEATURES])[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(_logit(p))[:, 1]
        return np.asarray(p, dtype=float)

    def summary(self, index: int) -> dict:
        return {
            "layer": index,
            "learner": self.name,
            "train_samples": self.train_samples,
            "oof_errors": self.oof_errors,
            "oof_balanced_accuracy": round(float(self.oof_balanced_accuracy), 4),
            "routing_threshold": round(float(self.threshold), 4),
            "selection_exits": self.selection_exits,
            "selection_error_rate": self.selection_error_rate,
            "probability_transform": "OOF logistic sigmoid calibration" if self.calibrator else "native",
        }


@dataclass
class CaRSELClassifier:
    """Research cascaded ensemble model.

    fit is implemented by training.train_carsel. The instantiated frozen model
    only accepts X, never ground truth. Stages exit on high confidence and
    the remaining records are handled by a pre-fitted KNN fallback.
    """
    stages: list[Stage]
    fallback: Pipeline
    fallback_train_samples: int
    fallback_source: str
    version: str = "0.2.0"
    classes_: np.ndarray = field(default_factory=lambda: np.array([0, 1]))

    def predict_with_trace(self, frame: pd.DataFrame) -> list[dict]:
        if not len(frame):
            return []
        X = frame[FEATURES].copy().reset_index(drop=True)
        active = np.ones(len(X), dtype=bool)
        predictions = np.zeros(len(X), dtype=int)
        probabilities = np.full(len(X), np.nan, dtype=float)
        decision_layers = [""] * len(X)
        trails: list[list[dict]] = [[] for _ in range(len(X))]

        for idx, stage in enumerate(self.stages, start=1):
            subset_ids = np.flatnonzero(active)
            if not len(subset_ids):
                break
            p = stage.predict_positive(X.iloc[subset_ids])
            confidence = np.maximum(p, 1 - p)
            accepted = confidence >= stage.threshold
            for j, rid in enumerate(subset_ids):
                trails[rid].append({
                    "layer": idx, "model": stage.name,
                    "positive_probability": round(float(p[j]), 6),
                    "confidence": round(float(confidence[j]), 6),
                    "threshold": round(float(stage.threshold), 4),
                    "action": "accepted" if accepted[j] else "forwarded",
                })
                if accepted[j]:
                    predictions[rid] = int(p[j] >= 0.5)
                    probabilities[rid] = float(p[j])
                    decision_layers[rid] = f"stage_{idx}"
                    active[rid] = False

        subset_ids = np.flatnonzero(active)
        if len(subset_ids):
            p = self.fallback.predict_proba(X.iloc[subset_ids])[:, 1]
            for j, rid in enumerate(subset_ids):
                probabilities[rid] = float(p[j])
                predictions[rid] = int(p[j] >= 0.5)
                decision_layers[rid] = "knn_fallback"
                trails[rid].append({
                    "layer": "fallback", "model": "KNN",
                    "positive_probability": round(float(p[j]), 6),
                    "confidence": round(float(max(p[j], 1 - p[j])), 6),
                    "threshold": None,
                    "action": "accepted",
                })
        assert np.isfinite(probabilities).all()
        return [
            {
                "predicted_class": int(predictions[i]),
                "positive_probability": round(float(probabilities[i]), 6),
                "decision_layer": decision_layers[i],
                "trace": trails[i],
            }
            for i in range(len(X))
        ]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([item["predicted_class"] for item in self.predict_with_trace(frame)], dtype=int)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = np.array([item["positive_probability"] for item in self.predict_with_trace(frame)])
        return np.column_stack([1 - p, p])

    def model_summary(self) -> dict:
        return {
            "layers": [stage.summary(i + 1) for i, stage in enumerate(self.stages)],
            "fallback": {"learner": "KNN", "train_samples": self.fallback_train_samples,
                         "origin": self.fallback_source},
            "features": list(FEATURES),
        }
