"""The saved model bundle: estimator + everything needed to use it on new data.

The API, the training script and (later) the maintenance copilot all go
through this one class, so the feature logic can't drift between them.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, RISK_HIGH, RISK_MEDIUM, RUL_CAP, SENSOR_NAMES, SHORT_WINDOW
from .features import build_features, feature_columns, sensor_of


def nasa_score(y_true, y_pred) -> float:
    """Scoring function from the PHM08 challenge (lower is better).

    Late predictions (saying the engine has more life than it really does)
    cost more than early ones, because a late call means a missed repair.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def risk_level(rul: float) -> str:
    if rul <= RISK_HIGH:
        return "high"
    if rul <= RISK_MEDIUM:
        return "medium"
    return "low"


@dataclass
class RULModel:
    estimator: object
    sensors: list[str]
    healthy_reference: dict[str, float]
    name: str = "gradient_boosting"
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def feature_cols(self) -> list[str]:
        return feature_columns(self.sensors)

    # ---- prediction -------------------------------------------------------
    def latest_features(self, readings: pd.DataFrame) -> pd.DataFrame:
        """Features for the most recent cycle of each engine in `readings`."""
        if "unit" not in readings:
            readings = readings.assign(unit=0)
        feats = build_features(readings, self.sensors)
        last_idx = readings.sort_values(["unit", "cycle"]).groupby("unit").tail(1).index
        return feats.loc[last_idx, self.feature_cols]

    def predict_features(self, X: pd.DataFrame) -> np.ndarray:
        # The model was trained on capped labels, so keep outputs in 0..cap.
        return np.clip(self.estimator.predict(X), 0, RUL_CAP)

    def predict(self, readings: pd.DataFrame) -> np.ndarray:
        return self.predict_features(self.latest_features(readings))

    # ---- explanation ------------------------------------------------------
    def explain_features(self, x: pd.DataFrame) -> list[dict]:
        """How much each sensor moves this prediction, in cycles.

        For one sensor at a time, we swap its features for the values a
        healthy, early-life engine typically shows and re-predict. The change
        is that sensor's impact: a positive number means "this sensor's
        current state is costing the engine that many cycles of life".
        It's a simple what-if, so impacts don't add up exactly to the total.
        """
        base = float(self.predict_features(x)[0])
        groups: dict[str, list[str]] = {}
        for col in self.feature_cols:
            groups.setdefault(sensor_of(col), []).append(col)

        rows = []
        for sensor, cols in groups.items():
            what_if = x.copy()
            for c in cols:
                what_if[c] = self.healthy_reference[c]
            impact = float(self.predict_features(what_if)[0]) - base
            label = "operating age (cycles run)" if sensor == "cycle" else SENSOR_NAMES[sensor]
            rows.append(
                {
                    "feature": sensor,
                    "description": label,
                    "impact_cycles": round(impact, 1),
                    "current_value": round(float(x[sensor].iloc[0]), 4),
                    "healthy_value": round(float(self.healthy_reference[sensor]), 4),
                    "trend_per_cycle": None
                    if sensor == "cycle"
                    else round(float(x[f"{sensor}_slope{SHORT_WINDOW}"].iloc[0]), 5),
                }
            )
        rows.sort(key=lambda r: abs(r["impact_cycles"]), reverse=True)
        return rows

    # ---- persistence ------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "RULModel":
        return joblib.load(path)
