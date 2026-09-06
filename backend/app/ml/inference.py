"""
STEP 4 of the ML pipeline: live inference on the interactive dashboard's
running twin pair.

Deliberately reuses the EXACT SAME feature-construction code
(features.add_rolling_features / feature_columns) that trained the
models, rather than re-deriving similar-looking features by hand. This
matters: a common way ML systems silently break in production is
"training/serving skew" — the live feature pipeline drifting out of sync
with the one used to train the model. Sharing the code outright is the
simplest way to make that class of bug structurally impossible here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .dataset_builder import CHANNELS
from .features import add_rolling_features, feature_columns, WINDOW
from ..engine_model import AeroPistonEngine, EngineInputs
from ..health import compute_health

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class MLModels:
    """Loads the three trained artifacts once and caches them."""

    _instance = None

    def __init__(self):
        self.classifier = joblib.load(MODEL_DIR / "fault_classifier.joblib")
        self.rul_regressor = joblib.load(MODEL_DIR / "rul_regressor.joblib")
        self.autoencoder = joblib.load(MODEL_DIR / "anomaly_autoencoder.joblib")
        self.scaler = joblib.load(MODEL_DIR / "anomaly_scaler.joblib")
        with open(MODEL_DIR / "metadata.json") as f:
            self.meta = json.load(f)
        self.feat_cols = self.meta["feature_columns"]
        self.anomaly_threshold = self.meta["anomaly_threshold"]

    @classmethod
    def get(cls) -> "MLModels":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def buffer_to_dataframe(buffer: list[dict]) -> pd.DataFrame:
    """buffer: list of {run_id, t_s, fault_type, severity, throttle,
    altitude_m, airspeed_m_s, ambient_temp_offset_K, load_factor,
    expected: EngineState, sensed: EngineState} — same shape produced by
    fault_injection.run_twin_pair, so dataset_builder.build_rows works
    unmodified on live data too.
    """
    from .dataset_builder import build_rows

    rows, _ = build_rows(buffer, run_id=0)
    return pd.DataFrame(rows)


def analyze(buffer: list[dict]) -> dict:
    """Runs all three models on the most recent sample of the buffer.
    Returns None-valued fields if there isn't enough history yet (the
    rolling window needs WINDOW samples to mean anything).
    """
    if len(buffer) < 3:
        return {"ready": False, "reason": f"need at least 3 samples, have {len(buffer)}"}

    models = MLModels.get()
    df = buffer_to_dataframe(buffer)
    df["run_id"] = 0
    df = add_rolling_features(df)
    last = df.iloc[[-1]]
    X = last[models.feat_cols].values

    # --- anomaly score ---
    X_s = models.scaler.transform(X)
    recon = models.autoencoder.predict(X_s)
    recon_error = float(np.mean((X_s - recon) ** 2))
    is_anomaly = recon_error > models.anomaly_threshold

    # --- fault classification ---
    proba = models.classifier.predict_proba(X)[0]
    classes = models.classifier.classes_
    top_idx = int(np.argmax(proba))
    predicted_fault = str(classes[top_idx])
    fault_probs = sorted(
        [{"fault_type": str(c), "probability": float(p)} for c, p in zip(classes, proba)],
        key=lambda d: -d["probability"],
    )

    # --- RUL (only meaningful if a degrading fault is actually suspected) ---
    predicted_rul_s = None
    if predicted_fault != "none" or is_anomaly:
        predicted_rul_s = float(models.rul_regressor.predict(X)[0])

    # --- current rule-based health, for side-by-side comparison ---
    sensed_state = buffer[-1]["sensed"]
    indices, health_score, health_status = compute_health(sensed_state)

    return {
        "ready": True,
        "t_s": buffer[-1]["t_s"],
        "anomaly_score": recon_error,
        "anomaly_threshold": models.anomaly_threshold,
        "is_anomaly": bool(is_anomaly),
        "predicted_fault_type": predicted_fault,
        "fault_probabilities": fault_probs[:4],
        "predicted_rul_s": predicted_rul_s,
        "rule_based_health_status": health_status,
        "rule_based_health_score": health_score,
    }


def forecast_if_continued(expected_engine: AeroPistonEngine, last_inputs: EngineInputs, horizon_s: float, dt_s: float = 2.0):
    """Projects the CLEAN twin forward with the current commanded inputs
    held constant — "what the engine should read if nothing changes and
    nothing is wrong". Operates on a deep copy so it never disturbs the
    live simulation.
    """
    sim = copy.deepcopy(expected_engine)
    t0 = sim.state.t_s
    points = []
    n_steps = max(1, int(horizon_s / dt_s))
    for i in range(n_steps):
        s = sim.step(copy.copy(last_inputs), dt_s)
        points.append(
            {
                "t_s": s.t_s,
                "rpm": s.rpm,
                "cht_C": s.cht_K - 273.15,
                "egt_C": s.egt_K - 273.15,
                "oil_temp_C": s.oil_temp_K - 273.15,
                "oil_pressure_bar": s.oil_pressure_Pa / 1e5,
            }
        )
    return points
