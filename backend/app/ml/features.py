"""
STEP 2 of the ML pipeline: feature engineering.

The raw dataset already has, per timestep: sensed_<channel>,
expected_<channel> (the clean twin's prediction for the same commanded
inputs), and residual_<channel> = sensed - expected.

Two reasons rolling-window features are added on top of the raw
per-timestep values:

1. Single-sample noise. The physics model + faults include some
   step-to-step variability (dither in the commanded inputs, the
   stochastic misfire kick, etc.). A short window's mean/std separates a
   real sustained shift from single-sample noise.

2. Trend information. "Is CHT residual rising over the last N samples"
   is a much stronger degradation signal than "what is CHT residual right
   now" — this is the whole point of trending toward a RUL estimate rather
   than reacting to instantaneous threshold crossings (the "reactive,
   threshold-based" limitation called out in the problem statement).

Rolling stats are computed PER run_id (a run boundary must never leak
into another run's window) using a fixed sample window, not a fixed time
window, because the dataset's dt is constant (2s) within a run.
"""

import numpy as np
import pandas as pd

from .dataset_builder import CHANNELS

WINDOW = 15  # samples (~30s at dt=2s)

# Channels we build rolling residual/sensed features for. Keep this list
# focused on what actually helps each fault type (see fault_injection.py) —
# every channel would work, but a smaller feature set trains faster and is
# easier to inspect for feature importance.
FEATURE_CHANNELS = [
    "rpm",
    "cht_K",
    "egt_K",
    "oil_temp_K",
    "oil_pressure_Pa",
    "vibration_g",
    "fuel_mass_flow_kg_s",
    "brake_power_W",
]


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["run_id", "t_s"]).reset_index(drop=True)
    grouped = df.groupby("run_id", sort=False)

    new_cols = {}
    for ch in FEATURE_CHANNELS:
        res_col = f"residual_{ch}"
        sen_col = f"sensed_{ch}"

        roll_res = grouped[res_col].rolling(WINDOW, min_periods=1)
        new_cols[f"roll_mean_residual_{ch}"] = roll_res.mean().reset_index(level=0, drop=True)
        new_cols[f"roll_std_residual_{ch}"] = roll_res.std().reset_index(level=0, drop=True).fillna(0.0)

        # rate of change of the residual over the window — the primary
        # trend signal for the RUL regressor
        roll_min = grouped[res_col].rolling(WINDOW, min_periods=1).min().reset_index(level=0, drop=True)
        roll_max = grouped[res_col].rolling(WINDOW, min_periods=1).max().reset_index(level=0, drop=True)
        new_cols[f"roll_slope_residual_{ch}"] = (df[res_col] - grouped[res_col].shift(WINDOW - 1)) / WINDOW

        roll_sen = grouped[sen_col].rolling(WINDOW, min_periods=1)
        new_cols[f"roll_std_sensed_{ch}"] = roll_sen.std().reset_index(level=0, drop=True).fillna(0.0)

    feat_df = pd.DataFrame(new_cols, index=df.index)
    feat_df = feat_df.fillna(0.0)
    out = pd.concat([df, feat_df], axis=1)
    return out


def feature_columns() -> list[str]:
    """The full list of model-input feature column names (raw + rolling).
    Deliberately EXCLUDES: fault_type, severity, health_score, health_status,
    RUL_s, reaches_failure, run_id, t_s — those are labels/identifiers, not
    inputs a real monitoring system would use to predict itself.
    """
    cols = ["throttle", "altitude_m", "airspeed_m_s", "ambient_temp_offset_K", "load_factor"]
    for ch in CHANNELS:
        cols += [f"sensed_{ch}", f"residual_{ch}"]
    for ch in FEATURE_CHANNELS:
        cols += [
            f"roll_mean_residual_{ch}",
            f"roll_std_residual_{ch}",
            f"roll_slope_residual_{ch}",
            f"roll_std_sensed_{ch}",
        ]
    return cols
