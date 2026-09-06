"""
Threshold-based health indexing.

This intentionally mirrors the "conventional, reactive" monitoring called
out in the problem statement as the baseline being improved upon. It is
the placeholder for the AI/ML anomaly-detection & RUL layer that will
replace/augment it in the next project phase. Thresholds below are
ASSUMED operating limits (typical small air-cooled SI aero-engine ranges)
and must be replaced with the actual engine's flight manual limits.
"""

from dataclasses import dataclass
from .engine_model import EngineState


@dataclass
class Limits:
    # NOTE: same issue as EGT above — this model's own nominal cruise CHT
    # (~55-90C, a function of the ASSUMED thermal-mass/htc coefficients in
    # EngineConfig, not a real airframe) runs cooler than typical real
    # air-cooled aero-engine CHT (which is why 210/232C, borrowed from real
    # type-certificate limits, never got approached even under fault
    # conditions). Recalibrated to this engine's own nominal range instead.
    cht_caution_K: float = 403.15   # 130 C
    cht_warning_K: float = 423.15   # 150 C
    # NOTE: these were originally set from generic small-SI-engine literature
    # figures independent of app.engine_model's own defaults, and turned out
    # to sit *below* this engine's normal full-throttle/climb EGT (~840-900C
    # from AeroPistonEngine's own assumed exhaust-energy-fraction curve),
    # which made ordinary climb-power operation false-trip "warning" in the
    # synthetic dataset (caught via generate_dataset.py's failure-rate sanity
    # check: 100% of nominal, fault-free runs were "reaching failure").
    # Recalibrated to sit above this specific model's own nominal high-power
    # EGT with real margin, instead of an independent literature figure.
    egt_caution_K: float = 1173.15  # 900 C
    egt_warning_K: float = 1223.15  # 950 C
    oil_temp_caution_K: float = 363.15  # 90 C — see CHT note above
    oil_temp_warning_K: float = 378.15  # 105 C
    oil_pressure_min_caution_Pa: float = 150_000.0
    oil_pressure_min_warning_Pa: float = 100_000.0
    vibration_caution_g: float = 0.35
    vibration_warning_g: float = 0.55


def _status_from_normalized(n: float) -> str:
    if n >= 1.0:
        return "warning"
    if n >= 0.7:
        return "caution"
    return "nominal"


def compute_health(state: EngineState, limits: Limits | None = None):
    lim = limits or Limits()
    indices = []

    def add_upper(name, value, caution, warning):
        normalized = (value - caution) / max(warning - caution, 1e-6)
        normalized = max(0.0, normalized) if value > caution else (value / caution) * 0.7
        indices.append((name, value, normalized, _status_from_normalized(normalized)))

    def add_lower(name, value, caution, warning):
        # lower is worse (e.g. oil pressure)
        if value <= warning:
            normalized = 1.0 + (warning - value) / max(warning, 1e-6)
        elif value <= caution:
            normalized = 0.7 + 0.3 * (caution - value) / max(caution - warning, 1e-6)
        else:
            normalized = (caution / value) * 0.7 if value > 0 else 1.0
        indices.append((name, value, normalized, _status_from_normalized(normalized)))

    add_upper("CHT", state.cht_K, lim.cht_caution_K, lim.cht_warning_K)
    add_upper("EGT", state.egt_K, lim.egt_caution_K, lim.egt_warning_K)
    add_upper("Oil Temperature", state.oil_temp_K, lim.oil_temp_caution_K, lim.oil_temp_warning_K)
    add_lower("Oil Pressure", state.oil_pressure_Pa, lim.oil_pressure_min_caution_Pa, lim.oil_pressure_min_warning_Pa)
    add_upper("Vibration", state.vibration_g, lim.vibration_caution_g, lim.vibration_warning_g)

    worst_normalized = max(i[2] for i in indices)
    overall_score = max(0.0, 100.0 * (1.0 - min(worst_normalized, 1.0)))
    overall_status = _status_from_normalized(worst_normalized)

    return indices, overall_score, overall_status
