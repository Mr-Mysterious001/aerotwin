"""
Turns a raw twin-pair run (list of per-timestep expected/sensed state
dicts from fault_injection.run_twin_pair) into flat, labeled rows ready
for a dataframe: sensed values, expected values, residuals (sensed -
expected), the current rule-based health status/score (from health.py,
computed on the *sensed* state — exactly what a real monitoring system
would see), and — for runs that actually reach a failure threshold — a
Remaining-Useful-Life (RUL) label in seconds.

RUL definition: "failure" = the rule-based health status is "warning"
for a sustained debounce window (avoids labeling single-sample noise as
failure). RUL(t) = time_to_that_point, clipped at 0 after the point.
Runs that never reach "warning" get RUL_s = None (right-censored — used
for classification/anomaly training, excluded from RUL regression).
"""

from ..health import compute_health

CHANNELS = [
    "rpm",
    "cht_K",
    "egt_K",
    "oil_temp_K",
    "oil_pressure_Pa",
    "vibration_g",
    "fuel_mass_flow_kg_s",
    "brake_power_W",
    "alternator_voltage_V",
    "battery_soc_pct",
]


def build_rows(records, debounce_s: float = 6.0, run_id: int = 0):
    if not records:
        return [], None

    dt_s = records[1]["t_s"] - records[0]["t_s"] if len(records) > 1 else 1.0
    debounce_n = max(1, int(round(debounce_s / max(dt_s, 1e-6))))

    statuses = []
    for r in records:
        _, score, status = compute_health(r["sensed"])
        statuses.append((score, status))

    failure_idx = None
    for i in range(len(statuses) - debounce_n + 1):
        if all(statuses[j][1] == "warning" for j in range(i, i + debounce_n)):
            failure_idx = i
            break
    t_failure = records[failure_idx]["t_s"] if failure_idx is not None else None

    rows = []
    for i, r in enumerate(records):
        score, status = statuses[i]
        exp, sen = r["expected"], r["sensed"]
        row = {
            "run_id": run_id,
            "t_s": r["t_s"],
            "fault_type": r["fault_type"],
            "severity": r["severity"],
            "throttle": r["throttle"],
            "altitude_m": r["altitude_m"],
            "airspeed_m_s": r["airspeed_m_s"],
            "ambient_temp_offset_K": r["ambient_temp_offset_K"],
            "load_factor": r["load_factor"],
            "health_score": score,
            "health_status": status,
        }
        for ch in CHANNELS:
            sv, ev = getattr(sen, ch), getattr(exp, ch)
            row[f"sensed_{ch}"] = sv
            row[f"expected_{ch}"] = ev
            row[f"residual_{ch}"] = sv - ev
        if t_failure is not None:
            row["RUL_s"] = max(0.0, t_failure - r["t_s"])
            row["reaches_failure"] = True
        else:
            row["RUL_s"] = None
            row["reaches_failure"] = False
        rows.append(row)
    return rows, t_failure
