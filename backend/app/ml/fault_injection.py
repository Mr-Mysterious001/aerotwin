"""
Fault / degradation injection for synthetic dataset generation.

Core idea (this IS the "digital twin comparison" the project asks for):
for every simulated mission we run TWO copies of the physics engine in
lockstep, fed the identical pilot-commanded inputs (throttle/altitude/
airspeed/load):

  - "expected" twin : the clean AeroPistonEngine, no faults — what a
                       healthy engine should be doing right now, for
                       these exact commanded inputs.
  - "sensed" twin    : the same engine with a fault/degradation injected
                       — either into its physical parameters (real
                       mechanical degradation) or into its measurement
                       layer only (sensor faults, e.g. drift) — standing
                       in for the telemetry actually coming off the
                       aircraft.

residual(channel) = sensed(channel) - expected(channel)

That residual stream, not the raw sensed values alone, is the primary
signal fed to the ML layer: it isolates "how far is reality from what
physics says should be happening right now" independent of whatever the
pilot happens to be commanding at that moment. This mirrors real
gas-path-analysis / residual-based diagnostics used in aircraft engine
health management programs.

FAULT TYPES implemented (matching the problem statement's list):
  misfire_conditions, injector_abnormalities, cooling_degradation,
  lubrication_issues, sensor_drift, combustion_instability,
  overheating_trend, abnormal_vibration, none (nominal)

All severity/progression shapes below are ASSUMED — built to generate
plausible, exactly-labeled run-to-failure training data. They are not
measured failure-mode data from a real engine test program.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Optional

from ..engine_model import AeroPistonEngine, EngineConfig, EngineInputs, EngineState

FAULT_TYPES = [
    "none",
    "misfire_conditions",
    "injector_abnormalities",
    "cooling_degradation",
    "lubrication_issues",
    "sensor_drift",
    "combustion_instability",
    "overheating_trend",
    "abnormal_vibration",
]

# Which sensed channel(s) sensor_drift can corrupt — picked per-run.
DRIFT_CHANNELS = ["cht_K", "egt_K", "oil_pressure_Pa", "oil_temp_K"]


@dataclass
class FaultScenario:
    fault_type: str = "none"
    onset_s: float = 0.0            # when degradation begins
    ramp_s: float = 600.0           # time to reach full severity (severity=1.0)
    max_severity: float = 1.0       # cap below 1.0 for a "near miss" / non-failure run
    rng_seed: int = 0
    drift_channel: str = "cht_K"    # sensor_drift only
    drift_sign: float = 1.0         # sensor_drift / injector_abnormalities direction

    def severity(self, t_s: float) -> float:
        if t_s <= self.onset_s:
            return 0.0
        frac = min(1.0, (t_s - self.onset_s) / max(self.ramp_s, 1e-3))
        return frac * self.max_severity


def degraded_config(base_cfg: EngineConfig, scenario: FaultScenario, severity: float) -> EngineConfig:
    """Copy of base_cfg with physical-degradation faults applied at the
    given severity. Sensor-only faults are handled separately (post-hoc,
    on the sensed state) since a broken sensor doesn't change the engine.
    """
    cfg = copy.deepcopy(base_cfg)
    ft = scenario.fault_type
    if ft == "cooling_degradation":
        # Scale BOTH the natural-convection and forced (airspeed) cooling
        # terms — the airspeed term dominates total h at cruise speeds, so
        # scaling only the base term (as originally implemented) left
        # total cooling almost unchanged and the fault had no visible effect.
        factor = 1.0 - 0.65 * severity
        cfg.head_htc_base_W_per_K *= factor
        cfg.head_htc_per_airspeed *= factor
        cfg.oil_htc_base_W_per_K *= factor
        cfg.oil_htc_per_airspeed *= factor
    elif ft == "overheating_trend":
        # narrower fault: cooling-fan-type loss on the head circuit only
        factor = 1.0 - 0.75 * severity
        cfg.head_htc_base_W_per_K *= factor
        cfg.head_htc_per_airspeed *= factor
    elif ft == "lubrication_issues":
        cfg.oil_pump_gain_Pa_per_rpm_per_cP *= (1.0 - 0.7 * severity)
        cfg.fmep_base_Pa *= (1.0 + 0.5 * severity)
    elif ft == "combustion_instability":
        # torque_fluct_gain alone is too small a contributor next to
        # vibration_base_g to move the needle (checked empirically); add a
        # direct vibration contribution for rough-running combustion, on
        # top of the torque-fluctuation and efficiency-loss effects, at a
        # lower magnitude than the abnormal_vibration fault so the two
        # remain distinguishable to the classifier.
        cfg.vibration_torque_fluct_gain *= (1.0 + 3.0 * severity)
        cfg.otto_realization_factor *= (1.0 - 0.15 * severity)
        cfg.vibration_base_g += 0.42 * severity
    elif ft == "abnormal_vibration":
        cfg.vibration_base_g += 0.5 * severity
    elif ft == "misfire_conditions":
        # baseline combustion efficiency droop that worsens with severity;
        # the intermittent per-event drop + vibration kick is applied
        # separately, per-step, in run_twin_pair
        cfg.combustion_efficiency *= (1.0 - 0.10 * severity)
    return cfg


def sensed_inputs_for_fault(base_inputs: EngineInputs, scenario: FaultScenario, severity: float, rng: random.Random) -> EngineInputs:
    """Input-level fault effects (things that change what enters the
    cylinders, not the hardware itself)."""
    inp = copy.copy(base_inputs)
    if scenario.fault_type == "injector_abnormalities" and severity > 0:
        inp.afr_target = (inp.afr_target or 14.7) + scenario.drift_sign * 4.0 * severity
    return inp


def apply_sensor_fault(state: EngineState, scenario: FaultScenario, severity: float) -> EngineState:
    """Broken-sensor fault: engine is fine, reported value is wrong."""
    if scenario.fault_type != "sensor_drift" or severity <= 0:
        return state
    s = copy.copy(state)
    drift_magnitude = {
        "cht_K": 90.0,
        "egt_K": 130.0,
        "oil_pressure_Pa": 220_000.0,
        "oil_temp_K": 70.0,
    }.get(scenario.drift_channel, 50.0)
    delta = scenario.drift_sign * drift_magnitude * severity
    setattr(s, scenario.drift_channel, getattr(s, scenario.drift_channel) + delta)
    return s


def run_twin_pair(operating_points, scenario: FaultScenario, base_cfg: Optional[EngineConfig] = None):
    """Runs the expected (clean) and sensed (faulted) twins in lockstep.

    Returns a list of per-timestep dicts containing both twins' full
    state, the operating inputs, and the fault label/severity — this is
    the row format the dataset generator writes to disk.
    """
    base_cfg = base_cfg or EngineConfig()
    expected_engine = AeroPistonEngine(copy.deepcopy(base_cfg))
    sensed_engine = AeroPistonEngine(copy.deepcopy(base_cfg))
    rng = random.Random(scenario.rng_seed)

    records = []
    prev_t = operating_points[0].t_s
    for i, op in enumerate(operating_points):
        dt = (op.t_s - prev_t) if i > 0 else 1.0
        dt = max(dt, 1e-3)
        prev_t = op.t_s

        severity = scenario.severity(op.t_s)

        base_inputs = EngineInputs(
            throttle=op.throttle,
            altitude_m=op.altitude_m,
            airspeed_m_s=op.airspeed_m_s,
            ambient_temp_offset_K=op.ambient_temp_offset_K,
            load_factor=op.load_factor,
        )

        # --- expected/clean twin ---
        expected_state = expected_engine.step(copy.copy(base_inputs), dt)

        # --- sensed/faulted twin: physical config degradation this step ---
        sensed_engine.cfg = degraded_config(base_cfg, scenario, severity)
        sensed_inputs = sensed_inputs_for_fault(base_inputs, scenario, severity, rng)

        # misfire: intermittent single-step combustion dropout on top of
        # the baseline droop already in degraded_config. A real misfire
        # causes a torque-impulse imbalance between cylinders — a sharp
        # vibration KICK, not a dip. The engine model's vibration term is
        # tied to average torque magnitude, so a lower-torque step from a
        # dropped cylinder would (wrongly) show LOWER vibration; that kick
        # is added explicitly here instead of relying on the torque term.
        misfire_kick = 0.0
        if scenario.fault_type == "misfire_conditions" and severity > 0 and rng.random() < 0.30 * severity:
            sensed_engine.cfg = copy.deepcopy(sensed_engine.cfg)
            sensed_engine.cfg.combustion_efficiency *= 0.55
            misfire_kick = 0.35 * severity

        sensed_state = sensed_engine.step(sensed_inputs, dt)
        if misfire_kick > 0:
            sensed_state = copy.copy(sensed_state)
            sensed_state.vibration_g += misfire_kick
        sensed_state = apply_sensor_fault(sensed_state, scenario, severity)

        records.append(
            {
                "t_s": op.t_s,
                "fault_type": scenario.fault_type,
                "severity": severity,
                "throttle": op.throttle,
                "altitude_m": op.altitude_m,
                "airspeed_m_s": op.airspeed_m_s,
                "ambient_temp_offset_K": op.ambient_temp_offset_K,
                "load_factor": op.load_factor,
                "expected": expected_state,
                "sensed": sensed_state,
            }
        )

    return records
