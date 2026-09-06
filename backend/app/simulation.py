import copy
import random
from collections import deque
from typing import Deque

from .engine_model import AeroPistonEngine, EngineConfig, EngineInputs, EngineState
from .ml.fault_injection import FaultScenario, degraded_config, sensed_inputs_for_fault, apply_sensor_fault

MAX_HISTORY = 20_000  # cap in-memory replay buffer
_RNG = random.Random(0)


class SimulationManager:
    """Owns a PAIRED twin (expected/clean + sensed/possibly-faulted) plus
    their shared time-history.

    Phase 1 (previous version of this file) ran a single engine instance.
    Phase 2 needs both twins live in the interactive dashboard too, not
    just in offline dataset generation, because the ML analysis endpoint
    (app.ml.inference.analyze) consumes exactly the same buffer shape
    fault_injection.run_twin_pair produces for training -- so the person
    using the dashboard can manually dial in a fault and watch the
    anomaly/classification/RUL outputs react to it live.
    """

    def __init__(self):
        self.base_cfg = EngineConfig()
        self.expected_engine = AeroPistonEngine(copy.deepcopy(self.base_cfg))
        self.sensed_engine = AeroPistonEngine(copy.deepcopy(self.base_cfg))
        self.history: Deque[dict] = deque(maxlen=MAX_HISTORY)

    def reset(self):
        self.expected_engine = AeroPistonEngine(copy.deepcopy(self.base_cfg))
        self.sensed_engine = AeroPistonEngine(copy.deepcopy(self.base_cfg))
        self.history.clear()

    def step(
        self,
        inputs: EngineInputs,
        dt_s: float,
        injected_fault_type: str = "none",
        injected_severity: float = 0.0,
    ) -> dict:
        # expected/clean twin: same commanded inputs, never faulted
        expected_state = self.expected_engine.step(copy.copy(inputs), dt_s)

        # sensed twin: apply the manually-dialed fault at a constant
        # (not time-ramped -- the person controls it live) severity
        scenario = FaultScenario(fault_type=injected_fault_type, drift_sign=1.0, drift_channel="cht_K")
        if injected_severity > 0 and injected_fault_type != "none":
            self.sensed_engine.cfg = degraded_config(self.base_cfg, scenario, injected_severity)
            sensed_inputs = sensed_inputs_for_fault(inputs, scenario, injected_severity, _RNG)
        else:
            self.sensed_engine.cfg = copy.deepcopy(self.base_cfg)
            sensed_inputs = inputs
        sensed_state = self.sensed_engine.step(copy.copy(sensed_inputs), dt_s)
        sensed_state = apply_sensor_fault(sensed_state, scenario, injected_severity)

        record = {
            "t_s": expected_state.t_s,
            "fault_type": injected_fault_type,
            "severity": injected_severity,
            "throttle": inputs.throttle,
            "altitude_m": inputs.altitude_m,
            "airspeed_m_s": inputs.airspeed_m_s,
            "ambient_temp_offset_K": inputs.ambient_temp_offset_K,
            "load_factor": inputs.load_factor,
            "expected": expected_state,
            "sensed": sensed_state,
        }
        self.history.append(record)
        return record

    def current(self) -> dict:
        if self.history:
            return self.history[-1]
        return {
            "t_s": 0.0,
            "fault_type": "none",
            "severity": 0.0,
            "throttle": 0.0,
            "altitude_m": 0.0,
            "airspeed_m_s": 0.0,
            "ambient_temp_offset_K": 0.0,
            "load_factor": 1.0,
            "expected": self.expected_engine.state,
            "sensed": self.sensed_engine.state,
        }

    def get_history(self, last_n: int | None = None) -> list[dict]:
        if last_n is None:
            return list(self.history)
        return list(self.history)[-last_n:]


# Module-level singleton used by the API layer (adequate for a single-engine
# prototype; swap for a keyed registry to simulate a fleet).
manager = SimulationManager()
