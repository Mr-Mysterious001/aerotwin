import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .engine_model import EngineInputs, EngineState
from .health import compute_health
from .schemas import (
    EngineStateResponse,
    ForecastPoint,
    ForecastRequest,
    HealthIndex,
    HealthResponse,
    MLAnalysisResponse,
    RunRequest,
    StepRequest,
    TwinStepResponse,
)
from .simulation import manager

app = FastAPI(
    title="MALE-UAV Aero Piston Engine Digital Twin",
    description=(
        "Phase 1: manual-input digital twin core, physics-based engine "
        "simulation, threshold-based health indices, mission-profile batch "
        "run, post-flight replay. Phase 2: synthetic-data-trained ML/DL "
        "layer (anomaly detection, fault classification, RUL estimation) "
        "layered on top of the same physics twin."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prototype only -- restrict in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_state_response(s: EngineState) -> EngineStateResponse:
    return EngineStateResponse(
        t_s=s.t_s,
        rpm=s.rpm,
        map_pa=s.map_pa,
        air_mass_flow_kg_s=s.air_mass_flow_kg_s,
        fuel_mass_flow_kg_s=s.fuel_mass_flow_kg_s,
        indicated_power_W=s.indicated_power_W,
        brake_power_W=s.brake_power_W,
        torque_Nm=s.torque_Nm,
        cht_K=s.cht_K,
        egt_K=s.egt_K,
        oil_temp_K=s.oil_temp_K,
        oil_pressure_Pa=s.oil_pressure_Pa,
        vibration_g=s.vibration_g,
        alternator_voltage_V=s.alternator_voltage_V,
        battery_soc_pct=s.battery_soc_pct,
        ambient_temp_K=s.ambient_temp_K,
        ambient_pressure_Pa=s.ambient_pressure_Pa,
        cht_C=s.cht_K - 273.15,
        egt_C=s.egt_K - 273.15,
        oil_temp_C=s.oil_temp_K - 273.15,
        brake_power_hp=s.brake_power_W / 745.7,
        oil_pressure_bar=s.oil_pressure_Pa / 1e5,
    )


def _to_health_response(s: EngineState) -> HealthResponse:
    indices, overall_score, overall_status = compute_health(s)
    return HealthResponse(
        t_s=s.t_s,
        indices=[
            HealthIndex(name=n, value=v, normalized=norm, status=st)
            for (n, v, norm, st) in indices
        ],
        overall_score_pct=overall_score,
        overall_status=overall_status,
    )


def _record_to_twin_response(record: dict) -> TwinStepResponse:
    return TwinStepResponse(
        sensed=_to_state_response(record["sensed"]),
        expected=_to_state_response(record["expected"]),
        fault_type=record["fault_type"],
        severity=record["severity"],
    )


@app.get("/api/state/current", response_model=TwinStepResponse)
def get_current_state():
    return _record_to_twin_response(manager.current())


@app.get("/api/health/current", response_model=HealthResponse)
def get_current_health():
    return _to_health_response(manager.current()["sensed"])


@app.post("/api/simulate/step", response_model=TwinStepResponse)
def simulate_step(req: StepRequest):
    inputs = EngineInputs(
        throttle=req.throttle,
        altitude_m=req.altitude_m,
        airspeed_m_s=req.airspeed_m_s,
        ambient_temp_offset_K=req.ambient_temp_offset_K,
        afr_target=req.afr_target,
        load_factor=req.load_factor,
    )
    record = manager.step(
        inputs, req.dt_s,
        injected_fault_type=req.injected_fault_type,
        injected_severity=req.injected_severity,
    )
    return _record_to_twin_response(record)


@app.post("/api/simulate/run", response_model=list[TwinStepResponse])
def simulate_run(req: RunRequest):
    """Batch-run a mission profile (list of steps) -- supports mission
    planning / what-if simulation ahead of flight."""
    if req.reset_first:
        manager.reset()
    results = []
    for step_req in req.profile:
        inputs = EngineInputs(
            throttle=step_req.throttle,
            altitude_m=step_req.altitude_m,
            airspeed_m_s=step_req.airspeed_m_s,
            ambient_temp_offset_K=step_req.ambient_temp_offset_K,
            afr_target=step_req.afr_target,
            load_factor=step_req.load_factor,
        )
        record = manager.step(
            inputs, step_req.dt_s,
            injected_fault_type=step_req.injected_fault_type,
            injected_severity=step_req.injected_severity,
        )
        results.append(_record_to_twin_response(record))
    return results


@app.get("/api/history", response_model=list[TwinStepResponse])
def get_history(last_n: int | None = None):
    """Post-flight analysis / mission replay: returns recorded history."""
    return [_record_to_twin_response(r) for r in manager.get_history(last_n)]


@app.post("/api/reset")
def reset_simulation():
    manager.reset()
    return {"status": "reset"}


@app.get("/api/ml/analyze", response_model=MLAnalysisResponse)
def ml_analyze():
    """Runs the trained anomaly detector + fault classifier + RUL
    regressor on the current session's recent history. Falls back to a
    'not ready' response with an explanation if models aren't trained yet
    or there isn't enough history."""
    from .ml import inference

    try:
        result = inference.analyze(list(manager.history))
    except FileNotFoundError:
        return MLAnalysisResponse(ready=False, reason="Models not trained yet -- run train_models.py first.")
    return MLAnalysisResponse(**result)


@app.post("/api/ml/forecast", response_model=list[ForecastPoint])
def ml_forecast(req: ForecastRequest):
    """Projects the clean/expected twin forward from its current state
    with the last-commanded inputs held constant -- 'what should this
    engine read if the current setting is maintained and nothing is
    wrong'. Does not touch the live simulation state."""
    from .ml import inference
    from .engine_model import EngineInputs as _Inputs

    current = manager.current()
    last_inputs = _Inputs(
        throttle=current["throttle"],
        altitude_m=current["altitude_m"],
        airspeed_m_s=current["airspeed_m_s"],
        ambient_temp_offset_K=current["ambient_temp_offset_K"],
        load_factor=current["load_factor"],
    )
    points = inference.forecast_if_continued(manager.expected_engine, last_inputs, req.horizon_s)
    return [ForecastPoint(**p) for p in points]


@app.websocket("/ws/stream")
async def stream_state(websocket: WebSocket):
    """Push the current state + health once per second for a live dashboard."""
    await websocket.accept()
    try:
        while True:
            record = manager.current()
            payload = {
                "twin": _record_to_twin_response(record).model_dump(),
                "health": _to_health_response(record["sensed"]).model_dump(),
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


@app.get("/")
def root():
    return {
        "name": "MALE-UAV Aero Piston Engine Digital Twin -- backend",
        "docs": "/docs",
        "endpoints": [
            "POST /api/simulate/step",
            "POST /api/simulate/run",
            "GET  /api/state/current",
            "GET  /api/health/current",
            "GET  /api/history",
            "POST /api/reset",
            "GET  /api/ml/analyze",
            "POST /api/ml/forecast",
            "WS   /ws/stream",
        ],
    }
