from typing import Optional
from pydantic import BaseModel, Field


class StepRequest(BaseModel):
    throttle: float = Field(..., ge=0.0, le=1.0, description="Throttle position 0-1")
    altitude_m: float = Field(0.0, ge=0.0, le=11000.0)
    airspeed_m_s: float = Field(30.0, ge=0.0, le=150.0)
    ambient_temp_offset_K: float = Field(0.0, ge=-40.0, le=40.0, description="ISA deviation, e.g. hot-day +K")
    afr_target: Optional[float] = Field(None, gt=8.0, lt=25.0)
    load_factor: float = Field(1.0, ge=0.0, le=1.0)
    dt_s: float = Field(1.0, gt=0.0, le=10.0)
    injected_fault_type: str = Field("none", description="For live ML demo: manually dial in a fault type")
    injected_severity: float = Field(0.0, ge=0.0, le=1.0, description="Fault severity 0-1, held constant (not ramped)")


class RunRequest(BaseModel):
    """Batch mission-profile run: a sequence of StepRequests applied in order."""
    profile: list[StepRequest]
    reset_first: bool = True


class EngineStateResponse(BaseModel):
    t_s: float
    rpm: float
    map_pa: float
    air_mass_flow_kg_s: float
    fuel_mass_flow_kg_s: float
    indicated_power_W: float
    brake_power_W: float
    torque_Nm: float
    cht_K: float
    egt_K: float
    oil_temp_K: float
    oil_pressure_Pa: float
    vibration_g: float
    alternator_voltage_V: float
    battery_soc_pct: float
    ambient_temp_K: float
    ambient_pressure_Pa: float
    # convenience Celsius/derived fields for the dashboard
    cht_C: float
    egt_C: float
    oil_temp_C: float
    brake_power_hp: float
    oil_pressure_bar: float


class TwinStepResponse(BaseModel):
    sensed: EngineStateResponse
    expected: EngineStateResponse
    fault_type: str
    severity: float


class MLAnalysisResponse(BaseModel):
    ready: bool
    reason: Optional[str] = None
    t_s: Optional[float] = None
    anomaly_score: Optional[float] = None
    anomaly_threshold: Optional[float] = None
    is_anomaly: Optional[bool] = None
    predicted_fault_type: Optional[str] = None
    fault_probabilities: Optional[list] = None
    predicted_rul_s: Optional[float] = None
    rule_based_health_status: Optional[str] = None
    rule_based_health_score: Optional[float] = None


class ForecastPoint(BaseModel):
    t_s: float
    rpm: float
    cht_C: float
    egt_C: float
    oil_temp_C: float
    oil_pressure_bar: float


class ForecastRequest(BaseModel):
    horizon_s: float = Field(300.0, gt=0.0, le=3600.0)


class HealthIndex(BaseModel):
    name: str
    value: float
    normalized: float  # 0 (nominal) .. 1+ (at/over limit)
    status: str        # "nominal" | "caution" | "warning"


class HealthResponse(BaseModel):
    t_s: float
    indices: list[HealthIndex]
    overall_score_pct: float
    overall_status: str
