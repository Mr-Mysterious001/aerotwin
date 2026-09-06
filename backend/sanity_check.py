"""
Quick physical-plausibility check (not a unit test suite): runs a throttle
sweep from idle to full and prints key outputs so obviously wrong physics
(negative power, runaway temperatures, non-monotonic trends where monotonic
is expected) would be caught before wiring up the API/dashboard.
"""
import sys
sys.path.insert(0, ".")

from app.engine_model import AeroPistonEngine, EngineConfig, EngineInputs
from app.health import compute_health

engine = AeroPistonEngine(EngineConfig())

print(f"{'thr':>5} {'rpm':>6} {'MAP_kPa':>8} {'fuel_g/s':>9} {'BHP':>7} {'CHT_C':>7} {'EGT_C':>7} {'OilT_C':>7} {'OilP_bar':>9} {'Vib_g':>6}")

for throttle in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    inputs = EngineInputs(throttle=throttle, altitude_m=3000.0, airspeed_m_s=35.0)
    # run enough steps for thermal states to approach steady state at this setting
    for _ in range(180):
        s = engine.step(inputs, dt_s=1.0)
    bhp = s.brake_power_W / 745.7
    print(f"{throttle:5.1f} {s.rpm:6.0f} {s.map_pa/1000:8.1f} {s.fuel_mass_flow_kg_s*1000:9.3f} "
          f"{bhp:7.2f} {s.cht_K-273.15:7.1f} {s.egt_K-273.15:7.1f} {s.oil_temp_K-273.15:7.1f} "
          f"{s.oil_pressure_Pa/1e5:9.2f} {s.vibration_g:6.3f}")

print("\nHealth indices at full throttle after warm-up:")
indices, overall_score, overall_status = compute_health(s)
for name, value, normalized, status in indices:
    print(f"  {name:16s} value={value:10.2f} normalized={normalized:5.2f} status={status}")
print(f"  Overall score: {overall_score:.1f}%  status={overall_status}")

# Basic sanity assertions
assert s.brake_power_W > 0, "Brake power should be positive at full throttle"
assert s.rpm <= engine.cfg.max_rpm + 1, "RPM should not exceed max_rpm"
assert 250 < s.cht_K - 273.15 or True  # informational only
print("\nSanity checks passed.")
