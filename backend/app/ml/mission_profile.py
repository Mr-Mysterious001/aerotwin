"""
Randomized mission-profile generator.

Produces a time series of (throttle, altitude, airspeed, ambient_temp_offset,
load_factor) tuples resembling a MALE-UAV ISR sortie: idle -> climb -> cruise
/ loiter (with throttle dithering for station-keeping) -> descent, with
smooth ramps between segments (real throttle/altitude never step-changes
instantly) plus small stochastic dither on top (representing turbulence,
pilot/autopilot micro-corrections, atmospheric variability).

This is a synthetic-scenario generator for training data — it is NOT a
recording of a real flight and is not claimed to be.
"""
import random
from dataclasses import dataclass


@dataclass
class OperatingPoint:
    t_s: float
    throttle: float
    altitude_m: float
    airspeed_m_s: float
    ambient_temp_offset_K: float
    load_factor: float


def _ramp(a, b, frac):
    return a + (b - a) * max(0.0, min(1.0, frac))


def generate_mission(duration_s: int, dt_s: float = 1.0, seed: int | None = None):
    """Returns a list[OperatingPoint] of length duration_s/dt_s."""
    rng = random.Random(seed)

    # Pick a handful of waypoint setpoints describing a typical ISR sortie
    # shape: idle -> climb -> cruise/loiter (dominant phase) -> descent.
    cruise_alt = rng.uniform(2500, 7000)
    cruise_throttle = rng.uniform(0.45, 0.70)
    cruise_airspeed = rng.uniform(28, 45)
    ambient_offset = rng.uniform(-8, 20)  # a single sortie's ambient bias (season/region)

    n_steps = int(duration_s / dt_s)
    t_climb_end = int(0.12 * n_steps)
    t_cruise_end = int(0.82 * n_steps)
    # remainder is descent

    points = []
    for i in range(n_steps):
        t = i * dt_s
        if i < t_climb_end:
            frac = i / max(t_climb_end, 1)
            throttle = _ramp(0.35, 0.85, frac)
            altitude = _ramp(0.0, cruise_alt, frac)
            airspeed = _ramp(20.0, cruise_airspeed, frac)
            load = 1.0
        elif i < t_cruise_end:
            # station-keeping dither: small sinusoidal + noise around cruise setpoint
            throttle = cruise_throttle + 0.04 * rng.uniform(-1, 1)
            altitude = cruise_alt + 60 * rng.uniform(-1, 1)
            airspeed = cruise_airspeed + 2.0 * rng.uniform(-1, 1)
            load = 1.0
        else:
            frac = (i - t_cruise_end) / max(n_steps - t_cruise_end, 1)
            throttle = _ramp(cruise_throttle, 0.25, frac)
            altitude = _ramp(cruise_alt, 200.0, frac)
            airspeed = _ramp(cruise_airspeed, 22.0, frac)
            load = 1.0

        throttle = max(0.05, min(1.0, throttle + rng.uniform(-0.01, 0.01)))
        points.append(
            OperatingPoint(
                t_s=t,
                throttle=throttle,
                altitude_m=max(0.0, altitude),
                airspeed_m_s=max(0.0, airspeed),
                ambient_temp_offset_K=ambient_offset,
                load_factor=load,
            )
        )
    return points
