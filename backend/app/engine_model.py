"""
Physics-based digital twin core for a MALE-UAV aero piston engine.

SCOPE / HONESTY NOTE
---------------------
This is a *first-principles lumped-parameter* model, not a curve-fit to any
specific real engine. It uses:
  - Ideal-gas relations for intake charge mass
  - Air-standard Otto-cycle thermal efficiency as the physics baseline
  - Global fuel-energy balance (work / coolant-and-head heat / exhaust
    enthalpy / unaccounted losses) to derive CHT and EGT
  - Newton's law of cooling (lumped thermal capacitance) for CHT and oil
    temperature dynamics
  - A viscosity-temperature relation (Andrade-type) for oil pressure

Several sub-relations (volumetric efficiency vs. RPM, friction MEP vs. RPM,
energy-balance split fractions, convective heat-transfer coefficients) do
NOT have a public first-principles closed form for a specific unnamed
engine — real engines get these from dynamometer/test-rig mapping. Those
are implemented here as clearly-labeled ASSUMED/TYPICAL shapes taken from
published small-SI-engine literature ranges, exposed as `EngineConfig`
fields so they can be replaced with calibrated test-rig data once available.
Nothing here should be read as verified performance data for any real
engine model.

No ML / degradation / fault-injection logic lives in this file — that is
intentionally deferred to a later module per the project plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Physical constants (exact / standard values — not assumptions)
# ---------------------------------------------------------------------------
R_AIR = 287.05          # J/(kg.K) specific gas constant, dry air
GAMMA_AIR = 1.4          # ratio of specific heats, cold air-standard
G0 = 9.80665             # m/s^2
T0_ISA = 288.15          # K, sea-level ISA temperature
P0_ISA = 101325.0        # Pa, sea-level ISA pressure
LAPSE_RATE = 0.0065      # K/m, ISA troposphere lapse rate


def isa_atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """ISA troposphere model (valid 0-11000 m). Exact standard relations.

    Returns (pressure_Pa, temperature_K, density_kg_m3).
    """
    alt = max(0.0, min(altitude_m, 11000.0))
    temp = T0_ISA - LAPSE_RATE * alt
    pressure = P0_ISA * (temp / T0_ISA) ** (G0 / (LAPSE_RATE * R_AIR))
    density = pressure / (R_AIR * temp)
    return pressure, temp, density


# ---------------------------------------------------------------------------
# Engine configuration — replace defaults with OEM / test-rig data
# ---------------------------------------------------------------------------
@dataclass
class EngineConfig:
    # --- Geometry (ASSUMED representative of a small 2-cyl 4-stroke
    #     air-cooled aero engine in the ~35-45 hp class used on MALE-class
    #     UAVs; replace with actual engine datasheet values) ---
    cylinders: int = 2
    displacement_L: float = 1.0          # total swept volume, litres
    compression_ratio: float = 8.5

    # --- Combustion / fuel (ASSUMED: gasoline/avgas-like fuel) ---
    fuel_lhv_J_per_kg: float = 43.0e6    # lower heating value
    afr_stoich: float = 14.7
    combustion_efficiency: float = 0.98  # fraction of fuel energy released

    # --- Rated operating point (ASSUMED) ---
    rated_rpm: float = 5800.0
    idle_rpm: float = 1700.0
    max_rpm: float = 6800.0

    # --- Volumetric efficiency curve (ASSUMED shape: peaks near rated RPM) ---
    ve_peak: float = 0.85
    ve_idle: float = 0.55

    # --- Otto-cycle efficiency realization factor (ASSUMED: real engines
    #     achieve roughly 40-55% of the ideal air-standard Otto efficiency
    #     once heat transfer, pumping and combustion losses are included) ---
    otto_realization_factor: float = 0.48

    # --- Friction mean effective pressure, FMEP (ASSUMED linear-in-RPM
    #     approximation, Pa; typical small SI engines: 100-250 kPa) ---
    fmep_base_Pa: float = 90_000.0
    fmep_rpm_coeff_Pa_per_rpm: float = 20.0

    # --- Fuel energy balance split at rated load (ASSUMED, typical small
    #     SI-engine bookkeeping: ~28% work, ~28% coolant/head, ~36% exhaust,
    #     ~8% radiation/unaccounted) ---
    frac_to_head: float = 0.28
    frac_to_exhaust: float = 0.36

    # --- Thermal masses / cooling (ASSUMED lumped-parameter values) ---
    head_thermal_mass_J_per_K: float = 3500.0
    head_htc_base_W_per_K: float = 45.0       # natural convection, static air
    head_htc_per_airspeed: float = 6.0        # extra W/K per (m/s) of airspeed
    oil_thermal_mass_J_per_K: float = 2200.0
    oil_htc_base_W_per_K: float = 18.0
    oil_htc_per_airspeed: float = 2.5
    oil_from_friction_fraction: float = 0.55   # fraction of friction heat into oil
    oil_from_head_conduction_W_per_K: float = 4.0  # conductive coupling coefficient

    # --- Oil pressure model (ASSUMED Andrade-type viscosity law + RPM-driven
    #     gear pump with relief valve cap) ---
    oil_visc_ref_cP: float = 12.0     # reference kinematic-ish viscosity index at 80C
    oil_visc_ref_temp_K: float = 353.15
    oil_visc_andrade_B: float = 900.0  # K, Andrade constant (assumed)
    oil_pump_gain_Pa_per_rpm_per_cP: float = 70.0
    oil_pressure_relief_Pa: float = 550_000.0  # ~5.5 bar relief cap
    oil_pressure_min_idle_Pa: float = 100_000.0

    # --- Alternator / battery (ASSUMED simple regulator model) ---
    alternator_cutin_rpm: float = 2200.0
    alternator_regulated_voltage: float = 14.2
    battery_nominal_voltage: float = 12.6
    battery_capacity_Ah: float = 18.0
    avionics_load_A: float = 8.0

    # --- Vibration (ASSUMED placeholder physically-motivated synthetic
    #     signature — firing-frequency tone whose RMS scales with torque
    #     fluctuation; NOT a substitute for real accelerometer data) ---
    vibration_base_g: float = 0.15
    vibration_torque_fluct_gain: float = 0.02


@dataclass
class EngineInputs:
    """Externally-supplied / manually-entered operating inputs for one tick."""
    throttle: float            # 0..1 pilot/throttle-servo commanded position
    altitude_m: float = 0.0
    airspeed_m_s: float = 30.0
    ambient_temp_offset_K: float = 0.0   # ISA deviation, e.g. hot-day offset
    afr_target: Optional[float] = None   # None -> stoichiometric
    load_factor: float = 1.0             # 0..1 external mechanical load (e.g. generator, prop pitch)


@dataclass
class EngineState:
    t_s: float = 0.0
    rpm: float = 0.0
    map_pa: float = 0.0
    air_mass_flow_kg_s: float = 0.0
    fuel_mass_flow_kg_s: float = 0.0
    indicated_power_W: float = 0.0
    brake_power_W: float = 0.0
    torque_Nm: float = 0.0
    cht_K: float = 0.0
    egt_K: float = 0.0
    oil_temp_K: float = 0.0
    oil_pressure_Pa: float = 0.0
    vibration_g: float = 0.0
    alternator_voltage_V: float = 0.0
    battery_soc_pct: float = 100.0
    ambient_temp_K: float = 0.0
    ambient_pressure_Pa: float = 0.0


class AeroPistonEngine:
    """Lumped-parameter physics-based digital twin of one engine."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.cfg = config or EngineConfig()
        self.state = EngineState(
            rpm=self.cfg.idle_rpm,
            cht_K=T0_ISA + 20.0,
            oil_temp_K=T0_ISA + 15.0,
            battery_soc_pct=100.0,
        )

    # -- sub-models -----------------------------------------------------
    def _volumetric_efficiency(self, rpm: float) -> float:
        """ASSUMED bell-shaped VE curve peaking near rated RPM."""
        cfg = self.cfg
        x = (rpm - cfg.rated_rpm) / max(cfg.rated_rpm - cfg.idle_rpm, 1.0)
        shape = math.exp(-0.9 * x * x)
        return cfg.ve_idle + (cfg.ve_peak - cfg.ve_idle) * shape

    def _fmep_pa(self, rpm: float) -> float:
        """ASSUMED approx-linear friction MEP vs RPM."""
        return self.cfg.fmep_base_Pa + self.cfg.fmep_rpm_coeff_Pa_per_rpm * rpm

    def _target_rpm(self, inputs: EngineInputs) -> float:
        """Simple governor: commanded RPM interpolated by throttle between
        idle and max, then reduced slightly by external load factor.
        (ASSUMED first-order governor response, not a specific FADEC map.)
        """
        cfg = self.cfg
        cmd = cfg.idle_rpm + inputs.throttle * (cfg.max_rpm - cfg.idle_rpm)
        cmd -= inputs.load_factor * 0.05 * (cfg.max_rpm - cfg.idle_rpm)
        return max(cfg.idle_rpm, min(cmd, cfg.max_rpm))

    # -- main integration step -------------------------------------------
    def step(self, inputs: EngineInputs, dt_s: float) -> EngineState:
        cfg = self.cfg
        s = self.state

        # --- Atmosphere (exact ISA relations) ---
        p_amb, t_amb_isa, _ = isa_atmosphere(inputs.altitude_m)
        t_amb = t_amb_isa + inputs.ambient_temp_offset_K
        rho_amb = p_amb / (R_AIR * t_amb)

        # --- Governor: first-order lag toward target RPM ---
        rpm_target = self._target_rpm(inputs)
        tau_rpm = 1.2  # s, ASSUMED mechanical/governor time constant
        rpm = s.rpm + (rpm_target - s.rpm) * (1 - math.exp(-dt_s / tau_rpm))
        rpm = max(0.0, rpm)

        # --- Intake manifold pressure: naturally aspirated throttle model ---
        # Physics: partially-closed throttle plate creates pressure drop;
        # exact loss coefficient is engine-specific, so the throttle->MAP
        # mapping below is an ASSUMED monotonic approximation.
        map_pa = p_amb * (0.20 + 0.80 * inputs.throttle)
        rho_intake = map_pa / (R_AIR * t_amb)

        # --- Volumetric efficiency & air mass flow (ideal-gas + VE, exact
        # given VE) ---
        ve = self._volumetric_efficiency(rpm)
        # 4-stroke: one intake event per cylinder per 2 crank revolutions
        rev_per_s = rpm / 60.0
        displacement_m3 = cfg.displacement_L * 1e-3
        air_mass_flow = ve * displacement_m3 * rho_intake * (rev_per_s / 2.0)

        # --- Fuel flow from AFR target (exact given AFR) ---
        afr = inputs.afr_target or cfg.afr_stoich
        fuel_mass_flow = air_mass_flow / afr if afr > 0 else 0.0

        # --- Combustion energy release rate (exact given LHV & eta_comb) ---
        q_in_rate = fuel_mass_flow * cfg.fuel_lhv_J_per_kg * cfg.combustion_efficiency

        # --- Ideal air-standard Otto efficiency (exact thermodynamic
        # relation for the given compression ratio) ---
        eta_otto_ideal = 1.0 - cfg.compression_ratio ** (1.0 - GAMMA_AIR)
        eta_actual = eta_otto_ideal * cfg.otto_realization_factor

        indicated_power = q_in_rate * eta_actual

        # --- Friction losses -> brake power (exact given FMEP definition) ---
        fmep = self._fmep_pa(rpm)
        friction_power = fmep * displacement_m3 * (rev_per_s / 2.0)
        brake_power = max(0.0, indicated_power - friction_power)

        omega = 2.0 * math.pi * rpm / 60.0
        torque = brake_power / omega if omega > 1e-3 else 0.0

        # --- Fuel-energy balance -> heat to head & exhaust enthalpy
        # (ASSUMED split fractions; remainder is radiation/unaccounted).
        # The exhaust fraction is scaled up with RPM: at higher engine
        # speed there is proportionally less time per cycle for in-cylinder
        # heat transfer to the walls, so a larger share of combustion
        # energy leaves as exhaust enthalpy (this is why EGT rises with
        # load/RPM in real SI engines, not just because more fuel is
        # burned). The head fraction is reduced correspondingly; the gap
        # is absorbed into the unaccounted/radiation share. Coefficients
        # here are an ASSUMED illustrative trend, not a calibrated map.
        rpm_ratio = rpm / cfg.rated_rpm
        exhaust_rpm_scale = 0.55 + 0.55 * min(rpm_ratio, 1.3)
        head_rpm_scale = 1.15 - 0.15 * min(rpm_ratio, 1.3)
        heat_to_head_rate = cfg.frac_to_head * head_rpm_scale * q_in_rate
        heat_to_exhaust_rate = cfg.frac_to_exhaust * exhaust_rpm_scale * q_in_rate

        # --- CHT: lumped thermal capacitance, Newton cooling to airflow ---
        h_head = cfg.head_htc_base_W_per_K + cfg.head_htc_per_airspeed * inputs.airspeed_m_s
        d_cht = (heat_to_head_rate - h_head * (s.cht_K - t_amb)) / cfg.head_thermal_mass_J_per_K
        cht = s.cht_K + d_cht * dt_s

        # --- EGT: exhaust enthalpy energy balance ---
        # m_dot_exhaust*cp_exhaust*(T_egt - T_amb) ~= heat_to_exhaust_rate
        cp_exhaust = 1150.0  # J/(kg.K), ASSUMED typical hot exhaust gas cp
        m_dot_exhaust = air_mass_flow + fuel_mass_flow
        if m_dot_exhaust > 1e-6:
            egt = t_amb + heat_to_exhaust_rate / (m_dot_exhaust * cp_exhaust)
        else:
            egt = t_amb

        # --- Oil temperature: lumped node heated by friction + head
        # conduction, cooled by airflow ---
        friction_heat_rate = friction_power  # all friction work -> heat
        h_oil = cfg.oil_htc_base_W_per_K + cfg.oil_htc_per_airspeed * inputs.airspeed_m_s
        oil_heat_in = (
            cfg.oil_from_friction_fraction * friction_heat_rate
            + cfg.oil_from_head_conduction_W_per_K * (cht - s.oil_temp_K)
        )
        d_oil = (oil_heat_in - h_oil * (s.oil_temp_K - t_amb)) / cfg.oil_thermal_mass_J_per_K
        oil_temp = s.oil_temp_K + d_oil * dt_s

        # --- Oil pressure: Andrade viscosity law + RPM-driven pump, capped
        # by relief valve. (Earlier version added a separate RPM-scaled
        # "idle floor" term that ended up dominating the pump-gain term by
        # ~5x, which meant the lubrication_issues fault — which only
        # scales pump gain — barely moved the output. Simplified to a
        # single pump-gain term so a degraded pump actually shows up.) ---
        visc_cP = cfg.oil_visc_ref_cP * math.exp(
            cfg.oil_visc_andrade_B * (1.0 / oil_temp - 1.0 / cfg.oil_visc_ref_temp_K)
        )
        oil_pressure = min(
            cfg.oil_pressure_relief_Pa,
            cfg.oil_pump_gain_Pa_per_rpm_per_cP * rpm * (visc_cP / cfg.oil_visc_ref_cP),
        )

        # --- Vibration: firing-frequency tone, RMS scaled with torque
        # fluctuation proxy (placeholder, no accelerometer input yet) ---
        firing_freq_hz = rev_per_s * (cfg.cylinders / 2.0)
        torque_fluct_proxy = abs(torque) / max(1.0, cfg.rated_rpm * 0.02)
        vibration_g = cfg.vibration_base_g + cfg.vibration_torque_fluct_gain * torque_fluct_proxy
        vibration_g *= 1.0 + 0.05 * math.sin(2 * math.pi * firing_freq_hz * s.t_s)

        # --- Alternator / battery bookkeeping ---
        if rpm >= cfg.alternator_cutin_rpm:
            alt_voltage = cfg.alternator_regulated_voltage
            net_current_A = 25.0 - cfg.avionics_load_A  # ASSUMED alternator max ~25A
        else:
            alt_voltage = cfg.battery_nominal_voltage
            net_current_A = -cfg.avionics_load_A
        d_soc = (net_current_A * dt_s / 3600.0) / cfg.battery_capacity_Ah * 100.0
        battery_soc = max(0.0, min(100.0, s.battery_soc_pct + d_soc))

        # --- Commit new state ---
        new_state = EngineState(
            t_s=s.t_s + dt_s,
            rpm=rpm,
            map_pa=map_pa,
            air_mass_flow_kg_s=air_mass_flow,
            fuel_mass_flow_kg_s=fuel_mass_flow,
            indicated_power_W=indicated_power,
            brake_power_W=brake_power,
            torque_Nm=torque,
            cht_K=cht,
            egt_K=egt,
            oil_temp_K=oil_temp,
            oil_pressure_Pa=oil_pressure,
            vibration_g=vibration_g,
            alternator_voltage_V=alt_voltage,
            battery_soc_pct=battery_soc,
            ambient_temp_K=t_amb,
            ambient_pressure_Pa=p_amb,
        )
        self.state = new_state
        return new_state
