import React, { useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";
import {
  simulateStep,
  getCurrentHealth,
  getHistory,
  resetSimulation,
  mlAnalyze,
  mlForecast,
} from "./api.js";

const DEFAULT_INPUTS = {
  throttle: 0.5,
  altitude_m: 3000,
  airspeed_m_s: 35,
  ambient_temp_offset_K: 0,
  load_factor: 1.0,
  dt_s: 2.0,
};

const FAULT_TYPES = [
  "none",
  "misfire_conditions",
  "injector_abnormalities",
  "cooling_degradation",
  "lubrication_issues",
  "sensor_drift",
  "combustion_instability",
  "overheating_trend",
  "abnormal_vibration",
];

const MAX_TREND_POINTS = 600;

function Slider({ label, value, unit, min, max, step, onChange }) {
  return (
    <div className="control">
      <div className="control-label">
        <span>{label}</span>
        <span className="control-value">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

function Readout({ label, value, unit, status }) {
  return (
    <div className={`readout ${status || "nominal"}`}>
      <div className="readout-label">{label}</div>
      <div className="readout-value">
        {value}
        <span className="readout-unit">{unit}</span>
      </div>
    </div>
  );
}

function statusFor(healthIndices, name) {
  const idx = healthIndices?.find((h) => h.name === name);
  return idx ? idx.status : "nominal";
}

export default function App() {
  const [inputs, setInputs] = useState(DEFAULT_INPUTS);
  const [faultType, setFaultType] = useState("none");
  const [faultSeverity, setFaultSeverity] = useState(0.0);
  const [current, setCurrent] = useState(null); // {sensed, expected, fault_type, severity}
  const [health, setHealth] = useState(null);
  const [ml, setMl] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [trend, setTrend] = useState([]);
  const [autoRun, setAutoRun] = useState(false);
  const [connOk, setConnOk] = useState(null);
  const intervalRef = useRef(null);

  const set = (key) => (val) => setInputs((prev) => ({ ...prev, [key]: val }));

  const pushTrendPoint = (twin) => {
    setTrend((prev) => {
      const next = [
        ...prev,
        {
          t_s: twin.sensed.t_s,
          cht_C: twin.sensed.cht_C,
          egt_C: twin.sensed.egt_C,
          oil_temp_C: twin.sensed.oil_temp_C,
          expected_cht_C: twin.expected.cht_C,
          expected_egt_C: twin.expected.egt_C,
          expected_oil_temp_C: twin.expected.oil_temp_C,
        },
      ];
      return next.length > MAX_TREND_POINTS ? next.slice(next.length - MAX_TREND_POINTS) : next;
    });
  };

  const doStep = async () => {
    try {
      const twin = await simulateStep({
        ...inputs,
        injected_fault_type: faultType,
        injected_severity: faultSeverity,
      });
      setCurrent(twin);
      setConnOk(true);
      pushTrendPoint(twin);
      const h = await getCurrentHealth();
      setHealth(h);
      try {
        const analysis = await mlAnalyze();
        setMl(analysis);
      } catch (e) {
        setMl({ ready: false, reason: "models not trained yet" });
      }
    } catch (e) {
      console.error(e);
      setConnOk(false);
    }
  };

  const toggleAutoRun = () => setAutoRun((prev) => !prev);

  useEffect(() => {
    if (autoRun) {
      intervalRef.current = setInterval(doStep, Math.max(inputs.dt_s, 0.2) * 1000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => intervalRef.current && clearInterval(intervalRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRun, inputs, faultType, faultSeverity]);

  const handleReset = async () => {
    setAutoRun(false);
    try {
      await resetSimulation();
      setTrend([]);
      setCurrent(null);
      setHealth(null);
      setMl(null);
      setForecast(null);
      setConnOk(true);
    } catch (e) {
      setConnOk(false);
    }
  };

  const handleLoadReplay = async () => {
    try {
      const hist = await getHistory(MAX_TREND_POINTS);
      setTrend(
        hist.map((twin) => ({
          t_s: twin.sensed.t_s,
          cht_C: twin.sensed.cht_C,
          egt_C: twin.sensed.egt_C,
          oil_temp_C: twin.sensed.oil_temp_C,
          expected_cht_C: twin.expected.cht_C,
          expected_egt_C: twin.expected.egt_C,
          expected_oil_temp_C: twin.expected.oil_temp_C,
        }))
      );
      setConnOk(true);
    } catch (e) {
      setConnOk(false);
    }
  };

  const handleForecast = async () => {
    try {
      const points = await mlForecast(300);
      setForecast(points);
    } catch (e) {
      console.error(e);
    }
  };

  const overallStatus = health?.overall_status || "nominal";
  const sensed = current?.sensed;
  const expected = current?.expected;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>AERO PISTON ENGINE — DIGITAL TWIN + ML</h1>
          <div className="subtitle">MALE-UAV ground control / health monitoring prototype — Phase 2 (ML/DL layer)</div>
        </div>
        <div className="header-right">
          <span>
            <span className={`conn-dot ${connOk === false ? "err" : "ok"}`} />
            {connOk === false ? "backend unreachable" : "backend ok"}
          </span>
          <button className="btn" onClick={handleLoadReplay}>
            Load full replay
          </button>
          <button className="btn danger" onClick={handleReset}>
            Reset
          </button>
        </div>
      </header>

      <div className="layout">
        <aside className="panel">
          <p className="panel-title">MANUAL INPUT PARAMETERS</p>

          <Slider label="Throttle" value={inputs.throttle.toFixed(2)} unit="" min={0} max={1} step={0.01} onChange={set("throttle")} />
          <Slider label="Altitude" value={inputs.altitude_m} unit=" m" min={0} max={9000} step={100} onChange={set("altitude_m")} />
          <Slider label="Airspeed" value={inputs.airspeed_m_s} unit=" m/s" min={0} max={80} step={1} onChange={set("airspeed_m_s")} />
          <Slider label="Ambient temp offset" value={inputs.ambient_temp_offset_K} unit=" K" min={-20} max={30} step={1} onChange={set("ambient_temp_offset_K")} />
          <Slider label="External load factor" value={inputs.load_factor.toFixed(2)} unit="" min={0} max={1} step={0.05} onChange={set("load_factor")} />
          <Slider label="Step size (dt)" value={inputs.dt_s.toFixed(1)} unit=" s" min={0.2} max={5} step={0.2} onChange={set("dt_s")} />

          <div className="control-actions">
            <button className="btn primary" onClick={doStep} disabled={autoRun}>
              Step once
            </button>
            <button className="btn" onClick={toggleAutoRun}>
              {autoRun ? "Pause" : "Run continuously"}
            </button>
          </div>

          <p className="panel-title" style={{ marginTop: 22 }}>
            LIVE FAULT INJECTION (demo)
          </p>
          <div className="control">
            <div className="control-label">
              <span>Fault type</span>
            </div>
            <select
              value={faultType}
              onChange={(e) => setFaultType(e.target.value)}
              style={{
                width: "100%",
                background: "#172227",
                color: "#dce7e4",
                border: "1px solid #223035",
                borderRadius: 3,
                padding: "6px 8px",
                fontFamily: "JetBrains Mono",
                fontSize: 12,
              }}
            >
              {FAULT_TYPES.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>
          <Slider
            label="Fault severity"
            value={faultSeverity.toFixed(2)}
            unit=""
            min={0}
            max={1}
            step={0.05}
            onChange={setFaultSeverity}
          />
          <p className="empty-note" style={{ marginTop: -8 }}>
            Dials a fault into the "sensed" twin only — the "expected" twin
            keeps running clean, so the gap between them is what the ML
            layer below is reacting to.
          </p>
        </aside>

        <div className="main-col">
          <section className="panel">
            <p className="panel-title">INSTRUMENT CLUSTER (sensed vs. expected)</p>
            {!sensed ? (
              <p className="empty-note">No data yet — press "Step once" to feed the twin its first input.</p>
            ) : (
              <div className="instrument-grid">
                <Readout label="ENGINE SPEED" value={sensed.rpm.toFixed(0)} unit=" rpm" />
                <Readout label="FUEL FLOW" value={(sensed.fuel_mass_flow_kg_s * 3600).toFixed(2)} unit=" kg/h" />
                <Readout label="BRAKE POWER" value={sensed.brake_power_hp.toFixed(1)} unit=" hp" />
                <Readout
                  label="CYL. HEAD TEMP"
                  value={`${sensed.cht_C.toFixed(1)} (exp ${expected.cht_C.toFixed(1)})`}
                  unit=" °C"
                  status={statusFor(health?.indices, "CHT")}
                />
                <Readout
                  label="EXHAUST GAS TEMP"
                  value={`${sensed.egt_C.toFixed(0)} (exp ${expected.egt_C.toFixed(0)})`}
                  unit=" °C"
                  status={statusFor(health?.indices, "EGT")}
                />
                <Readout
                  label="OIL TEMPERATURE"
                  value={`${sensed.oil_temp_C.toFixed(1)} (exp ${expected.oil_temp_C.toFixed(1)})`}
                  unit=" °C"
                  status={statusFor(health?.indices, "Oil Temperature")}
                />
                <Readout
                  label="OIL PRESSURE"
                  value={`${sensed.oil_pressure_bar.toFixed(2)} (exp ${expected.oil_pressure_bar.toFixed(2)})`}
                  unit=" bar"
                  status={statusFor(health?.indices, "Oil Pressure")}
                />
                <Readout
                  label="VIBRATION"
                  value={sensed.vibration_g.toFixed(3)}
                  unit=" g"
                  status={statusFor(health?.indices, "Vibration")}
                />
              </div>
            )}
          </section>

          <section className="panel">
            <p className="panel-title">RULE-BASED HEALTH (threshold system — the "before")</p>
            {!health ? (
              <p className="empty-note">Awaiting first simulation step.</p>
            ) : (
              <div className="health-row">
                <div className={`health-score ${overallStatus}`}>{health.overall_score_pct.toFixed(0)}%</div>
                {health.indices.map((idx) => (
                  <div className="health-chip" key={idx.name}>
                    <span className={`dot ${idx.status}`} />
                    {idx.name}: {idx.status}
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="panel">
            <p className="panel-title">ML/DL ANALYSIS (residual-based — the "after")</p>
            {!ml || !ml.ready ? (
              <p className="empty-note">{ml?.reason || "Awaiting enough history (needs a few steps)."}</p>
            ) : (
              <div>
                <div className="health-row" style={{ marginBottom: 14 }}>
                  <div className={`health-score ${ml.is_anomaly ? "warning" : "nominal"}`}>
                    {ml.is_anomaly ? "ANOMALY" : "NOMINAL"}
                  </div>
                  <div className="health-chip">
                    anomaly score: {ml.anomaly_score.toFixed(4)} (threshold {ml.anomaly_threshold.toFixed(4)})
                  </div>
                  {ml.predicted_rul_s != null && (
                    <div className="health-chip">
                      predicted RUL: {(ml.predicted_rul_s / 60).toFixed(1)} min
                    </div>
                  )}
                </div>
                <div className="instrument-grid">
                  {ml.fault_probabilities.map((fp) => (
                    <div key={fp.fault_type} className={`readout ${fp.probability > 0.5 ? "warning" : "nominal"}`}>
                      <div className="readout-label">{fp.fault_type}</div>
                      <div className="readout-value" style={{ fontSize: 16 }}>
                        {(fp.probability * 100).toFixed(0)}
                        <span className="readout-unit">%</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="control-actions" style={{ marginTop: 14, maxWidth: 260 }}>
                  <button className="btn" onClick={handleForecast}>
                    Forecast next 5 min if unchanged
                  </button>
                </div>
                {forecast && (
                  <p className="empty-note" style={{ marginTop: 8 }}>
                    Expected twin at +5min if current inputs hold: CHT{" "}
                    {forecast[forecast.length - 1].cht_C.toFixed(1)}°C, EGT{" "}
                    {forecast[forecast.length - 1].egt_C.toFixed(0)}°C, oil{" "}
                    {forecast[forecast.length - 1].oil_temp_C.toFixed(1)}°C — this projects the CLEAN
                    twin forward, i.e. what should happen if nothing is wrong.
                  </p>
                )}
              </div>
            )}
          </section>

          <section className="panel chart-panel">
            <p className="panel-title">MISSION TREND / REPLAY (solid = sensed, dashed = expected)</p>
            {trend.length < 2 ? (
              <p className="empty-note">Trend needs at least two data points — keep stepping.</p>
            ) : (
              <Plot
                data={[
                  { x: trend.map((p) => p.t_s), y: trend.map((p) => p.cht_C), name: "CHT sensed", type: "scattergl", mode: "lines", line: { color: "#3fd6c8", width: 2 } },
                  { x: trend.map((p) => p.t_s), y: trend.map((p) => p.expected_cht_C), name: "CHT expected", type: "scattergl", mode: "lines", line: { color: "#3fd6c8", width: 1, dash: "dot" } },
                  { x: trend.map((p) => p.t_s), y: trend.map((p) => p.egt_C), name: "EGT sensed", type: "scattergl", mode: "lines", line: { color: "#f2b33d", width: 2 }, yaxis: "y2" },
                  { x: trend.map((p) => p.t_s), y: trend.map((p) => p.expected_egt_C), name: "EGT expected", type: "scattergl", mode: "lines", line: { color: "#f2b33d", width: 1, dash: "dot" }, yaxis: "y2" },
                  { x: trend.map((p) => p.t_s), y: trend.map((p) => p.oil_temp_C), name: "Oil sensed", type: "scattergl", mode: "lines", line: { color: "#e5484d", width: 2 } },
                  { x: trend.map((p) => p.t_s), y: trend.map((p) => p.expected_oil_temp_C), name: "Oil expected", type: "scattergl", mode: "lines", line: { color: "#e5484d", width: 1, dash: "dot" } },
                ]}
                layout={{
                  autosize: true,
                  height: 340,
                  margin: { l: 50, r: 50, t: 10, b: 40 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  font: { color: "#7c9a97", family: "JetBrains Mono", size: 11 },
                  xaxis: { title: "t (s)", gridcolor: "#1c2a2e", zeroline: false },
                  yaxis: { title: "CHT / Oil (°C)", gridcolor: "#1c2a2e", zeroline: false },
                  yaxis2: { title: "EGT (°C)", overlaying: "y", side: "right", gridcolor: "transparent", zeroline: false },
                  legend: { orientation: "h", y: -0.2 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
                useResizeHandler
              />
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
