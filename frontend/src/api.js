// Point this at the running backend (see backend/README.md).
export const API_BASE = "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} -> ${res.status}: ${text}`);
  }
  return res.json();
}

export function simulateStep(inputs) {
  return request("/api/simulate/step", {
    method: "POST",
    body: JSON.stringify(inputs),
  });
}

export function getCurrentHealth() {
  return request("/api/health/current");
}

export function getHistory(lastN) {
  const q = lastN ? `?last_n=${lastN}` : "";
  return request(`/api/history${q}`);
}

export function resetSimulation() {
  return request("/api/reset", { method: "POST" });
}

export function mlAnalyze() {
  return request("/api/ml/analyze");
}

export function mlForecast(horizonS) {
  return request("/api/ml/forecast", {
    method: "POST",
    body: JSON.stringify({ horizon_s: horizonS }),
  });
}
