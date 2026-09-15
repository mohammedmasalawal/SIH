const API_BASE = "/api";

async function fetchHotspots() {
  const response = await fetch(`${API_BASE}/hotspots`);
  if (!response.ok) throw new Error(`Failed to fetch hotspots: ${response.status}`);
  return response.json();
}

async function fetchClasses() {
  const response = await fetch(`${API_BASE}/classes`);
  if (!response.ok) throw new Error(`Failed to fetch classes: ${response.status}`);
  return response.json();
}

async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error(`Failed to fetch health: ${response.status}`);
  return response.json();
}

async function classifyHotspot(payload) {
  const response = await fetch(`${API_BASE}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && body.detail ? JSON.stringify(body.detail) : response.statusText;
    throw new Error(`Classify failed (${response.status}): ${detail}`);
  }
  return body;
}

window.Api = { fetchHotspots, fetchClasses, fetchHealth, classifyHotspot };
