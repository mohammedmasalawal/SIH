function initMap() {
  const map = L.map("map").setView([22.3, 71.0], 7);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  return map;
}

function popupHtml(hotspot) {
  const escape = (value) =>
    String(value ?? "n/a").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  return (
    `<b>${escape(hotspot.label)}</b><br>` +
    `source: ${escape(hotspot.label_source)}<br>` +
    `date: ${escape(hotspot.acq_date)}<br>` +
    `frp: ${escape(hotspot.frp)}<br>` +
    `recurrence: ${escape(hotspot.recurrence_count)}`
  );
}

function plotHotspots(map, hotspots, colors) {
  const markersByLabel = {};
  for (const hotspot of hotspots) {
    const color = colors[hotspot.label] || "#000000";
    const marker = L.circleMarker([hotspot.latitude, hotspot.longitude], {
      radius: 7,
      color,
      fillColor: color,
      fillOpacity: 0.85,
      weight: 1,
    }).bindPopup(popupHtml(hotspot));
    marker.addTo(map);
    if (!markersByLabel[hotspot.label]) markersByLabel[hotspot.label] = [];
    markersByLabel[hotspot.label].push(marker);
  }
  return markersByLabel;
}

function applyFilter(map, markersByLabel, activeLabels) {
  for (const [label, markers] of Object.entries(markersByLabel)) {
    for (const marker of markers) {
      const shouldShow = activeLabels.has(label);
      const isShown = map.hasLayer(marker);
      if (shouldShow && !isShown) marker.addTo(map);
      if (!shouldShow && isShown) map.removeLayer(marker);
    }
  }
}

window.MapView = { initMap, plotHotspots, applyFilter };
