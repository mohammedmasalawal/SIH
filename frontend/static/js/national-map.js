// National thermal-detection map: all packed detections live on the GPU
// (deck.gl ScatterplotLayer over MapLibre); month and class filtering happen in the
// shader via DataFilterExtension, so changing filters never re-uploads the points.
(async function () {
  "use strict";

  const LANDCOVER = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
    60: "Bare / sparse", 70: "Snow / ice", 80: "Water", 90: "Herbaceous wetland",
    95: "Mangroves", 100: "Moss / lichen",
  };
  const INDIA = { lng: 80.5, lat: 22.5, zoom: 4.1 };
  const PLAY_INTERVAL_MS = 1200;

  const params = new URLSearchParams(location.search);
  const statusEl = document.getElementById("status");
  const slider = document.getElementById("month-slider");
  const monthLabel = document.getElementById("month-label");
  const allYearBox = document.getElementById("all-year");
  const playButton = document.getElementById("play");
  const timings = { start: performance.now() };

  const esc = (value) =>
    String(value ?? "n/a").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const hexToRgba = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).concat(255);
  const fmtCount = (n) => n.toLocaleString("en-US");

  async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status} ${await response.text()}`);
    return response.json();
  }

  async function fetchPoints() {
    const response = await fetch("/api/map/points.bin");
    if (!response.ok) throw new Error(`/api/map/points.bin: ${response.status} ${await response.text()}`);
    return response.arrayBuffer();
  }

  const map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: [Number(params.get("lng") ?? INDIA.lng), Number(params.get("lat") ?? INDIA.lat)],
    zoom: Number(params.get("zoom") ?? INDIA.zoom),
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  const mapLoaded = new Promise((resolve) => map.once("load", resolve));

  let meta, classesInfo, buffer;
  try {
    [meta, classesInfo, buffer] = await Promise.all([
      fetchJson("/api/map/meta"), fetchJson("/api/classes"), fetchPoints(), mapLoaded,
    ]);
  } catch (error) {
    statusEl.textContent = "Failed to load: " + error.message;
    statusEl.classList.add("error");
    console.error(error);
    return;
  }
  timings.fetched = performance.now();

  // --- unpack the struct-of-arrays binary (views, no copies) ---------------------
  const N = meta.count;
  const L = meta.layout;
  const positions = new Float32Array(buffer, L.positions.offset, L.positions.length);
  const rowIds = new Uint32Array(buffer, L.row_id.offset, N);
  const days = new Uint16Array(buffer, L.day.offset, N);
  const classIds = new Uint8Array(buffer, L.class_id.offset, N);

  const palette = meta.classes.map((label) => hexToRgba(classesInfo.colors_dark[label] || "#6b6a65"));
  const colors = new Uint8Array(N * 4);
  for (let i = 0; i < N; i++) colors.set(palette[classIds[i]], i * 4);
  const dayValues = Float32Array.from(days); // filter values are float attributes

  const pointData = {
    length: N,
    attributes: {
      getPosition: { value: positions, size: 2 },
      getFillColor: { value: colors, size: 4 },
      getFilterValue: { value: dayValues, size: 1 },
    },
  };
  const getFilterCategory = (_, { index }) => classIds[index];
  const filterExtension = new deck.DataFilterExtension({ filterSize: 1, categorySize: 1, countItems: true });

  // --- months -------------------------------------------------------------------
  const base = Date.parse(meta.base_date + "T00:00:00Z");
  const dayOf = (ms) => Math.round((ms - base) / 86400000);
  const months = [];
  for (let d = new Date(base); dayOf(d.getTime()) <= meta.max_day; ) {
    const next = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1));
    months.push({
      key: `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`,
      label: d.toLocaleString("en-US", { month: "short", year: "numeric", timeZone: "UTC" }),
      start: Math.max(0, dayOf(d.getTime())),
      end: Math.min(meta.max_day, dayOf(next.getTime()) - 1),
    });
    d = next;
  }

  const requestedMonth = months.findIndex((m) => m.key === params.get("month"));
  const state = {
    monthIndex: requestedMonth >= 0 ? requestedMonth : months.length - 1,
    allYear: params.get("all") === "1",
    active: new Set(meta.classes.map((_, i) => i)),
    visible: null,
  };

  // --- class toggles (the legend) -------------------------------------------------
  const togglesEl = document.getElementById("class-toggles");
  meta.classes.forEach((label, i) => {
    const row = document.createElement("label");
    row.className = "class-toggle";
    row.title = `${fmtCount(meta.class_counts[label])} detections in the full period`;
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.addEventListener("change", () => {
      if (box.checked) state.active.add(i);
      else state.active.delete(i);
      row.classList.toggle("off", !box.checked);
      render();
    });
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = classesInfo.colors_dark[label];
    const name = document.createElement("span");
    name.textContent = label;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = fmtCount(meta.class_counts[label]);
    row.append(box, swatch, name, count);
    togglesEl.appendChild(row);
  });

  // --- layer --------------------------------------------------------------------
  const firstSymbolId = map.getStyle().layers.find((layer) => layer.type === "symbol")?.id;
  const radiusForZoom = (zoom) => Math.round(Math.min(3, Math.max(0.9, 0.9 + (zoom - 4) * 0.3)) * 10) / 10;
  let radiusScale = radiusForZoom(map.getZoom());

  function filterRange() {
    if (state.allYear) return [0, meta.max_day];
    const month = months[state.monthIndex];
    return [month.start, month.end];
  }

  function buildLayer() {
    return new deck.ScatterplotLayer({
      id: "detections",
      data: pointData,
      beforeId: firstSymbolId, // points under place labels
      radiusUnits: "pixels",
      getRadius: 1,
      radiusScale,
      stroked: false,
      opacity: 0.85,
      pickable: true,
      extensions: [filterExtension],
      getFilterCategory,
      filterRange: filterRange(),
      filterCategories: [...state.active],
      onFilteredItemsChange: ({ count }) => {
        state.visible = count;
        updateStatus();
      },
    });
  }

  let popup = null;
  const overlay = new deck.MapboxOverlay({
    interleaved: true,
    pickingRadius: 5, // hit target larger than the 2-3 px marks
    layers: [],
    onAfterRender: () => {
      if (!timings.firstRender) {
        timings.firstRender = performance.now();
        updateStatus();
      }
    },
    onClick: (info) => {
      if (info.layer && info.index >= 0) showDetection(rowIds[info.index], info.coordinate);
      else if (popup) popup.remove(); // click on empty map closes it (see Popup options below)
    },
  });
  map.addControl(overlay);

  function render() {
    overlay.setProps({ layers: [buildLayer()] });
  }

  map.on("zoom", () => {
    const next = radiusForZoom(map.getZoom());
    if (next !== radiusScale) {
      radiusScale = next;
      render();
    }
  });

  // --- time controls --------------------------------------------------------------
  slider.max = String(months.length - 1);
  let playTimer = null;

  function syncControls() {
    slider.value = String(state.monthIndex);
    slider.disabled = state.allYear;
    monthLabel.textContent = state.allYear
      ? `${months[0].label} – ${months[months.length - 1].label}`
      : months[state.monthIndex].label;
    allYearBox.checked = state.allYear;
    playButton.innerHTML = playTimer ? "&#10074;&#10074;" : "&#9654;";
    playButton.setAttribute("aria-label", playTimer ? "Pause" : "Play months");
    const url = new URL(location.href);
    url.searchParams.set("month", months[state.monthIndex].key);
    if (state.allYear) url.searchParams.set("all", "1");
    else url.searchParams.delete("all");
    history.replaceState(null, "", url);
  }

  function update() {
    syncControls();
    render();
  }

  function stopPlay() {
    clearInterval(playTimer);
    playTimer = null;
  }

  slider.addEventListener("input", () => {
    state.monthIndex = Number(slider.value);
    update();
  });
  allYearBox.addEventListener("change", () => {
    state.allYear = allYearBox.checked;
    if (state.allYear) stopPlay();
    update();
  });
  playButton.addEventListener("click", () => {
    if (playTimer) {
      stopPlay();
    } else {
      state.allYear = false;
      playTimer = setInterval(() => {
        state.monthIndex = (state.monthIndex + 1) % months.length;
        update();
      }, PLAY_INTERVAL_MS);
    }
    update();
  });

  function updateStatus() {
    // since navigation start: includes page, libraries, basemap style and the point download
    const loadSeconds = (timings.firstRender ?? timings.fetched) / 1000;
    const shown = state.visible == null ? "" : ` · ${fmtCount(state.visible)} shown`;
    statusEl.textContent = `${fmtCount(N)} detections loaded in ${loadSeconds.toFixed(1)} s${shown}`;
  }

  // --- click popup ------------------------------------------------------------------
  const fmtDistance = (m) => (m == null ? "n/a" : m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`);

  function fmtTime(date, hhmm) {
    if (!hhmm) return esc(date);
    const minutes = Number(hhmm.slice(0, 2)) * 60 + Number(hhmm.slice(2, 4));
    const ist = (minutes + 330) % 1440;
    const pad = (n) => String(n).padStart(2, "0");
    return `${esc(date)} ${hhmm.slice(0, 2)}:${hhmm.slice(2, 4)} UTC (${pad(Math.floor(ist / 60))}:${pad(ist % 60)} IST)`;
  }

  function facility(type, distance) {
    return type ? `${esc(type.replaceAll("_", " "))} · ${fmtDistance(distance)}` : fmtDistance(distance);
  }

  function detectionHtml(d) {
    const color = classesInfo.colors_dark[d.label] || "#6b6a65";
    const rows = [
      ["Date", fmtTime(d.acq_date, d.acq_time)],
      ["Day/night", d.daynight === "D" ? "Day" : d.daynight === "N" ? "Night" : "n/a"],
      ["FRP", d.frp == null ? "n/a" : `${esc(d.frp)} MW`],
      ["Recurrence", d.recurrence_count == null ? "n/a" : `${esc(d.recurrence_count)} day${d.recurrence_count === 1 ? "" : "s"} at this cell`],
      ["Anomalous", d.is_anomalous ? "Yes" : "No"],
      ["Landcover", d.landcover_class == null ? "n/a" : `${esc(LANDCOVER[d.landcover_class] ?? "Unknown")} (${esc(d.landcover_class)})`],
      ["Heat industry", facility(d.nearest_heat_facility_type, d.dist_to_heat_industry_m)],
      ["Flare-capable", facility(d.nearest_flare_facility_type, d.dist_to_flare_capable_m)],
      ["Coal mine", `${fmtDistance(d.dist_to_coal_mine_m)}${d.nearest_coal_source ? ` (${esc(d.nearest_coal_source)})` : ""}`],
      ["OSM industrial", `${fmtDistance(d.dist_to_industrial_m)}${d.osm_industrial_tag ? ` · inside ${esc(d.osm_industrial_tag)}` : ""}`],
      ["Label source", esc(d.label_source)],
      ["Location", `${d.latitude.toFixed(5)}, ${d.longitude.toFixed(5)}`],
    ];
    return (
      `<div class="detection"><h2><span class="swatch" style="background:${color}"></span>${esc(d.label)}</h2><dl>` +
      rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") +
      "</dl></div>"
    );
  }

  async function showDetection(id, coordinate) {
    if (popup) popup.remove();
    // closeOnClick off: MapLibre's own click event fires after deck's onClick for the
    // same click, and would otherwise close the popup the moment it opens.
    popup = new maplibregl.Popup({ maxWidth: "340px", closeOnClick: false })
      .setLngLat(coordinate)
      .setHTML('<div class="detection">Loading…</div>')
      .addTo(map);
    const target = popup;
    try {
      const record = await fetchJson(`/api/detection/${id}`);
      if (target === popup) target.setHTML(detectionHtml(record));
    } catch (error) {
      if (target === popup) target.setHTML(`<div class="detection">Could not load detection ${id}: ${esc(error.message)}</div>`);
    }
  }

  update();
  // Debug/measurement handle (used by the headless benchmark; harmless otherwise).
  window.__nationalMap = { map, overlay, state, months, meta, timings, update, showDetection };
})();
