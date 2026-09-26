// Agninetra dashboard: national map + situation panels + timeline + alerts.
// Reads only static files under data/ (built by training/export_site.py), so it runs
// on any static host -- Vercel in production, the local FastAPI server in development.
(async function () {
  "use strict";

  const DATA = "data/";
  const LANDCOVER = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
    60: "Bare / sparse", 70: "Snow / ice", 80: "Water", 90: "Herbaceous wetland",
    95: "Mangroves", 100: "Moss / lichen",
  };
  const INDIA = { lng: 80.5, lat: 22.5, zoom: 4.1 };
  const PLAY_INTERVAL_MS = 1400;
  const TOP_STATES = 7;
  // Stack order in charts: bulk classes first, rare ones on top where they stay visible.
  const STACK_ORDER = ["agricultural burning", "wildfire", "unknown", "industrial", "gas flare"];
  const VERIFIED_STATES = new Set(["Gujarat"]); // where gold-set review has checked the labels
  const ALL_STATES = [0, 255]; // state filter range meaning "no state filter"
  // Hollow rings, kept clear of the class colours: GEM, OSM critical, other OSM industrial.
  const FACILITY_COLORS = ["#ffffff", "#45c1d6", "#8a97a4"];
  // Why a detection has its label (label_source), in words.
  const LABEL_REASONS = {
    rule: {
      "gas flare": "Rule match: within 1 km of a flare-capable GEM facility, active on 5+ days",
      industrial: "Rule match: at or near mapped heavy industry, an OSM industrial site, or a coal mine",
      "agricultural burning": "Rule match: WorldCover cropland, more than 2 km from mapped industry",
      wildfire: "Rule match: WorldCover tree cover / shrubland / grassland, more than 2 km from mapped industry",
    },
    unknown: "Needs review: no labelling rule matched",
    conflict: "Needs review: more than one labelling rule matched",
    model: "Model prediction",
  };

  const params = new URLSearchParams(location.search);
  const $ = (id) => document.getElementById(id);
  const statusEl = $("status");
  const timings = { start: performance.now() };

  const esc = (value) =>
    String(value ?? "n/a").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtCount = (n) => Math.round(n).toLocaleString("en-US");
  const pad = (n) => String(n).padStart(2, "0");
  const hexToRgba = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).concat(255);

  // --- static data (gzip-aware: decompress in the browser unless the host already did) ---
  async function fetchBytes(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      return new Uint8Array(await new Response(stream).arrayBuffer());
    }
    return bytes;
  }
  const fetchJson = async (url) => JSON.parse(new TextDecoder().decode(await fetchBytes(url)));

  // --- map ---------------------------------------------------------------------------
  const map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: [Number(params.get("lng") ?? INDIA.lng), Number(params.get("lat") ?? INDIA.lat)],
    zoom: Number(params.get("zoom") ?? INDIA.zoom),
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  const mapLoaded = new Promise((resolve) => map.once("load", resolve));

  let popup = null;
  function openPopup(lngLat, html) {
    if (popup) popup.remove();
    // closeOnClick off: MapLibre's click fires after deck's onClick for the same click
    // and would close the popup the moment it opens.
    popup = new maplibregl.Popup({ maxWidth: "340px", closeOnClick: false }).setLngLat(lngLat).setHTML(html).addTo(map);
    return popup;
  }

  // Alerts panel reads its own static CSV; start it now so it works even if the rest fails.
  let classColors = null;
  let displayNames = { unknown: "Unclassified — needs review" }; // replaced by classes.json
  const colorFor = (label) => classColors?.[label] ?? "#6b6a65";
  const nameFor = (label) => displayNames[label] ?? label;
  let onAlertSelect = null;
  const alertsReady = AlertsPanel.init({ map, openPopup, colorFor, nameFor, onSelect: (alert) => onAlertSelect?.(alert) });

  setupMethodDrawer();

  let meta, classesInfo, stats, buffer, detailIndex, rowStates, statesInfo;
  try {
    [meta, classesInfo, stats, buffer, detailIndex, rowStates, statesInfo] = await Promise.all([
      fetchJson(DATA + "meta.json"),
      fetchJson(DATA + "classes.json"),
      fetchJson(DATA + "stats.json.gz"),
      fetchBytes(DATA + "points.bin.gz"),
      fetchJson(DATA + "details/index.json"),
      fetchBytes(DATA + "point_state.bin.gz"),
      fetchJson(DATA + "states.json.gz"),
      mapLoaded,
    ]);
  } catch (error) {
    statusEl.textContent = `Detections unavailable: ${error.message}`;
    statusEl.classList.add("error");
    console.error(error);
    return;
  }
  timings.fetched = performance.now();
  classColors = classesInfo.colors_dark;
  displayNames = { ...displayNames, ...(classesInfo.display ?? {}) };
  alertsReady.then(({ recolor }) => recolor());
  document.querySelectorAll("[data-swatch]").forEach((dt) =>
    dt.insertAdjacentHTML("afterbegin", `<span class="swatch" style="background:${colorFor(dt.dataset.swatch)}"></span>`));
  $("method-range").textContent = `${meta.base_date} to ${meta.max_date} (UTC dates)`;

  const CLASSES = meta.classes;
  const stackOrder = STACK_ORDER.filter((c) => CLASSES.includes(c)).map((c) => CLASSES.indexOf(c));

  // --- packed points (struct-of-arrays views, no copies) ----------------------------------
  const N = meta.count;
  const L = meta.layout;
  const bin = buffer.buffer;
  const positions = new Float32Array(bin, buffer.byteOffset + L.positions.offset, L.positions.length);
  const rowIds = new Uint32Array(bin, buffer.byteOffset + L.row_id.offset, N);
  const days = new Uint16Array(bin, buffer.byteOffset + L.day.offset, N);
  const classIds = new Uint8Array(bin, buffer.byteOffset + L.class_id.offset, N);
  const palette = CLASSES.map((label) => hexToRgba(colorFor(label)));
  const colors = new Uint8Array(N * 4);
  const filterValues = new Float32Array(N * 2); // [day, state index] per point
  for (let i = 0; i < N; i++) {
    colors.set(palette[classIds[i]], i * 4);
    filterValues[i * 2] = days[i];
    filterValues[i * 2 + 1] = rowStates[rowIds[i]]; // point_state.bin is indexed by row id
  }
  const pointData = {
    length: N,
    attributes: {
      getPosition: { value: positions, size: 2 },
      getFillColor: { value: colors, size: 4 },
      getFilterValue: { value: filterValues, size: 2 },
    },
  };
  const getFilterCategory = (_, { index }) => classIds[index];
  const filterExtension = new deck.DataFilterExtension({ filterSize: 2, categorySize: 1, countItems: true });

  // --- dates and months ---------------------------------------------------------------------
  const base = Date.parse(meta.base_date + "T00:00:00Z");
  const dayOf = (ms) => Math.round((ms - base) / 86400000);
  const dayIso = (d) => new Date(base + d * 86400000).toISOString().slice(0, 10);
  const fmtDay = (d) => {
    const t = new Date(base + d * 86400000);
    return `${t.getUTCDate()} ${t.toLocaleString("en-US", { month: "short", timeZone: "UTC" })} ${t.getUTCFullYear()}`;
  };
  const months = [];
  for (let d = new Date(base); dayOf(d.getTime()) <= meta.max_day; ) {
    const next = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1));
    months.push({
      key: `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}`,
      label: d.toLocaleString("en-US", { month: "short", year: "numeric", timeZone: "UTC" }),
      short: d.toLocaleString("en-US", { month: "short", timeZone: "UTC" }),
      start: Math.max(0, dayOf(d.getTime())),
      end: Math.min(meta.max_day, dayOf(next.getTime()) - 1),
      fullDays: dayOf(next.getTime()) - dayOf(d.getTime()),
    });
    d = next;
  }

  const requestedMonth = months.findIndex((m) => m.key === params.get("month"));
  const requestedState = stats.states.indexOf(params.get("state"));
  const state = {
    monthIndex: requestedMonth >= 0 ? requestedMonth : months.length - 1,
    allYear: params.get("all") === "1",
    active: new Set(CLASSES.map((_, i) => i)),
    region: requestedState >= 0 ? requestedState : null, // index into stats.states, null = all India
    visible: null,
  };
  const range = () => (state.allYear ? [0, meta.max_day] : [months[state.monthIndex].start, months[state.monthIndex].end]);
  const periodLabel = () =>
    state.allYear ? `${months[0].label} – ${months[months.length - 1].label}` : months[state.monthIndex].label;

  // --- aggregates from stats.json (detections per day x class x state) ----------------------
  const nDays = meta.max_day + 1;
  const dayClass = Array.from({ length: nDays }, () => new Float64Array(CLASSES.length));
  const dayAnom = Array.from({ length: nDays }, () => new Float64Array(CLASSES.length));
  function buildDaily() { // per-day totals for the selected region
    dayClass.forEach((row) => row.fill(0));
    dayAnom.forEach((row) => row.fill(0));
    for (let i = 0; i < stats.n.length; i++) {
      if (state.region != null && stats.state[i] !== state.region) continue;
      dayClass[stats.day[i]][stats.cls[i]] += stats.n[i];
      dayAnom[stats.day[i]][stats.cls[i]] += stats.anom[i];
    }
  }
  buildDaily();

  function aggregate([from, to]) { // byClass / anom: selected region; byState: all states
    const byClass = new Float64Array(CLASSES.length);
    let anom = 0;
    for (let d = from; d <= to; d++) {
      for (const c of state.active) {
        byClass[c] += dayClass[d][c];
        anom += dayAnom[d][c];
      }
    }
    const byState = new Map();
    for (let i = 0; i < stats.n.length; i++) {
      const d = stats.day[i];
      if (d < from || d > to || !state.active.has(stats.cls[i])) continue;
      const row = byState.get(stats.state[i]) ?? byState.set(stats.state[i], new Float64Array(CLASSES.length)).get(stats.state[i]);
      row[stats.cls[i]] += stats.n[i];
    }
    return { byClass, total: byClass.reduce((a, b) => a + b, 0), anom, byState, days: to - from + 1 };
  }

  // --- layer ------------------------------------------------------------------------------------
  const firstSymbolId = map.getStyle().layers.find((layer) => layer.type === "symbol")?.id;
  const radiusForZoom = (zoom) => Math.round(Math.min(3, Math.max(0.9, 0.9 + (zoom - 4) * 0.3)) * 10) / 10;
  let radiusScale = radiusForZoom(map.getZoom());

  function buildLayer() {
    return new deck.ScatterplotLayer({
      id: "detections",
      data: pointData,
      beforeId: firstSymbolId,
      radiusUnits: "pixels",
      getRadius: 1,
      radiusScale,
      stroked: false,
      opacity: 0.85,
      pickable: true,
      extensions: [filterExtension],
      getFilterCategory,
      filterRange: [range(), state.region == null ? ALL_STATES : [state.region, state.region]],
      filterCategories: [...state.active],
      onFilteredItemsChange: ({ count }) => { state.visible = count; updateStatus(); },
    });
  }

  const overlay = new deck.MapboxOverlay({
    interleaved: true,
    pickingRadius: 5,
    layers: [],
    onAfterRender: () => {
      if (!timings.firstRender) { timings.firstRender = performance.now(); updateStatus(); }
    },
    onClick: (info) => {
      if (info.layer?.id === "facilities" && info.index >= 0) showFacility(info.index);
      else if (info.layer && info.index >= 0) showDetection(rowIds[info.index], info.coordinate);
      else if (popup) popup.remove();
    },
  });
  map.addControl(overlay);

  // --- known facilities (loaded on first use) -------------------------------------------------
  let facilities = null, facilityLayer = null;
  function buildFacilityLayer() {
    const f = facilities;
    const count = f.lat.length;
    const pos = new Float32Array(count * 2), lineColors = new Uint8Array(count * 4);
    const rgba = FACILITY_COLORS.map(hexToRgba);
    for (let i = 0; i < count; i++) {
      pos[i * 2] = f.lon[i] / 1e5; pos[i * 2 + 1] = f.lat[i] / 1e5;
      lineColors.set(rgba[f.group[i]], i * 4);
    }
    return new deck.ScatterplotLayer({
      id: "facilities",
      data: { length: count, attributes: { getPosition: { value: pos, size: 2 }, getLineColor: { value: lineColors, size: 4 } } },
      radiusUnits: "pixels", getRadius: 4, radiusMinPixels: 3,
      filled: true, getFillColor: [8, 12, 17, 90], stroked: true, lineWidthUnits: "pixels", getLineWidth: 1.5,
      pickable: true,
    });
  }
  const facilitiesToggle = $("facilities-toggle");
  facilitiesToggle.addEventListener("change", async () => {
    if (facilitiesToggle.checked && !facilities) {
      facilitiesToggle.disabled = true;
      try {
        facilities = await fetchJson(DATA + "facilities.json.gz");
        facilityLayer = buildFacilityLayer();
        const counts = facilities.groups.map((_, g) => facilities.group.filter((x) => x === g).length);
        $("facilities-legend").innerHTML = facilities.groups.map((name, g) =>
          `<div style="color:${FACILITY_COLORS[g]}"><span class="ring"></span><span style="color:var(--text-secondary)">${esc(name)} · ${fmtCount(counts[g])}</span></div>`).join("");
      } catch (error) {
        facilitiesToggle.checked = false;
        $("facilities-legend").textContent = `Facilities unavailable: ${error.message}`;
        $("facilities-legend").hidden = false;
        console.warn(error);
      } finally {
        facilitiesToggle.disabled = false;
      }
    }
    $("facilities-legend").hidden = !facilitiesToggle.checked && !!facilities;
    syncControls();
    renderMap();
  });
  function showFacility(i) {
    const f = facilities;
    const lngLat = [f.lon[i] / 1e5, f.lat[i] / 1e5];
    const rows = [
      ["Type", esc(f.kinds[f.kind[i]])],
      ["Group", esc(f.groups[f.group[i]])],
      ["Source", f.sources[f.source[i]] === "OSM" ? "OpenStreetMap (a point inside the mapped outline)" : "Global Energy Monitor"],
      ["Location", `${lngLat[1].toFixed(5)}, ${lngLat[0].toFixed(5)}`],
    ];
    openPopup(lngLat, `<div class="detection"><h2><span class="swatch" style="background:transparent;border:2px solid ${FACILITY_COLORS[f.group[i]]}"></span>${esc(f.name[i] ?? "Unnamed facility")}</h2><dl>` +
      rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") + "</dl></div>");
  }

  const renderMap = () => overlay.setProps({
    layers: [buildLayer(), ...(facilitiesToggle.checked && facilityLayer ? [facilityLayer] : [])],
  });
  map.on("zoom", () => {
    const next = radiusForZoom(map.getZoom());
    if (next !== radiusScale) { radiusScale = next; renderMap(); }
  });

  // --- header status --------------------------------------------------------------------------
  function istStamp(isoUtc) {
    const t = new Date(Date.parse(isoUtc) + 330 * 60000);
    return `${t.getUTCDate()} ${t.toLocaleString("en-US", { month: "short", timeZone: "UTC" })} ${pad(t.getUTCHours())}:${pad(t.getUTCMinutes())} IST`;
  }
  function updateStatus() {
    const exported = meta.exported_at ? ` · updated ${istStamp(meta.exported_at)}` : "";
    const shown = state.visible == null ? "" : ` · ${fmtCount(state.visible)} on map`;
    statusEl.textContent = `Data through ${fmtDay(meta.max_day)}${exported}${shown}`;
    const ageHours = meta.exported_at ? (Date.now() - Date.parse(meta.exported_at)) / 3.6e6 : Infinity;
    $("live-dot").className = `live-dot ${ageHours <= 12 ? "live" : "stale"}`;
    $("live-dot").title = ageHours <= 12 ? "Updated within the last 12 hours" : "Not updated in over 12 hours";
  }

  // --- situation KPIs -------------------------------------------------------------------------
  let alertList = [];
  function renderKpis(agg) {
    const [from, to] = range();
    const perDay = agg.total / agg.days;
    let compare = "";
    if (!state.allYear && state.monthIndex > 0) {
      const prev = months[state.monthIndex - 1];
      const prevAgg = aggregate([prev.start, prev.end]);
      const prevPerDay = prevAgg.total / prevAgg.days;
      if (prevPerDay > 0) {
        const change = (perDay / prevPerDay - 1) * 100;
        const cls = change >= 0 ? "delta-up" : "delta-down";
        compare = `<span class="${cls}">${change >= 0 ? "▲" : "▼"} ${Math.abs(change).toFixed(0)}%</span> vs ${esc(prev.short)} (${fmtCount(prevPerDay)}/day)`;
      }
    }
    const fromIso = dayIso(from), toIso = dayIso(to);
    const alertsIn = alertList.filter((a) => a.date >= fromIso && a.date <= toIso).length;
    const partial = !state.allYear && months[state.monthIndex].end - months[state.monthIndex].start + 1 < months[state.monthIndex].fullDays;
    $("kpi-period").textContent = (state.region != null ? `${stats.states[state.region]} · ` : "") + periodLabel() + (partial ? " · to date" : "");
    $("kpis").innerHTML = [
      ["Detections", fmtCount(agg.total), `${agg.days} day${agg.days === 1 ? "" : "s"}`],
      ["Per day", fmtCount(perDay), compare || "average"],
      ["Anomalous", fmtCount(agg.anom), agg.total ? `${((agg.anom / agg.total) * 100).toFixed(2)}% of detections` : ""],
      ["Alerts", fmtCount(alertsIn), "raised in period"],
    ].map(([label, value, sub]) =>
      `<div class="kpi"><div class="kpi-label">${esc(label)}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`
    ).join("");
  }

  // --- class mix: ring gauges that double as the class legend + toggles -------------------------
  function renderClassMix(agg) {
    const [from, to] = range();
    const totals = new Float64Array(CLASSES.length);
    for (let d = from; d <= to; d++) for (let c = 0; c < CLASSES.length; c++) totals[c] += dayClass[d][c];
    const all = totals.reduce((a, b) => a + b, 0) || 1;
    const R = 20, C = 2 * Math.PI * R;
    $("class-mix").innerHTML = stackOrder.map((c) => {
      const share = totals[c] / all;
      const on = state.active.has(c);
      return `<button type="button" class="ring-btn" data-cls="${c}" aria-pressed="${on}" title="${esc(nameFor(CLASSES[c]))}: ${fmtCount(totals[c])} (${(share * 100).toFixed(1)}%)">
        <svg width="52" height="52" viewBox="0 0 52 52" aria-hidden="true">
          <circle cx="26" cy="26" r="${R}" fill="none" stroke="#1e2a36" stroke-width="5"/>
          <circle cx="26" cy="26" r="${R}" fill="none" stroke="${colorFor(CLASSES[c])}" stroke-width="5" stroke-linecap="round"
            stroke-dasharray="${Math.max(0.001, share * C)} ${C}" transform="rotate(-90 26 26)"/>
          <text x="26" y="30" text-anchor="middle" class="ring-pct">${share >= 0.1 ? Math.round(share * 100) : (share * 100).toFixed(1)}%</text>
        </svg>
        <span class="ring-name">${esc(nameFor(CLASSES[c]))}</span>
        <span class="ring-count">${fmtCount(totals[c])}</span>
      </button>`;
    }).join("");
  }
  $("class-mix").addEventListener("click", (event) => {
    const btn = event.target.closest(".ring-btn");
    if (!btn) return;
    const c = Number(btn.dataset.cls);
    if (state.active.has(c)) state.active.delete(c);
    else state.active.add(c);
    update({ classesChanged: true });
  });

  // --- top states ---------------------------------------------------------------------------
  const statesTip = $("states-tip");
  function renderStates(agg) {
    const rows = [...agg.byState.entries()]
      .map(([s, byClass]) => ({ index: s, name: stats.states[s], byClass, total: byClass.reduce((a, b) => a + b, 0) }))
      .filter((r) => r.total > 0)
      .sort((a, b) => b.total - a.total)
      .slice(0, TOP_STATES);
    if (!rows.length) {
      $("states").innerHTML = '<p class="empty-note">No detections in this period for the selected classes.</p>';
      return;
    }
    const max = rows[0].total;
    $("states").innerHTML = rows.map((r, i) =>
      `<div class="state-row${r.index === state.region ? " selected" : ""}" data-i="${i}" title="Show ${esc(r.name)}">
        <div class="state-line"><span class="state-name">${esc(r.name)}</span><span class="state-count">${fmtCount(r.total)}</span></div>
        <div class="state-bar" style="width:${Math.max(4, (r.total / max) * 100)}%">${
          stackOrder.filter((c) => r.byClass[c] > 0).map((c) =>
            `<span style="flex:${r.byClass[c]};background:${colorFor(CLASSES[c])}"></span>`).join("")
        }</div>
      </div>`
    ).join("");
    $("states").onmousemove = (event) => {
      const row = event.target.closest(".state-row");
      if (!row) { statesTip.hidden = true; return; }
      const r = rows[Number(row.dataset.i)];
      statesTip.innerHTML = `<div class="tip-title">${esc(r.name)}</div>` + stackOrder.filter((c) => r.byClass[c] > 0).map((c) =>
        `<div class="tip-row"><span class="swatch" style="background:${colorFor(CLASSES[c])}"></span>${esc(nameFor(CLASSES[c]))}<b>${fmtCount(r.byClass[c])}</b></div>`).join("");
      placeTip(statesTip, event);
    };
    $("states").onmouseleave = () => { statesTip.hidden = true; };
    $("states").onclick = (event) => {
      const row = event.target.closest(".state-row");
      if (row) selectRegion(rows[Number(row.dataset.i)].index, { fly: true });
    };
  }

  function placeTip(tip, event) {
    tip.hidden = false;
    const { innerWidth: w, innerHeight: h } = window;
    const rect = tip.getBoundingClientRect();
    tip.style.left = `${Math.min(event.clientX + 14, w - rect.width - 8)}px`;
    tip.style.top = `${Math.min(Math.max(8, event.clientY - rect.height - 10), h - rect.height - 8)}px`;
  }

  // --- timeline: detections per day, stacked by class --------------------------------------------
  const chartEl = $("timeline-chart");
  const tlTip = $("timeline-tip");
  let timelineGeom = null;
  function renderTimeline() {
    const width = chartEl.clientWidth, height = chartEl.clientHeight;
    if (width < 50 || height < 40) return;
    const m = { l: 34, r: 6, t: 4, b: 16 };
    const plotW = width - m.l - m.r, plotH = height - m.t - m.b;
    const step = plotW / nDays;
    let max = 1;
    const totals = dayClass.map((row) => { let t = 0; for (const c of state.active) t += row[c]; if (t > max) max = t; return t; });
    const y = (v) => m.t + plotH - (v / max) * plotH;
    const barW = Math.max(0.6, step - (step > 3 ? 1 : 0));
    const paths = stackOrder.filter((c) => state.active.has(c)).map((c) => {
      let d = "";
      for (let day = 0; day < nDays; day++) {
        let below = 0;
        for (const o of stackOrder) { if (o === c) break; if (state.active.has(o)) below += dayClass[day][o]; }
        const v = dayClass[day][c];
        if (!v) continue;
        const x = m.l + day * step, y0 = y(below), y1 = y(below + v);
        d += `M${x.toFixed(1)} ${y0.toFixed(1)}V${y1.toFixed(1)}h${barW.toFixed(2)}V${y0.toFixed(1)}z`;
      }
      return `<path d="${d}" fill="${colorFor(CLASSES[c])}"/>`;
    }).join("");
    const ticks = [0, max / 2, max].map((v) =>
      `<g class="tl-axis"><line x1="${m.l}" x2="${width - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 4}" y="${y(v) + 3}" text-anchor="end">${fmtCount(v)}</text></g>`).join("");
    const monthBands = months.map((mo, i) => {
      const x = m.l + mo.start * step, w = (mo.end - mo.start + 1) * step;
      return `<rect class="tl-month" data-month="${i}" x="${x}" y="${m.t}" width="${w}" height="${plotH}"/>` +
        `<text class="tl-label" x="${x + 2}" y="${height - 4}" fill="#6f7d8b" font-size="10">${esc(mo.short)}</text>`;
    }).join("");
    chartEl.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Detections per day, stacked by class">
      ${ticks}<rect id="tl-selected" class="tl-selected" x="0" y="${m.t}" width="0" height="${plotH}"/>${paths}${monthBands}
      <line id="tl-cursor" class="tl-cursor" x1="0" x2="0" y1="${m.t}" y2="${m.t + plotH}" visibility="hidden"/></svg>`;
    timelineGeom = { m, step, plotH, totals };
    markTimelineSelection();
  }
  function markTimelineSelection() {
    const sel = chartEl.querySelector("#tl-selected");
    if (!sel || !timelineGeom) return;
    const [from, to] = range();
    sel.setAttribute("x", timelineGeom.m.l + from * timelineGeom.step);
    sel.setAttribute("width", (to - from + 1) * timelineGeom.step);
  }
  chartEl.addEventListener("mousemove", (event) => {
    if (!timelineGeom) return;
    const rect = chartEl.getBoundingClientRect();
    const day = Math.floor((event.clientX - rect.left - timelineGeom.m.l) / timelineGeom.step);
    const cursor = chartEl.querySelector("#tl-cursor");
    if (day < 0 || day >= nDays) { tlTip.hidden = true; cursor?.setAttribute("visibility", "hidden"); return; }
    const x = timelineGeom.m.l + (day + 0.5) * timelineGeom.step;
    cursor?.setAttribute("x1", x); cursor?.setAttribute("x2", x); cursor?.setAttribute("visibility", "visible");
    tlTip.innerHTML = `<div class="tip-title">${esc(fmtDay(day))} · ${fmtCount(timelineGeom.totals[day])}</div>` +
      stackOrder.filter((c) => state.active.has(c) && dayClass[day][c] > 0).slice().reverse().map((c) =>
        `<div class="tip-row"><span class="swatch" style="background:${colorFor(CLASSES[c])}"></span>${esc(nameFor(CLASSES[c]))}<b>${fmtCount(dayClass[day][c])}</b></div>`).join("");
    placeTip(tlTip, event);
  });
  chartEl.addEventListener("mouseleave", () => {
    tlTip.hidden = true;
    chartEl.querySelector("#tl-cursor")?.setAttribute("visibility", "hidden");
  });
  chartEl.addEventListener("click", (event) => {
    const band = event.target.closest(".tl-month");
    if (!band) return;
    stopPlay();
    state.allYear = false;
    state.monthIndex = Number(band.dataset.month);
    update();
  });
  new ResizeObserver(() => renderTimeline()).observe(chartEl);

  // --- region selector -----------------------------------------------------------------------
  const stateSelect = $("state-select");
  const regionNames = stats.states.map((name, i) => ({ name, i })).sort((a, b) =>
    (a.name.startsWith("(") - b.name.startsWith("(")) || a.name.localeCompare(b.name));
  stateSelect.insertAdjacentHTML("beforeend", regionNames.map(({ name, i }) =>
    `<option value="${i}">${esc(name)}${VERIFIED_STATES.has(name) ? " ✓ verified" : ""}</option>`).join(""));
  map.addSource("region-outline", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({ id: "region-outline", type: "line", source: "region-outline",
                 paint: { "line-color": "#45c1d6", "line-width": 1.6, "line-opacity": 0.9 } }, firstSymbolId);
  function selectRegion(index, { fly = false } = {}) {
    state.region = index;
    stateSelect.value = index == null ? "" : String(index);
    const name = index == null ? null : stats.states[index];
    const note = $("state-validation");
    if (name == null) note.textContent = "Accuracy verified in Gujarat; other states not yet validated.";
    else if (VERIFIED_STATES.has(name)) note.innerHTML = '<span class="badge verified">verified</span>Accuracy verified in Gujarat (Jamnagar gold sets); other states not yet validated.';
    else note.innerHTML = `<span class="badge unvalidated">not validated</span>Accuracy verified in Gujarat; ${esc(name)} not yet validated.`;
    const outline = statesInfo.outlines.features.filter((f) => f.properties.name === name);
    map.getSource("region-outline").setData({ type: "FeatureCollection", features: outline });
    if (fly) {
      const box = name != null && statesInfo.bbox[name];
      if (box) map.fitBounds([[box[0], box[1]], [box[2], box[3]]], { padding: 40, duration: 900 });
      else if (index == null) map.flyTo({ center: [INDIA.lng, INDIA.lat], zoom: INDIA.zoom, duration: 900 });
    }
    buildDaily();
    update({ classesChanged: true });
  }
  stateSelect.addEventListener("change", () => selectRegion(stateSelect.value === "" ? null : Number(stateSelect.value), { fly: true }));

  // --- time controls ---------------------------------------------------------------------------
  const playButton = $("play"), allYearBox = $("all-year");
  let playTimer = null;
  function stopPlay() { clearInterval(playTimer); playTimer = null; }
  allYearBox.addEventListener("change", () => { state.allYear = allYearBox.checked; if (state.allYear) stopPlay(); update(); });
  playButton.addEventListener("click", () => {
    if (playTimer) stopPlay();
    else {
      state.allYear = false;
      playTimer = setInterval(() => { state.monthIndex = (state.monthIndex + 1) % months.length; update(); }, PLAY_INTERVAL_MS);
    }
    update();
  });

  function syncControls() {
    $("period-chip").textContent = periodLabel();
    allYearBox.checked = state.allYear;
    playButton.innerHTML = playTimer ? "&#10074;&#10074;" : "&#9654;";
    playButton.setAttribute("aria-label", playTimer ? "Pause" : "Play months");
    const url = new URL(location.href);
    url.searchParams.set("month", months[state.monthIndex].key);
    if (state.allYear) url.searchParams.set("all", "1"); else url.searchParams.delete("all");
    if (state.region != null) url.searchParams.set("state", stats.states[state.region]); else url.searchParams.delete("state");
    if (facilitiesToggle.checked) url.searchParams.set("facilities", "1"); else url.searchParams.delete("facilities");
    history.replaceState(null, "", url);
  }

  function update({ classesChanged = false } = {}) {
    syncControls();
    renderMap();
    const agg = aggregate(range());
    renderKpis(agg);
    renderClassMix(agg);
    renderStates(agg);
    if (classesChanged) renderTimeline(); else markTimelineSelection();
    updateStatus();
  }

  // --- click details from static shards -----------------------------------------------------------
  const shardCache = new Map();
  async function detailRecord(rowId) {
    const shard = Math.floor(rowId / detailIndex.shard_rows);
    if (!shardCache.has(shard)) shardCache.set(shard, fetchJson(`${DATA}details/${String(shard).padStart(4, "0")}.json.gz`));
    const cols = await shardCache.get(shard);
    const i = rowId - shard * detailIndex.shard_rows;
    const dict = (f) => (cols[f][i] >= 0 ? detailIndex.dicts[f][cols[f][i]] : null);
    const int = (f) => cols[f][i];
    return {
      latitude: cols.lat[i] / 1e5, longitude: cols.lon[i] / 1e5,
      acq_date: dayIso(cols.day[i]), acq_time: int("acq_time") == null ? null : String(int("acq_time")).padStart(4, "0"),
      frp: cols.frp[i] == null ? null : cols.frp[i] / 100, is_anomalous: !!cols.anom[i],
      recurrence_count: int("recurrence_count"), landcover_class: int("landcover_class"),
      dist_to_heat_industry_m: int("dist_to_heat_industry_m"), dist_to_flare_capable_m: int("dist_to_flare_capable_m"),
      dist_to_coal_mine_m: int("dist_to_coal_mine_m"), dist_to_industrial_m: int("dist_to_industrial_m"),
      label: dict("label"), satellite: dict("satellite"), daynight: dict("daynight"),
      nearest_heat_facility_type: dict("nearest_heat_facility_type"), nearest_flare_facility_type: dict("nearest_flare_facility_type"),
      nearest_coal_source: dict("nearest_coal_source"), osm_industrial_tag: dict("osm_industrial_tag"), label_source: dict("label_source"),
    };
  }

  const fmtDistance = (m) => (m == null ? "n/a" : m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`);
  function fmtTime(date, hhmm) { // IST first (can roll into the next day), FIRMS' UTC in brackets
    if (!hhmm) return esc(date);
    const t = new Date(Date.parse(`${date}T${hhmm.slice(0, 2)}:${hhmm.slice(2, 4)}:00Z`) + 330 * 60000);
    return `${esc(t.toISOString().slice(0, 10))} ${pad(t.getUTCHours())}:${pad(t.getUTCMinutes())} IST (${esc(date)} ${hhmm.slice(0, 2)}:${hhmm.slice(2, 4)} UTC)`;
  }
  const facility = (type, distance) => (type ? `${esc(type.replaceAll("_", " "))} · ${fmtDistance(distance)}` : fmtDistance(distance));

  function detectionHtml(d) {
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
    const reason = d.label_source === "rule" ? LABEL_REASONS.rule[d.label] : LABEL_REASONS[d.label_source];
    rows.unshift(["Why", `<span class="reason">${esc(reason ?? `label source: ${d.label_source ?? "n/a"}`)}</span>`]);
    return `<div class="detection"><h2><span class="swatch" style="background:${colorFor(d.label)}"></span>${esc(nameFor(d.label))}</h2><dl>` +
      rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") + "</dl></div>";
  }

  async function showDetection(rowId, coordinate) {
    const target = openPopup(coordinate, '<div class="detection">Loading…</div>');
    try {
      const record = await detailRecord(rowId);
      if (target === popup) target.setHTML(detectionHtml(record));
    } catch (error) {
      if (target === popup) target.setHTML(`<div class="detection">Could not load detection ${rowId}: ${esc(error.message)}</div>`);
    }
  }

  // --- Validation & Method drawer ---------------------------------------------------------------
  function setupMethodDrawer() {
    const drawer = $("method-drawer"), backdrop = $("method-backdrop"), opener = $("method-open");
    const setOpen = (open) => {
      drawer.hidden = backdrop.hidden = !open;
      opener.setAttribute("aria-expanded", String(open));
      if (open) $("method-close").focus(); else opener.focus();
    };
    opener.addEventListener("click", () => setOpen(true));
    $("method-close").addEventListener("click", () => setOpen(false));
    backdrop.addEventListener("click", () => setOpen(false));
    document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !drawer.hidden) setOpen(false); });
    if (params.get("method") === "1") setOpen(true);
  }

  // --- alerts: count them per period; an alert click shows its month ---------------------------
  alertsReady.then(({ alerts }) => { alertList = alerts; update(); });
  onAlertSelect = (alert) => {
    const index = months.findIndex((mo) => mo.key === alert.date.slice(0, 7));
    if (state.allYear || index < 0 || index === state.monthIndex) return;
    stopPlay();
    state.monthIndex = index;
    update();
  };

  if (state.region != null) selectRegion(state.region, { fly: !params.has("lat") });
  else update({ classesChanged: true });
  if (params.get("facilities") === "1") {
    facilitiesToggle.checked = true;
    facilitiesToggle.dispatchEvent(new Event("change"));
  }
  if (params.get("alert")) {
    alertsReady.then(({ select }) => {
      if (!select(params.get("alert"))) console.warn(`alert ${params.get("alert")} not found`);
    });
  }
  window.__dashboard = { map, overlay, state, months, meta, timings, update, showDetection, detailRecord, selectRegion };
})();
