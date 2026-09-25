// Alerts side panel (all alert types, filterable): reads alerts_history.csv as a plain static file and builds
// everything (list, popup) from its rows -- no API calls, so it works on any static
// host that serves data/alerts_history.csv next to index.html.
(function () {
  "use strict";

  const ALERTS_URL = "data/alerts_history.csv";

  const esc = (value) =>
    String(value ?? "n/a").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // RFC 4180: quoted fields may contain commas, quotes ("") and newlines.
  function parseCsv(text) {
    const rows = [];
    let row = [], field = "", quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (c === '"') quoted = false;
        else field += c;
      } else if (c === '"') quoted = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(field); field = "";
        if (row.length > 1 || row[0] !== "") rows.push(row);
        row = [];
      } else field += c;
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    const [header, ...body] = rows;
    return body.map((cells) => Object.fromEntries(header.map((name, i) => [name, cells[i] ?? ""])));
  }

  const num = (v) => (v === "" || v == null ? null : Number(v));
  const pad = (n) => String(n).padStart(2, "0");

  // Alert types (training/alerts.py). Rows written before types existed are industrial_anomaly.
  const TYPES = {
    industrial_anomaly: "Industrial anomaly",
    new_activity_at_critical_site: "New activity at critical site",
    new_unmapped_source: "New unmapped source",
    large_fire_event: "Large fire event",
  };

  function toAlert(r) {
    const time = r.acq_time ? String(r.acq_time).padStart(4, "0") : "";
    const type = TYPES[r.alert_type] ? r.alert_type : "industrial_anomaly";
    return {
      type, id: r.alert_id,
      lat: num(r.latitude), lng: num(r.longitude),
      date: r.acq_date, time, sortKey: `${r.acq_date}T${time || "9999"}`,
      label: r.label, frp: num(r.frp), normal: num(r.site_normal_frp), ratio: num(r.frp_vs_normal),
      priorDays: num(r.prior_active_days), daynight: r.daynight, satellite: r.satellite,
      facility: (r.facility_type || "").replaceAll("_", " "), facilityName: r.facility_name || "",
      facilityDistance: num(r.facility_distance_m),
      activeDays: num(r.active_days_30d), recurrence: num(r.recurrence_count), firstSeen: r.first_seen || "",
      eventCount: num(r.event_count), dominantClass: r.dominant_class || "",
    };
  }

  function fmtDay(isoDate) {
    const d = new Date(`${isoDate}T00:00:00Z`);
    const month = d.toLocaleString("en-US", { month: "short", timeZone: "UTC" }); // "Sep", matching the slider
    return `${d.getUTCDate()} ${month} ${d.getUTCFullYear()}`;
  }
  // FIRMS reports UTC; IST (UTC+5:30) can fall on the next calendar day, so shift the
  // whole timestamp, not just the clock.
  function istParts(isoDate, hhmm) {
    const t = new Date(Date.parse(`${isoDate}T${hhmm.slice(0, 2)}:${hhmm.slice(2)}:00Z`) + 330 * 60000);
    return { day: fmtDay(t.toISOString().slice(0, 10)), time: `${pad(t.getUTCHours())}:${pad(t.getUTCMinutes())}` };
  }
  const fmtDate = (a) => {
    if (!a.time) return fmtDay(a.date); // day-level alerts (events, first seen) keep the FIRMS date
    const ist = istParts(a.date, a.time);
    return `${ist.day} ${ist.time} IST`;
  };
  const fmtUtc = (a) => `${fmtDay(a.date)} ${a.time.slice(0, 2)}:${a.time.slice(2)} UTC`;
  const fmtFrp = (v) => (v == null ? "n/a" : `${v.toFixed(v < 10 ? 2 : 1)} MW`);
  const fmtDistance = (m) => (m == null ? "n/a" : m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`);
  const fmtSite = (a) => `${a.lat.toFixed(4)}°N, ${a.lng.toFixed(4)}°E`;
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const siteName = (a) => (a.facilityName && a.facilityName !== "unnamed" ? a.facilityName : `Unnamed ${a.facility.split(" · ")[0]}`);
  const swatchLabel = (a) => (a.type === "large_fire_event" ? a.dominantClass : a.label);

  // What each type shows: list title, the number on the right, one detail line, and popup rows.
  const VIEW = {
    industrial_anomaly: {
      title: (a) => a.facility,
      metric: (a) => `${a.ratio?.toFixed(1)}× normal`,
      metricHint: "FRP as a multiple of this site's normal (median of its earlier days)",
      detail: (a) => `${fmtSite(a)} · ${fmtFrp(a.frp)} vs ${fmtFrp(a.normal)} normal`,
      rows: (a) => [
        ["FRP", `${esc(fmtFrp(a.frp))} — <strong>${esc(a.ratio?.toFixed(1))}×</strong> the site's normal`],
        ["Site normal", `${esc(fmtFrp(a.normal))} (median of ${esc(a.priorDays)} prior days)`],
        ["Facility", `${esc(a.facility)} · ${esc(fmtDistance(a.facilityDistance))}`],
      ],
    },
    new_activity_at_critical_site: {
      title: siteName,
      metric: (a) => (a.facilityDistance != null && a.facilityDistance < 1 ? "inside" : `${fmtDistance(a.facilityDistance)} away`),
      metricHint: "Distance to the facility; \"inside\" = within its mapped outline",
      detail: (a) => `${fmtSite(a)} · ${a.facility}`,
      rows: (a) => [
        ["Facility", `${esc(siteName(a))}<br>${esc(a.facility)}`],
        ["Why", "no other activity within ~1 km in the previous 90 days"],
        ["Distance", a.facilityDistance != null && a.facilityDistance < 1 ? "inside the facility's mapped outline" : esc(fmtDistance(a.facilityDistance))],
        ["FRP", esc(fmtFrp(a.frp))],
      ],
    },
    new_unmapped_source: {
      title: () => "Unmapped heat source",
      metric: (a) => `${a.activeDays} of 30 days`,
      metricHint: "Active days at this cell in the last 30 days",
      detail: (a) => `${fmtSite(a)} · first seen ${fmtDay(a.firstSeen)}`,
      rows: (a) => [
        ["Active days", `${esc(a.activeDays)} in the last 30 days (${esc(plural(a.recurrence, "day"))} ever)`],
        ["First seen", esc(fmtDay(a.firstSeen))],
        ["Max FRP", esc(fmtFrp(a.frp))],
        ["Nearby", "no known facility within 2 km"],
      ],
    },
    large_fire_event: {
      title: (a) => `${plural(a.eventCount, "fire detection")} in one day`,
      metric: (a) => `${a.eventCount} fires`,
      metricHint: "Crop/wildfire detections in the same-day cluster",
      detail: (a) => `${fmtSite(a)} · mostly ${a.dominantClass} · ${fmtFrp(a.frp)} total`,
      rows: (a) => [
        ["Detections", `${esc(a.eventCount)} crop/wildfire detections in one same-day cluster, each within 5 km of another (centre shown)`],
        ["Excludes", "detections whose ~1 km area was active on more than 5 of the previous 30 days (industrial sites, coal-seam fires)"],
        ["Mostly", esc(a.dominantClass)],
        ["Total FRP", esc(fmtFrp(a.frp))],
      ],
    },
  };

  let nameFor = (label) => label; // display name for a class; set by init()

  function popupHtml(a, color) {
    const satellite = `https://www.google.com/maps?q=${a.lat},${a.lng}&t=k`; // built here, not taken from the file
    const when = a.time ? `${esc(fmtDate(a))} (${esc(fmtUtc(a))})` : `${esc(fmtDate(a))} (UTC date)`;
    const rows = [
      ["Type", esc(TYPES[a.type])],
      ["Date", when],
      ...(a.type === "large_fire_event" ? [] : [["Class", esc(nameFor(a.label))]]),
      ...VIEW[a.type].rows(a),
      ["Location", `${esc(fmtSite(a))} · <a href="${esc(satellite)}" target="_blank" rel="noopener">satellite view</a>`],
    ];
    return (
      `<div class="detection"><h2><span class="swatch" style="background:${esc(color)}"></span>${esc(VIEW[a.type].title(a))}</h2><dl>` +
      rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") +
      "</dl></div>"
    );
  }

  /**
   * @param {object} opts
   * @param {maplibregl.Map} opts.map
   * @param {(lngLat: [number, number], html: string) => void} opts.openPopup
   * @param {(label: string) => string} opts.colorFor  class colour (neutral until known)
   * @param {(label: string) => string} [opts.nameFor]  class display name (e.g. "unknown" -> "Unclassified — needs review")
   * @param {(alert: object) => void} [opts.onSelect]  e.g. switch the month filter
   */
  async function init({ map, openPopup, colorFor, nameFor: displayName, onSelect }) {
    if (displayName) nameFor = displayName;
    const panel = document.getElementById("alerts-panel");
    const list = document.getElementById("alerts-list");
    const countEl = document.getElementById("alerts-count");
    const toggle = document.getElementById("alerts-toggle");
    const filtersEl = document.getElementById("alerts-filters");

    const setCollapsed = (collapsed) => {
      panel.classList.toggle("collapsed", collapsed);
      toggle.setAttribute("aria-expanded", String(!collapsed));
      toggle.textContent = collapsed ? "Show" : "Hide";
    };
    setCollapsed(window.matchMedia("(max-width: 640px)").matches);
    toggle.addEventListener("click", () => setCollapsed(!panel.classList.contains("collapsed")));
    const none = { alerts: [], recolor() {} };

    let alerts;
    try {
      const response = await fetch(ALERTS_URL, { cache: "no-cache" });
      if (response.status === 404) {
        list.innerHTML = '<p class="alerts-empty">No alerts recorded yet.</p>';
        countEl.textContent = "0";
        return none;
      }
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      alerts = parseCsv(await response.text()).map(toAlert).filter((a) => Number.isFinite(a.lat) && Number.isFinite(a.lng));
    } catch (error) {
      list.innerHTML = `<p class="alerts-empty">Alerts unavailable: ${esc(error.message)}</p>`;
      countEl.textContent = "–";
      return none;
    }

    alerts.sort((a, b) => (a.sortKey < b.sortKey ? 1 : a.sortKey > b.sortKey ? -1 : 0));
    countEl.textContent = alerts.length.toLocaleString("en-US");
    if (!alerts.length) {
      list.innerHTML = '<p class="alerts-empty">No alerts recorded yet.</p>';
      return { alerts, recolor() {} };
    }

    const fragment = document.createDocumentFragment();
    alerts.forEach((a, i) => {
      const view = VIEW[a.type];
      const item = document.createElement("button");
      item.type = "button";
      item.className = "alert-item";
      item.dataset.index = String(i);
      item.dataset.type = a.type;
      item.innerHTML =
        `<span class="alert-top"><span class="swatch" data-label="${esc(swatchLabel(a))}" style="background:${esc(colorFor(swatchLabel(a)))}"></span>` +
        `<span class="alert-facility">${esc(view.title(a))}</span>` +
        `<span class="alert-ratio" title="${esc(view.metricHint)}">${esc(view.metric(a))}</span></span>` +
        `<span class="alert-meta"><span class="alert-badge">${esc(TYPES[a.type])}</span>${a.type === "large_fire_event" ? "" : `<span data-class-label="${esc(a.label)}">${esc(nameFor(a.label))}</span> · `}${esc(fmtDate(a))}</span>` +
        `<span class="alert-meta">${esc(view.detail(a))}</span>`;
      fragment.appendChild(item);
    });
    list.replaceChildren(fragment);

    // Type filter: "All" plus one toggle per type that has alerts, each with its count.
    const counts = Object.fromEntries(Object.keys(TYPES).map((t) => [t, alerts.filter((a) => a.type === t).length]));
    let active = "all";
    const applyFilter = () => {
      let shown = 0;
      list.querySelectorAll(".alert-item").forEach((item) => {
        const visible = active === "all" || item.dataset.type === active;
        item.hidden = !visible;
        shown += visible;
      });
      countEl.textContent = shown.toLocaleString("en-US");
      filtersEl.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.type === active)));
    };
    const chip = (type, text) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "alert-filter";
      b.dataset.type = type;
      b.textContent = text;
      b.addEventListener("click", () => { active = type; applyFilter(); });
      return b;
    };
    filtersEl.replaceChildren(
      chip("all", `All ${alerts.length}`),
      ...Object.entries(TYPES).filter(([t]) => counts[t]).map(([t, name]) => chip(t, `${name} ${counts[t]}`)),
    );
    applyFilter();

    list.addEventListener("click", (event) => {
      const item = event.target.closest(".alert-item");
      if (!item) return;
      const alert = alerts[Number(item.dataset.index)];
      list.querySelector(".alert-item.selected")?.classList.remove("selected");
      item.classList.add("selected");
      onSelect?.(alert);
      const zoom = alert.type === "large_fire_event" ? 10 : 12; // an event spans kilometres
      map.flyTo({ center: [alert.lng, alert.lat], zoom: Math.max(map.getZoom(), zoom), essential: true });
      openPopup([alert.lng, alert.lat], popupHtml(alert, colorFor(swatchLabel(alert))));
    });

    // class colours and names may arrive after the list is built (they come with the point data)
    const recolor = () => {
      list.querySelectorAll(".swatch[data-label]").forEach((s) => { s.style.background = colorFor(s.dataset.label); });
      list.querySelectorAll("[data-class-label]").forEach((s) => { s.textContent = nameFor(s.dataset.classLabel); });
    };
    return { alerts, recolor };
  }

  window.AlertsPanel = { init, parseCsv };
})();
