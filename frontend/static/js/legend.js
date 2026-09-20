function renderLegend(colors) {
  const container = document.getElementById("legend");
  container.innerHTML = "";
  for (const [label, color] of Object.entries(colors)) {
    const row = document.createElement("div");
    row.className = "legend-row";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = color;
    row.appendChild(swatch);
    row.appendChild(document.createTextNode(label));
    container.appendChild(row);
  }
}

window.Legend = { renderLegend };
