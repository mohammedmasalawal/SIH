function renderFilters(classes, onChange) {
  const container = document.getElementById("filters");
  container.innerHTML = "";
  const active = new Set(classes);

  for (const label of classes) {
    const row = document.createElement("label");
    row.className = "filter-row";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) active.add(label);
      else active.delete(label);
      onChange(active);
    });

    row.appendChild(checkbox);
    row.appendChild(document.createTextNode(" " + label));
    container.appendChild(row);
  }
}

window.Filters = { renderFilters };
