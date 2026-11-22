import {applyPalette, sharePalette} from "./palette-theme.js";

const $ = (id) => document.getElementById(id);
let state = null;
let picked = [];

async function request(path, body) {
  const response = await fetch(path, {method: body ? "POST" : "GET", credentials: "same-origin", cache: "no-store",
    headers: body ? {"Content-Type": "application/json"} : {}, body: body ? JSON.stringify(body) : undefined});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}

// Scatter the swatches over the field so each square hangs in space with
// ample margin between its neighbours and the edges, never overlapping.
// A jittered grid guarantees placement on any viewport size.
function scatterChips(ids) {
  const field = $("field");
  field.replaceChildren();
  const rect = field.getBoundingClientRect();
  const chip = 96; // one inch at the CSS reference pixel density
  const marginX = Math.max(24, rect.width * 0.07);
  const marginY = Math.max(24, rect.height * 0.12);
  const availW = Math.max(chip, rect.width - 2 * marginX);
  const availH = Math.max(chip, rect.height - 2 * marginY);
  // Sparsest column count whose grid fits both axes at generous spacing,
  // relaxing toward plain non-overlap on small viewports.
  let cols = 1;
  for (const attempt of [5, 4, 3, 2, 1]) {
    const rows = Math.ceil(ids.length / attempt);
    const spacingW = availW / attempt;
    const spacingH = availH / rows;
    if (spacingW >= chip * 1.35 && spacingH >= chip * 1.15) { cols = attempt; break; }
  }
  const rows = Math.ceil(ids.length / cols);
  const cellW = availW / cols, cellH = availH / rows;
  const cells = [];
  for (let row = 0; row < rows; ++row) {
    const inRow = Math.min(cols, ids.length - row * cols);
    const rowOffset = (cols - inRow) * cellW / 2; // center a short last row
    for (let col = 0; col < inRow; ++col) cells.push({row, col, inRow, rowOffset});
  }
  for (let i = cells.length - 1; i > 0; --i) { // shuffle cells, not colors
    const j = Math.floor(Math.random() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  const looseX = Math.max(0, cellW - chip), looseY = Math.max(0, cellH - chip);
  ids.forEach((id, index) => {
    const cell = cells[index];
    const x = marginX + cell.rowOffset + cell.col * cellW + cellW / 2 + (Math.random() - 0.5) * looseX;
    const y = marginY + cell.row * cellH + cellH / 2 + (Math.random() - 0.5) * looseY;
    const chipNode = document.createElement("button");
    chipNode.type = "button";
    chipNode.className = "chip";
    chipNode.style.left = `${x}px`;
    chipNode.style.top = `${y}px`;
    chipNode.style.background = state.swatches[id].color;
    chipNode.setAttribute("aria-label", `Choose the color ${state.swatches[id].color}`);
    chipNode.addEventListener("click", () => choose(id, chipNode));
    field.append(chipNode);
  });
}

function choose(id, node) {
  node.remove();
  picked.push(id);
  if (picked.length === state.swatches.length) submitOrder();
}

async function submitOrder() {
  $("overlay").hidden = true;
  $("page").inert = false;
  try {
    state = await request("/api/palette/rank", {ranking: picked});
    render();
    $("notice").textContent = "Your personal design is ready. Rate it after viewing the home page.";
  } catch (error) { $("notice").textContent = error.message; }
}

function openField() {
  picked = [];
  $("overlay").hidden = false;
  $("page").inert = true;
  scatterChips(state.swatches.map((swatch) => swatch.id));
}

function render() {
  const current = state.candidate;
  applyPalette(current);
  sharePalette(current);
  $("candidate").textContent = current ? `Current accent: ${current.color}` : "Choose the colors to begin.";
  $("good").disabled = $("bad").disabled = !current;
  renderSaved();
}

// The participant's saved palettes, oldest first; the active one is pressed.
function renderSaved() {
  const list = state.saved || [];
  $("saved").replaceChildren(...list.map((palette, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", String(palette.id === state.active));
    button.setAttribute("aria-label", `Palette ${index + 1}: ${palette.color} and ${palette.secondary}`);
    button.title = `${palette.color} · ${palette.secondary}`;
    for (const color of [palette.color, palette.secondary]) {
      const swatch = document.createElement("span");
      swatch.style.background = color;
      button.append(swatch);
    }
    button.addEventListener("click", () => select(palette.id));
    return button;
  }));
  $("prev").disabled = $("next").disabled = list.length < 2;
}

async function select(id) {
  try {
    state = await request("/api/palette/select", {id});
    render();
    $("notice").textContent = "Palette applied across the site.";
  } catch (error) { $("notice").textContent = error.message; }
}

function step(by) {
  const list = state.saved || [];
  if (list.length < 2) return;
  const at = list.findIndex((palette) => palette.id === state.active);
  select(list[(at + by + list.length) % list.length].id);
}

async function update(path, body) {
  try {
    state = await request(path, body);
    render();
    $("notice").textContent = "Noted. The next candidate reflects your answer.";
  } catch (error) { $("notice").textContent = error.message; }
}

$("good").addEventListener("click", () => update("/api/palette/feedback", {nonce: state.candidate.nonce, rating: "good"}));
$("bad").addEventListener("click", () => update("/api/palette/feedback", {nonce: state.candidate.nonce, rating: "bad"}));
$("again").addEventListener("click", () => openField());
$("prev").addEventListener("click", () => step(-1));
$("next").addEventListener("click", () => step(1));

try {
  state = await request("/api/palette");
  render();
  if (!state.candidate) openField(); // first visit: the field greets you
} catch (error) { $("notice").textContent = error.message; }
