import {applyPalette, pageState, sharePalette} from "./palette-theme.js";

const $ = (id) => document.getElementById(id);
let state = null;
let picked = [];

async function request(path, body) {
  const response = await fetch(path, {method: body ? "POST" : "GET", credentials: "same-origin", cache: "no-store",
    headers: body ? {"Content-Type": "application/json"} : {}, body: body ? JSON.stringify(body) : undefined});
  const data = await response.json();
  if (!response.ok) throw Object.assign(new Error(data.error || `Request failed: ${response.status}`), {status: response.status});
  return data;
}

// A design drawn as a small page (palette_designs.preview): nav bar, header, a card with text, muted text, a link, an
// input, a primary and a secondary button and a badge, a stripe, and the palette's colors along the bottom. Each part
// wears its element's color and, for boxes, its radius, border width and box-shadow.
const PREVIEW_KEY = /^[a-z0-9_]{1,24}$/;
function mockup(view, className = "mock") {
  const node = document.createElement("span");
  node.className = className;
  for (const [key, value] of Object.entries(view.preview || {})) {
    if (!PREVIEW_KEY.test(key)) continue;
    node.style.setProperty(`--m-${key.replaceAll("_", "-")}`, typeof value === "number" ? `${value}px` : String(value));
  }
  const part = (name, ...children) => {
    const element = document.createElement("span");
    element.className = `m-${name}`;
    element.append(...children);
    return element;
  };
  const strip = part("palette", ...(view.colors || []).map((color) => {
    const swatch = document.createElement("span");
    swatch.style.background = color;
    return swatch;
  }));
  node.append(part("stripe"), part("nav", part("dot"), part("dot"), part("dot")), part("header", part("title")),
    part("card", part("text"), part("muted"), part("link"), part("input"),
         part("row", part("button"), part("button2"), part("badge"))), strip);
  node.title = `${(view.names || []).join(" · ")} (${view.style}, ${view.scheme})`;
  return node;
}

// Scatter the round over the field so each square hangs in space with ample margin between its neighbours and the
// edges, never overlapping. A jittered grid guarantees placement on any viewport size.
function scatterChips(designs) {
  const field = $("field");
  field.replaceChildren();
  const rect = field.getBoundingClientRect();
  const chip = 160; // a small page: 156 by 116 CSS pixels
  const marginX = Math.max(24, rect.width * 0.07);
  const marginY = Math.max(24, rect.height * 0.12);
  const availW = Math.max(chip, rect.width - 2 * marginX);
  const availH = Math.max(chip, rect.height - 2 * marginY);
  let cols = 1;
  for (const attempt of [5, 4, 3, 2, 1]) {
    const rows = Math.ceil(designs.length / attempt);
    if (availW / attempt >= chip * 1.35 && availH / rows >= chip * 1.15) { cols = attempt; break; }
  }
  const rows = Math.ceil(designs.length / cols);
  const cellW = availW / cols, cellH = availH / rows;
  const cells = [];
  for (let row = 0; row < rows; ++row) {
    const inRow = Math.min(cols, designs.length - row * cols);
    const rowOffset = (cols - inRow) * cellW / 2;
    for (let col = 0; col < inRow; ++col) cells.push({row, col, rowOffset});
  }
  for (let i = cells.length - 1; i > 0; --i) {
    const j = Math.floor(Math.random() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  const looseX = Math.max(0, cellW - chip), looseY = Math.max(0, cellH - chip);
  designs.forEach((design, index) => {
    const cell = cells[index];
    const node = document.createElement("button");
    node.type = "button";
    node.className = "chip";
    node.style.left = `${marginX + cell.rowOffset + cell.col * cellW + cellW / 2 + (Math.random() - 0.5) * looseX}px`;
    node.style.top = `${marginY + cell.row * cellH + cellH / 2 + (Math.random() - 0.5) * looseY}px`;
    node.append(mockup(design));
    node.setAttribute("aria-label", `Choose the design of ${design.names.join(", ")}`);
    node.addEventListener("click", () => choose(design.id, node));
    field.append(node);
  });
}

function choose(id, node) {
  node.remove();
  picked.push(id);
  if (picked.length === state.round.designs.length) submitOrder();
}

function closeField() {
  $("overlay").hidden = true;
  $("page").inert = false;
}

function openField() {
  picked = [];
  $("overlay").hidden = false;
  $("page").inert = true;
  scatterChips(state.round.designs);
}

async function submitOrder() {
  closeField();
  try {
    state = await request("/api/palette/rank", {nonce: state.round.nonce, ranking: picked});
    render();
    $("notice").textContent = "Ranked. The network is learning from it; score the design it proposes.";
  } catch (error) {
    if (error.status === 409) {
      // The round moved on (another tab, or a first visit whose identity settled after this page loaded):
      // read the current round and choose again.
      state = await request("/api/palette");
      render();
      $("notice").textContent = "Your round changed; choose again.";
      openField();
      return;
    }
    $("notice").textContent = error.message;
  }
}

function renderDesign(target, wear) {
  target.replaceChildren();
  if (!wear) {
    target.append(Object.assign(document.createElement("p"), {className: "muted", textContent: "Rank a round to begin."}));
    return;
  }
  const text = document.createElement("div");
  text.append(Object.assign(document.createElement("strong"), {textContent: wear.design.names.join(" · ")}),
    document.createElement("br"),
    Object.assign(document.createElement("span"), {className: "muted",
      textContent: `${wear.design.colors.length} color${wear.design.colors.length === 1 ? "" : "s"}, ` +
                   `${wear.design.scheme}, ${wear.design.style}`}));
  target.append(mockup(wear.design), text);
}

function card(title, wear) {
  const node = document.createElement("div");
  node.className = "card";
  node.style.setProperty("--sample-accent", wear.color);
  node.style.setProperty("--sample-secondary", wear.secondary);
  node.append(Object.assign(document.createElement("h3"), {textContent: title}));
  const row = document.createElement("div");
  row.className = "design";
  renderDesign(row, wear);
  const sample = document.createElement("div");
  sample.className = "sample";
  sample.append(Object.assign(document.createElement("b"), {textContent: "A heading in this accent"}),
    document.createElement("br"), document.createTextNode("Body text on the site's background."));
  const scores = document.createElement("p");
  scores.className = "muted";
  scores.textContent = `Your network ${wear.score.yours.toFixed(3)} · shared network ${wear.score.everyone.toFixed(3)}`;
  node.append(row, sample, scores);
  return node;
}

function render() {
  const current = state.candidate;
  applyPalette(current);
  sharePalette(current);
  renderDesign($("candidate"), current);
  $("rating").disabled = $("score").disabled = !current;
  const compare = state.comparison;
  $("compare").replaceChildren(...(compare ? [card("Yours", compare.yours), card("Everyone's default", compare.everyone)] : []));
  const model = state.model, history = state.history;
  $("model").textContent = (model.state === "ready"
      ? (model.shared_steps ? `The network has trained ${model.shared_steps} shared steps.`
                            : "The network is ready and has not learned from anyone yet.")
      : model.state === "compiling" ? "The network is compiling; designs are random until it is ready."
      : `The network is unavailable: ${model.error}`) +
    ` You have ranked ${history.rounds} round${history.rounds === 1 ? "" : "s"} and scored ${history.ratings} design${history.ratings === 1 ? "" : "s"}.`;
  renderSaved();
}

function renderSaved() {
  const list = state.saved || [];
  $("saved").replaceChildren(...list.map((palette, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", String(palette.id === state.active));
    button.setAttribute("aria-label", `Palette ${index + 1}: ${palette.names.join(", ")}`);
    button.title = `${palette.names.join(" · ")} (${palette.style})`;
    for (const color of palette.colors) {
      const swatch = document.createElement("span");
      swatch.style.background = color;
      button.append(swatch);
    }
    button.addEventListener("click", () => act("/api/palette/select", {id: palette.id}, "Palette applied across the site."));
    return button;
  }));
  $("prev").disabled = $("next").disabled = list.length < 2;
}

async function act(path, body, message) {
  try {
    state = await request(path, body);
    render();
    $("notice").textContent = message;
    return true;
  } catch (error) { $("notice").textContent = error.message; return false; }
}

function step(by) {
  const list = state.saved || [];
  if (list.length < 2) return;
  const at = list.findIndex((palette) => palette.id === state.active);
  act("/api/palette/select", {id: list[(at + by + list.length) % list.length].id}, "Palette applied across the site.");
}

$("rating").addEventListener("input", () => { $("rating-value").value = Number($("rating").value).toFixed(2); });
$("score").addEventListener("click", () => act("/api/palette/feedback",
  {nonce: state.candidate.nonce, rating: Number($("rating").value)}, "Scored. The next design reflects what the network learned."));
$("again").addEventListener("click", async () => { if (await act("/api/palette/round", {}, "")) openField(); });
$("clear").addEventListener("click", async () => {
  if (!window.confirm("Forget your rankings, scores, palettes and personal layer? The shared network keeps what it learned.")) return;
  if (await act("/api/palette/clear", {}, "Your history is cleared.")) openField();
});
$("prev").addEventListener("click", () => step(-1));
$("next").addEventListener("click", () => step(1));

try {
  state = await pageState(); // the page's one first request (palette-theme.js), so a first visit has one identity
  render();
  if (!state.history.rounds) openField(); // first visit: the field greets you
} catch (error) { $("notice").textContent = error.message; }
