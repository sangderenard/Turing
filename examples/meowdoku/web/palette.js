import {applyPalette} from "./palette-theme.js";

const $ = (id) => document.getElementById(id);
let state = null;
let order = [];

async function request(path, body) {
  const response = await fetch(path, {method: body ? "POST" : "GET", credentials: "same-origin", cache: "no-store",
    headers: body ? {"Content-Type": "application/json"} : {}, body: body ? JSON.stringify(body) : undefined});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}

function render() {
  const list = $("swatches");
  list.replaceChildren();
  for (const [position, id] of order.entries()) {
    const row = document.createElement("div");
    row.className = "swatch";
    const sample = document.createElement("span");
    sample.className = "sample";
    sample.style.background = state.swatches[id].color;
    const label = document.createElement("span");
    label.textContent = `${position + 1}. ${state.swatches[id].color}`;
    row.append(sample, label);
    for (const [name, shift] of [["Up", -1], ["Down", 1]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = name;
      button.setAttribute("aria-label", `Move ${state.swatches[id].color} ${name.toLowerCase()}`);
      button.disabled = position + shift < 0 || position + shift >= order.length;
      button.addEventListener("click", () => {
        [order[position], order[position + shift]] = [order[position + shift], order[position]];
        render();
      });
      row.append(button);
    }
    list.append(row);
  }
  const current = state.candidate;
  applyPalette(current);
  $("candidate").textContent = current ? `Current accent: ${current.color}` : "Rank the colors to begin.";
  $("good").disabled = $("bad").disabled = !current;
}

async function update(path, body) {
  try {
    state = await request(path, body);
    order = state.ranking ? [...state.ranking] : state.swatches.map((swatch) => swatch.id);
    render();
    $("notice").textContent = "Your personal design is ready. Rate it after viewing the home page.";
  } catch (error) { $("notice").textContent = error.message; }
}

$("rank").addEventListener("click", () => update("/api/palette/rank", {ranking: order}));
for (const rating of ["good", "bad"])
  $(rating).addEventListener("click", () => update("/api/palette/feedback", {nonce: state.candidate.nonce, rating}));

try {
  state = await request("/api/palette");
  order = state.ranking ? [...state.ranking] : state.swatches.map((swatch) => swatch.id);
  render();
} catch (error) { $("notice").textContent = error.message; }
