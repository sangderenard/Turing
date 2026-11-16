// [CONDITION] The operators' switch for the site's condition card (site_condition.py). Never a public file: the server
// answers this module only to a signed-in operator or admin (GET /api/condition/controls.js) and checks the role again on
// every switch. Loaded by site-nav.js (the account menu) and by the cards themselves, so a card can be taken down from it.
const CARDS = {
  yellow: { label: "Yellow card: mild attack", background: "#ffd60a", color: "#000" },
  red: { label: "Red card: threat or down", background: "#b00020", color: "#fff" },
};

async function call(method, body) {
  const options = { method, credentials: "same-origin" };
  if (body) Object.assign(options, { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const res = await fetch("/api/condition", options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "HTTP " + res.status);
  return data;
}

export async function mount(target, { onCard } = {}) {
  const box = document.createElement("div");
  box.style.cssText = "display:grid;gap:6px;padding:6px 10px;text-align:left;";
  const status = document.createElement("div");
  status.style.cssText = "font-size:12px;opacity:.85;";
  box.append(status);
  const buttons = {};
  function render(state) {
    status.textContent = state.card
      ? (state.card === "red" ? "Red" : "Yellow") + " card up" + (state.line ? ", " + state.line : "")
      : "Site condition: open, no card posted";
    for (const [card, button] of Object.entries(buttons)) {
      const up = state.card === card;
      button.textContent = up ? "Take the " + card + " card down" : CARDS[card].label;
      button.setAttribute("aria-pressed", String(up));
    }
  }
  for (const [card, look] of Object.entries(CARDS)) {
    const button = document.createElement("button");
    button.type = "button";
    button.style.cssText = "all:unset;box-sizing:border-box;display:block;width:100%;padding:7px 10px;border-radius:6px;" +
      "cursor:pointer;font-weight:700;text-align:center;background:" + look.background + ";color:" + look.color + ";";
    button.addEventListener("click", async () => {
      const up = button.getAttribute("aria-pressed") === "true";
      if (!up && !window.confirm("Post the " + card + " card? Every new visitor will see it instead of the site until it is taken down.")) return;
      button.disabled = true;
      try {
        const state = await call("POST", { card: up ? "none" : card, note: "from the menu" });
        render(state);
        if (onCard) onCard(state);
      } catch (error) {
        status.textContent = "Could not switch: " + error.message;
      } finally {
        button.disabled = false;
      }
    });
    buttons[card] = button;
    box.append(button);
  }
  target.append(box);
  try { render(await call("GET")); } catch (error) { status.textContent = "Condition unknown: " + error.message; }
}
