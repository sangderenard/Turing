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

// [NOTICE] The header notice's form (site_condition.py post_notice). The server checks the role and every field again.
async function noticeCall(body) {
  const res = await fetch("/api/condition/notice", { method: "POST", credentials: "same-origin",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "HTTP " + res.status);
  return data.notice;
}

function mountNoticeForm(box) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Header notice";
  summary.style.cssText = "cursor:pointer;font-weight:700;";
  details.append(summary);
  const field = (label, node) => {
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:grid;gap:2px;font-size:12px;margin-top:4px;";
    wrap.append(document.createTextNode(label), node);
    details.append(wrap);
    return node;
  };
  const input = (type) => { const n = document.createElement("input"); n.type = type; n.style.cssText = "font:inherit;"; return n; };
  const title = field("Title", input("text"));
  const body = field("Text (blank line = new paragraph)", document.createElement("textarea"));
  body.rows = 5;
  const eta = field("Estimate (when public services are ready)", input("text"));
  const mode = field("Placement", document.createElement("select"));
  for (const [value, label] of [["push", "Take the space at the top"], ["overlay", "On top of the page"]]) {
    const option = document.createElement("option"); option.value = value; option.textContent = label; mode.append(option);
  }
  const block = field("Block the page under it (overlay only; never the header)", input("checkbox"));
  const dismissible = field("Visitors may dismiss it (not when blocking)", input("checkbox"));
  dismissible.checked = true;
  const paths = field("Only on paths starting with (comma separated; empty = every page)", input("text"));
  const said = document.createElement("div");
  said.style.cssText = "font-size:12px;opacity:.85;margin-top:4px;";
  const post = document.createElement("button");
  post.type = "button"; post.textContent = "Post notice";
  const clear = document.createElement("button");
  clear.type = "button"; clear.textContent = "Clear notice";
  for (const b of [post, clear]) b.style.cssText = "margin:6px 6px 0 0;";
  details.append(post, clear, said);
  post.addEventListener("click", async () => {
    if (block.checked && !window.confirm("A blocking notice makes every page unusable under it until it is cleared. Post it?")) return;
    try {
      const n = await noticeCall({ title: title.value, body: body.value, eta: eta.value, mode: mode.value, block: block.checked,
        dismissible: dismissible.checked, paths: paths.value.split(",").map(p => p.trim()).filter(Boolean) });
      said.textContent = "Posted, version " + n.version + ". New page loads show it.";
    } catch (error) { said.textContent = "Not posted: " + error.message; }
  });
  clear.addEventListener("click", async () => {
    try { await noticeCall({ clear: true }); said.textContent = "Cleared."; } catch (error) { said.textContent = "Not cleared: " + error.message; }
  });
  box.append(details);
  call("GET").then(state => {
    const n = state.notice;
    if (!n) return;
    title.value = n.title || ""; body.value = n.body || ""; eta.value = n.eta || ""; mode.value = n.mode || "push";
    block.checked = !!n.block; dismissible.checked = !!n.dismissible; paths.value = (n.paths || []).join(", ");
    said.textContent = "Current: version " + n.version + (n.posted ? ", " + n.posted : "");
  }).catch(() => {});
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
  mountNoticeForm(box);
  try { render(await call("GET")); } catch (error) { status.textContent = "Condition unknown: " + error.message; }
}
