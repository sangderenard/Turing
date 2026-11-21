const $ = (id) => document.getElementById(id);
const SYMBOLS = ["BTC/USD", "SPY", "QQQ", "IWM"];
const COLORS = { "BTC/USD": [0.95, 0.70, 0.25, 1], SPY: [0.30, 0.72, 0.95, 1],
                 QQQ: [0.66, 0.55, 0.96, 1], IWM: [0.44, 0.82, 0.55, 1],
                 AAPL: [0.98, 0.46, 0.52, 1], MSFT: [0.98, 0.46, 0.52, 1],
                 NVDA: [0.98, 0.46, 0.52, 1], TSLA: [0.98, 0.46, 0.52, 1] };
const VERTEX = `#version 300 es
in vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
`;
const FRAGMENT = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }
`;

function shader(gl, type, source) {
  const value = gl.createShader(type);
  gl.shaderSource(value, source);
  gl.compileShader(value);
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value));
  return value;
}

function renderer(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: true });
  if (!gl) throw new Error("WebGL2 is unavailable in this browser.");
  const program = gl.createProgram();
  const vertex = shader(gl, gl.VERTEX_SHADER, VERTEX);
  const fragment = shader(gl, gl.FRAGMENT_SHADER, FRAGMENT);
  gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
  gl.deleteShader(vertex); gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const buffer = gl.createBuffer();
  const position = gl.getAttribLocation(program, "aPosition");
  const color = gl.getUniformLocation(program, "uColor");
  function clear() {
    const ratio = devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(.06, .11, .17, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  }
  function draw(points, mode, rgba) {
    if (points.length < 2) return;
    gl.uniform4fv(color, rgba);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.DYNAMIC_DRAW);
    gl.drawArrays(mode === "line" ? gl.LINE_STRIP : gl.TRIANGLES, 0, points.length / 2);
  }
  return { gl, clear, draw };
}

const fmt = (number) => Number(number).toLocaleString(undefined, { maximumFractionDigits: 2 });
let adminTraining = false, trainingInstance = null, trainingPending = null;
let trainingBusy = false, trainingReading = false, trainingAction = null, clearingRun = null;
let qualificationPayload = null;

function drawQualification(payload) {
  qualificationPayload = payload;
  const root = $("qualification-runs"); root.replaceChildren();
  for (const run of payload.runs || []) {
    const card = document.createElement("article"); card.className = "qualification-run";
    const title = document.createElement("h3"); title.textContent = `${run.market} · ${run.status}`;
    const meta = document.createElement("div"); meta.className = "metrics";
    const last = (run.progress || []).at(-1);
    const fields = [`${run.model} · ${run.worker}`, `phase ${run.phase}`,
                    run.planned_epochs ? `${run.completed_epochs}/${run.planned_epochs} epochs`
                                       : `${run.completed_epochs} epochs · continuous`];
    if (last?.loss != null) fields.push(`loss ${Number(last.loss).toFixed(5)}`);
    if (last?.validation_mean != null) fields.push(`validation mean ${(Number(last.validation_mean) * 100).toFixed(2)}% of order cap`);
    if (run.live) {
      const observed = run.live.observed_at ? ` · last AAPL bar ${new Date(run.live.observed_at).toLocaleString()}` : "";
      fields.push(`market feed ${run.live.source}${observed}`);
      fields.push(`${run.live.matured_frames} live frames with known outcomes`);
      fields.push(run.live.live_batch
        ? `${run.live.historical_batch} historical + ${run.live.live_batch} live labeled frames per batch`
        : "historical training while live outcomes mature");
    }
    if (run.result?.holdout_mean_return_on_order_cap != null)
      fields.push(`holdout mean ${(Number(run.result.holdout_mean_return_on_order_cap) * 100).toFixed(2)}% of order cap`);
    for (const value of fields) { const item = document.createElement("span"); item.textContent = value; meta.append(item); }
    const bar = document.createElement("progress");
    if (run.planned_epochs) {
      bar.max = run.planned_epochs;
      bar.value = Math.min(bar.max, run.completed_epochs);
    } else {
      bar.removeAttribute("value");
    }
    const checks = document.createElement("ol"); checks.className = "checks";
    for (const check of (run.progress || []).filter((item) => item.validation_mean != null).slice(-8)) {
      const line = document.createElement("li");
      const position = [check.seed != null ? `seed ${check.seed}` : null,
                        check.epoch != null ? `epoch ${check.epoch}` : null].filter(Boolean).join(", ");
      line.textContent = `${position ? position + ": " : ""}validation ${(Number(check.validation_mean) * 100).toFixed(2)}% · ${check.validation_passed ? "gate met" : "gate unmet"}`;
      checks.append(line);
    }
    card.append(title, meta, bar, checks);
    if (adminTraining) {
      const controls = document.createElement("div"); controls.className = "row qualification-actions";
      const managed = trainingInstance?.name === run.worker;
      const active = managed && ["running", "starting", "backoff"].includes(trainingInstance.state);
      const stopped = managed && ["stopped", "failed"].includes(trainingInstance.state) && trainingInstance.enabled !== false;
      for (const [label, action, enabled] of [["Start worker", "start", stopped], ["Stop worker", "shutdown", active]]) {
        const button = document.createElement("button");
        button.type = "button"; button.textContent = label;
        button.disabled = !enabled || !!trainingPending || trainingBusy || trainingReading;
        button.addEventListener("click", () => requestTrainingControl(false, action, run.worker));
        controls.append(button);
      }
      const clear = document.createElement("button"); clear.type = "button"; clear.textContent = "Clear run";
      clear.disabled = clearingRun !== null || run.status === "running";
      clear.addEventListener("click", () => clearQualification(run.id));
      controls.append(clear);
      const note = document.createElement("small"); note.className = "muted";
      note.textContent = managed ? "Start and Stop control the shared worker; Clear removes this run from the watch list."
                                 : "This worker is not managed here; Clear removes this run from the watch list.";
      controls.append(note); card.append(controls);
    }
    root.append(card);
  }
  $("qualification-state").textContent = root.childElementCount
    ? `${root.childElementCount} recent run(s). Updated ${new Date(payload.asof * 1000).toLocaleTimeString()}.`
    : "No qualification run has started on this shared arena.";
}

async function updateQualification() {
  drawQualification(await api("/api/paper/qualification/live"));
}
async function clearQualification(id) {
  if (!adminTraining || clearingRun !== null) return;
  clearingRun = id;
  if (qualificationPayload) drawQualification(qualificationPayload);
  let clearError = null;
  try {
    const response = await fetch("/api/paper/qualification/clear", {
      method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Clear refused (${response.status})`);
    await updateQualification();
  } catch (error) {
    clearError = `${error.message}. Refresh before retrying; the request may have reached the server.`;
  } finally {
    clearingRun = null;
    if (qualificationPayload) drawQualification(qualificationPayload);
    if (clearError) $("qualification-state").textContent = clearError;
  }
}
setInterval(() => { if (!document.hidden) updateQualification().catch((error) => { $("qualification-state").textContent = error.message; }); }, 2000);
updateQualification().catch((error) => { $("qualification-state").textContent = error.message; });

// Use the existing audited supervisor request queue, never a second trainer runner.
// This page exposes the shortcut to admins only; the control API rechecks authority.
const TRAINING_INSTANCE = "nodus-training";
async function refreshTrainingControl() {
  if (trainingReading || trainingBusy) return;
  trainingReading = true;
  $("training-toggle").disabled = true;
  $("training-cancel").disabled = true;
  trainingAction = null; trainingPending = null;
  try {
    const who = await api("/api/whoami");
    const allowed = who.role === "admin" && (who.capabilities || []).includes("account_manage");
    adminTraining = allowed;
    $("training-controls").hidden = !allowed;
    if (!allowed) { trainingInstance = null; if (qualificationPayload) drawQualification(qualificationPayload); return; }
    const data = await api("/api/control/instances");
    const instance = (data.instances || []).find((item) => item.name === TRAINING_INSTANCE);
    trainingInstance = instance || null;
    const requests = (data.requests || []).filter((item) => item.instance === TRAINING_INSTANCE || item.instance === "*");
    trainingPending = requests.find((item) => item.state === "pending") || null;
    if (qualificationPayload) drawQualification(qualificationPayload);
    $("training-cancel").hidden = !trainingPending || trainingPending.instance !== TRAINING_INSTANCE;
    $("training-cancel").disabled = !trainingPending || trainingPending.instance !== TRAINING_INSTANCE;
    if (!instance) {
      $("training-toggle").textContent = "Trainer unavailable";
      $("training-control-state").textContent = "The supervisor has no registered nodus-training instance.";
      return;
    }
    const active = ["running", "starting", "backoff"].includes(instance.state);
    const stopped = ["stopped", "failed"].includes(instance.state);
    trainingAction = active ? "shutdown" : stopped && instance.enabled !== false ? "start" : null;
    $("training-toggle").textContent = active ? "Stop training" : stopped ? "Start training" : "Training " + instance.state;
    $("training-toggle").disabled = !trainingAction || !!trainingPending;
    const last = requests[0];
    $("training-control-state").textContent = `Supervisor: ${instance.state}. ` + (trainingPending
      ? `Request ${trainingPending.id}: ${trainingPending.action} pending. ${trainingPending.detail || "Waiting for the supervisor; this is not yet completed."}`
      : last ? `Last request ${last.id}: ${last.state}. ${last.detail || ""}` : "No recent control request.");
  } catch (error) {
    adminTraining = false; trainingInstance = null;
    $("training-controls").hidden = true;
    if (qualificationPayload) drawQualification(qualificationPayload);
    $("training-control-state").textContent = `Could not verify training state: ${error.message}. Refresh before acting.`;
    $("training-toggle").textContent = "Training status unavailable";
  } finally { trainingReading = false; if (qualificationPayload) drawQualification(qualificationPayload); }
}
async function requestTrainingControl(cancel = false, requestedAction = null, worker = TRAINING_INSTANCE) {
  if (trainingBusy || trainingReading || !adminTraining || worker !== TRAINING_INSTANCE ||
      (cancel ? !trainingPending || trainingPending.instance !== TRAINING_INSTANCE : !(requestedAction || trainingAction))) return;
  const action = requestedAction || trainingAction, pending = trainingPending;
  if (!cancel && (trainingPending || !trainingInstance ||
      (action === "start" && !["stopped", "failed"].includes(trainingInstance.state)) ||
      (action === "shutdown" && !["running", "starting", "backoff"].includes(trainingInstance.state)))) return;
  trainingBusy = true;
  if (qualificationPayload) drawQualification(qualificationPayload);
  $("training-toggle").disabled = true; $("training-cancel").disabled = true;
  $("training-refresh").disabled = true;
  try {
    const response = await fetch("/api/control/instances" + (cancel ? "/cancel" : ""), {
      method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cancel ? { id: pending.id } : {
        instance: TRAINING_INSTANCE, action, after_jobs: action !== "shutdown",
        reason: "Paper market watch admin training control",
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Control refused (${response.status})`);
    $("training-control-state").textContent = `Request ${result.request.id}: ${result.request.state}. Waiting for supervisor state.`;
  } catch (error) {
    // A lost response is not proof the request failed. Never resend automatically.
    $("training-control-state").textContent = `${error.message}. Check supervisor status before retrying; the request may have reached it.`;
  } finally {
    trainingBusy = false; trainingAction = null; trainingPending = null;
    $("training-refresh").disabled = false;
    await refreshTrainingControl();
  }
}
$("training-toggle").addEventListener("click", () => requestTrainingControl());
$("training-cancel").addEventListener("click", () => requestTrainingControl(true));
$("training-refresh").addEventListener("click", refreshTrainingControl);
setInterval(() => { if (!document.hidden) refreshTrainingControl(); }, 3000);
refreshTrainingControl();

const historyView = renderer($("history"));
const bookView = renderer($("book"));
let lastHistory = null, lastBook = null;

function plotHistory(payload) {
  lastHistory = payload;
  historyView.clear();
  const symbols = [...SYMBOLS, payload.stock];
  const series = symbols.map((symbol) => {
    const bars = (payload.bars[symbol] || []).filter((bar) => Number.isFinite(bar.c) && Date.parse(bar.t));
    const first = bars[0]?.c;
    return { symbol, bars: first > 0 ? bars.map((bar) => ({ t: Date.parse(bar.t), pct: (bar.c / first - 1) * 100 })) : [] };
  });
  const all = series.flatMap((item) => item.bars);
  if (!all.length) { $("status").textContent = "No history bars were available for this selection."; return; }
  const left = Math.min(...all.map((bar) => bar.t));
  const right = Math.max(...all.map((bar) => bar.t));
  const lo = Math.min(...all.map((bar) => bar.pct));
  const hi = Math.max(...all.map((bar) => bar.pct));
  const spread = Math.max(hi - lo, .01);
  const ymin = lo - spread * .08, ymax = hi + spread * .08;
  const legend = $("legend"); legend.replaceChildren();
  for (const { symbol, bars } of series) {
    const swatch = document.createElement("span");
    swatch.textContent = `${symbol}: ${bars.length ? bars.at(-1).pct.toFixed(2) + "%" : "no bars"}`;
    const rgb = COLORS[symbol].slice(0, 3).map((v) => Math.round(v * 255));
    swatch.style.color = `rgb(${rgb.join(",")})`;
    legend.append(swatch);
    let segment = [], lastTime = null;
    const flush = () => { historyView.draw(segment, "line", COLORS[symbol]); segment = []; };
    for (const bar of bars) {
      if (lastTime !== null && bar.t - lastTime > 3 * 3600 * 1000) flush();
      segment.push(-.95 + 1.9 * (bar.t - left) / Math.max(right - left, 1),
                   -.85 + 1.7 * (bar.pct - ymin) / (ymax - ymin));
      lastTime = bar.t;
    }
    flush();
  }
}

function plotBook(book, note) {
  lastBook = { book, note };
  bookView.clear();
  const levels = $("book-levels"); levels.replaceChildren();
  if (!book || !Array.isArray(book.b) || !Array.isArray(book.a)) {
    $("book-state").textContent = note || "Order book unavailable on this feed.";
    return;
  }
  const bid = book.b.slice(0, 10), ask = book.a.slice(0, 10);
  const max = Math.max(1e-9, ...bid.concat(ask).map((level) => Number(level.s) || 0));
  const rect = (x0, x1, y0, y1) => [x0,y0,x1,y0,x0,y1,x0,y1,x1,y0,x1,y1];
  bid.forEach((level, i) => {
    const y = .8 - i * .17, width = .8 * (Number(level.s) || 0) / max;
    bookView.draw(rect(-.96, -.96 + width, y - .055, y + .055), "triangles", [0.25,.78,.55,1]);
  });
  ask.forEach((level, i) => {
    const y = .8 - i * .17, width = .8 * (Number(level.s) || 0) / max;
    bookView.draw(rect(.96 - width, .96, y - .055, y + .055), "triangles", [.94,.39,.43,1]);
  });
  const headerBid = document.createElement("strong"); headerBid.textContent = "Bid · price / BTC";
  const headerAsk = document.createElement("strong"); headerAsk.textContent = "Ask · price / BTC";
  levels.append(headerBid, headerAsk);
  for (let i = 0; i < Math.max(bid.length, ask.length); i++) {
    for (const level of [bid[i], ask[i]]) {
      const cell = document.createElement("span");
      cell.textContent = level ? `$${fmt(level.p)} / ${Number(level.s).toFixed(6)}` : "—";
      levels.append(cell);
    }
  }
  $("book-state").textContent = `Green bids · red asks · top ${Math.min(10,bid.length)} levels · book time ${book.t || "unknown"}`;
}

function tickers(payload) {
  const root = $("tickers"); root.replaceChildren();
  for (const symbol of [...SYMBOLS, payload.stock]) {
    const bar = payload.bars[symbol];
    const tile = document.createElement("div"); tile.className = "ticker";
    const name = document.createElement("strong"); name.textContent = symbol;
    const price = document.createElement("span"); price.textContent = bar ? `$${fmt(bar.c)}` : "No current bar";
    const time = document.createElement("small"); time.textContent = bar?.t ? new Date(bar.t).toLocaleString() : "";
    tile.append(name, price, time); root.append(tile);
  }
  const q = payload.rate_limit || {};
  $("rate").textContent = `Refresh: latest data every 15 seconds, history every 5 minutes. ` +
    `Alpaca remaining requests: stocks ${q.stocks?.remaining ?? "?"}/${q.stocks?.limit ?? "?"}, ` +
    `crypto ${q.crypto?.remaining ?? "?"}/${q.crypto?.limit ?? "?"}.`;
}

async function api(path) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Market data request failed (${response.status})`);
  return body;
}
const stock = () => $("wildcard").value;
async function credentialState() {
  const payload = await api("/api/user-secrets/alpaca-paper");
  $("key-state").textContent = payload.configured ? "Paper key configured for this account." : "Enter your paper key to load the markets.";
}
$("key-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("key-form").querySelector("button");
  button.disabled = true;
  try {
    const response = await fetch("/api/user-secrets/alpaca-paper", {
      method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json", "X-Meowdoku-Secret": "1" },
      body: JSON.stringify({ key_id: $("key-id").value.trim(), secret_key: $("key-secret").value.trim() }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not save the paper key.");
    $("key-form").reset();
    await credentialState();
    await update();
  } catch (error) { $("key-state").textContent = error.message; }
  finally { button.disabled = false; }
});
async function history() {
  const requested = stock();
  const payload = await api(`/api/markets/history?stock=${encodeURIComponent(requested)}`);
  if (requested === stock()) plotHistory(payload);
}
async function now() {
  const requested = stock();
  const payload = await api(`/api/markets/now?stock=${encodeURIComponent(requested)}`);
  if (requested === stock()) { tickers(payload); plotBook(payload.book, payload.book_note); }
}
async function update() {
  try { await Promise.all([history(), now()]); $("status").textContent = "Market data loaded. Stock feed: IEX."; }
  catch (error) { $("status").textContent = error.message; }
}
$("wildcard").addEventListener("change", update);
window.addEventListener("resize", () => { if (lastHistory) plotHistory(lastHistory); if (lastBook) plotBook(lastBook.book, lastBook.note); });
document.addEventListener("visibilitychange", () => { if (!document.hidden) update(); });
setInterval(() => { if (!document.hidden) now().catch((error) => { $("status").textContent = error.message; }); }, 15000);
setInterval(() => { if (!document.hidden) history().catch((error) => { $("status").textContent = error.message; }); }, 300000);
update();
credentialState().catch((error) => { $("key-state").textContent = error.message; });

const dmTemplates = {
  "look:account": () => ({}), "look:bots": () => ({}), "look:presence": () => ({}),
  "look:strategies": () => ({}),
  "look:orders": () => ({}), "look:mock_orders": () => ({}),
  "look:mock_ledger": () => ({ strategy_id: "" }),
  "look:scoreboard": () => ({}), "look:map": () => ({}), "look:picture": () => ({}),
  "look:qualification": () => ({}),
  "look:proposals": () => ({ stock: "AAPL", offset: 0, limit: 100 }),
  "examine:node": () => ({ radius: 1 }),
  "make:strategy": () => ({ request_id: crypto.randomUUID().replaceAll("-", ""), name: "Strategy", risk: "cautious", policy: "momentum", outcomes: ["fixed_horizon", "closed_trade", "fill_quality"] }),
  "make:bot": () => ({ request_id: crypto.randomUUID().replaceAll("-", ""), name: "Market bot", kind: "python", source: { ref: "divider:YOUR_JOB_ID" } }),
  "make:order": () => ({ request_id: crypto.randomUUID().replaceAll("-", ""), strategy_id: "", symbol: "BTC/USD", side: "buy", amount: "5" }),
  "make:mock_order": () => ({ request_id: crypto.randomUUID().replaceAll("-", ""), strategy_id: "", symbol: "BTC/USD", side: "buy", amount: "5" }),
  "use:reconcile": () => ({ id: "" }),
  "use:nodus_ui": () => ({ instance: "nodus-messenger", action: { op: "viewport", expected_revision: 0, width: 800, height: 600 } }),
};
$("dm-command").addEventListener("change", async () => {
  const command = $("dm-command").value;
  const template = dmTemplates[command]();
  if (command === "use:nodus_ui") {
    try {
      const response = await fetch(`/api/nodus/digest?instance=${encodeURIComponent(template.instance)}`, { credentials: "same-origin", cache: "no-store" });
      if (response.ok) template.action.expected_revision = (await response.json()).revision;
    } catch (_) { /* The action response reports unavailable hosts. */ }
  }
  if ($("dm-command").value === command) $("dm-input").value = JSON.stringify(template, null, 2);
});
$("dm-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("dm-form").querySelector("button");
  button.disabled = true;
  try {
    const input = JSON.parse($("dm-input").value);
    if (input === null || Array.isArray(input) || typeof input !== "object") throw new Error("Input must be a JSON object.");
    const [verb, target] = $("dm-command").value.split(":");
    const response = await fetch("/api/paper/dm", {
      method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ verb, target, input }),
    });
    const result = await response.json();
    $("dm-output").textContent = JSON.stringify(result, null, 2);
    if (!response.ok) $("dm-output").textContent = `Command failed (${response.status})\n` + $("dm-output").textContent;
    const picture = $("dm-picture");
    picture.hidden = !(response.ok && verb === "look" && target === "picture" && result.view);
    if (!picture.hidden) {
      const view = new URL(result.view, location.href);
      view.searchParams.set("picture", "1");
      picture.src = view.href;
      $("dm-output").textContent = "Living Data map picture";
    } else picture.removeAttribute("src");
  } catch (error) { $("dm-picture").hidden = true; $("dm-picture").removeAttribute("src"); $("dm-output").textContent = error.message; }
  finally { button.disabled = false; }
});

let latestPresence = null;
let presenceCards = [];
function drawPresence() {
  const root = $("bot-presence"); root.replaceChildren();
  presenceCards = [];
  const bots = latestPresence?.bots || [];
  for (const bot of bots) {
    const age = Math.max(0, Date.now() / 1000 - bot.last_seen);
    if (age >= (latestPresence?.fade_seconds || 300)) continue;
    const card = document.createElement("div"); card.className = "bot-unit";
    card.style.opacity = String(Math.max(.12, 1 - age / (latestPresence.fade_seconds || 300)));
    const avatar = document.createElement("span"); avatar.className = "bot-avatar";
    const head = document.createElement("span"); head.className = "bot-head";
    const body = document.createElement("span"); body.className = "bot-body";
    avatar.append(head, body);
    const copy = document.createElement("span"); copy.className = "bot-copy";
    const name = document.createElement("strong"); name.textContent = bot.name;
    const kind = document.createElement("small"); kind.textContent = bot.kind.replaceAll("_", " ");
    const source = document.createElement("small");
    const channel = bot.activity?.channel === "mcp_reported" ? "MCP (reported)" : bot.activity?.channel || "REST";
    source.textContent = `${channel} · ${bot.activity?.interaction || "active"}`;
    copy.append(name, kind, source); card.append(avatar, copy); root.append(card);
    presenceCards.push({ card, seen: bot.last_seen });
  }
  $("bot-presence-state").textContent = root.childElementCount ? `${root.childElementCount} bots active in the last five minutes.` : "No registered bots active in the last five minutes.";
}
function fadePresence() {
  const seconds = latestPresence?.fade_seconds || 300;
  for (const item of presenceCards) {
    const age = Math.max(0, Date.now() / 1000 - item.seen);
    item.card.style.opacity = age >= seconds ? "0" : String(Math.max(.12, 1 - age / seconds));
  }
}
async function updatePresence() {
  let offset = 0, combined = [], latest = null;
  for (let page = 0; page < 20; page++) {
    latest = await api(`/api/paper/presence?offset=${offset}`);
    combined.push(...(latest.bots || []));
    if (latest.next_offset === null || latest.next_offset === undefined) break;
    offset = latest.next_offset;
  }
  latestPresence = { ...latest, bots: combined };
  drawPresence();
}
setInterval(fadePresence, 1000);
setInterval(() => { if (!document.hidden) updatePresence().catch((error) => { $("bot-presence-state").textContent = error.message; }); }, 15000);
updatePresence().catch((error) => { $("bot-presence-state").textContent = error.message; });
