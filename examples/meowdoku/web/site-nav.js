// Meowdoku site controls. HAND-AUTHORED, not produced by the Turing compiler.
// The local HLS decoder is loaded only after a listener explicitly starts audio.
// Opt in: <script type="module" src="/site-nav.js" data-site-nav="allow"></script>.  Without that attribute it does nothing.
// Signed out: a small "Sign in" link to /login.html.  Signed in: "<username> · <server>" opening a menu of the pages this account's
// capabilities reach (GET /api/whoami), the Stable / Development server switch (GET/POST /api/route, dev_ui only) and Sign out
// (POST /api/account/logout).  Built with createElement/textContent only, inside a closed Shadow DOM; the server enforces every
// capability, this only decides what to show.
const VERSION = 9;

function allowed() {
  const here = import.meta.url;
  for (const s of document.querySelectorAll('script[type="module"][src]')) {
    if (s.src === here && s.getAttribute("data-site-nav") === "allow") return true;
  }
  return false;
}

function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "text") n.textContent = v;
    else if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid) n.append(kid);
  return n;
}

async function getJson(path) {
  const r = await fetch(path, { credentials: "same-origin", cache: "no-store" });
  if (!r.ok) throw new Error(path + " " + r.status);
  return r.json();
}

async function postJson(path, body) {
  const r = await fetch(path, { method: "POST", credentials: "same-origin", cache: "no-store",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  let j = {};
  try { j = await r.json(); } catch (e) { /* empty body */ }
  return { status: r.status, body: j };
}

const CSS = `
:host { all: initial; }
.wrap { position: fixed; top: max(8px, env(safe-area-inset-top)); right: max(8px, env(safe-area-inset-right)); z-index: 2147483000; font: 13px/1.3 system-ui, sans-serif; color: #e8e8ee; display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 6px; max-width: calc(100vw - 32px); padding: 6px; border: 1px solid #454556; border-radius: 10px; background: #14141eee; }
.actions { display: contents; }
.chip.toggle { min-height: 32px; box-sizing: border-box; }
.label { font-size: 11px; color: #b6b6ca; margin: 0 4px; }
.chip { all: unset; cursor: pointer; padding: 5px 10px; border-radius: 14px; background: rgba(28,28,36,.82); border: 1px solid rgba(255,255,255,.18);
        color: #e8e8ee; font: inherit; text-decoration: none; display: inline-block; }
.chip:hover, .chip:focus-visible { background: rgba(48,48,62,.95); outline: 2px solid #7aa2ff; outline-offset: 1px; }
.chip.quiet { opacity: .55; } .chip.quiet:hover, .chip.quiet:focus-visible { opacity: 1; }
.menu { position: absolute; top: calc(100% + 6px); right: 0; min-width: 210px; max-width: calc(100vw - 48px); max-height: 65vh; overflow: auto; background: #1c1c24; border: 1px solid rgba(255,255,255,.18);
        border-radius: 8px; padding: 6px; box-shadow: 0 6px 24px rgba(0,0,0,.5); }
.menu a, .menu button { all: unset; display: block; box-sizing: border-box; width: 100%; padding: 6px 10px; border-radius: 5px; cursor: pointer;
        color: #e8e8ee; font: inherit; }
.menu a:hover, .menu button:hover, .menu a:focus-visible, .menu button:focus-visible { background: #2c2c3a; }
.menu a[aria-current="page"] { color: #9fb8ff; }
.menu button[aria-pressed="true"] { color: #9fb8ff; font-weight: 600; }
.menu button:disabled { opacity: .45; cursor: default; }
.head { padding: 4px 10px 6px; color: #a8a8b8; font-size: 12px; }
.sep { height: 1px; margin: 5px 4px; background: rgba(255,255,255,.12); }
.suite { position: absolute; right: 0; top: calc(100% + 6px); box-sizing: border-box; width: min(420px, calc(100vw - 32px)); max-height: min(70vh, 560px); overflow: auto; padding: 12px; border: 1px solid #454556; border-radius: 10px; background: #14141e; }
.binding { display: grid; gap: 3px; padding: 7px; margin: 4px 0; border: 1px solid #454556; border-radius: 4px; overflow-wrap: anywhere; }
.binding span { font-size: 11px; color: #b6b6ca; }
.binding.active { border-color: #ffd166; background: #80551e; }
.radio-bands { display: flex; gap: 4px; align-items: center; min-height: 30px; }
.radio-bands meter { flex: 1; min-width: 0; width: 40px; height: 24px; }
.radio-audio { width: 100%; margin: 8px 0; }
.suite[aria-label="Nodus Radio"] { border-radius: 2px; background: #171b20; border-color: #5f747d; font: 12px/1.35 ui-monospace, Consolas, monospace; }
.suite[aria-label="Nodus Radio"] .radio-bands { padding: 4px; border: 1px solid #435960; background: #071317; }
.suite[aria-label="Nodus Radio"] .chip { border-radius: 2px; }
.suite[aria-label="Nodus Radio"] input[type="search"], .suite[aria-label="Nodus Radio"] select { box-sizing: border-box; width: 100%; margin: 4px 0; padding: 5px; color: #e8f7f4; background: #101a1e; border: 1px solid #5f747d; font: inherit; }
[hidden] { display: none !important; }
.note { padding: 4px 10px; color: #ffb3a8; font-size: 12px; }
`;

// Pages supply their existing policy and sampler. This is presentation only:
// no key listeners, pointer capture, device permissions or second control engine.
const suites = new Map();
let suiteMount = null;
let dismissHeaderPanel = null;
export function registerControlSuite(spec) {
  if (!spec || typeof spec.identity !== "string" || !spec.identity ||
      !Array.isArray(spec.policy?.bindings)) throw new TypeError("Control suite needs identity and policy.bindings");
  if (spec.sample !== undefined && typeof spec.sample !== "function") throw new TypeError("sample must be a function");
  if (suites.has(spec.identity)) throw new Error("Control suite already registered: " + spec.identity);
  if (spec.policy.bindings.some(binding => typeof binding.action !== "string" || !Array.isArray(binding.inputs) || binding.inputs.some(input => typeof input !== "string"))) throw new TypeError("Invalid control bindings");
  if ((spec.actions || []).some(action => typeof action.run !== "function")) throw new TypeError("Control action needs run callback");
  const record = {spec, dispose: null};
  suites.set(spec.identity, record);
  if (suiteMount) record.dispose = suiteMount(spec);
  return () => {
    if (suites.get(spec.identity) !== record) return;
    record.dispose?.();
    suites.delete(spec.identity);
  };
}

function mountControlSuite(wrap, host, spec) {
  const chip = el("button", {class: "chip", type: "button", text: spec.label || "Controls", "aria-expanded": "false"});
  const panel = el("section", {class: "suite", "aria-label": spec.label || "Controls", hidden: ""});
  const status = el("div", {class: "note", role: "status"});
  const close = el("button", {class: "chip", type: "button", text: "Close controls"});
  panel.append(close);
  const rows = spec.policy.bindings.map(binding => {
    const row = el("div", {class: "binding"});
    row.append(el("strong", {text: binding.action}), el("span", {text: binding.inputs.join(" · ")}));
    panel.append(row);
    return {row, action: binding.action};
  });
  for (const action of spec.actions || []) {
    if (typeof action.run !== "function") throw new TypeError("Control action needs run callback");
    const button = el("button", {class: "chip", type: "button", text: action.label});
    button.addEventListener("click", async () => {
      button.disabled = true;
      try { if (action.close) setOpen(false); await action.run(); status.textContent = ""; }
      catch (error) { status.textContent = String(error.message || error); }
      finally { button.disabled = false; }
    });
    panel.append(button);
  }
  panel.append(status);
  let frame = 0;
  function refresh() {
    if (panel.hidden) return;
    try {
      for (const {row, action} of rows) {
        const value = spec.sample?.(action);
        row.classList.toggle("active", Number.isFinite(value) && value !== 0);
      }
    } catch (error) { status.textContent = "Control telemetry unavailable: " + String(error.message || error); }
    if (spec.sample) frame = requestAnimationFrame(refresh);
  }
  function setOpen(open, focus = true) {
    if (open === !panel.hidden) return;
    try { (open ? spec.onOpen : spec.onClose)?.(); }
    catch (error) { status.textContent = String(error.message || error); return; }
    if (open && dismissHeaderPanel && dismissHeaderPanel !== dismiss) dismissHeaderPanel();
    dismissHeaderPanel = open ? dismiss : (dismissHeaderPanel === dismiss ? null : dismissHeaderPanel);
    panel.hidden = !open;
    chip.setAttribute("aria-expanded", String(open));
    cancelAnimationFrame(frame);
    if (open) { refresh(); if (focus) close.focus(); }
    else if (focus) chip.focus();
  }
  const dismiss = () => setOpen(false, false);
  chip.addEventListener("click", () => setOpen(panel.hidden));
  close.addEventListener("click", () => setOpen(false));
  panel.addEventListener("keydown", event => {
    if (event.key === "Escape") { event.stopPropagation(); setOpen(false); }
  });
  const outside = event => { if (!event.composedPath().includes(host)) setOpen(false, false); };
  document.addEventListener("pointerdown", outside);
  wrap.append(chip, panel);
  return () => {
    setOpen(false, false);
    cancelAnimationFrame(frame);
    document.removeEventListener("pointerdown", outside);
    chip.remove(); panel.remove();
  };
}

// Nodus owns capture/FFT; Python reads its existing UDP relay inbox.
// This widget shows NRAD bands and an opt-in HLS player for the authored file station.
function mountRadio(wrap, host, canControl = false) {
  const chip = el("button", {class: "chip", type: "button", text: "Nodus Radio", "aria-expanded": "false"});
  const panel = el("section", {class: "suite", hidden: "", "aria-label": "Nodus Radio"});
  const close = el("button", {class: "chip", type: "button", text: "Close radio"});
  const title = el("strong", {text: "Server station · Nodus Radio"});
  const status = el("p", {role: "status", text: "Waiting for Nodus"});
  const bands = el("div", {class: "radio-bands", "aria-label": "Spectral power bands"});
  const audio = el("audio", {class: "radio-audio", controls: "", preload: "none", "aria-label": "Nodus Radio audio"});
  const nativeHls = !!audio.canPlayType("application/vnd.apple.mpegurl");
  const listen = el("button", {class: "chip", type: "button", text: "Listen in this browser"});
  const audioNote = el("p", {role: "status", text: "Audio starts when you choose Listen."});
  audio.hidden = audioNote.hidden = listen.hidden = true;
  panel.append(close, title, status, bands, listen, audio, audioNote);
  let hlsEngine = null, audioEpoch = 0, audioAvailable = false;
  let fftFrames = [], fftTitle = null, historyBusy = false, historyTimer = null;
  function renderBands(frame) {
    bands.replaceChildren();
    for (const [index, value] of (frame.bands || []).slice(0, 128).entries()) {
      const level = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
      bands.append(el("meter", {min: "0", max: "1", value: String(frame.status === "live" ? level : 0),
        "aria-label": "Band " + (index + 1), title: "Band " + (index + 1) + ": " + level.toFixed(3)}));
    }
  }
  function stopAudio() {
    audioEpoch++;
    audio.pause();
    if (hlsEngine) { hlsEngine.destroy(); hlsEngine = null; }
    audio.removeAttribute("src"); audio.load();
    audio.hidden = true;
    listen.disabled = false;
    fftFrames = []; fftTitle = null;
  }
  listen.addEventListener("click", async () => {
    if (!audioAvailable) return;
    const epoch = ++audioEpoch;
    listen.disabled = true;
    audioNote.textContent = "Connecting to the station…";
    try {
      if (nativeHls) {
        audio.src = "/api/radio/hls/radio.m3u8";
        audio.hidden = false;
        await audio.play();
      } else {
        const {default: Hls} = await import("./radio-hls.min.js");
        if (epoch !== audioEpoch || !audioAvailable) return;
        if (!Hls.isSupported()) throw new Error("This browser cannot decode the radio stream");
        hlsEngine = new Hls({lowLatencyMode: true});
        hlsEngine.on(Hls.Events.MEDIA_ATTACHED, () => hlsEngine?.loadSource("/api/radio/hls/radio.m3u8"));
        hlsEngine.on(Hls.Events.MANIFEST_PARSED, () => {
          if (epoch === audioEpoch) audio.play().catch(error => { audioNote.textContent = String(error.message || error); });
        });
        hlsEngine.on(Hls.Events.ERROR, (_event, data) => {
          if (epoch === audioEpoch && data.fatal) {
            stopAudio();
            audioNote.textContent = "Radio stream error: " + data.details;
          }
        });
        audio.hidden = false;
        hlsEngine.attachMedia(audio);
      }
      if (epoch === audioEpoch) audioNote.textContent = "Listening here. Server speakers are separate.";
    } catch (error) {
      if (epoch === audioEpoch) {
        stopAudio();
        audioNote.textContent = "Could not start radio: " + String(error.message || error);
      }
    } finally { listen.disabled = !!audio.getAttribute("src") || !!hlsEngine; }
  });
  function showAudibleFrame() {
    if (audio.paused || !Number.isFinite(audio.currentTime) || !fftFrames.length) return;
    // Both NRAD captured_frames and HLS audio PTS start at the file-source
    // sample origin. Select by media time, rather than packet arrival time.
    const target = audio.currentTime;
    let selected = null;
    for (const frame of fftFrames) {
      if (Number(frame.frame) / frame.sample_rate <= target) selected = frame;
      else break;
    }
    if (!selected || target - Number(selected.frame) / selected.sample_rate > 0.25) return;
    renderBands({...selected, status: "live"});
    window.dispatchEvent(new CustomEvent("nodus-radio-audible-frame", {detail: selected}));
  }
  async function pollHistory(epoch) {
    if (epoch !== generation || panel.hidden) return;
    if (!historyBusy && audioAvailable) {
      historyBusy = true;
      try {
        const data = await getJson("/api/radio/history");
        if (epoch === generation && !panel.hidden) {
          if (fftTitle !== data.title) fftFrames = [];
          fftTitle = data.title;
          fftFrames = data.frames || [];
          showAudibleFrame();
        }
      } catch (_error) { if (epoch === generation) fftFrames = []; }
      finally { historyBusy = false; }
    }
    if (epoch === generation && !panel.hidden) historyTimer = setTimeout(() => pollHistory(epoch), 250);
  }
  audio.addEventListener("timeupdate", showAudibleFrame);
  const queueStatus = el("p", {role: "status", text: "Jukebox: checking configuration"});
  const search = el("input", {type: "search", placeholder: "Search station tracks", "aria-label": "Search station tracks"});
  const songs = el("select", {"aria-label": "Song to request"});
  const request = el("button", {class: "chip", type: "button", text: "Request song"});
  const queue = el("div", {"aria-label": "Radio request lists"});
  const charts = el("div", {"aria-label": "Radio billboard"});
  const rating = el("select", {"aria-label": "Rate the playing song"});
  rating.append(el("option", {value: "", text: "No rating"}));
  for (let n = 0; n <= 5; n++) rating.append(el("option", {value: String(n), text: String(n)}));
  const vote = el("button", {class: "chip", type: "button", text: "Vote 0–5"});
  const adminButtons = [];
  let jukeboxBusy = false, currentPlay = null;
  request.disabled = vote.disabled = true;
  panel.append(queueStatus, search, songs, request, rating, vote, queue, charts);
  let availableSongs = [];
  function filterSongs() {
    const selected = songs.value;
    const query = String(search.value || "").trim().toLocaleLowerCase();
    const matching = availableSongs.filter(song => song.title.toLocaleLowerCase().includes(query));
    songs.replaceChildren(...matching.map(song => el("option", {value: song.id, text: song.title})));
    if (matching.some(song => song.id === selected)) songs.value = selected;
    request.disabled = jukeboxBusy || !matching.length;
  }
  search.addEventListener("input", filterSongs);
  function showJukebox(data) {
    const catalogue = data.catalogue || [];
    const names = new Map(catalogue.map(s => [s.id, s.title]));
    availableSongs = catalogue;
    filterSongs();
    currentPlay = data.current?.id || null;
    queueStatus.textContent = data.error || data.runner?.error || (data.paused ? "Jukebox paused" : "Jukebox running") +
      (data.current ? ": " + (names.get(data.current.song_id) || data.current.song_id) : ": no song playing");
    vote.disabled = jukeboxBusy || !currentPlay;
    for (const button of adminButtons) button.disabled = jukeboxBusy || !!data.error;
    queue.replaceChildren();
    for (const row of (data.requests || [])) {
      const line = el("div", {text: (row.mine ? "Your request: " : "Request: ") + (names.get(row.song_id) || row.song_id)});
      if (row.mine || canControl) {
        const remove = el("button", {type: "button", text: "Remove"});
        remove.disabled = jukeboxBusy;
        remove.addEventListener("click", () => commandJukebox({action: "remove", request_id: row.id}));
        line.append(remove);
      }
      queue.append(line);
    }
    charts.replaceChildren(el("strong", {text: "Billboard · average rating, then vote count"}));
    for (const row of (data.charts || [])) charts.append(el("p", {text:
      (names.get(row.song_id) || row.song_id) + " · " + Number(row.average).toFixed(2) + "/5 · " + row.votes + " votes"}));
  }
  async function commandJukebox(body) {
    if (jukeboxBusy) return;
    jukeboxBusy = true; request.disabled = vote.disabled = true;
    for (const button of adminButtons) button.disabled = true;
    const epoch = generation;
    try {
      const response = await fetch("/api/radio/jukebox", {method: "POST", credentials: "same-origin", cache: "no-store",
        headers: {"Content-Type": "application/json", "X-Meowdoku-Radio": "1"}, body: JSON.stringify(body)});
      const data = await response.json();
      jukeboxBusy = false;
      if (epoch === generation && !panel.hidden) showJukebox(data);
    } catch (error) {jukeboxBusy = false; if (epoch === generation && !panel.hidden) showJukebox({error: "Request failed; refresh status before retrying"});}
  }
  request.addEventListener("click", () => commandJukebox({action: "request", song_id: songs.value}));
  vote.addEventListener("click", () => {
    if (rating.value !== "" && currentPlay) commandJukebox({action: "rate", play_id: currentPlay, rating: Number(rating.value)});
  });
  if (canControl) for (const [action, label] of [["resume", "Start / resume jukebox"], ["pause", "Pause after this song"], ["skip", "Skip current song"]]) {
    const button = el("button", {class: "chip", type: "button", text: label});
    button.disabled = true; button.addEventListener("click", () => commandJukebox({action}));
    adminButtons.push(button); panel.append(button);
  }
  let paTimer = null, paBusy = false;
  const paStatus = el("p", {role: "status", text: "Server speakers: checking configuration"});
  const play = el("button", {class: "chip", type: "button", text: "Play on server speakers"});
  const stop = el("button", {class: "chip", type: "button", text: "Stop my server playback"});
  play.disabled = stop.disabled = true;
  function showPA(data) {
    paStatus.textContent = "Server speakers: " + (data.error || data.status || "unavailable");
    play.disabled = paBusy || !data.configured || !["stopped", "failed"].includes(data.status);
    stop.disabled = paBusy || !data.can_stop;
  }
  async function pollPA(epoch) {
    try {
      const data = await getJson("/api/radio/jukebox");
      if (epoch !== generation || panel.hidden) return;
      if (!jukeboxBusy) showJukebox(data);
    } catch (error) {
      if (epoch !== generation || panel.hidden) return;
      showJukebox({error: "Jukebox unavailable"});
    }
    if (canControl) try {
      const data = await getJson("/api/radio/playback");
      if (epoch !== generation || panel.hidden) return;
      showPA(data);
    } catch (error) {
      if (epoch !== generation || panel.hidden) return;
      showPA({error: "unavailable"});
    }
    if (epoch === generation && !panel.hidden) paTimer = setTimeout(() => pollPA(epoch), 1000);
  }
  async function commandPA(action) {
    if (paBusy) return;
    paBusy = true; play.disabled = stop.disabled = true;
    try {
      const response = await fetch("/api/radio/playback", {method: "POST", credentials: "same-origin", cache: "no-store",
        headers: {"Content-Type": "application/json", "X-Meowdoku-Radio": "1"}, body: JSON.stringify({action})});
      const data = await response.json();
      paBusy = false;
      showPA(data);
    } catch (error) {paBusy = false; showPA({error: "request failed; check status before retrying"});}
  }
  if (canControl) {
    panel.append(el("p", {text: "Admin PA control: plays on the server's selected speakers, not this browser. Continues until Stop or the source ends."}), paStatus, play, stop);
    play.addEventListener("click", () => commandPA("start"));
    stop.addEventListener("click", () => commandPA("stop"));
  }
  let generation = 0, timer = null;
  const dismiss = () => setOpen(false);
  async function poll(epoch) {
    try {
      const data = await getJson("/api/radio");
      if (epoch !== generation || panel.hidden) return;
      window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: data}));
      title.textContent = data.title || "Nodus Radio";
      const available = data.audio === true;
      audioAvailable = available;
      listen.hidden = !available;
      audioNote.hidden = !available;
      if (!available && (audio.getAttribute("src") || hlsEngine)) stopAudio();
      status.textContent = data.status === "live" ? "Live · RMS " + Number(data.rms).toFixed(3)
        : data.status === "stale" ? "Signal stopped · last packet " + Number(data.age_s).toFixed(1) + "s ago"
        : data.status === "waiting" ? "Waiting for Nodus packets" : "Radio is not broadcasting";
      if (audio.paused) renderBands(data);
    } catch (error) {
      if (epoch !== generation || panel.hidden) return;
      window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: {status: "unavailable"}}));
      status.textContent = "Radio unavailable";
      bands.replaceChildren();
    }
    if (epoch === generation && !panel.hidden) timer = setTimeout(() => poll(epoch), 100);
  }
  function setOpen(open) {
    if (open && dismissHeaderPanel && dismissHeaderPanel !== dismiss) dismissHeaderPanel();
    dismissHeaderPanel = open ? dismiss : (dismissHeaderPanel === dismiss ? null : dismissHeaderPanel);
    generation++; clearTimeout(timer); timer = null; clearTimeout(paTimer); paTimer = null;
    clearTimeout(historyTimer); historyTimer = null;
    if (!open) stopAudio();
    if (!open) window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: {status: "unavailable"}}));
    panel.hidden = !open; chip.setAttribute("aria-expanded", String(open));
    if (open) {poll(generation); pollPA(generation); pollHistory(generation);}
  }
  chip.addEventListener("click", () => setOpen(panel.hidden));
  close.addEventListener("click", () => {setOpen(false); chip.focus();});
  panel.addEventListener("keydown", event => {
    if (event.key === "Escape") {event.stopPropagation(); setOpen(false); chip.focus();}
  });
  document.addEventListener("pointerdown", event => {if (!event.composedPath().includes(host)) setOpen(false);});
  window.addEventListener("pagehide", () => setOpen(false));
  document.addEventListener("visibilitychange", () => {if (document.hidden) setOpen(false);});
  wrap.append(chip, panel);
}

async function main() {
  if (!allowed()) { console.info("[site-nav] not loaded: this page's script tag lacks data-site-nav=\"allow\""); return; }

  const host = document.createElement("div");
  host.style.cssText = "all:initial !important;display:block !important;flex:0 0 auto !important;grid-column:1 / -1 !important";
  const root = host.attachShadow({ mode: "closed" });
  root.append(el("style", { text: CSS }));
  const wrap = el("nav", { class: "wrap", "aria-label": "Site controls" });
  const actions = el("div", {class: "actions", id: "site-actions"});
  const toggle = el("button", {class: "chip toggle", type: "button",
    text: "Hide menu", "aria-expanded": "true", "aria-controls": "site-actions"});
  actions.append(el("span", { class: "label", text: "Site controls" }));
  wrap.append(actions, toggle);
  root.append(wrap);
  // The fixed banner used to cover the page's first controls. Reserve its
  // measured height in normal flow, including wrapping on narrow screens.
  document.body.prepend(host);
  function reserveSpace() {
    host.style.setProperty("height", Math.ceil(wrap.getBoundingClientRect().bottom + 8) + "px", "important");
  }
  toggle.addEventListener("click", () => {
    const collapse = !actions.hidden;
    if (collapse) dismissHeaderPanel?.();
    actions.hidden = collapse;
    toggle.textContent = collapse ? "Show site menu" : "Hide menu";
    toggle.setAttribute("aria-expanded", String(!collapse));
    reserveSpace();
    toggle.focus();
  });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(reserveSpace).observe(wrap);
  window.addEventListener("resize", reserveSpace);
  reserveSpace();
  suiteMount = spec => mountControlSuite(actions, host, spec);
  for (const record of suites.values()) record.dispose = suiteMount(record.spec);
  window.dispatchEvent(new Event("site-controls-ready"));
  let who;
  try { who = await getJson("/api/whoami"); } catch (e) { return; }
  const caps = new Set(who.capabilities || []);

  if (!who.signed_in) {
    actions.append(el("a", { class: "chip" + (who.shader ? " quiet" : ""), href: "/login.html", text: "Sign in" }),
      el("a", { class: "chip quiet", href: "/register.html", text: "Register" }));   // MOCKUP: registration is by invitation
    reserveSpace();
    return;
  }

  if (caps.has("view_site")) mountRadio(actions, host, caps.has("account_manage"));

  if (!new URLSearchParams(location.search).has("viewer")) {
    // Tickets is a small menu: the account's issue tickets, and (dev_ui) the change reviews an admin approves.
    const ticketsChip = el("button", { class: "chip", type: "button", "aria-haspopup": "true", "aria-expanded": "false",
      text: "Tickets ▾", title: "Issue tickets and change reviews" });
    const ticketsMenu = el("div", { class: "menu", role: "menu", hidden: "" });
    ticketsMenu.append(el("a", { href: "/issues.html", role: "menuitem", text: "Issue tickets" }));
    if (caps.has("dev_ui")) ticketsMenu.append(el("a", { href: "/reviews.html", role: "menuitem",
      text: who.role === "admin" ? "Change reviews (approve)" : "Change reviews" }));
    const dismissTickets = () => setTicketsOpen(false);
    function setTicketsOpen(open) {
      if (open && dismissHeaderPanel && dismissHeaderPanel !== dismissTickets) dismissHeaderPanel();
      dismissHeaderPanel = open ? dismissTickets : (dismissHeaderPanel === dismissTickets ? null : dismissHeaderPanel);
      ticketsMenu.hidden = !open;
      ticketsChip.setAttribute("aria-expanded", String(open));
    }
    ticketsChip.addEventListener("click", () => setTicketsOpen(ticketsMenu.hidden));
    wrap.addEventListener("keydown", (e) => { if (e.key === "Escape" && !ticketsMenu.hidden) { setTicketsOpen(false); ticketsChip.focus(); } });
    document.addEventListener("pointerdown", (e) => { if (!ticketsMenu.hidden && e.target !== host) setTicketsOpen(false); });
    actions.append(ticketsChip, ticketsMenu);
    actions.append(el("button", { class: "chip", type: "button", text: "Report issue", onclick: () => {
      // Frozen game pages can still own the legacy /api/tickets form. The
      // shared header always opens the account-owned SQL issue interface.
      location.assign("/issues.html?new=1&source=" + encodeURIComponent(location.pathname));
    } }));
  }

  let route = {};
  const pageRouteControl = who.signed_in && (caps.has("dev_ui") || location.pathname === "/" ||
    location.pathname === "/index.html");
  if (pageRouteControl) { try { route = await getJson("/api/route"); } catch (e) { route = {}; } }
  const onDev = !!(route.dev_route && route.route === route.dev_route);
  const serverName = pageRouteControl ? (onDev ? "development" : "stable") : "";
  const chip = el("button", { class: "chip", "aria-haspopup": "true", "aria-expanded": "false",
    text: who.username + (serverName ? " · " + serverName : "") });
  const menu = el("div", { class: "menu", role: "menu", hidden: "" });
  const note = el("div", { class: "note", hidden: "" });

  const here = location.pathname === "/index.html" ? "/" : location.pathname;
  function link(label, href, show) {
    if (show === false) return null;
    const a = el("a", { href, role: "menuitem", text: label });
    if (href === here) a.setAttribute("aria-current", "page");
    return a;
  }
  async function choose(site) {
    const res = await postJson("/api/route", { site });
    if (res.status === 200) { location.reload(); return; }
    note.textContent = (res.body && res.body.error) || "could not switch server";
    note.hidden = false;
  }
  function serverButton(label, site, pressed, disabled) {
    const b = el("button", { role: "menuitemradio", "aria-pressed": String(pressed), text: label, onclick: () => choose(site) });
    if (disabled) b.disabled = true;
    return b;
  }

  menu.append(el("div", { class: "head", text: "Signed in as " + who.username + " (" + (who.role || "member") + ")" }));
  if (caps.has("message")) menu.append(el("button", { role: "menuitem", type: "button", text: "Messages", onclick: () => {
    setOpen(false);
    if (window.MeowdokuMessenger) window.MeowdokuMessenger.open();
    else location.assign("/login.html?messages=1");
  } }));
  for (const item of [
    link("Home", "/"),
    link("Games", "/games.html", !onDev || caps.has("dev_ui")),
    link("Profile and servers", "/profile.html"),
    link("Keys", "/keys.html", caps.has("message")),
    link("Connected services", "/connected-services.html"),
    link("My files", "/files.html", caps.has("message")),
    link("Paper markets", "/markets.html", caps.has("message")),
    link("Browse the site", "/browse"),
    link("Help: the manual", "/manual/", caps.has("view_site")),
    link("Repository world map", "/worldmap.html", caps.has("dev_ui")),
    link("Math lab", "/mathlab.html", caps.has("compile_submit")),
    link("Source divider", "/divider.html", caps.has("compile_submit")),
    link("Marble studio", "/marble_studio.html", caps.has("dev_ui")),
    link("T4 wave arena", "/wave_arena.html", caps.has("dev_ui")),
    link("Creature studio", "/mechanical_creature_studio.html", caps.has("dev_ui")),
    link("Account and password", "/login.html"),
    link("Invite someone", "/invite.html"),                                         // MOCKUP: any account may invite
  ]) if (item) menu.append(item);
  if (pageRouteControl) {
    menu.append(el("div", { class: "sep" }), el("div", { class: "head", text: "Server: " + (who.instance || serverName) }),
      serverButton("Stable server", "stable", !onDev, false),
      serverButton("Development server", "dev", onDev, route.dev_available === false));
  }
  menu.append(note, el("div", { class: "sep" }),
    el("button", { role: "menuitem", text: "Sign out", onclick: async () => { await postJson("/api/account/logout", {}); location.reload(); } }));

  const dismiss = () => setOpen(false);
  function setOpen(open) {
    if (open && dismissHeaderPanel && dismissHeaderPanel !== dismiss) dismissHeaderPanel();
    dismissHeaderPanel = open ? dismiss : (dismissHeaderPanel === dismiss ? null : dismissHeaderPanel);
    menu.hidden = !open;
    chip.setAttribute("aria-expanded", String(open));
  }
  chip.addEventListener("click", () => setOpen(menu.hidden));
  wrap.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { setOpen(false); chip.focus(); } });
  document.addEventListener("pointerdown", (e) => { if (!menu.hidden && e.target !== host) setOpen(false); });
  actions.append(chip, menu);
  reserveSpace();
}

window.MeowdokuSiteNav = { version: VERSION, registerControlSuite };
main();
