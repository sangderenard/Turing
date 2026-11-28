// Meowdoku site controls. HAND-AUTHORED, not produced by the Turing compiler.
// The local HLS decoder is loaded only after a listener explicitly starts audio.
// Opt in: <script type="module" src="/site-nav.js" data-site-nav="allow"></script>.  Without that attribute it does nothing.
// Signed out: a small "Sign in" link to /login.html.  Signed in: "<username> · <server>" opening a menu of the pages this account's
// capabilities reach (GET /api/whoami), the Stable / Development server switch (GET/POST /api/route, dev_ui only) and Sign out
// (POST /api/account/logout).  Built with createElement/textContent only, inside a closed Shadow DOM; the server enforces every
// capability, this only decides what to show.
import {loadPersonalPalette} from "./palette-theme.js";

const VERSION = 12;

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
.wrap { position: fixed; top: max(8px, env(safe-area-inset-top)); right: max(8px, env(safe-area-inset-right)); z-index: 2147483000; font: 13px/1.3 system-ui, sans-serif; color: var(--palette-fg, #e8e8ee); display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 6px; max-width: calc(100vw - 32px); padding: 6px; border: var(--palette-nav-border, 1px) solid var(--palette-line, #454556); border-radius: var(--palette-nav-radius, 10px); background: color-mix(in srgb, var(--palette-nav-bg, var(--palette-surface, #14141e)) 93%, transparent); box-shadow: var(--palette-nav-shadow, none); }
.actions { display: contents; }
.chip.toggle { min-height: 32px; box-sizing: border-box; }
.label { font-size: 11px; color: var(--palette-muted, #b6b6ca); margin: 0 4px; }
.chip { all: unset; cursor: pointer; padding: 5px 10px; border-radius: var(--palette-button2-radius, 14px); background: color-mix(in srgb, var(--palette-button2-bg, var(--palette-raised, #1c1c24)) 82%, transparent); border: 1px solid color-mix(in srgb, var(--palette-line, #ffffff) 18%, transparent);
        color: var(--palette-fg, #e8e8ee); font: inherit; text-decoration: none; display: inline-block; }
.chip:hover, .chip:focus-visible { background: color-mix(in srgb, var(--palette-raised, #30303e) 95%, transparent); outline: 2px solid var(--palette-accent, #7aa2ff); outline-offset: 1px; }
.chip.quiet { opacity: .55; } .chip.quiet:hover, .chip.quiet:focus-visible { opacity: 1; }
.chip:disabled, .chip.quiet:disabled { opacity: .35; cursor: default; }
.menu { position: absolute; top: calc(100% + 6px); right: 0; min-width: 210px; max-width: calc(100vw - 48px); max-height: 65vh; overflow: auto; background: var(--palette-surface, #1c1c24); border: 1px solid color-mix(in srgb, var(--palette-line, #ffffff) 18%, transparent);
        border-radius: 8px; padding: 6px; box-shadow: 0 6px 24px rgba(0,0,0,.5); }
.menu a, .menu button { all: unset; display: block; box-sizing: border-box; width: 100%; padding: 6px 10px; border-radius: 5px; cursor: pointer;
        color: var(--palette-fg, #e8e8ee); font: inherit; }
.menu a:hover, .menu button:hover, .menu a:focus-visible, .menu button:focus-visible { background: var(--palette-raised, #2c2c3a); }
.menu a[aria-current="page"] { color: var(--palette-ink, #9fb8ff); }
.menu button[aria-pressed="true"] { color: var(--palette-ink, #9fb8ff); font-weight: 600; }
.menu button:disabled { opacity: .45; cursor: default; }
.head { padding: 4px 10px 6px; color: var(--palette-muted, #a8a8b8); font-size: 12px; }
.sep { height: 1px; margin: 5px 4px; background: color-mix(in srgb, var(--palette-line, #ffffff) 12%, transparent); }
.suite { position: absolute; right: 0; top: calc(100% + 6px); box-sizing: border-box; width: min(420px, calc(100vw - 32px)); max-height: min(70vh, 560px); overflow: auto; padding: 12px; border: 1px solid var(--palette-line, #454556); border-radius: 10px; background: var(--palette-surface, #14141e); }
.binding { display: grid; gap: 3px; padding: 7px; margin: 4px 0; border: 1px solid var(--palette-line, #454556); border-radius: 4px; overflow-wrap: anywhere; }
.binding span { font-size: 11px; color: var(--palette-muted, #b6b6ca); }
.binding.active { border-color: #ffd166; background: #80551e; }
.radio-bands { display: flex; gap: 4px; align-items: center; min-height: 30px; }
.radio-bands meter { flex: 1; min-width: 0; width: 40px; height: 24px; }
.radio-audio { width: 100%; margin: 8px 0; }
.suite[aria-label="Nodus Radio"] { border-radius: 2px; background: var(--palette-surface, #171b20); border-color: var(--palette-line, #5f747d); font: 12px/1.35 ui-monospace, Consolas, monospace; }
.suite[aria-label="Nodus Radio"] .radio-bands { padding: 4px; border: 1px solid var(--palette-line, #435960); background: var(--palette-bg, #071317); }
.suite[aria-label="Nodus Radio"] .chip { border-radius: 2px; }
.suite[aria-label="Nodus Radio"] input[type="search"], .suite[aria-label="Nodus Radio"] select { box-sizing: border-box; width: 100%; margin: 4px 0; padding: 5px; color: var(--palette-fg, #e8f7f4); background: var(--palette-raised, #101a1e); border: 1px solid var(--palette-line, #5f747d); font: inherit; }
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

// Nodus analyzes uploaded tracks once; the browser reads their stored NRAD bands
// against the HLS media clock. The live relay remains a station diagnostic.
function mountRadio(wrap, host, canControl = false) {
  const chip = el("button", {class: "chip", type: "button", text: "Nodus Radio", "aria-expanded": "false"});
  const panel = el("section", {class: "suite", hidden: "", "aria-label": "Nodus Radio"});
  const close = el("button", {class: "chip", type: "button", text: "Close radio"});
  const title = el("strong", {text: "Server station · Nodus Radio"});
  const status = el("p", {role: "status", text: "Waiting for Nodus"});
  const bands = el("div", {class: "radio-bands", "aria-label": "Spectral power bands"});
  const audio = el("audio", {class: "radio-audio", preload: "none", "aria-label": "Nodus Radio audio"});
  const nativeHls = !!audio.canPlayType("application/vnd.apple.mpegurl");
  const listen = el("button", {class: "chip", type: "button", text: "Listen in this browser"});
  const stopListening = el("button", {class: "chip", type: "button", text: "Stop listening"});
  const audioNote = el("p", {role: "status", text: "Audio starts when you choose Listen."});
  audio.hidden = audioNote.hidden = listen.hidden = stopListening.hidden = true;
  panel.append(close, title, status, bands, listen, stopListening, audioNote);
  // Keep the media element outside the collapsible menu so iOS can keep playing.
  document.body.append(audio);
  const listeningKey = "meowdoku.radio.listening";
  let wantsAudio = false;
  try { wantsAudio = sessionStorage.getItem(listeningKey) === "1"; } catch (_error) {}
  let hlsEngine = null, audioEpoch = 0, audioAvailable = false, connecting = false, audioRetryAt = 0;
  let analysisView = null, analysisSong = null, analysisRequest = 0, analysisRetryAt = 0, audibleRms = null;
  function renderBands(frame) {
    bands.replaceChildren();
    for (const [index, value] of (frame.bands || []).slice(0, 128).entries()) {
      const level = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
      bands.append(el("meter", {min: "0", max: "1", value: String(frame.status === "live" ? level : 0),
        "aria-label": "Band " + (index + 1), title: "Band " + (index + 1) + ": " + level.toFixed(3)}));
    }
  }
  function updateAudioControls() {
    const playing = !audio.paused && !audio.ended;
    chip.textContent = playing ? "Nodus Radio · listening" : wantsAudio ? "Nodus Radio · resume" : "Nodus Radio";
    listen.textContent = wantsAudio ? "Resume listening" : "Listen in this browser";
    listen.hidden = !audioAvailable || playing;
    listen.disabled = connecting;
    stopListening.hidden = !wantsAudio;
    audioNote.hidden = !audioAvailable && !wantsAudio;
  }
  function stopAudio() {
    audioEpoch++;
    connecting = false;
    audioRetryAt = Date.now() + 5000;
    audio.pause();
    if (hlsEngine) { hlsEngine.destroy(); hlsEngine = null; }
    audio.removeAttribute("src"); audio.load();
    audio.hidden = true;
    audibleRms = null;
    bands.replaceChildren();
    updateAudioControls();
  }
  async function tryPlay(epoch) {
    try {
      await audio.play();
    } catch (error) {
      if (epoch !== audioEpoch) return;
      audioNote.textContent = error?.name === "NotAllowedError"
        ? "Tap Resume listening to let this browser play Radio."
        : "Could not start radio: " + String(error.message || error);
      updateAudioControls();
    }
  }
  async function startAudio() {
    if (!wantsAudio || !audioAvailable || connecting) return;
    const epoch = ++audioEpoch;
    if (audio.getAttribute("src") || hlsEngine) { await tryPlay(epoch); return; }
    connecting = true;
    audioNote.textContent = "Connecting to the station…";
    updateAudioControls();
    try {
      if (nativeHls) {
        audio.src = "/api/radio/hls/radio.m3u8";
        audio.hidden = false;
        connecting = false;
        updateAudioControls();
        await tryPlay(epoch);
      } else {
        const {default: Hls} = await import("./radio-hls.min.js");
        if (epoch !== audioEpoch || !audioAvailable) return;
        if (!Hls.isSupported()) throw new Error("This browser cannot decode the radio stream");
        hlsEngine = new Hls({lowLatencyMode: true});
        hlsEngine.on(Hls.Events.MEDIA_ATTACHED, () => { if (epoch === audioEpoch) hlsEngine?.loadSource("/api/radio/hls/radio.m3u8"); });
        hlsEngine.on(Hls.Events.MANIFEST_PARSED, () => {
          if (epoch === audioEpoch) {
            connecting = false;
            updateAudioControls();
            tryPlay(epoch);
          }
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
    } catch (error) {
      if (epoch === audioEpoch) {
        stopAudio();
        audioNote.textContent = "Could not start radio: " + String(error.message || error);
      }
    }
  }
  listen.addEventListener("click", () => {
    if (!audioAvailable) return;
    wantsAudio = true;
    try { sessionStorage.setItem(listeningKey, "1"); } catch (_error) {}
    updateAudioControls();
    startAudio();
  });
  stopListening.addEventListener("click", () => {
    wantsAudio = false;
    try { sessionStorage.removeItem(listeningKey); } catch (_error) {}
    stopAudio();
    audioNote.textContent = "Audio starts when you choose Listen.";
  });
  audio.addEventListener("playing", () => {
    connecting = false;
    audioNote.textContent = "Listening here. Server speakers are separate.";
    updateAudioControls();
  });
  audio.addEventListener("pause", updateAudioControls);
  audio.addEventListener("ended", updateAudioControls);
  function showAudibleFrame() {
    if (audio.paused || !Number.isFinite(audio.currentTime) || !analysisView) return;
    // NRAD frames use the source's 48 kHz clock, which also starts HLS PTS.
    const view = analysisView, count = view.byteLength / 48;
    const target = audio.currentTime * 48000;
    let lo = 0, hi = count;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (Number(view.getBigUint64(mid * 48 + 16, true)) <= target) lo = mid + 1;
      else hi = mid;
    }
    if (!lo) return;
    const at = (lo - 1) * 48, frameNumber = Number(view.getBigUint64(at + 16, true));
    if (target - frameNumber > 12000) return;
    const frame = {status: "live", sample_rate: 48000, frame: String(frameNumber),
      rms: view.getFloat32(at + 24, true), bands: Array.from({length: 5}, (_, n) => view.getFloat32(at + 28 + 4 * n, true))};
    audibleRms = frame.rms;
    renderBands(frame);
    window.dispatchEvent(new CustomEvent("nodus-radio-audible-frame", {detail: frame}));
  }
  async function loadLevels(songId) {
    if (songId === analysisSong && (analysisView || Date.now() < analysisRetryAt)) return;
    analysisSong = songId; analysisView = null;
    analysisRetryAt = Date.now() + 5000;
    audibleRms = null;
    bands.replaceChildren();
    const requestId = ++analysisRequest;
    if (!songId?.startsWith("upload:") && !songId?.startsWith("suno:")) return;
    try {
      const response = await fetch("/api/radio/levels?id=" + encodeURIComponent(songId),
        {credentials: "same-origin", cache: "force-cache"});
      if (!response.ok) return;
      if (response.headers.get("X-Radio-Track-Id") !== songId ||
          response.headers.get("X-Radio-Analysis-Version") !== "1" ||
          response.headers.get("X-Radio-Frame-Hop") !== "256" ||
          response.headers.get("X-Radio-Band-Count") !== "5" ||
          response.headers.get("X-Radio-Sample-Rate") !== "48000") return;
      const buffer = await response.arrayBuffer();
      if (requestId !== analysisRequest || !buffer.byteLength || buffer.byteLength > 16 * 1024 * 1024 || buffer.byteLength % 48) return;
      const view = new DataView(buffer);
      if (view.getUint32(0, true) !== 0x4441524e || view.getUint32(4, true) !== 1 ||
          view.getUint32(8, true) !== 5 || view.getUint32(12, true) !== 48000) return;
      analysisView = view;
      showAudibleFrame();
    } catch (_error) { /* The station still plays if stored levels are unavailable. */ }
  }
  let meterFrame = 0;
  function tickMeters() {
    showAudibleFrame();
    if (!audio.paused && !audio.ended) meterFrame = requestAnimationFrame(tickMeters);
    else meterFrame = 0;
  }
  audio.addEventListener("playing", () => { if (!meterFrame) meterFrame = requestAnimationFrame(tickMeters); });
  audio.addEventListener("seeked", showAudibleFrame);
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
  const uploadFile = el("input", {type: "file", accept: ".flac,audio/flac", "aria-label": "FLAC master to add to the jukebox"});
  const uploadTitle = el("input", {type: "text", maxlength: "200", placeholder: "Track title", "aria-label": "Jukebox track title"});
  const uploadButton = el("button", {class: "chip", type: "button", text: "Upload FLAC to jukebox"});
  const uploadStatus = el("p", {role: "status", text: "Choose a FLAC master to offer to the shared jukebox. Upload requires the server Radio codec to be configured."});
  uploadFile.addEventListener("change", () => {
    if (uploadFile.files?.[0] && !uploadTitle.value.trim()) uploadTitle.value = uploadFile.files[0].name.replace(/\.flac$/i, "");
  });
  uploadButton.addEventListener("click", async () => {
    const file = uploadFile.files?.[0], name = uploadTitle.value.trim();
    if (!file || !/\.flac$/i.test(file.name) || !name || name.length > 200) {
      uploadStatus.textContent = "Choose a FLAC file and enter a track title.";
      return;
    }
    if (file.size < 42 || file.size > 256 * 1024 * 1024) {
      uploadStatus.textContent = "FLAC must be between 42 bytes and 256 MiB.";
      return;
    }
    uploadButton.disabled = true;
    uploadStatus.textContent = "Uploading " + file.name + "…";
    try {
      const response = await fetch("/api/radio/uploads?title=" + encodeURIComponent(name), {
        method: "POST", credentials: "same-origin", cache: "no-store",
        headers: {"Content-Type": "audio/flac", "X-Meowdoku-Radio": "1"}, body: file
      });
      const isJson = response.headers.get("content-type")?.includes("application/json");
      const data = isJson ? await response.json() : {};
      if (!response.ok) throw new Error(data.error || "Radio upload failed (HTTP " + response.status + ")");
      uploadStatus.textContent = "Added " + data.title + " to the jukebox · SHA-256 " + data.sha256;
      uploadFile.value = "";
      const latest = await getJson("/api/radio/jukebox");
      if (!panel.hidden) showJukebox(latest);
    } catch (error) {
      uploadStatus.textContent = String(error.message || error);
    } finally {
      uploadButton.disabled = false;
    }
  });
  panel.append(el("strong", {text: "Add a lossless track"}), uploadFile, uploadTitle, uploadButton, uploadStatus);
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
  // [RADIO] Suno playlists onto the jukebox (suno_playlists.py). Shown only to an account with a Suno connection, its
  // own or one an owner shares with it; each song is saved as MP3 and requested in the playlist's order.
  const sunoBox = el("div", {"aria-label": "Suno playlists", hidden: ""});
  const sunoLists = el("select", {"aria-label": "Suno playlist"});
  const sunoMore = el("button", {class: "chip", type: "button", text: "More playlists", hidden: ""});
  const sunoQueue = el("button", {class: "chip", type: "button", text: "Put playlist on the jukebox"});
  const sunoCredit = el("button", {class: "chip", type: "button", hidden: ""});
  const sunoStatus = el("p", {role: "status", text: ""});
  sunoBox.append(el("strong", {text: "Suno playlists"}), sunoLists, sunoMore, sunoQueue, sunoCredit, sunoStatus);
  panel.append(sunoBox);
  let sunoCursor = null, sunoBusy = false, sunoTimer = null, sunoLoaded = false, sunoRunning = false;
  async function loadPlaylists(more) {
    if (sunoBusy) return;
    sunoBusy = true; sunoMore.disabled = sunoQueue.disabled = true;
    try {
      const data = await getJson("/api/radio/suno/playlists" + (more && sunoCursor ? "?cursor=" + encodeURIComponent(sunoCursor) : ""));
      if (!more) sunoLists.replaceChildren();
      for (const p of data.playlists || [])
        sunoLists.append(el("option", {value: p.id, text: p.name + (p.songs != null ? " · " + p.songs + " songs" : "")}));
      sunoCursor = data.next_cursor || null;
      sunoMore.hidden = !sunoCursor;
      sunoBox.hidden = false; sunoLoaded = true;
      if (!more && !sunoRunning) sunoStatus.textContent = "Playlists of " + data.owner + "'s Suno" +
        (data.shared ? " (shared with you)" : "") + ". Songs are saved as MP3 and requested in order.";
    } catch (_error) {
      if (!more) sunoBox.hidden = true;            // no Suno connection for this account: the section stays hidden
    } finally {
      sunoBusy = false; sunoMore.disabled = false; sunoQueue.disabled = sunoRunning || !sunoLists.value;
    }
  }
  function showSunoJob(job) {
    if (!job) return;
    sunoRunning = job.state === "running";
    const counts = {};
    for (const song of job.songs || []) {
      const kind = String(song.outcome).split(":")[0];
      counts[kind] = (counts[kind] || 0) + 1;
    }
    const summary = Object.entries(counts).map(([kind, n]) => n + " " + kind).join(", ");
    const credit = job.needs_credit ? " " + job.needs_credit + " need a Suno download credit" +
      (job.credits_remaining != null ? " (" + job.credits_remaining + " credits left)" : "") + "." : "";
    sunoStatus.textContent = sunoRunning
      ? "Putting the playlist on the jukebox: " + job.done + " of " + (job.total ?? "?") + " songs" + (summary ? " (" + summary + ")" : "")
      : job.state === "failed" ? "The playlist could not be put on the jukebox: " + job.error
      : "Done: " + (summary || "no songs") + "." + credit;
    sunoCredit.hidden = !(job.state === "done" && job.needs_credit > 0 && !job.confirm_credit);
    sunoCredit.textContent = "Spend " + job.needs_credit + " Suno download credit(s) and add those songs";
    sunoQueue.disabled = sunoRunning || sunoBusy || !sunoLists.value;
  }
  async function pollSunoJob(epoch) {
    try {
      const data = await getJson("/api/radio/suno/playlist-job");
      if (epoch !== generation || panel.hidden) return;
      showSunoJob(data.job);
      if (data.job?.state === "running") sunoTimer = setTimeout(() => pollSunoJob(epoch), 2000);
    } catch (_error) { /* not signed in, or no job: nothing to show */ }
  }
  async function queuePlaylist(confirmCredit) {
    const id = sunoLists.value;
    if (!id || sunoBusy || sunoRunning) return;
    sunoBusy = true; sunoQueue.disabled = true; sunoCredit.hidden = true;
    try {
      const response = await fetch("/api/radio/suno/playlists/" + encodeURIComponent(id) + "/queue", {method: "POST",
        credentials: "same-origin", cache: "no-store", headers: {"Content-Type": "application/json", "X-Meowdoku-Radio": "1"},
        body: JSON.stringify({confirm_credit: confirmCredit})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "HTTP " + response.status);
      sunoBusy = false;
      showSunoJob(data.job);
      clearTimeout(sunoTimer); pollSunoJob(generation);
    } catch (error) {
      sunoStatus.textContent = error.message;
    } finally {
      sunoBusy = false; sunoQueue.disabled = sunoRunning || !sunoLists.value;
    }
  }
  sunoMore.addEventListener("click", () => loadPlaylists(true));
  sunoQueue.addEventListener("click", () => queuePlaylist(false));
  sunoCredit.addEventListener("click", () => queuePlaylist(true));
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
      if (epoch !== generation || (panel.hidden && !wantsAudio)) return;
      window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: data}));
      title.textContent = data.title || "Nodus Radio";
      const available = data.audio === true;
      audioAvailable = available;
      loadLevels(data.song_id || null);
      if (wantsAudio && available && audio.paused && !connecting &&
          !audio.getAttribute("src") && !hlsEngine && Date.now() >= audioRetryAt) startAudio();
      if (!available && (audio.getAttribute("src") || hlsEngine)) stopAudio();
      updateAudioControls();
      status.textContent = !audio.paused && audibleRms !== null ? "Listening · RMS " + audibleRms.toFixed(3)
        : !audio.paused && !analysisView ? "Listening · stored levels unavailable"
        : data.status === "live" ? "Live · RMS " + Number(data.rms).toFixed(3)
        : data.status === "stale" ? "Signal stopped · last packet " + Number(data.age_s).toFixed(1) + "s ago"
        : data.status === "waiting" ? "Waiting for Nodus packets" : "Radio is not broadcasting";
      if (!analysisView) renderBands({bands: [], status: "waiting"});
      else showAudibleFrame();
    } catch (error) {
      if (epoch !== generation || (panel.hidden && !wantsAudio)) return;
      window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: {status: "unavailable"}}));
      status.textContent = "Radio unavailable";
      bands.replaceChildren();
    }
    if (epoch === generation && (!panel.hidden || wantsAudio)) timer = setTimeout(() => poll(epoch), panel.hidden ? 1000 : 100);
  }
  function setOpen(open) {
    if (open === !panel.hidden) return;
    if (open && dismissHeaderPanel && dismissHeaderPanel !== dismiss) dismissHeaderPanel();
    dismissHeaderPanel = open ? dismiss : (dismissHeaderPanel === dismiss ? null : dismissHeaderPanel);
    generation++; clearTimeout(timer); timer = null; clearTimeout(paTimer); paTimer = null;
    clearTimeout(sunoTimer); sunoTimer = null;
    if (!open && audio.paused) window.dispatchEvent(new CustomEvent("nodus-radio-signal", {detail: {status: "unavailable"}}));
    panel.hidden = !open; chip.setAttribute("aria-expanded", String(open));
    if (open) {poll(generation); pollPA(generation); if (!sunoLoaded) loadPlaylists(false); pollSunoJob(generation);}
    else if (wantsAudio) poll(generation);
  }
  chip.addEventListener("click", () => setOpen(panel.hidden));
  close.addEventListener("click", () => {setOpen(false); chip.focus();});
  panel.addEventListener("keydown", event => {
    if (event.key === "Escape") {event.stopPropagation(); setOpen(false); chip.focus();}
  });
  document.addEventListener("pointerdown", event => {if (!event.composedPath().includes(host)) setOpen(false);});
  window.addEventListener("pagehide", () => setOpen(false));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) setOpen(false);
    else if (wantsAudio && audioAvailable && audio.paused) startAudio();
  });
  window.addEventListener("pageshow", event => {
    if (event.persisted && wantsAudio && audioAvailable && audio.paused) startAudio();
  });
  wrap.append(chip, panel);
  updateAudioControls();
  if (wantsAudio) poll(generation);
}

// [NOTICE] The site's header notice (site_condition.py; GET /api/condition/notice, public). mountNotice is exported so a page
// can block one element of its own the same way:
//   mountNotice(element, {title, body, mode: "overlay", block: true})  ->  {remove()}
// mode "push" takes the space it needs at the top of the target; "overlay" sits on top of it (as tall as it needs, up to the
// target's height, scrolling inside). With block, the target's other content is made inert (no click, focus or typing);
// for the whole page the site header and its menu are never blocked. Text only: nothing in a notice is read as HTML.
const NOTICE_CSS = `
:host { all: initial; }
.notice { box-sizing: border-box; font: 15px/1.45 system-ui, sans-serif; color: var(--palette-fg, #f3f6f8); background: var(--palette-surface, #13202a);
  border-bottom: 3px solid #e0a526; padding: 14px 18px; max-height: 100%; overflow: auto; }
.notice.overlay { box-shadow: 0 8px 28px rgba(0,0,0,.45); }
.notice.block { height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;
  background: color-mix(in srgb, var(--palette-bg, #0a1016) 94%, transparent); border-bottom: none; }
.notice.block > * { max-width: 72ch; }
h2 { margin: 0 0 6px; font-size: 18px; color: #ffd26a; }
p { margin: 6px 0; white-space: pre-wrap; }
.eta { font-weight: 700; color: #ffd26a; }
.countdown { font: 700 22px/1.3 ui-monospace, monospace; color: #ffd26a; margin: 8px 0; }
.countdown small { display: block; font: 600 12px/1.3 system-ui, sans-serif; opacity: .8; }
.posted { font-size: 12px; opacity: .7; }
button { font: inherit; margin-top: 8px; padding: 5px 12px; border-radius: 6px; border: 1px solid var(--palette-line, #5f747d); background: var(--palette-raised, #1d2c35);
  color: inherit; cursor: pointer; }
`;

export function mountNotice(target, notice, { top = 0, keep = [] } = {}) {
  const host = document.createElement("div");
  host.setAttribute("data-site-notice", "");
  const shadow = host.attachShadow({ mode: "open" });
  const overlay = notice.mode === "overlay";
  const block = overlay && !!notice.block;
  const box = el("section", { class: "notice" + (overlay ? " overlay" : "") + (block ? " block" : ""),
    role: block ? "alertdialog" : "region", "aria-label": notice.title || "Notice" });
  if (notice.title) box.append(el("h2", { text: notice.title }));
  for (const para of String(notice.body || "").split(/\n\s*\n/)) if (para.trim()) box.append(el("p", { text: para.trim() }));
  if (notice.eta) box.append(el("p", { class: "eta", text: notice.eta }));
  let timer = 0;
  if (Number.isFinite(Number(notice.until)) && Number(notice.until) > 0) {   // a live countdown to notice.until (epoch s)
    const end = Number(notice.until) * 1000;
    const label = el("small", { text: notice.until_label || "Time remaining" });
    const clock = el("span", { role: "timer" });
    box.append(el("div", { class: "countdown" }, label, clock));
    const tick = () => {
      let left = Math.max(0, Math.floor((end - Date.now()) / 1000));
      if (left === 0) { clock.textContent = "Now"; clearInterval(timer); return; }
      const d = Math.floor(left / 86400); left %= 86400;
      const h = Math.floor(left / 3600); left %= 3600;
      const m = Math.floor(left / 60), s = left % 60;
      clock.textContent = (d ? d + "d " : "") + String(h).padStart(2, "0") + "h " + String(m).padStart(2, "0") + "m " +
        String(s).padStart(2, "0") + "s";
    };
    tick();
    timer = setInterval(tick, 1000);
    clock.title = "until " + new Date(end).toLocaleString();
  }
  const blocked = [];
  const remove = () => { clearInterval(timer); host.remove(); for (const node of blocked) node.inert = false; };
  if (notice.dismissible && !block) {
    box.append(el("button", { type: "button", text: "Dismiss", onclick: () => { remove(); if (notice.onDismiss) notice.onDismiss(); } }));
  }
  shadow.append(el("style", { text: NOTICE_CSS }), box);
  const whole = target === document.body;
  if (!overlay) {
    host.style.cssText = "display:block;position:relative;z-index:1;";
    if (whole && keep[0] && keep[0].parentNode === target) keep[0].after(host); else target.prepend(host);
    return { remove };
  }
  if (whole) {
    host.style.cssText = `position:fixed;left:0;right:0;top:${top}px;z-index:2147482000;` +
      (block ? "bottom:0;" : `max-height:calc(100vh - ${top}px);`);
    target.append(host);
  } else {
    if (getComputedStyle(target).position === "static") target.style.position = "relative";
    host.style.cssText = "position:absolute;left:0;right:0;top:0;z-index:2147482000;" + (block ? "bottom:0;" : "max-height:100%;");
    target.append(host);
  }
  if (block) {
    for (const child of Array.from(target.children)) {
      if (child !== host && !keep.includes(child)) { child.inert = true; blocked.push(child); }
    }
  }
  return { remove };
}

async function showSiteNotice(header, headerBottom) {
  let notice;
  try { notice = (await getJson("/api/condition/notice")).notice; } catch (e) { return; }
  if (!notice) return;
  const paths = Array.isArray(notice.paths) ? notice.paths : [];
  if (paths.length && !paths.some(prefix => location.pathname.startsWith(prefix))) return;
  const key = "meowdoku_notice_dismissed";
  let dismissed = null;
  try { dismissed = sessionStorage.getItem(key); } catch (e) { dismissed = null; }
  if (notice.dismissible && dismissed === String(notice.version)) return;
  notice.onDismiss = () => { try { sessionStorage.setItem(key, String(notice.version)); } catch (e) { /* storage off */ } };
  mountNotice(document.body, notice, { top: headerBottom(), keep: [header] });
}

async function main() {
  if (!allowed()) { console.info("[site-nav] not loaded: this page's script tag lacks data-site-nav=\"allow\""); return; }
  loadPersonalPalette();      // the participant's color game palette, on every page that carries the site controls

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
  showSiteNotice(host, () => Math.ceil(wrap.getBoundingClientRect().bottom));   // [NOTICE] for everyone, signed in or not
  let who;
  try { who = await getJson("/api/whoami"); } catch (e) { return; }
  const caps = new Set(who.capabilities || []);
  // Sign in from any page returns to that page: login.js reads ?next= and accepts only a same-site path.
  const signInHref = () => location.pathname === "/login.html" ? "/login.html"
    : "/login.html?next=" + encodeURIComponent(location.pathname + location.search + location.hash);

  if (!who.signed_in) {
    try { sessionStorage.removeItem("meowdoku.radio.listening"); } catch (_error) {}
    actions.append(el("a", { class: "chip", href: "/palette.html", text: "Color game" }),
      el("a", { class: "chip" + (who.shader ? " quiet" : ""), href: signInHref(), text: "Sign in" }),
      el("button", { class: "chip quiet", type: "button", disabled: "", text: "Register", title: "General registration is not open" }),
      el("a", { class: "chip quiet", href: "/register.html#alpha-form", text: "Alpha user request" }));
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
    ticketsMenu.append(el("a", { href: "/research.html", role: "menuitem", text: "Research" }));
    ticketsMenu.append(el("a", { href: "/development.html", role: "menuitem", text: "Development" }));
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
  // [LOAN] the dev Apache sends a request carrying meowdoku_loan=<port> to that branch server (deploy/apache/dev-proxy.conf);
  // choosing Stable or Development forgets it.
  const loanCookie = (document.cookie.match(/(?:^|;\s*)meowdoku_loan=(\d+)/) || [])[1] || "";
  function setLoan(port) {
    document.cookie = "meowdoku_loan=" + (port || "") + "; path=/; Secure; SameSite=Lax" + (port ? "" : "; Max-Age=0");
  }
  async function choose(site) {
    setLoan("");
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
  if (who.role === "admin" || who.role === "operator") {
    // [CONDITION] The yellow and red card switch. Its code is not in this public file: the server answers the module only to
    // an operator or an admin and checks the role again on every switch (site_condition.py).
    const slot = el("div", { class: "condition" });
    menu.append(slot, el("div", { class: "sep" }));
    import("/api/condition/controls.js").then(m => m.mount(slot)).catch(() => slot.remove());
  }
  if (caps.has("message")) menu.append(el("button", { role: "menuitem", type: "button", text: "Messages", onclick: () => {
    setOpen(false);
    if (window.MeowdokuMessenger) window.MeowdokuMessenger.open();
    else location.assign("/login.html?messages=1");
  } }));
  for (const item of [
    link("Home", "/"),
    link("Color game", "/palette.html"),
    link("Newsletter", "/newsletter.html"),
    link("Games", "/games.html", !onDev || caps.has("dev_ui")),
    link("Profile and servers", "/profile.html"),
    link("Keys", "/keys.html", caps.has("message")),
    link("Connected services", "/connected-services.html"),
    link("My files", "/files.html", caps.has("message")),
    link("Paper markets", "/markets.html", caps.has("message")),
    link("Browse the site", "/browse", caps.has("dev_ui")),                        // /browse answers 403 without dev_ui (server.py _browse_get)
    link("Help: the manual", "/manual/", caps.has("view_site")),
    link("Repository world map", "/worldmap.html", caps.has("dev_ui")),
    link("Math lab", "/mathlab.html", caps.has("compile_submit")),
    link("Source divider", "/divider.html", caps.has("compile_submit")),
    link("Marble studio", "/marble_studio.html", caps.has("dev_ui")),
    link("T4 wave arena", "/wave_arena.html", caps.has("dev_ui")),
    link("Creature studio", "/mechanical_creature_studio.html", caps.has("dev_ui")),
    link("People (accounts and groups)", "/people.html", caps.has("account_manage")),
    link("Operator control", "/operator-control.html", caps.has("deploy_scripts")),
    link("Agent workspace", "/agent-workspace.html", caps.has("operator_workspace")),
    link("Agent controls", "/agent-controls.html", caps.has("operator_workspace")),
    link("Agent training arena", "/agent-training-arena.html", caps.has("operator_workspace")),
    link("Rooms", "/rooms.html"),
    link("Account and password", "/login.html"),
    link("Invite someone", "/invite.html"),                                         // [INVITE] invitations.py: a link and a QR code, once
  ]) if (item) menu.append(item);
  if (pageRouteControl) {
    menu.append(el("div", { class: "sep" }), el("div", { class: "head", text: "Server: " + (who.instance || serverName) }),
      serverButton("Stable server", "stable", !onDev, false),
      serverButton("Development server", "dev", onDev && !loanCookie, route.dev_available === false));
    if (onDev && caps.has("dev_ui")) {                 // [LOAN] every ready branch server, hosted by the dev site
      let loans = [];
      try { loans = ((await getJson("/api/loans")).loans || []).filter(loan => loan.state === "ready"); } catch (e) { loans = []; }
      for (const loan of loans) {
        const b = el("button", { role: "menuitemradio", "aria-pressed": String(String(loan.port) === loanCookie),
          text: "Branch " + loan.branch + " (" + loan.repository + ", " + (loan.username || "?") + ")",
          onclick: () => { setLoan(String(loan.port)); location.reload(); } });
        menu.append(b);
      }
    }
  }
  menu.append(note, el("div", { class: "sep" }),
    el("button", { role: "menuitem", text: "Sign out", onclick: async () => {
      const result = await postJson("/api/account/logout", {});
      if (result.status === 200) try { sessionStorage.removeItem("meowdoku.radio.listening"); } catch (_error) {}
      location.reload();
    } }));

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

window.MeowdokuSiteNav = { version: VERSION, registerControlSuite, mountNotice };
main();
