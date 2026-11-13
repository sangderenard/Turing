// The site console: a text adventure over the site's own walking verbs, for dev_ui accounts.  HAND-AUTHORED.
// Places come from maps the server already has; walking is the Living Data cursor (/api/living-data/sessions: one per account,
// dev_ui).  Two maps:
//   pictures   your photo jobs (GET /api/marble_photo/jobs), one place each; "develop" re-develops the one you stand on from its
//              saved linear exposure through POST /api/marble_photo/retouch, "show" puts the picture on screen.
//   repo       a stored repository world map (GET /api/worldmap/index, /api/worldmap/map; files are rooms, folders hold them,
//              imports are passages).  A whole repository can exceed the cursor's 4096 places, so the console hands the cursor the
//              neighbourhood of where you stand (the folder, its parent chain and subfolders, and the files a file links with) and
//              re-centres it after every step.  "examine" reads a file through the core's Git (/api/git/file, core_git).
// The server enforces every capability; this page only stops a non-dev account early.  textContent only, no innerHTML.
const byId = id => document.getElementById(id);
const out = byId("out");
const SESSION_KEY = "meowdoku.console.session";
const MAX_PLACES = 4000;
// Repository world maps are named owner/name; the core's Git names its repositories (deploy/branch_policy.json) differently.
const GIT_NAMES = { "sangderenard/Turing": "turing", "sangderenard/nodus": "nodus", "sangderenard/speaktome": "speaktome",
                    "sangderenard/spectral-analyzer": "spectral-analyzer", "sangderenard/nogodsnomasters": "workspace" };
const DEVELOP = {   // the retouch endpoint's own settings and ranges (marble_photo.retouch / _display_params)
  spectral_look: ["visible_reference", "compressed", "nir_red", "uv_violet"], exposure: [-6, 6], red: [0, 3], green: [0, 3],
  blue: [0, 3], whitebalance: [-3, 3], tint: [-3, 3], normalization: ["auto99", "percentile", "manual", "specular"],
  normalization_percentile: [1, 100], normalization_white: [1e-6, 1e12], black_point: [0, 0.5],
  tone_curve: ["exponential", "linear"], display_look: ["reference", "color_priority"], saturation: [0, 2], vibrance: [-1, 1],
  gamma: [0.2, 3], intensity: [0, 3], firefly_filter: ["true", "false"], firefly_threshold: [2, 64], firefly_strength: [0, 1],
};

let who = null, caps = new Set();
let cursor = null;          // the last cursor receipt
let place = null;           // {map: "pictures"|"repo", ...}
let developed = {};         // settings kept per picture between develop commands

function print(text, cls = "say") {
  const line = document.createElement("div"); line.className = cls; line.textContent = text; out.append(line);
  out.scrollTop = out.scrollHeight;
}
function picture(src, alt) {
  const img = document.createElement("img"); img.src = src; img.alt = alt; img.onload = () => { out.scrollTop = out.scrollHeight; };
  out.append(img);
}
function remember(id) { try { id ? localStorage.setItem(SESSION_KEY, id) : localStorage.removeItem(SESSION_KEY); } catch (e) { /* private window */ } }
function stored() { try { return localStorage.getItem(SESSION_KEY); } catch (e) { return null; } }
function rid() { return "console-" + crypto.getRandomValues(new Uint32Array(4)).join("-"); }

async function call(path, body) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store",
    ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }) });
  let data = {};
  try { data = await response.json(); } catch (e) { /* no body */ }
  if (!response.ok) { const error = new Error(data.error || `HTTP ${response.status}`); error.status = response.status; throw error; }
  return data;
}

// ---- the cursor ------------------------------------------------------------------------------------------------------------------
async function closeStored() {
  const id = stored();
  if (!id) return;
  try { const now = await call(`/api/living-data/sessions/${id}`); await call(`/api/living-data/sessions/${id}/close`, { expected_revision: now.revision }); }
  catch (error) { if (error.status !== 404) throw error; }
  remember(null);
}
async function stand(map, at) {
  await closeStored();
  try { cursor = await call("/api/living-data/sessions", { map, at, request_id: rid() }); }
  catch (error) {
    if (error.status === 409) throw new Error(error.message + ". Another client holds this account's cursor (it lapses after an hour unused).");
    throw error;
  }
  remember(cursor.session_id);
  return cursor;
}
function describe() {
  const at = cursor.at || {};
  print(`You are at ${at.name || at.identity}${at.kind ? " (" + at.kind + ")" : ""}.`);
  cursor.exits.forEach((e, i) => print(`  ${i + 1}. ${e.direction === "outgoing" ? "→" : "←"} ${e.relationship}: ${label(e.to)}`, "exit"));
  if (cursor.exit_total > cursor.exits.length) print(`  … ${cursor.exit_total - cursor.exits.length} more exits not listed.`, "note");
  if (!cursor.exits.length) print("  No exits.", "note");
}
function label(identity) {
  if (place && place.names && place.names.has(identity)) return place.names.get(identity);
  return identity;
}
function exitFrom(word) {
  const n = Number(word);
  if (Number.isInteger(n) && n >= 1 && n <= cursor.exits.length) return cursor.exits[n - 1];
  const lowered = word.toLowerCase();
  const found = cursor.exits.filter(e => label(e.to).toLowerCase() === lowered || e.to.toLowerCase() === lowered
    || label(e.to).toLowerCase().endsWith("/" + lowered));
  return found.length === 1 ? found[0] : null;
}

// ---- pictures --------------------------------------------------------------------------------------------------------------------
async function enterPictures(prefix) {
  const listing = await call("/api/marble_photo/jobs");
  const entries = (listing.entries || []).filter(e => e.kind === "photo" || e.kind === undefined);
  const root = "pictures:" + (who.username || "mine");
  const names = new Map([[root, "your pictures"]]);
  const regions = [], tracks = [];
  for (const entry of entries) {
    const identity = "picture:" + entry.digest;
    const name = `${entry.digest.slice(0, 12)} · ${entry.state}${entry.created_at ? " · " + new Date(entry.created_at * 1000).toLocaleString() : ""}`;
    names.set(identity, name);
    regions.push({ identity, kind: "picture", name, buildings: [] });
    tracks.push({ source: identity, target: root, relationship: "contained-by" });
  }
  const at = prefix ? [...names.keys()].find(k => k.startsWith("picture:" + prefix)) : root;
  if (!at) throw new Error(`no picture of yours starts with ${prefix}`);
  place = { map: "pictures", names, entries: new Map(entries.map(e => ["picture:" + e.digest, e])) };
  await stand({ schema: "abstract-ui-div-map-v0", identity: root, kind: "pictures", name: "your pictures", regions, tracks }, at);
  print(`${entries.length} pictures.`, "note");
  describe();
}
function digestHere() {
  const at = cursor && cursor.at && cursor.at.identity;
  if (!place || place.map !== "pictures" || !at || !at.startsWith("picture:")) throw new Error("stand on a picture first (pictures, then go to one)");
  return at.slice("picture:".length);
}
async function showPicture(digest) {
  const status = await call(`/api/marble_photo?digest=${encodeURIComponent(digest)}`);
  const name = status.retouched_image || status.image;
  if (!name) { print(`The picture is ${status.state}; it has no image yet.`, "note"); return status; }
  print(`${status.state} · ${name}${status.retouch_params ? " · developed " + JSON.stringify(status.retouch_params) : ""}`, "note");
  picture(`/api/marble_photo/image?digest=${encodeURIComponent(digest)}&name=${encodeURIComponent(name)}&t=${Date.now()}`, "picture " + digest.slice(0, 12));
  return status;
}
function parseSettings(words) {
  const settings = {};
  for (const word of words) {
    const [key, raw] = word.split("=");
    if (!(key in DEVELOP) || raw === undefined) throw new Error(`unknown setting ${word}; type settings`);
    const range = DEVELOP[key];
    if (typeof range[0] === "string") {
      if (!range.includes(raw)) throw new Error(`${key} is one of ${range.join(", ")}`);
      settings[key] = key === "firefly_filter" ? raw === "true" : raw;
    } else {
      const value = Number(raw);
      if (!Number.isFinite(value) || value < range[0] || value > range[1]) throw new Error(`${key} is a number from ${range[0]} to ${range[1]}`);
      settings[key] = value;
    }
  }
  return settings;
}
async function develop(words) {
  const digest = digestHere();
  if (words[0] === "reset") { developed[digest] = {}; words = words.slice(1); }
  const settings = { ...(developed[digest] || {}), ...parseSettings(words) };
  print("Developing from the saved linear exposure…", "note");
  await call("/api/marble_photo/retouch", { digest, ...settings });
  developed[digest] = settings;
  await showPicture(digest);
}

// ---- repository world maps -------------------------------------------------------------------------------------------------------
async function loadRepo(name) {
  const index = await call("/api/worldmap/index");
  const wanted = (name || "turing").toLowerCase();
  const repo = (index.repos || []).find(r => r.repo.toLowerCase() === wanted || r.repo.split("/")[1].toLowerCase() === wanted);
  if (!repo) throw new Error(`no stored world map for ${name}; there are: ${(index.repos || []).map(r => r.repo).join(", ")}`);
  const map = await call(`/api/worldmap/map?repo=${encodeURIComponent(repo.dir)}&commit=${encodeURIComponent(repo.commit)}`);
  const regionById = new Map(map.regions.map(r => [r.id, r]));
  const fileOf = room => { const dir = regionById.get(room.region).path; return dir ? dir + "/" + room.name : room.name; };
  const roomByPath = new Map(map.rooms.map(r => [fileOf(r), r]));
  const children = new Map();
  for (const region of map.regions) if (region.parent !== null) children.set(region.parent, [...(children.get(region.parent) || []), region]);
  const files = new Map();
  for (const room of map.rooms) files.set(room.region, [...(files.get(room.region) || []), room]);
  const out = new Map(), into = new Map();
  for (const [from, to, , kind] of map.portals) {
    out.set(from, [...(out.get(from) || []), [to, kind]]);
    into.set(to, [...(into.get(to) || []), [from, kind]]);
  }
  return { repo, map, regionById, fileOf, roomByPath, children, files, out, into,
           regionByPath: new Map(map.regions.map(r => [r.path, r])) };
}

// The neighbourhood of one place, as the cursor's own map form: a folder shows its files, its subfolders and its parents; a file
// shows its folder's files and every file it links with.  Identities are dir:<path> and file:<path>.
function neighbourhood(world, identity) {
  const regions = new Map(), tracks = [], names = new Map();
  const add = (id, kind, name) => { if (!regions.has(id) && regions.size < MAX_PLACES) { regions.set(id, { identity: id, kind, name, buildings: [] }); names.set(id, name); } return regions.has(id); };
  const dirId = region => "dir:" + region.path, fileId = room => "file:" + world.fileOf(room);
  const dirName = region => (region.path || world.repo.repo) + "/";
  let region = identity.startsWith("file:") ? world.regionById.get(world.roomByPath.get(identity.slice(5)).region)
                                            : world.regionByPath.get(identity.slice(4));
  // the parent chain
  for (let r = region; r; r = r.parent === null ? null : world.regionById.get(r.parent)) {
    add(dirId(r), "folder", dirName(r));
    if (r.parent !== null && add(dirId(world.regionById.get(r.parent)), "folder", dirName(world.regionById.get(r.parent))))
      tracks.push({ source: dirId(r), target: dirId(world.regionById.get(r.parent)), relationship: "inside" });
  }
  for (const child of world.children.get(region.id) || [])
    if (add(dirId(child), "folder", dirName(child))) tracks.push({ source: dirId(child), target: dirId(region), relationship: "inside" });
  for (const room of world.files.get(region.id) || [])
    if (add(fileId(room), "file", world.fileOf(room))) tracks.push({ source: fileId(room), target: dirId(region), relationship: "inside" });
  if (identity.startsWith("file:")) {
    const room = world.roomByPath.get(identity.slice(5));
    add(identity, "file", world.fileOf(room));
    for (const [to, kind] of world.out.get(room.id) || []) {
      const other = world.map.rooms[to];
      if (add(fileId(other), "file", world.fileOf(other))) tracks.push({ source: identity, target: fileId(other), relationship: world.map.kinds[kind] || "links" });
    }
    for (const [from, kind] of world.into.get(room.id) || []) {
      const other = world.map.rooms[from];
      if (add(fileId(other), "file", world.fileOf(other))) tracks.push({ source: fileId(other), target: identity, relationship: world.map.kinds[kind] || "links" });
    }
  }
  const root = "repo:" + world.repo.repo;
  return { names, model: { schema: "abstract-ui-div-map-v0", identity: root, kind: "repository", name: world.repo.repo,
                           regions: [...regions.values()], tracks }, truncated: regions.size >= MAX_PLACES };
}
async function standInRepo(world, identity) {
  const { names, model, truncated } = neighbourhood(world, identity);
  place = { map: "repo", world, names };
  await stand(model, identity);
  if (truncated) print(`Only the first ${MAX_PLACES} places around here are walkable.`, "note");
  describe();
}
function repoPlace(world, path) {
  const clean = (path || "").replace(/^\/+|\/+$/g, "");
  if (world.roomByPath.has(clean)) return "file:" + clean;
  if (world.regionByPath.has(clean)) return "dir:" + clean;
  return null;
}
async function enterRepo(words) {
  const world = await loadRepo(words[0]);
  // Unasked, you stand on this console's own page among the site's web files, or on its folder in an older map.
  const wanted = words[1] || "examples/meowdoku/web/console.html";
  const at = repoPlace(world, wanted) || (words[1] ? null : repoPlace(world, "examples/meowdoku/web")) || repoPlace(world, "");
  if (!at) throw new Error(`${world.repo.repo} at ${world.repo.commit.slice(0, 12)} has no file or folder ${wanted}`);
  print(`${world.repo.repo} at ${world.repo.commit.slice(0, 12)}: ${world.map.rooms.length} files, ${world.map.portals.length} links.`, "note");
  await standInRepo(world, at);
}
async function examineFile() {
  const at = cursor.at.identity;
  if (!at.startsWith("file:")) { print(`${label(at)} is a folder; go into one of its files to read it.`, "note"); return; }
  const repository = GIT_NAMES[place.world.repo.repo];
  if (!repository || !caps.has("core_git")) { print("Reading files needs core_git and a repository the core's Git knows.", "note"); return; }
  const result = await call("/api/git/file", { repository, ref: place.world.repo.commit, path: at.slice(5) });
  const lines = result.output.split("\n");
  print(lines.slice(0, 80).join("\n"), "say");
  if (lines.length > 80) print(`… ${lines.length - 80} more lines.`, "note");
}

// ---- verbs -----------------------------------------------------------------------------------------------------------------------
const HELP = [
  "pictures [digest-prefix]   stand among your photo jobs, or on one",
  "repo [name] [path]         stand on a file or folder of a repository's world map (default: this page in Turing)",
  "where / look               where you stand and its exits",
  "go <n | name>              take an exit",
  "examine                    read the file, or show the picture, you stand on",
  "show                       put the picture on screen",
  "develop key=value …        re-develop the picture from its linear exposure; develop reset … starts over",
  "settings                   the develop settings and their ranges",
  "leave                      close your cursor",
];
async function run(text) {
  const [verb, ...words] = text.trim().split(/\s+/);
  if (!verb) return;
  print("> " + text.trim(), "cmd");
  try {
    switch (verb.toLowerCase()) {
      case "help": HELP.forEach(h => print(h, "note")); break;
      case "pictures": await enterPictures(words[0]); break;
      case "repo": await enterRepo(words); break;
      case "where": case "look":
        if (!cursor) throw new Error("you are nowhere yet: pictures, or repo");
        cursor = await call(`/api/living-data/sessions/${cursor.session_id}`); describe(); break;
      case "go": {
        if (!cursor) throw new Error("you are nowhere yet: pictures, or repo");
        const exit = exitFrom(words.join(" "));
        if (!exit) throw new Error("no such exit; look lists them");
        cursor = await call(`/api/living-data/sessions/${cursor.session_id}/move`,
          { to: exit.to, relationship: exit.relationship, direction: exit.direction, expected_revision: cursor.revision, request_id: rid() });
        if (place.map === "repo") await standInRepo(place.world, exit.to); else describe();
        break;
      }
      case "examine":
        if (!cursor) throw new Error("you are nowhere yet: pictures, or repo");
        if (place.map === "repo") await examineFile(); else await showPicture(digestHere());
        break;
      case "show": await showPicture(digestHere()); break;
      case "develop": await develop(words); break;
      case "settings":
        for (const [key, range] of Object.entries(DEVELOP))
          print(`  ${key}: ${typeof range[0] === "string" ? range.join(" | ") : range[0] + " to " + range[1]}`, "note");
        break;
      case "leave": await closeStored(); cursor = place = null; print("You left; the cursor is closed.", "note"); break;
      default: print(`I don't know "${verb}". Type help.`, "err");
    }
  } catch (error) { print(error.message || String(error), "err"); }
}

async function main() {
  try { who = await call("/api/whoami"); } catch (error) { print(error.message, "err"); return; }
  caps = new Set(who.capabilities || []);
  if (!who.signed_in) { print("Sign in first (/login.html).", "err"); return; }
  if (!caps.has("dev_ui")) { print("The console is for developer accounts (dev_ui).", "err"); return; }
  byId("form").hidden = false;
  print(`Signed in as ${who.username}. Type help.`, "note");
  const history = [];
  let back = 0;
  byId("form").addEventListener("submit", async event => {
    event.preventDefault();
    const line = byId("line"), text = line.value;
    line.value = ""; history.push(text); back = history.length;
    line.disabled = true;
    try { await run(text); } finally { line.disabled = false; line.focus(); }
  });
  byId("line").addEventListener("keydown", event => {
    if (event.key === "ArrowUp" && back > 0) { back--; byId("line").value = history[back]; event.preventDefault(); }
    if (event.key === "ArrowDown" && back < history.length) { back++; byId("line").value = history[back] || ""; event.preventDefault(); }
  });
  byId("line").focus();
  const start = new URLSearchParams(location.search).get("do");
  if (start) await run(start);
}
main();
