// [ACCESS] Which tier is this page?  The SERVER decides (access.py: the socket peer and the Host header -> lan/dns/ip -> edit/dev/play/shader)
// and enforces it on every call; this module only asks (/api/whoami) so the page can hide or show its development options.  The choice made
// here is cosmetic: a hidden button the server would refuse anyway.  Imported first by loading_panel.js and app.js (top-level await: they
// start once the tier is known); a failed answer means "play" (the normal site, no development options).
let who = { origin: "ip", tier: "play", dev: false, edit: false, shader: false, role: "guest", signed_in: false, capabilities: Object.freeze([]) };
try {
  const r = await fetch("/api/whoami", { cache: "no-store" });
  if (r.ok) { const j = await r.json(); who = { origin: String(j.origin), tier: String(j.tier), dev: !!j.dev, edit: !!j.edit, shader: !!j.shader,
    role: j.signed_in ? String(j.role || "member") : "guest", signed_in: !!j.signed_in,
    capabilities: Object.freeze(Array.isArray(j.capabilities) ? j.capabilities.filter(c => typeof c === "string") : []) }; }
} catch { /* the normal site */ }
window.meowdokuAccess = Object.freeze(who);

if (typeof document !== "undefined") {
  document.documentElement.dataset.tier = who.tier;
  // Owner-only controls (they act on the whole process: the shared learner, the Lichess client, the demo-learning switch).
  // edit: as before.  dev: shown, read-only.  anything else: gone.
  const OWNER = ["demo-learn", "learn-status", "arena-save", "arena-stop", "lichess-stop", "lichess-resume"];
  const ids = OWNER.map((id) => "#" + id);
  const style = document.createElement("style");
  style.textContent = `
    html[data-tier="play"] :is(${ids.join(",")}, #learn-panel), html[data-tier="shader"] :is(${ids.join(",")}, #learn-panel) { display: none !important; }
    html[data-tier="dev"] :is(${ids.join(",")}) { pointer-events: none; opacity: .55; }`;
  document.head.appendChild(style);
  if (!who.edit) {
    const lock = () => { for (const id of OWNER) { const el = document.getElementById(id); if (el) { el.disabled = true; el.title = "owner tool (edit tier)"; } } };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", lock); else lock();
  }
}
export default who;
