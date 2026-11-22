// Apply only this participant's server-selected color. Every page that loads site-nav.js (with the home feed and the color
// game) wears it: the two accent variables, and the color game's accent rules for links, buttons, focus, selection, form
// controls and a stripe along the top of the page. A page's own class rules still win over these element rules.
// A palette chosen or rated on the color game reaches every open page of the site at once (BroadcastChannel).
const HEX = /^#[0-9a-f]{6}$/i;
const CHANNEL = "speaktome-palette";
const RULES = `
:root { accent-color: var(--palette-accent); }
body { box-shadow: inset 0 4px 0 var(--palette-accent); }
a { color: var(--palette-accent); }
button:hover, button:focus-visible { border-color: var(--palette-accent); outline-color: var(--palette-accent); }
::selection { background: color-mix(in srgb, var(--palette-accent) 45%, transparent); }
`;

export function applyPalette(candidate) {
  const color = candidate?.color;
  if (typeof color !== "string" || !HEX.test(color)) return;
  const root = document.documentElement;
  root.style.setProperty("--palette-accent", color);
  // Derived shades the shared stylesheet (pages.css) reads: a dark tint for backgrounds, a mid edge for borders.
  root.style.setProperty("--palette-tint", `color-mix(in srgb, ${color} 35%, #101722)`);
  root.style.setProperty("--palette-edge", `color-mix(in srgb, ${color} 55%, #384d67)`);
  if (typeof candidate.secondary === "string" && HEX.test(candidate.secondary))
    root.style.setProperty("--palette-secondary", candidate.secondary);
  if (!document.getElementById("palette-theme")) {
    const style = document.createElement("style");
    style.id = "palette-theme";
    style.textContent = RULES;
    document.head.append(style);
  }
}

// Tell the site's other open pages which palette is now active.
export function sharePalette(candidate) {
  if (typeof BroadcastChannel !== "function" || !candidate) return;
  const channel = new BroadcastChannel(CHANNEL);
  channel.postMessage({color: candidate.color, secondary: candidate.secondary});
  channel.close();
}

let listening = false;
export async function loadPersonalPalette() {
  if (!listening && typeof BroadcastChannel === "function") {
    listening = true;
    new BroadcastChannel(CHANNEL).onmessage = (event) => applyPalette(event.data);
  }
  try {
    const response = await fetch("/api/palette", {credentials: "same-origin", cache: "no-store"});
    if (response.ok) applyPalette((await response.json()).candidate);
  } catch (_) { /* The site's default design remains usable offline. */ }
}
