// Apply only this participant's server-selected color. Every page that loads site-nav.js (with the home feed and the color
// game) wears it: the two accent variables, and the color game's accent rules for links, buttons, focus, selection, form
// controls and a stripe along the top of the page. A page's own class rules still win over these element rules.
// A palette chosen or rated on the color game reaches every open page of the site at once (BroadcastChannel).
const HEX = /^#[0-9a-f]{6}$/i;
// A design's bevel and shadow are box-shadow lists the server builds from numbers and rgba(); nothing else is worn.
const SHADOW = /^[-0-9a-z.,()% ]{1,240}$/i;
const CHANNEL = "speaktome-palette";
// Every page colors its surfaces and text through these variables, each with the page's own color as the fallback, so
// a page without a palette looks as it did. [name, [percent of the design's color, light base], [percent, dark base]].
// State colors (ok, bad, warn, release and dev badges) are not derived: they keep their meaning in every design.
const SHADES = [
  ["--palette-bg", [6, "#fbfaf7"], [12, "#101318"]],        // the page behind everything
  ["--palette-surface", [4, "#ffffff"], [16, "#191d25"]],   // cards, panels, dialogs
  ["--palette-raised", [12, "#f2f0ea"], [26, "#232935"]],   // controls, inputs, hovered or selected rows
  ["--palette-line", [28, "#d6d3cc"], [42, "#343b47"]],     // borders and dividers
  ["--palette-fg", [8, "#1d1d1f"], [8, "#eef0f4"]],         // body text
  ["--palette-muted", [22, "#66666c"], [22, "#a3a9b3"]],    // secondary text
  ["--palette-ink", [70, "#000000"], [70, "#ffffff"]],      // accent-colored text and links, readable on the surfaces
];

// Text drawn on a fill of the design's color: black on a light color, white on a dark one (WCAG relative luminance).
function onAccent(color) {
  const channel = (offset) => {
    const value = parseInt(color.slice(offset, offset + 2), 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  const luminance = 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
  return luminance > 0.179 ? "#000000" : "#ffffff";
}
const RULES = `
:root { accent-color: var(--palette-accent); }
body { box-shadow: inset 0 4px 0 var(--palette-accent), inset 0 6px 0 var(--palette-secondary, var(--palette-accent)); }
a { color: var(--palette-ink, var(--palette-accent)); }
a:hover { color: var(--palette-secondary, var(--palette-ink, var(--palette-accent))); }
button { box-shadow: var(--palette-bevel, 0 0 0 transparent), var(--palette-shadow, 0 0 0 transparent); }
button:hover, button:focus-visible { border-color: var(--palette-accent); outline-color: var(--palette-accent); }
::selection { background: color-mix(in srgb, var(--palette-accent) 45%, transparent); }
`;

export function applyPalette(candidate) {
  const color = candidate?.color;
  if (typeof color !== "string" || !HEX.test(color)) return;
  const root = document.documentElement;
  root.style.setProperty("--palette-accent", color);
  // The site's shades, all derived from the design's color (SHADES). Each is a light-dark() pair, so a page's own
  // color-scheme picks its branch: a page that is dark only must declare `color-scheme: dark`.
  for (const [name, light, dark] of SHADES) {
    root.style.setProperty(name, `light-dark(color-mix(in srgb, ${color} ${light[0]}%, ${light[1]}), ` +
                                 `color-mix(in srgb, ${color} ${dark[0]}%, ${dark[1]}))`);
  }
  root.style.setProperty("--palette-on-accent", onAccent(color));
  // The secondary color and the design's own bevel and shadow (worn by the site's buttons and cards). A value that is
  // omitted or fails its pattern is removed, so the page falls back to the stylesheet defaults instead of still wearing
  // the previous design's value.
  for (const [name, value, pattern] of [["--palette-secondary", candidate.secondary, HEX],
                                        ["--palette-bevel", candidate.bevel, SHADOW],
                                        ["--palette-shadow", candidate.shadow, SHADOW]]) {
    if (typeof value === "string" && pattern.test(value)) root.style.setProperty(name, value);
    else root.style.removeProperty(name);
  }
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
  channel.postMessage({color: candidate.color, secondary: candidate.secondary,
                       bevel: candidate.bevel, shadow: candidate.shadow});
  channel.close();
}

// One request for the page's first palette state, shared by every module that loads it (this module is loaded once
// per page). A first visit has no identity cookie yet; two concurrent requests would each be issued their own.
let firstState = null;
export function pageState() {
  if (!firstState) {
    firstState = fetch("/api/palette", {credentials: "same-origin", cache: "no-store"}).then(async (response) => {
      const data = await response.json();
      if (!response.ok) throw Object.assign(new Error(data.error || `Request failed: ${response.status}`), {status: response.status});
      return data;
    });
  }
  return firstState;
}

let listening = false;
export async function loadPersonalPalette() {
  if (!listening && typeof BroadcastChannel === "function") {
    listening = true;
    new BroadcastChannel(CHANNEL).onmessage = (event) => applyPalette(event.data);
  }
  try {
    applyPalette((await pageState()).candidate);
  } catch (_) { /* The site's default design remains usable offline. */ }
}
