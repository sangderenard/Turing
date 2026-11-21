// Apply only this participant's server-selected color to pages that opt in.
export function applyPalette(candidate) {
  const color = candidate?.color;
  if (typeof color !== "string" || !/^#[0-9a-f]{6}$/i.test(color)) return;
  document.documentElement.style.setProperty("--palette-accent", color);
  if (typeof candidate.secondary === "string" && /^#[0-9a-f]{6}$/i.test(candidate.secondary))
    document.documentElement.style.setProperty("--palette-secondary", candidate.secondary);
}

export async function loadPersonalPalette() {
  try {
    const response = await fetch("/api/palette", {credentials: "same-origin", cache: "no-store"});
    if (response.ok) applyPalette((await response.json()).candidate);
  } catch (_) { /* The site's default design remains usable offline. */ }
}
