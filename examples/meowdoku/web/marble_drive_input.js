// Keyboard intent only. The caller applies these axes through the existing
// native-law gravity control; this module never advances a body or a clock.
const AXES = new Map([
  ["ArrowLeft", [-1, 0]], ["KeyA", [-1, 0]],
  ["ArrowRight", [1, 0]], ["KeyD", [1, 0]],
  ["ArrowUp", [0, 1]], ["KeyW", [0, 1]],
  ["ArrowDown", [0, -1]], ["KeyS", [0, -1]],
]);

export function marbleDriveAxes(codes) {
  // Aliases are one action: W plus Up must not beat S plus Down.
  const held = new Set(codes);
  const action = (...names) => names.some(name => held.has(name)) ? 1 : 0;
  return [action("ArrowRight", "KeyD") - action("ArrowLeft", "KeyA"),
    action("ArrowUp", "KeyW") - action("ArrowDown", "KeyS")];
}

export function bindMarbleDriveInput(canvas, {
  enabled = () => true,
  onChange = () => {},
  window: host = globalThis.window,
  document: doc = globalThis.document,
} = {}) {
  const held = new Set(), remove = [];
  let axes = [0, 0];
  const originalTabIndex = canvas.getAttribute("tabindex");
  if (originalTabIndex === null) canvas.setAttribute("tabindex", "0");
  const publish = () => {
    const next = marbleDriveAxes(held);
    if (next[0] === axes[0] && next[1] === axes[1]) return;
    axes = next;
    onChange([...axes]);
  };
  const clear = () => { held.clear(); publish(); };
  const listen = (target, type, callback) => {
    target?.addEventListener(type, callback);
    remove.push(() => target?.removeEventListener(type, callback));
  };
  listen(canvas, "pointerdown", () => {
    if (enabled()) canvas.focus({preventScroll:true});
  });
  listen(canvas, "keydown", event => {
    if (!enabled()) { clear(); return; }
    if (event.isComposing || event.altKey || event.ctrlKey || event.metaKey || !AXES.has(event.code)) return;
    event.preventDefault();
    held.add(event.code);
    publish();
  });
  // The key may be released after focus moved away from the canvas.
  listen(host, "keyup", event => {
    if (!held.delete(event.code)) return;
    publish();
  });
  listen(canvas, "blur", clear);
  listen(host, "blur", clear);
  listen(doc, "visibilitychange", () => { if (doc.hidden) clear(); });
  return {
    clear,
    axes: () => { if (!enabled()) clear(); return [...axes]; },
    dispose: () => {
      clear();
      remove.splice(0).forEach(callback => callback());
      if (originalTabIndex === null) canvas.removeAttribute("tabindex");
    },
  };
}
