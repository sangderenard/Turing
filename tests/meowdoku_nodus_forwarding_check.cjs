// The Nodus page's input forwarding (web/nodus.js attachForwarding) never leaves a press without its release.
// node tests/meowdoku_nodus_forwarding_check.cjs   (exit 0 when every case holds)
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(path.join(__dirname, "..", "examples", "meowdoku", "web", "nodus.js"), "utf8");
const start = source.indexOf("function attachForwarding(");
const end = source.indexOf("\nfunction drawMap()");
if (start < 0 || end < 0) throw new Error("attachForwarding not found");
const body = source.slice(start, end);

function world({ refuseAfter = Infinity } = {}) {
  const sent = [];
  let revision = 1;
  const listeners = {};
  const canvas = {
    width: 100, height: 100, tabIndex: -1, classList: { set: new Set(), add(c) { this.set.add(c); }, remove(c) { this.set.delete(c); },
      contains(c) { return this.set.has(c); } },
    addEventListener(name, fn) { (listeners[name] ||= []).push(fn); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
    focus() {}, setPointerCapture() {}, hasPointerCapture: () => false, releasePointerCapture() {},
  };
  const toggleListeners = {};
  const toggle = { type: "", checked: false, addEventListener(name, fn) { toggleListeners[name] = fn; } };
  const card = { isConnected: true, append() {}, querySelector: () => toggle };
  const context = {
    state: { canEdit: true, forwarding: null, instance: "nodus-a", revision: 1, graphKey: null },
    el: (tag) => tag === "input" ? toggle : { append() {}, className: "" },
    document: { createTextNode: () => ({}) },
    $: () => ({ querySelectorAll: () => [] }),
    q: (p) => p + "?instance=nodus-a",
    get: async () => ({ status: 200, body: { instance: "nodus-a", revision } }),
    send: async (_path, payload) => {
      const action = payload.action;
      if (action.op !== "viewport") sent.push(action);
      if (sent.length > refuseAfter && action.op !== "viewport" && !(action.kind === "up" || action.action === 0)) {
        return { status: 409, body: { error: "revision changed" } };
      }
      revision += 1;
      return { status: 200, body: { revision, handled: true } };
    },
  };
  vm.createContext(context);
  vm.runInContext(body + "\nthis.attachForwarding = attachForwarding;", context);
  const status = { textContent: "" };
  const quad = { hasFrame: true, fullView: true };
  context.attachForwarding(card, canvas, status, quad, "ui-1");
  const fire = async (name, event = {}) => { for (const fn of listeners[name] || []) fn({ preventDefault() {}, clientX: 5, clientY: 5, button: 0,
    buttons: 1, pointerId: 1, keyCode: 0, ...event }); await settle(); };
  const on = async () => { toggle.checked = true; await toggleListeners.change(); };
  const off = async () => { toggle.checked = false; await toggleListeners.change(); await settle(); };
  return { sent, fire, on, off, context, toggle };
}
const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
const stillDown = (sent) => {
  const down = new Set();
  for (const a of sent) {
    if (a.op === "mouse" && a.kind === "down") down.add("b" + a.button);
    if (a.op === "mouse" && a.kind === "up") down.delete("b" + a.button);
    if (a.op === "key" && a.action === 1) down.add("k" + a.key);
    if (a.op === "key" && a.action === 0) down.delete("k" + a.key);
  }
  return [...down];
};
const cases = [];
const check = (name, fn) => cases.push([name, fn]);

check("switched off mid-drag and with a key down", async () => {
  const w = world(); await w.on();
  await w.fire("pointerdown"); await w.fire("keydown", { keyCode: 65 }); await w.off();
  return stillDown(w.sent);
});
check("the button comes up after forwarding stopped", async () => {
  const w = world(); await w.on();
  await w.fire("pointerdown"); w.context.state.forwarding = null; await w.fire("pointerup");
  return stillDown(w.sent);
});
check("a refused action mid-drag", async () => {
  const w = world({ refuseAfter: 1 }); await w.on();
  await w.fire("pointerdown"); await w.fire("pointermove", { clientX: 9 }); await settle(); await settle();
  return stillDown(w.sent);
});
check("the pointer is cancelled", async () => {
  const w = world(); await w.on();
  await w.fire("pointerdown"); await w.fire("pointercancel");
  return stillDown(w.sent);
});
check("focus leaves with a key down", async () => {
  const w = world(); await w.on();
  await w.fire("keydown", { keyCode: 16 }); await w.fire("blur");
  return stillDown(w.sent);
});
check("another instance is chosen", async () => {
  const w = world(); await w.on();
  await w.fire("pointerdown", { button: 2 }); w.context.state.releaseForwarding(); await settle(); await settle();
  return stillDown(w.sent);
});

(async () => {
  let failed = 0;
  for (const [name, fn] of cases) {
    const left = await fn();
    if (left.length) { failed += 1; console.log(`FAIL ${name}: still down ${left.join(", ")}`); }
    else console.log(`ok   ${name}`);
  }
  process.exit(failed ? 1 : 0);
})();
