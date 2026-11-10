import assert from "node:assert/strict";
import test from "node:test";
import {createObjectAutosave} from "../examples/meowdoku/web/marble_autosave.js";

const tick = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}
function harness(options = {}) {
  let value = 1, timerId = 0;
  const timers = new Map(), captures = [], prepares = [], stores = [];
  const saved = [], pending = [], errors = [], statuses = [];
  const hooks = {capture: null, prepare: null, store: null};
  const api = createObjectAutosave({
    capture() {
      if (hooks.capture) return hooks.capture();
      const raw = Object.freeze({value});
      captures.push(raw);
      return raw;
    },
    async prepare(raw, meta) {
      prepares.push({raw, meta});
      return hooks.prepare ? hooks.prepare(raw, meta) : Object.freeze({raw, verified: true});
    },
    async store(prepared, meta) {
      stores.push({prepared, meta});
      return hooks.store ? hooks.store(prepared, meta) : {digest: `object-${prepared.raw.value}`};
    },
    onSaved: (...args) => saved.push(args),
    onPending: (...args) => pending.push(args),
    onError: (...args) => errors.push(args),
    onStatus: status => statuses.push(status),
    setTimeoutFn(fn, delay) { timers.set(++timerId, {fn, delay}); return timerId; },
    clearTimeoutFn(id) { timers.delete(id); },
    ...options,
  });
  return {api, hooks, timers, captures, prepares, stores, saved, pending, errors, statuses,
    edit(next) { value = next; return api.changed(); },
    async due() {
      const queued = [...timers.values()];
      timers.clear();
      for (const timer of queued) timer.fn();
      await tick();
    },
  };
}

test("rapid edits synchronously capture immutable snapshots and debounce the newest", async () => {
  const h = harness();
  h.edit(1); h.edit(2); h.edit(3);
  assert.deepEqual(h.captures.map(raw => raw.value), [1, 2, 3]);
  assert.equal(h.pending.length, 3);
  assert.equal(h.api.dirty, true);
  assert.equal(h.api.pending, true);
  assert.equal(h.timers.size, 1);
  assert.equal([...h.timers.values()][0].delay, 800);
  assert.equal(h.prepares.length, 0);
  await h.due();
  assert.equal(h.prepares.length, 1);
  assert.equal(h.stores[0].prepared.raw, h.captures[2]);
  assert.equal(h.saved[0][2].revision, 3);
  assert.equal(h.api.savedRevision, 3);
  assert.equal(h.api.dirty, false);
  assert.equal(h.api.pending, false);
  assert.equal(h.api.state.phase, "saved");
});

test("an edit during preparation skips the obsolete POST and respects the new debounce", async () => {
  const h = harness(), gate = deferred();
  h.hooks.prepare = raw => raw.value === 1 ? gate.promise : {raw};
  h.edit(1); await h.due();
  h.edit(2);
  gate.resolve({raw: h.captures[0]}); await tick();
  assert.equal(h.stores.length, 0);
  assert.equal(h.prepares.length, 1);
  await h.due();
  assert.equal(h.stores.length, 1);
  assert.equal(h.stores[0].prepared.raw.value, 2);
  assert.deepEqual(h.saved.map(([result]) => result.digest), ["object-2"]);
});

test("edits during a POST never publish the old result and serialize the latest save", async () => {
  const h = harness(), gate = deferred();
  h.hooks.store = prepared => prepared.raw.value === 1 ? gate.promise : {digest: "new"};
  h.edit(1); await h.due();
  h.edit(2); h.edit(3); await h.due();
  assert.equal(h.prepares.length, 1, "new work waits for the in-flight POST");
  assert.equal(h.stores.length, 1);
  gate.resolve({digest: "old"}); await tick();
  assert.equal(h.stores.length, 2);
  assert.equal(h.stores[1].prepared.raw.value, 3);
  assert.deepEqual(h.saved.map(([result]) => result.digest), ["new"]);
  assert.equal(h.api.dirty, false);
});

test("manual flush saves an initial object and repeated clicks share its POST", async () => {
  const h = harness({enabled: false}), gate = deferred();
  h.hooks.store = () => gate.promise;
  const one = h.api.flush(), two = h.api.flush();
  await tick();
  assert.equal(h.captures.length, 1);
  assert.equal(h.stores.length, 1);
  gate.resolve({digest: "initial"});
  assert.deepEqual(await one, {digest: "initial"});
  assert.deepEqual(await two, {digest: "initial"});
  assert.deepEqual(await h.api.flush(), {digest: "initial"});
  assert.equal(h.stores.length, 1);
});

test("manual flush follows edits arriving during saving even with autosave disabled", async () => {
  const h = harness({enabled: false}), first = deferred(), last = deferred();
  h.hooks.store = prepared => prepared.raw.value === 1 ? first.promise : last.promise;
  h.edit(1);
  let resolved = false;
  const saving = h.api.flush().then(result => { resolved = true; return result; });
  await tick(); h.edit(2);
  first.resolve({digest: "old"}); await tick();
  assert.equal(resolved, false);
  assert.equal(h.stores.length, 2);
  assert.equal(h.saved.length, 0);
  last.resolve({digest: "current"});
  assert.deepEqual(await saving, {digest: "current"});
  assert.equal(h.api.dirty, false);
});

test("disable cancels the timer but preserves edits; reenable saves the newest", async () => {
  const h = harness();
  h.edit(1); h.api.setEnabled(false); h.edit(2);
  assert.equal(h.timers.size, 0);
  await h.due();
  assert.equal(h.prepares.length, 0);
  assert.equal(h.api.dirty, true);
  assert.equal(h.api.state.phase, "dirty");
  h.api.setEnabled(true);
  assert.equal(h.timers.size, 1);
  await h.due();
  assert.equal(h.stores[0].prepared.raw.value, 2);
});

test("disable during automatic preparation prevents an unsent POST", async () => {
  const h = harness(), gate = deferred();
  h.hooks.prepare = () => gate.promise;
  h.edit(1); await h.due();
  h.api.setEnabled(false);
  gate.resolve({raw: h.captures[0]}); await tick();
  assert.equal(h.stores.length, 0);
  assert.equal(h.api.dirty, true);
  await h.api.flush();
  assert.equal(h.stores.length, 1, "manual saving still works while disabled");
});

test("disable does not cancel an already sent save or a pending manual flush", async () => {
  const h = harness(), gate = deferred();
  h.hooks.store = () => gate.promise;
  h.edit(1); await h.due();
  const saving = h.api.flush();
  h.api.setEnabled(false);
  gate.resolve({digest: "sent"});
  assert.deepEqual(await saving, {digest: "sent"});
  assert.equal(h.saved.length, 1);
  assert.equal(h.api.dirty, false);
});

test("auto preparation failures retain dirty data without retry storms; a new edit retries", async () => {
  const h = harness(), failure = new Error("geometry failed");
  h.hooks.prepare = () => { throw failure; };
  h.edit(1); await h.due();
  assert.equal(h.errors[0][0], failure);
  assert.equal(h.errors[0][1].phase, "preparing");
  assert.equal(h.api.dirty, true);
  assert.equal(h.api.state.phase, "error");
  assert.equal(h.timers.size, 0);
  await h.due(); await tick();
  assert.equal(h.prepares.length, 1);
  h.hooks.prepare = null;
  h.edit(2); await h.due();
  assert.equal(h.api.dirty, false);
  assert.equal(h.stores[0].prepared.raw.value, 2);
});

test("manual failure rejects every waiting click; manual retry uses the retained snapshot", async () => {
  const h = harness(), gate = deferred(), failure = new Error("network failed");
  h.hooks.store = () => gate.promise;
  h.edit(1);
  const one = assert.rejects(h.api.flush(), error => error === failure);
  const two = assert.rejects(h.api.flush(), error => error === failure);
  await tick(); gate.reject(failure); await Promise.all([one, two]);
  assert.equal(h.api.dirty, true);
  assert.equal(h.errors[0][1].phase, "saving");
  assert.equal(h.timers.size, 0);
  h.hooks.store = null;
  await h.api.flush();
  assert.equal(h.stores[1].prepared.raw, h.captures[0]);
  assert.equal(h.api.dirty, false);
});

test("a stale save failure cannot reject a manual flush of a newer edit", async () => {
  const h = harness(), gate = deferred();
  h.hooks.store = prepared => prepared.raw.value === 1 ? gate.promise : {digest: "latest"};
  h.edit(1); await h.due();
  h.edit(2);
  const saving = h.api.flush();
  gate.reject(new Error("old request failed"));
  assert.deepEqual(await saving, {digest: "latest"});
  assert.equal(h.errors.length, 0);
});

test("off/on is an explicit retry after failure", async () => {
  const h = harness();
  h.hooks.store = () => { throw new Error("offline"); };
  h.edit(1); await h.due();
  h.hooks.store = null;
  h.api.setEnabled(false); h.api.setEnabled(true); await h.due();
  assert.equal(h.stores.length, 2);
  assert.equal(h.api.dirty, false);
});

test("an external load supersedes an outstanding POST and rejects old flushes", async () => {
  const h = harness(), gate = deferred();
  h.hooks.store = () => gate.promise;
  h.edit(1);
  const saving = assert.rejects(h.api.flush(), {name: "AbortError"});
  await tick();
  h.api.acceptExternal({digest: "loaded"});
  await saving;
  gate.resolve({digest: "obsolete"}); await tick();
  assert.equal(h.saved.length, 0);
  assert.equal(h.errors.length, 0);
  assert.equal(h.api.dirty, false);
  assert.deepEqual(await h.api.flush(), {digest: "loaded"});
  assert.equal(h.stores.length, 1);
});

test("load during preparation skips its POST, and subsequent edits remain serial", async () => {
  const h = harness(), gate = deferred();
  h.hooks.prepare = raw => raw.value === 1 ? gate.promise : {raw};
  h.edit(1); await h.due();
  h.api.supersede({digest: "loaded"});
  h.edit(2); await h.due();
  assert.equal(h.prepares.length, 1);
  gate.resolve({raw: h.captures[0]}); await tick();
  assert.equal(h.stores.length, 1);
  assert.equal(h.stores[0].prepared.raw.value, 2);
  assert.equal(h.saved.length, 1);
});

test("capture errors invalidate an old save, stay dirty, and can be manually retried", async () => {
  const h = harness(), gate = deferred();
  h.hooks.store = () => gate.promise;
  h.edit(1); await h.due();
  h.hooks.capture = () => { throw new Error("invalid edit"); };
  h.edit(2);
  assert.equal(h.api.dirty, true);
  assert.equal(h.errors[0][1].phase, "capture");
  gate.resolve({digest: "obsolete"}); await tick();
  assert.equal(h.saved.length, 0);
  await assert.rejects(h.api.flush(), /invalid edit/);
  h.hooks.capture = null; h.hooks.store = null;
  assert.deepEqual(await h.api.flush(), {digest: "object-2"});
  assert.equal(h.api.dirty, false);
});

test("dispose clears pending timers, rejects flushes, and ignores in-flight completion", async () => {
  const h = harness(), gate = deferred();
  h.hooks.store = () => gate.promise;
  h.edit(1);
  const saving = assert.rejects(h.api.flush(), {name: "AbortError"});
  await tick(); h.edit(2);
  h.api.dispose(); h.api.dispose(); await saving;
  assert.equal(h.timers.size, 0);
  const statusCount = h.statuses.length;
  gate.resolve({digest: "disposed"}); await tick();
  assert.equal(h.saved.length, 0);
  assert.equal(h.errors.length, 0);
  assert.equal(h.statuses.length, statusCount, "disposed observers are not called again");
  assert.equal(h.api.state.phase, "disposed");
  await assert.rejects(h.api.flush(), {name: "AbortError"});
  assert.throws(() => h.api.changed(), {name: "AbortError"});
  assert.throws(() => h.api.setEnabled(true), {name: "AbortError"});
});

test("dispose before debounce or during preparation never sends a POST", async () => {
  const queued = harness();
  queued.edit(1); queued.api.dispose(); await queued.due();
  assert.equal(queued.prepares.length, 0);
  const preparing = harness(), gate = deferred();
  preparing.hooks.prepare = () => gate.promise;
  preparing.edit(1); await preparing.due();
  preparing.api.dispose();
  gate.resolve({raw: preparing.captures[0]}); await tick();
  assert.equal(preparing.stores.length, 0);
});

test("observer exceptions cannot strand or duplicate a completed save", async () => {
  const h = harness({onSaved() { throw new Error("UI failed"); }});
  h.edit(1);
  assert.deepEqual(await h.api.flush(), {digest: "object-1"});
  assert.equal(h.api.dirty, false);
  assert.equal(h.errors[0][1].callback, "onSaved");
  assert.equal(h.timers.size, 0);
});

test("reentrant edits from onSaved keep a manual flush waiting for the latest", async () => {
  let h;
  h = harness({onSaved(result) { if (result.digest === "object-1") h.edit(2); }});
  h.edit(1);
  assert.deepEqual(await h.api.flush(), {digest: "object-2"});
  assert.equal(h.stores.length, 2);
  assert.equal(h.api.savedRevision, h.api.revision);
});

test("invalid options and asynchronous capture fail explicitly", async () => {
  assert.throws(() => createObjectAutosave(), /capture/);
  assert.throws(() => harness({delayMs: -1}), /delay/);
  const h = harness({capture: async () => ({value: 1})});
  await assert.rejects(h.api.flush(), /synchronous snapshot/);
  assert.equal(h.stores.length, 0);
});
