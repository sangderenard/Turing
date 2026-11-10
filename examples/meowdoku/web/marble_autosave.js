// Serial, revision-owned object saving. No render worker is stopped here.
// capture() must synchronously return a detached immutable snapshot; prepare()
// and store() must use that snapshot, not read changing Studio controls.
export function createObjectAutosave({
  capture, prepare, store, onSaved = () => {}, onPending = () => {},
  onError = () => {}, onStatus = () => {}, enabled = true, delayMs = 800,
  setTimeoutFn = globalThis.setTimeout.bind(globalThis),
  clearTimeoutFn = globalThis.clearTimeout.bind(globalThis),
} = {}) {
  for (const [name, value] of Object.entries({capture, prepare, store,
    onSaved, onPending, onError, onStatus, setTimeoutFn, clearTimeoutFn})) {
    if (typeof value !== "function") throw new TypeError(`${name} must be a function`);
  }
  if (!Number.isFinite(delayMs) || delayMs < 0) throw new RangeError("Invalid autosave delay");

  let revision = 0, savedRevision = 0, generation = 0;
  let latest = null, lastResult, hasResult = false, lastError = null;
  let timer = null, ready = false, active = null, disposed = false;
  let blockedRevision = null, waiters = [];
  enabled = Boolean(enabled);

  const dirty = () => revision !== savedRevision;
  const current = item => !disposed && item.generation === generation && item.revision === revision;
  const abortError = message => Object.assign(new Error(message), {name: "AbortError"});
  const assertOpen = () => { if (disposed) throw abortError("Object autosave is disposed"); };
  const metadata = item => Object.freeze({revision: item.revision, generation: item.generation});
  function state() {
    const working = active && current(active);
    const pending = !disposed && dirty() && (Boolean(working) || waiters.length > 0 ||
      (enabled && blockedRevision !== revision && (timer !== null || ready)));
    const phase = disposed ? "disposed" : working ? active.phase :
      blockedRevision === revision ? "error" : pending ? "queued" : dirty() ? "dirty" :
      hasResult ? "saved" : "idle";
    return Object.freeze({enabled, dirty: dirty(), pending, phase, revision, savedRevision,
      error: lastError});
  }
  // UI observers cannot strand the queue or turn a completed POST into a retry.
  // They are synchronous. Any async UI work must do its own revision check.
  function notify(callback, args, name) {
    try { callback(...args); }
    catch (error) {
      if (callback !== onError) {
        try { onError(error, {revision, generation, phase: "callback", callback: name}); }
        catch { /* A broken observer must not lose a pending save. */ }
      }
    }
  }
  const status = () => notify(onStatus, [state()], "onStatus");
  function clearTimer() {
    if (timer !== null) clearTimeoutFn(timer);
    timer = null;
  }
  function settle(error) {
    const pending = waiters;
    waiters = [];
    for (const waiter of pending) {
      if (error) waiter.reject(error);
      else waiter.resolve(lastResult);
    }
  }
  function failed(error, item, phase) {
    if (!current(item)) return;
    lastError = error;
    blockedRevision = revision;
    ready = false;
    clearTimer();
    // Reject first, so a retry explicitly requested by onError is a new batch.
    settle(error);
    notify(onError, [error, {...metadata(item), phase}], "onError");
  }
  function captureCurrent() {
    const item = {revision, generation, raw: null};
    try {
      item.raw = capture();
      if (item.raw && typeof item.raw.then === "function") {
        throw new TypeError("capture must return a synchronous snapshot");
      }
    } catch (error) {
      failed(error, item, "capture");
      return false;
    }
    if (!current(item)) return false;
    latest = item;
    notify(onPending, [item.raw, metadata(item)], "onPending");
    return current(item);
  }
  function schedule() {
    clearTimer();
    ready = false;
    if (!enabled || !dirty() || blockedRevision === revision || disposed) return;
    timer = setTimeoutFn(() => {
      timer = null;
      ready = true;
      pump();
      status();
    }, delayMs);
  }
  function canSave() {
    return !disposed && dirty() && latest !== null && blockedRevision !== revision &&
      (waiters.length > 0 || (enabled && ready));
  }
  function pump() {
    if (active || !canSave()) return;
    const item = {...latest, phase: "preparing"};
    active = item;
    ready = false;
    clearTimer();
    // Establish ownership before any callback or asynchronous preparation runs.
    void run(item);
  }
  async function run(item) {
    try {
      status();
      if (!current(item) || (!enabled && !waiters.length)) return;
      const prepared = await prepare(item.raw, metadata(item));
      // A slow geometry build must never POST an obsolete object, or an auto
      // save disabled while preparing. A manual flush remains allowed.
      if (!current(item) || (!enabled && !waiters.length)) return;
      item.phase = "saving";
      status();
      if (!current(item) || (!enabled && !waiters.length)) return;
      const result = await store(prepared, metadata(item));
      // A POST already sent cannot be recalled. Its content-addressed result
      // may persist, but it must not replace a newer edit or loaded object.
      if (!current(item)) return;
      savedRevision = item.revision;
      lastResult = result;
      hasResult = true;
      lastError = null;
      blockedRevision = null;
      notify(onSaved, [result, prepared, metadata(item)], "onSaved");
      if (current(item) && !dirty()) settle();
    } catch (error) {
      failed(error, item, item.phase);
    } finally {
      active = null;
      if (!disposed) {
        status();
        pump();
      }
    }
  }
  function changed() {
    assertOpen();
    revision++;
    latest = null;
    blockedRevision = null;
    lastError = null;
    clearTimer();
    ready = false;
    if (captureCurrent()) {
      schedule();
      pump();
    }
    status();
    return revision;
  }
  // Resolve only once the latest revision is saved, including edits arriving
  // during this flush. Concurrent clicks share work, not duplicate POSTs.
  function flush() {
    try { assertOpen(); } catch (error) { return Promise.reject(error); }
    if (!dirty() && hasResult) return Promise.resolve(lastResult);
    if (!dirty()) {
      changed();
      if (latest === null) return Promise.reject(lastError || abortError("Save superseded"));
    } else if (latest === null) {
      blockedRevision = null;
      lastError = null;
      if (!captureCurrent()) return Promise.reject(lastError || abortError("Save superseded"));
    }
    blockedRevision = null;
    lastError = null;
    clearTimer();
    ready = false;
    const promise = new Promise((resolve, reject) => waiters.push({resolve, reject}));
    pump();
    status();
    return promise;
  }
  function setEnabled(value) {
    assertOpen();
    const next = Boolean(value);
    if (next === enabled) return;
    enabled = next;
    clearTimer();
    ready = false;
    // A deliberate off/on toggle is also a user-requested retry.
    if (enabled) {
      blockedRevision = null;
      lastError = null;
      if (dirty() && latest === null) captureCurrent();
      schedule();
    }
    pump();
    status();
  }
  // Invoke immediately before applying a successfully loaded external object.
  // Existing flush callers must not mistake that load for their save result.
  function acceptExternal(result) {
    assertOpen();
    generation++;
    revision++;
    savedRevision = revision;
    latest = null;
    lastResult = result;
    hasResult = arguments.length > 0;
    blockedRevision = null;
    lastError = null;
    ready = false;
    clearTimer();
    settle(abortError("Object save superseded by an external object"));
    status();
    return revision;
  }
  function dispose() {
    if (disposed) return;
    disposed = true;
    generation++;
    clearTimer();
    ready = false;
    latest = null;
    settle(abortError("Object autosave is disposed"));
    status();
  }
  return {
    changed, flush, setEnabled, acceptExternal, supersede: acceptExternal, dispose,
    get revision() { return revision; },
    get savedRevision() { return savedRevision; },
    get dirty() { return dirty(); },
    get enabled() { return enabled; },
    get pending() { return state().pending; },
    get state() { return state(); },
  };
}
