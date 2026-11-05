// Generic page-shell binding for the independent repository-SSA Wasm backend.
// It knows only the published invocation contract (program.abi.json,
// schema turing.wasm-core-abi.v1). Placement, scheduling, inputs, and
// presentation remain decisions of the importing page.
//
// The core imports the libm names the LLVM and C lanes link (pow, exp, the
// trig family) under module "math"; the page binds them from Math, which is
// the same table the node harness supplies.
export const TURING_MATH_IMPORTS = Object.fromEntries(
  ['pow', 'exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
   'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'].map(name => [name, Math[name]]));

// A browser sampler over the viewport control-binding vocabulary
// (abstract_ui_viewports.ViewportControlBinding data: action -> inputs such
// as "keyboard:KeyW", "pointer:button-0", "gamepad:left-y-negative").  It
// answers a semantic action with the strongest live input bound to it, the
// same rule the Living Data Map's viewportInputValue applies, and lets a
// page add its own resolvers (a board page answers "pointer-cell-row" from
// its geometry).  Edge-triggered actions are armed by `pulse` and cleared
// by `settle` after the round that consumed them.
export class TuringControlSampler {
  constructor(bindings, resolvers = {}) {
    this.bindings = bindings;          // [{action, inputs: [...]}, ...]
    this.resolvers = resolvers;        // action -> (parameter) => value
    this.keys = new Set();
    this.pointer = {x: 0, y: 0, buttons: 0};
    this.pulses = new Map();           // action -> value for the next round
  }

  attach(target, options = {}) {
    const keyTarget = options.keyboard || window;
    keyTarget.addEventListener('keydown', event => this.keys.add(event.code));
    keyTarget.addEventListener('keyup', event => this.keys.delete(event.code));
    target.addEventListener('pointermove', event => {
      const bounds = target.getBoundingClientRect();
      this.pointer.x = (event.clientX - bounds.left) / Math.max(1, bounds.width);
      this.pointer.y = (event.clientY - bounds.top) / Math.max(1, bounds.height);
    });
    target.addEventListener('pointerdown', event => {
      this.pointer.buttons |= 1 << event.button;
      for (const binding of this.bindings) {
        if (binding.inputs.includes(`pointer:button-${event.button}`)) this.pulse(binding.action, 1);
      }
    });
    target.addEventListener('pointerup', event => { this.pointer.buttons &= ~(1 << event.button); });
    target.addEventListener('contextmenu', event => event.preventDefault());
    return this;
  }

  pulse(action, value = 1) { this.pulses.set(action, value); }
  settle() { this.pulses.clear(); }

  gamepad() {
    const pads = typeof navigator !== 'undefined' && navigator.getGamepads ? navigator.getGamepads() : [];
    return Array.from(pads || []).find(pad => pad && pad.connected) || null;
  }

  // A control hookup: one action, or an axis of signed actions
  // ("+move-forward -move-backward"), the grammar of
  // abstract_ui_viewports.control_terms.  Each term is sampled and summed.
  sample(control, parameter) {
    const tokens = String(control).trim().split(/\s+/);
    if (tokens.length > 1 || /^[+-]/.test(tokens[0])) {
      let total = 0;
      for (const token of tokens) {
        if (!/^[+-]./.test(token)) return undefined;
        const value = this.sampleAction(token.slice(1), parameter);
        if (value === undefined) return undefined;
        total += (token[0] === '-' ? -1 : 1) * value;
      }
      return total;
    }
    return this.sampleAction(tokens[0], parameter);
  }

  sampleAction(action, parameter) {
    if (action in this.resolvers) return this.resolvers[action](parameter, this);
    if (this.pulses.has(action)) return this.pulses.get(action);
    if (action === 'pointer.x') return this.pointer.x;
    if (action === 'pointer.y') return this.pointer.y;
    if (action === 'pointer.buttons') return this.pointer.buttons;
    let value = 0;
    const pad = this.gamepad();
    for (const binding of this.bindings) {
      if (binding.action !== action) continue;
      for (const input of binding.inputs) {
        if (input.startsWith('keyboard:') && this.keys.has(input.slice(9))) value = 1;
        if (input.startsWith('pointer:button-') && (this.pointer.buttons >> Number(input.slice(15))) & 1) value = 1;
        if (!pad) continue;
        const leftX = Number(pad.axes[0] || 0), leftY = Number(pad.axes[1] || 0);
        if (input === 'gamepad:left-y-negative') value = Math.max(value, Math.max(0, -leftY));
        if (input === 'gamepad:left-y-positive') value = Math.max(value, Math.max(0, leftY));
        if (input === 'gamepad:left-x-negative') value = Math.max(value, Math.max(0, -leftX));
        if (input === 'gamepad:left-x-positive') value = Math.max(value, Math.max(0, leftX));
        const button = /^gamepad:button-(\d+)$/.exec(input);
        if (button) value = Math.max(value, Number(pad.buttons[Number(button[1])]?.value || 0));
      }
      return value < 0.12 ? 0 : value;
    }
    return undefined;   // no binding and no resolver: the shell cannot supply it
  }
}

export class TuringWasmProgram {
  static async load(moduleUrl, contractUrl) {
    const [contract, instantiated] = await Promise.all([
      fetch(contractUrl).then(response => response.json()),
      WebAssembly.instantiateStreaming(fetch(moduleUrl), {math: TURING_MATH_IMPORTS}),
    ]);
    return new TuringWasmProgram(instantiated.instance, contract);
  }

  static async fromBytes(bytes, contract) {
    const {instance} = await WebAssembly.instantiate(bytes, {math: TURING_MATH_IMPORTS});
    return new TuringWasmProgram(instance, contract);
  }

  constructor(instance, contract) {
    this.instance = instance;
    this.contract = contract;
    const need = Math.ceil((contract.memory_bytes || 0) / 65536);
    const have = instance.exports.memory.buffer.byteLength / 65536;
    if (need > have) instance.exports.memory.grow(need - have);
  }

  // A fresh view every call: memory.grow detaches old ArrayBuffers.
  view(parameter) {
    const count = parameter.count || 1;
    const buffer = this.instance.exports.memory.buffer;
    switch (parameter.dtype) {
      case 'int64': return new BigInt64Array(buffer, parameter.offset, count);
      case 'int32': return new Int32Array(buffer, parameter.offset, count);
      case 'float32': return new Float32Array(buffer, parameter.offset, count);
      default: return new Float64Array(buffer, parameter.offset, count);
    }
  }

  write(parameter, value) {
    const view = this.view(parameter);
    const values = Array.isArray(value) || ArrayBuffer.isView(value) ? value : [value];
    for (let i = 0; i < view.length; i++) {
      const item = values[i < values.length ? i : values.length - 1];
      view[i] = parameter.dtype === 'int64' ? BigInt(item) : Number(item);
    }
  }

  read(parameter) {
    const view = this.view(parameter);
    if (view.length === 1) return view[0];
    return Array.from(view);
  }

  // The controller hookups: every input parameter whose contract descriptor
  // carries "control" (a semantic action such as "primary-action" or
  // "pointer-cell-row", or a shell resource such as "pointer.x") is filled
  // from `sample(control, parameter)` unless `feeds` already names it.  A
  // page supplies the sampler (its devices, its geometry); the program says
  // only what it needs.  A control the sampler cannot answer (undefined) is
  // refused by name rather than fed a silent zero.
  controlFeeds(sample, feeds = {}) {
    const filled = {...feeds};
    for (const parameter of this.contract.parameters) {
      if (!parameter.control || parameter.role === 'output' || parameter.name in filled) continue;
      const value = sample(parameter.control, parameter);
      if (value === undefined || value === null) {
        throw new Error(`control ${parameter.control} for ${parameter.name} is not supplied by this shell`);
      }
      filled[parameter.name] = value;
    }
    return filled;
  }

  // One round from the shell's devices: sample the controls, then invoke.
  invokeWithControls(sample, feeds = {}) {
    return this.invoke(this.controlFeeds(sample, feeds));
  }

  invoke(feeds) {
    const args = this.contract.parameters.map(parameter => {
      if (parameter.kind === 'buffer') {
        if (parameter.role !== 'output' && parameter.name in feeds) this.write(parameter, feeds[parameter.name]);
        return parameter.offset;
      }
      if (!(parameter.name in feeds)) throw new Error(`missing Turing feed: ${parameter.name}`);
      // A Python int is an i64 formal, which takes a BigInt; never truncate it.
      if (parameter.kind === 'i64') return BigInt(feeds[parameter.name]);
      return parameter.kind === 'i32' ? Number(feeds[parameter.name]) | 0 : Number(feeds[parameter.name]);
    });
    this.instance.exports[this.contract.entrypoint](...args);
    return Object.fromEntries(this.contract.parameters
      .filter(parameter => parameter.kind === 'buffer' && parameter.role === 'output')
      .map(parameter => [parameter.name, this.read(parameter)]));
  }
}
