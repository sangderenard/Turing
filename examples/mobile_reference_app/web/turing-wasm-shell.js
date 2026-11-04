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
