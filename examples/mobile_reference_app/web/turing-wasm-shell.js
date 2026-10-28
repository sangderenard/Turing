// Generic page-shell binding for the independent repository-SSA Wasm backend.
// It knows only the published invocation contract. Placement, scheduling,
// inputs, and presentation remain decisions of the importing page.
export class TuringWasmProgram {
  static async load(moduleUrl, contractUrl) {
    const [contract, instantiated] = await Promise.all([
      fetch(contractUrl).then(response => response.json()),
      WebAssembly.instantiateStreaming(fetch(moduleUrl), {}),
    ]);
    return new TuringWasmProgram(instantiated.instance, contract);
  }

  constructor(instance, contract) {
    this.instance = instance;
    this.contract = contract;
    this.memory = new DataView(instance.exports.memory.buffer);
  }

  invoke(feeds) {
    const args = this.contract.parameters.map(parameter => {
      if (parameter.kind === 'buffer') {
        if (parameter.role !== 'output' && parameter.name in feeds) {
          this.memory.setFloat64(parameter.offset, Number(feeds[parameter.name]), true);
        }
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
      .map(parameter => [parameter.name, parameter.dtype === 'int64'
        ? this.memory.getBigInt64(parameter.offset, true)
        : this.memory.getFloat64(parameter.offset, true)]));
  }
}
