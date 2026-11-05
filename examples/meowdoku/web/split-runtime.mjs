// The runtime half of split lowering (examples/meowdoku/split_lowering.py).
// The JS host module calls each WebAssembly region through
// `turingRuntime.call(descriptor, args)`; this binds that call to the region's
// TuringWasmProgram (turing-wasm-shell.js), feeding args by the descriptor's
// feedIds and returning the region's outputs in outputIds order.
import { TuringWasmProgram } from './turing-wasm-shell.js';

export async function loadSplitRuntime(split, loadBytes, loadJson) {
  const programs = new Map();
  await Promise.all(split.wasm_regions.map(async (name) => {
    const [bytes, contract] = await Promise.all([loadBytes(`regions/${name}.wasm`), loadJson(`regions/${name}.abi.json`)]);
    programs.set(name, await TuringWasmProgram.fromBytes(bytes, contract));
  }));
  const calls = new Map();
  return {
    calls,
    call(descriptor, args) {
      const program = programs.get(descriptor.callee);
      if (!program) throw new Error(`split runtime: no wasm region ${descriptor.callee}`);
      calls.set(descriptor.callee, (calls.get(descriptor.callee) || 0) + 1);
      const feeds = {};
      descriptor.feedIds.forEach((id, k) => { feeds[`value_${id}`] = args[k]; });
      const out = program.invoke(feeds);
      // `invoke` has already read each one-element output out of linear memory.
      return descriptor.outputIds.map((id) => out[`value_${id}`]);
    },
  };
}
