2026-11-10 WEB-LANE PRODUCTS DESCRIBE THEMSELVES (MEASURED, local Windows, worktree
bridging/web-lane-api cut from main f998b80f):

The Nodus importer reads the `turing-compiled-program-api-v1` contract that
compiled_program_api.py writes beside Fortran, standalone-shell and project
products. The Meowdoku web-lane products wrote only their JSON ABI families
(`turing.mobile-wasm-abi.v1`, `turing.fountain-products.v3`,
`turing.graph-buffer-abi.v1`) and no contract, so a program could not describe
its binaries to Nodus. The contract is now built by the emitter that produced
the artifact, from the same Function objects:

- `ssa_wasm_backend.emit_ssa_module_to_wasm_core`: `WasmCoreArtifact.api`
  (describe_fortran_function over the root Function, kind control, language
  wasm). entity_motion `_build_one` and policy_wasm `build_policy` write
  `<stem>.api.yaml` beside `.abi.json`. Output source_names are the lowering's
  (`result_0`), not the writer's authored names, which stay the JSON layer.
- `ssa_webgpu_backend.emit_tensor_program_passes`: `WGSLPassProgram.api`
  mirroring emit_module (language wgsl, one numerical entry per dispatch, no
  Parameters, slot metadata). fountain_build (3 sites) and fountain3d_build
  write `<wgsl stem>.api.yaml`.
- NOT done: `compile_graph_forward_to_wasm` / `BrowserGraphForward`
  (graph_wasm.py, fountain `build_fallback`): `_forward_graph_module` sets
  only named_outputs and no parameter_names, so a Function-derived contract
  would have unnamed inputs. Naming them is a tracing-layer change
  (`SSATensorOperations.input`), not taken here.

Emission, wasm bytes, WGSL and every JSON ABI are unchanged. Both writers skip
the API write on a source-digest cache hit, so existing products gain an
`.api.yaml` only on rebuild or force=True.

Measured: tests/test_meowdoku_product_api_yaml.py 2 passed (23.2 s: climb law
wasm core ~10 s; fountain size "one" WGSL ~18 s); tests/test_wasm_core_product.py
4 passed (10.4 s). Not exercised: policy_wasm and fountain3d write lines.
