# Program extraction contracts

`program_extraction.yaml` is the default exhaustive contract used by the live
compiler visualizer. It answers a different question from an OOP boundary
namespace: the namespace can spoof a particular program node, while this sheet
decides whether the compiler may cross from a resolved callable into its
implementation at all.

Every callable is classified as authored Python, repository Python,
third-party Python, standard-library Python, a builtin, a native extension, a
dynamic library, or unknown. The first matching rule wins; otherwise the
required class default wins. Loading fails if even one class has no default.

Available actions are:

- `ingest_python`: parse source and continue through reachable calls.
- `intrinsic`: retain semantic identity and lower through the named vocabulary.
- `python_host_call`: call the existing interpreter or a declared system port;
  do not inspect its implementation.
- `use_native`: retain the existing extension/DLL and declared ABI in place.
- `decompile_machine`: lift machine code only for an exact explicit opt-in with
  function, byte, and dependency-depth ceilings.
- `reject`: stop implementation pursuit at this identity.

The contract fingerprint participates in AOT checkpoint identity. Each choice
is attached to its call and receipts are stored in
`ProcessGraph.G.graph["extraction_contract_receipts"]` with identity,
provenance, rule, action, and parameters.

During Python ProcessGraph ingestion the same receipt is materialized on the
individual call node before its first live evolution event. Resolved identities
are then interpreted by `python_identity_programs.py`. Each replacement is
explicitly one of four kinds:

- `operator`: one canonical operator, such as `math.sqrt` -> `sqrt`,
  `random.gauss` -> stateful `random_source`, or `print` -> `stream_publish`;
- `program`: a short graph sequence, such as `len(x)` -> `shape(x)[0]`;
- `object`: compiler-owned state or schema meaning, such as a range domain,
  stable object identity, OOP `super` dispatch, or PRNG state;
- `compile_time`: provenance/logging metadata with no runtime operation.

Direct operators are materialized immediately and retain their authored
argument edges. Programs and objects remain call-shaped with their complete
declarative replacement attached for the sequence/OOP lowering passes. Other
choices also remain call-shaped so their authored argument dataflow is
preserved, with
`extraction_action`, `extraction_identity`, `extraction_rule`, and the complete
receipt in node attributes. This is terminal for recursive implementation
pursuit, not terminal for evaluation of the call's arguments. Declared
boundaries and rejected calls are also listed under
`extraction_boundary_calls` and `rejected_extraction_calls` on the graph.

Use another sheet with:

```powershell
python -m src.rendering.precompiled_graph_demo `
  --source examples/xor_project/train_xor.py `
  --entrypoint train `
  --extraction-contract extraction_contracts/program_extraction.yaml
```
