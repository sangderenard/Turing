# Binary-head IR continuation

**Paused:** 2026-08-12
**Reason:** full focus returned to recursively compiling the deep Python source ecosystem into one static repository-SSA program.

## Boundary

The binary head is a separate machine-state translation subsystem. Its retained
machine dialect may use the repository's `IRModule` and `Function` containers,
but that structural reuse does **not** make it repository SSA. A function is
eligible for static repository-SSA emission only after every machine operation
has been legalized to ordinary repository control, arithmetic, and memory
instructions. Never advertise retained machine instructions as repository SSA,
and never use them to make the source compiler appear complete.

The two experimental live/interposition demos and their tests were removed when
this track was paused because they displayed retained machine IR under a
"repository SSA" label. The underlying decoder, lifter, machine executor, and
stream interposer remain available for future work.

## Verified state at pause

- The scalar reference decoder is authoritative. Tensor lanes are optional
  accelerators/verifiers and cannot narrow reference decoding.
- Decode events distinguish `reference` from `tensor-verified` provenance.
- Reversible instruction layout is owned by authoritative `InstructionSpec`
  records, including legacy-prefix, REX.W, and 0F/0F38/0F3A map dimensions.
- The reversible grammar covers 308/308 authoritative tokens.
- A catalogue test synthesizes one canonical encoding per token, reference
  decodes it, writes it back byte-exactly, and verifies token identity across
  all 308 tensor lanes.
- A permanent bounded canonical-token regression lowers 308/308 authoritative
  tokens completely to ordinary repository SSA (2026-08-12: 308 complete,
  zero failures). Each form is independently encoded, decoded, CFG-lowered,
  and checked for retained machine-dialect operations.
- `SCASB` lowers to an 8-bit load from `[RDI]`, exact `AL - memory` arithmetic
  flags, and a DF-selected `RDI +/- 1`; plain SCASB does not consume RCX.
- `LOCK_ADD_RM8_R8` lowers to one sequentially-consistent atomic byte
  read-modify-write, exact arithmetic flags, and an unchanged source register.
  Its authoritative reversible grammar is explicitly memory-only: the scalar
  decoder, tensor verifier, and write head all reject a register ModRM form.
- Every current target that directly accepts repository SSA -- Fortran,
  WebGPU, SPIR-V, and the WebGL SSA adapter -- now rejects every retained
  machine-dialect occurrence at its front door. Both module and single-
  function entry points are covered where offered. Reusing
  `IRModule`/`Function` as decompiler containers can no longer be mistaken for
  repository-SSA legalization. Internal `FusedProgram` backends are separate
  numerical artifacts and are not reclassified as full-program SSA targets.
- The authoritative cached host-code ledger
  `bounded-host-ssa-authoritative-repository-complete.tsv` has zero rows; an
  older 16-item ledger is stale because those blockers were subsequently fixed.
- Recursive PE dependency completion is occurrence-exact. Unit extraction
  keeps its immutable raw blocker ledger, while assembled-library completeness
  subtracts an `external-machine-module` blocker only when the matching source
  cache key, import identity, and machine callsite address has a concrete
  decompiled target. Source ingestion and implementation selection consume
  this effective ledger plus an explicit machine-dialect scan; they no longer
  advertise `repository-ssa` merely because a host IRModule exists.
- The host SSA disk-cache schema is v2 so cached v1 results lacking exact
  import-occurrence identity cannot be reused under the stronger contract.
- The last user-authorized recursive `compile()` extraction reached cache
  serialization and failed on a Python `mappingproxy`. `_HostSSACachePickler`
  now handles that object and its focused tests pass. The failure occurred
  before a cache file was persisted.

## Focused evidence

- 80 focused read-head, stream, bidirectional, and PE tests passed.
- 245 semantic-family, read-head, and stream tests passed.
- 258 read-head, machine-semantic, stream, and host-cache tests passed.
- The 308-token catalogue roundtrip test passed independently.
- No long recursive compile was run after the user prohibited unrequested long
  runs.

## Safe restart order

1. Re-run the focused decoder/lifter/cache selections; do not start recursive
   host extraction as an orientation probe.
2. Keep the 308-token reference/tensor/write-head/ordinary-SSA catalogue audit
   green as new vocabulary is admitted.
3. Apply the same explicit machine-dialect rejection gate to any additional
   repository-SSA emitter entry points as they become target-capable.
4. Resume cached dependency recursion only with explicit user authorization for
   the long run, retaining every blocker occurrence and duplicate.

No numerical projection, Python/runtime fallback, external handler substitute,
or silent operation deletion is permitted on this path.
