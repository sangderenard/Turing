2026-11-03 Native returned-byte capacities and packet splitter parity (MEASURED)

Checkout: C:/Users/alber/.codex/worktrees/compressor-results/turing.

The source bytes-slice bound and exact input/output sequence descriptors now
drive the C module's existing caller-owned buffer ABI. Each output reserves
its exact source's runtime row capacity, while the native clipped-copy helper
writes its own logical length. The wrapper initializes incoming bytes length
and capacity and outgoing capacity from the public source extent. It does not
evaluate the Python slice on the host. Runtime views read only native extent
ports; repeated calls restore the original capacity buffers before invoking
the DLL. Runtime-ranked unfed buffers without a proven origin are refused.

The v2 ABI description is an EMISSION_ARTIFACT/RUNTIME_ABI fact derived from
BUFFER_ORDER, API_CONTRACT, the source capacity bound and both descriptors.
The runtime consumer checks the recorded description and digest. The audit
rejoins exact descriptor ownership and capacity/length indices, including a
custom exported entry name; it rejects a forged relation and unsupported shape
relations. The unchanged C/LLVM v1 ABI cases both pass.

The actual accepted `_read_header` -> `_split` module now emits complete C.
Its first DLL build failed because CModuleArtifact.compile linked the existing
validation runtime .c but omitted the header included by the generated module.
The DLL build now copies that existing header, as the standalone executable
build already does, and records its source-file receipt. No exception behavior
or codec source was changed.

Measured checks:
- tests/test_runtime_result_extent_ports.py: 23 passed in 6.69s, including
  native length/shape decoding, invalid bounds, custom entry and forged receipt.
- tests/test_linked_accepted_c_emission_owner.py and
  tests/test_public_byte_slice_capacity.py: 2 passed in 20.90s. Both compile DLLs.
  Actual splitter matches Python on empty command/literal streams, high bytes,
  and a 130-byte command stream plus all 256 literal byte values. Direct slices
  match Python on empty, ordinary, overlong and negative indices, with repeated
  native calls. The first tamper test only reversed a relation tuple (same
  relation facts); it was corrected to swap the actual length bindings.
- tests/test_runtime_sized_spans.py::test_runtime_abi_states_rank_extents_and_result_derivation:
  C and LLVM passed, 2 tests in 9.92s.
- tools/audit_identity_concordance.py mapping: 28 rows, zero findings,
  zero unsourced, CLOSED. git diff --check passed.

These are real codec-helper DLL checks. Whole compress_octets DLL creation and
compress/decompress round-trip parity have not completed. Full run31 stopped
after 428.13s at the reverse shape metadata writer; its separate receipt names
that exact row. A fresh full lowering is still required after integrating its
causal-writer correction.
