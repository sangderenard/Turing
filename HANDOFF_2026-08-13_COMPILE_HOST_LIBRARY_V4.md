# `compile()` host-library extraction v4 handoff

Date: 2026-08-13
Repository: `C:\dev\Powershell\turing`
Root target: CPython's native `compile()` / `Py_CompileString`
Run log: `build/compile-host-library-v4.stdout.log`
Error log: `build/compile-host-library-v4.stderr.log`
Authoritative occurrence ledger: `build/compile-host-library-report.json`

## Executive status

The v4 process **completed normally**. It did not crash, time out, or die in a
decoder exception. It ran from 2026-08-13 07:23:22 to 12:05:30, wrote an empty
stderr log, atomically produced the 7.23 MB JSON report, and exited after
16,920.010 seconds (4 h 42 m 0 s).

What completed:

- Recursive PE dependency discovery reached its worklist fixed point.
- 107 independently cached host-code units were collected.
- Those units materialized into one repository `IRModule` containing 32,114
  functions.
- The root was materialized as
  `pe_2e3f50323904bbf7__Py_CompileString`.
- All dependency and blocker occurrences were written to the report. The
  report deliberately retains duplicate occurrences.
- Every one of the 107 cache paths named by the report still exists.

What did not complete:

- `machine_state_complete` is `false`.
- `repository_ssa_complete` is `false`.
- There are 6,518 effective blocker occurrences remaining.
- Therefore this is a successful extraction/report checkpoint, **not yet a
  successfully deployable static SSA implementation of `compile()`**.

No serialized aggregate `IRModule` was written. The aggregate existed in the
finished process; the reusable persistent products are the per-unit pickle
caches plus the JSON dependency/occurrence ledger. A continuation should add
a cached library manifest/materialization checkpoint rather than rebuilding
the same aggregate discovery every time.

## Exact run totals

| Measure | Result |
|---|---:|
| Root cache key | `2e3f50323904bbf77cfa94faec5ab388d2c9c5e4b933c445289612201e5bdc28` |
| Units | 107 |
| Materialized functions | 32,114 |
| Dependency occurrences | 4,609 |
| Unresolved dependency occurrences | 3,910 |
| Raw blocker occurrences | 7,217 |
| Effective blocker occurrences | 6,518 |
| Elapsed | 16,920.0104728 seconds |

Raw blocker ledger:

| Kind | Occurrences |
|---|---:|
| `external-machine-module` | 4,609 |
| `indirect-call` | 2,190 |
| `decode` | 216 |
| `machine-control-decode` | 115 |
| `indirect-jump` | 87 |

Effective blocker ledger:

| Kind | Occurrences |
|---|---:|
| `external-machine-module` | 3,910 |
| `indirect-call` | 2,190 |
| `decode` | 216 |
| `machine-control-decode` | 115 |
| `indirect-jump` | 87 |

The 699-occurrence raw/effective difference is real progress: those external
references were discovered in source units and then satisfied by recursively
extracted dependency units. The remaining 3,910 external occurrences were not
satisfied. Do not report all 4,609 raw external references as unresolved, and
do not erase the duplicate occurrences when fixing them.

## Authoritative evidence and how to read it

The complete evidence is in `build/compile-host-library-report.json`:

- `units` records every cache key, source PE, requested symbol, hit/miss status,
  function count, blocker count, and pickle path.
- `dependency_occurrences` records every requesting unit, external identity,
  target unit when resolved, resolution result, and source address.
- `blocker_occurrences` is the complete raw ledger and includes its owning unit
  cache key.
- `effective_blocker_occurrences` is the post-link ledger. It retains
  duplicates, but currently does not repeat the owning unit cache key.

For diagnosis, join raw decode/control occurrences to `units` by
`unit_cache_key`. For unresolved externals, use `dependency_occurrences` rather
than attempting to recover ownership from the effective list. Never reduce
the report to a set and mistake one spelling for one failure; the occurrence
counts and call sites matter.

## Cache checkpoint — preserve this before doing anything expensive

This is the highest-priority handoff constraint:

> **Reuse and repair the existing cache. Do not rebuild these units from
> scratch merely because decoder or lowering code changed.**

Current persistent inventory:

- Cache directory: `C:\dev\Powershell\turing\.turing-cache\host-ssa`
- Pickle files currently present: 220
- Total cache size: approximately 5.006 GiB
- Reported v4 unit cache paths present: 107 of 107
- Root pickle:
  `.turing-cache\host-ssa\2e3f50323904bbf77cfa94faec5ab388d2c9c5e4b933c445289612201e5bdc28.pickle`
- Root pickle size: 633,557,355 bytes
- Current implementation digest observed at handoff:
  `bf77eb767c8810b7b54239d6f060fe203ff87bb94a107f1ecbe06620168c7b78`

The v4 run reported 38 cache hits and 69 misses, but those 38 hits contained
only 40 functions. The 69 misses produced 32,074 functions. The root miss alone
produced 20,171 functions and 2,784 raw blocker occurrences. In other words,
almost all expensive work was recomputed during v4 even though a cache existed.
Repeating that behavior after every additive instruction fix is unacceptable.

### Why the current cache policy will waste the checkpoint

`src/compiler/host_code_modules.py::_implementation_digest()` hashes whole
source files. Any edit to one hashed implementation file changes every unit
cache key, even if the edit affects only one opcode or one function.

There is an additive migration path, but
`_safe_additive_cache_result()` accepts only already-complete, repository-legal
units. That protects correctness, but it means the large incomplete units—the
ones containing nearly all successfully recovered functions—are rebuilt in
full. The root's 20,171 valid extracted functions are therefore treated as
disposable because other functions in the same unit still have blockers.

There is also a cache-versioning correctness gap: `binary_ingestion.py` affects
PE parsing but is not part of the implementation digest. Simply adding it to
the monolithic digest would correctly invalidate keys but would again discard
the expensive checkpoint. Introduce a layered/declared compatibility scheme
first, then bring every semantics-affecting module under explicit versioning.

### Required cache work before another full run

1. Preserve the existing 107 v4 pickles and report unchanged. Never overwrite
   a prior key in place; write repaired artifacts atomically under new keys.
2. Add an explicit cache manifest containing schema version, implementation
   component digests, source PE content digest, root/export identity, function
   keys, blocker-owner keys, import edges, and aggregate-library membership.
3. Split caching below the whole-export unit boundary. Cache lifted functions
   by PE content plus entry RVA/control-region identity and semantic decoder
   version. A changed opcode handler should relift the affected blocked owner
   regions, not all 20,171 functions rooted at `Py_CompileString`.
4. Build a repair operation that imports unaffected functions from an old
   incomplete unit, relifts only failed functions/regions under the new
   vocabulary, recomputes their edges, and emits a new unit atomically.
5. Cache the recursive library index (`root -> units -> dependency edges`) and
   materialized name mapping. Loading a completed worklist should not repeat
   dependency discovery merely to reconstruct the same aggregate module.
6. Record cache provenance in the JSON report, including component digests and
   whether each function was reused, repaired, or newly lifted. Cache reuse
   must be auditable, not inferred from elapsed time.
7. Only after that infrastructure is focused-tested should another full
   `extract_compile_host_library.py` run occur. The expected next run should
   consume the v4 inventory and perform bounded repair, not another five-hour
   rebuild.

Do not delete `.turing-cache`, change `TURING_HOST_SSA_CACHE`, touch source PE
files, or rewrite their timestamps as cleanup. Do not add the current digest to
an additive allow-list without proving the exact compatibility predicate. Do
not mark an incomplete unit complete merely to make it migratable.

## Remaining blocker families and next work

### 1. External PE dependencies: 3,910 effective occurrences

The largest exact unresolved resolution results are:

| Occurrences | Resolution |
|---:|---|
| 599 | `module-unavailable:api-ms-win-core-errorhandling-l1-1-0.dll!SetLastError` |
| 333 | `module-unavailable:api-ms-win-core-errorhandling-l1-1-0.dll!GetLastError` |
| 331 | `module-unavailable:api-ms-win-core-libraryloader-l1-1-0.dll!GetProcAddress` |
| 240 | `module-unavailable:api-ms-win-core-synch-l1-1-0.dll!LeaveCriticalSection` |
| 225 | `module-unavailable:api-ms-win-core-synch-l1-1-0.dll!EnterCriticalSection` |
| 222 | `module-unavailable:api-ms-win-core-processthreads-l1-1-0.dll!TlsGetValue` |
| 179 | `module-unavailable:api-ms-win-core-processthreads-l1-1-0.dll!TlsSetValue` |
| 125 | `module-unreadable:BinaryFormatError:ntdll.dll!NtClose` |
| 112 | `module-unreadable:BinaryFormatError:NTDLL.dll!RtlFreeHeap` |
| 78 | `module-unreadable:BinaryFormatError:ntdll.dll!RtlInitUnicodeString` |
| 76 | `module-unavailable:api-ms-win-core-libraryloader-l1-1-0.dll!LoadLibraryExW` |
| 72 | `module-unreadable:BinaryFormatError:ntdll.dll!NtOpenDirectoryObject` |
| 70 | `module-unavailable:api-ms-win-core-libraryloader-l1-1-0.dll!FreeLibrary` |

The implementation already contains an API-set namespace reader and host
resolver. The report proves that it did not resolve these API-set contracts in
this run. Diagnose that exact reader/path normalization against the local
`apisetschema.dll`; do not add hard-coded Python/runtime calls as substitutes.

The `ntdll.dll` group is a distinct failure: a path was found, but
`parse_pe_image` raised `BinaryFormatError`. Capture the exact parser exception
and PE structural field with a focused read of the same local image. Do not
classify it as merely another missing module.

After provider/parser fixes, resolve the recorded dependency occurrences from
the report and add or repair only the newly available units. There is no reason
to relift already cached Python, CRT, and KernelBase units to discover that the
same external names exist.

### 2. Decode vocabulary: 216 decode + 115 control-decode occurrences

The highest owning units are the root `python311.dll!Py_CompileString` (40),
`ucrtbase!__stdio_common_vfprintf` (35), KernelBase `CompareStringW` (31),
`MultiByteToWideChar` (30), and `WideCharToMultiByte` (30).

Frequent diagnostics include byte windows beginning with `0f 8a`, `0f 18`,
`a4`, `0f ae e8`, opcode `81 ModRM /1`, and opcode `83 ModRM /1`. Some reported
windows also begin with spellings that the vocabulary appears to support, such
as `ba`, `75`, `0f 29`, `0f 4c`, and `0f 47`. That is evidence to investigate
instruction boundary, prefix, operand-width, and control-region context—not a
license to add a second duplicate mnemonic implementation based on four preview
bytes.

For each occurrence family:

1. Join it to its exact PE unit, function RVA/name, address, and full bytes.
2. Establish whether the address is a true instruction boundary.
3. Add or correct the shared bidirectional machine vocabulary and semantic
   effect, not a special `compile()` path.
4. Validate decode -> machine SSA -> encode equivalence for the exact form.
5. Validate repository SSA legalization and interpreter state effects.
6. Repair only affected cached functions/units.

Eleven units currently have zero functions and one decode blocker each:
`strrchr`, `memcmp`, `strcmp`, `copysign`, `memmove`, `strchr`, `strncmp`,
`memchr`, `strncpy`, `strcspn`, and `__fpe_flt_rounds`. These are good focused
closure targets: each should move from a one-blocker negative cache result to a
real unit without requiring a full-library run.

### 3. Indirect control: 2,190 calls and 87 jumps

These are additional cases, not aliases for the decode or missing-module
counts. Do not delete them as “control only,” and do not weaken completeness to
ignore them.

Classify occurrences into:

- import/IAT targets that become static after dependency resolution;
- finite jump tables or callback tables whose complete target sets can be
  recovered from resident machine data;
- state-dependent but valid indirect control that must remain explicitly
  represented in repository SSA/machine state;
- genuinely unresolved memory provenance requiring stronger alias/table
  reasoning.

The architecture permits a loop or dynamic dispatch to remain. Completeness
means its state and target semantics are fully represented in the program, not
that every control edge is forced into a numeric projection or silently
removed.

## Architectural constraints for the continuation

- The target is a graph-produced complete program in repository SSA.
- Do not use numerical projection or fusion to make blockers disappear.
- Do not introduce Python, runtime-language, external-handler, or opaque-call
  fallback paths.
- Machine code sections may be retained as contextualized program state where
  translation is intentionally deferred, but that retention must remain inside
  the compiler's explicit machine/repository SSA contract.
- Do not directly forward AVX2 or any unsupported host instruction to this
  machine. Decode/emulate/translate through declared semantics and select a
  legal target vocabulary.
- Keep binary-specific IR out of the meaning of repository SSA. Conversion
  belongs at the existing machine dialect/repository SSA boundary.
- Use the graph-to-section renderer, not the design-flux `FusedProgram` path.
- Do not add capricious recursion, module, call-count, or loop cutoffs. The
  recursive library worklist must converge by stable identity and cached state.

## Recommended continuation order

1. Treat this report and its 107 cache paths as immutable input evidence.
2. Implement and test cache manifest + partial-unit/function repair first.
3. Fix API-set resolution and exact `ntdll` parsing with focused local images.
4. Repair newly resolvable dependency units from the saved edge ledger.
5. Fix decoder/semantic families occurrence-by-occurrence with exact roundtrip
   and state tests, repairing affected cached owners only.
6. Classify and lower indirect call/jump families without deleting dynamic
   semantics.
7. Re-materialize from cached units and write an aggregate checkpoint.
8. Run the full recursive entry only when the cache audit predicts reuse of the
   v4 functions. The launch should explicitly report reused/repaired/new
   function counts before expensive work starts.
9. Success requires zero effective blocker occurrences,
   `machine_state_complete=true`, and `repository_ssa_complete=true`, followed
   by execution/equivalence validation of the compiled `compile()` surface.

The v4 run bought a valuable 32,114-function map of the problem. The next turn
should reduce its exact blocker ledger while preserving that map, not spend
another 4 hours 42 minutes rediscovering it.
