# Windows parity receipt — 2026-10-24

## Scope and revision

This lane ran on Windows against a clean worktree made from fetched
`origin/main` at `d440cd27` (merge of PR #850).  The working branch is
`windows-parity-2026-10-24`.  No shared compiler pass and no
`lanes_plan.md` content was changed.

The words **observed**, **inferred**, and **unknown** below are deliberate.
Observed facts come from this run or an authored record.  Inferences are
labelled and are not promoted to facts.  Unknowns name evidence that the run
could not produce.

## Windows-only differences

| What | Where / records that disagreed | Evidence | Disposition |
|---|---|---|---|
| The Git Bash gate invoked `python3`, which is absent from the repository's sanctioned Windows Python installation. | `tools/gate/run_gate.sh` named `python3`; the checkout exposes `python`. | **Observed:** the clean gate stopped before collection with `python3: command not found`. | **Fixed** in `99575845`: invoke `python`. |
| A clean environment could not apply the gate's `--timeout` option. | Gate command requires pytest-timeout; `requirements.txt` did not declare it. | **Observed:** pytest rejected `--timeout=900` before collection. | **Fixed** in `ba061304`: declare `pytest-timeout>=2.4,<3`. |
| Checkout line-ending conversion inserted `\r` into baseline-list paths read by Bash command substitution. | The path record from `baseline_*.txt` disagreed with the filesystem path by one CR byte. | **Observed:** `cat` reported baseline filenames ending in `$'\r'`. | **Fixed** in `5fcc70e9`: require LF for gate shell scripts and text inputs in `.gitattributes`. |
| Validation-runtime symbols were present in the Windows DLL source but absent from its export table. | The C translation unit defined `turing_validation_error_reset`, `turing_validation_error_code`, and `turing_validation_error_detail`; `ctypes` could not resolve them from the DLL. | **Observed:** `test_a_bare_optional_field_read_stored_in_a_mapping_reads_its_presence` first failed on symbol lookup. | **Fixed** in `ef794189`: export the three authored ABI functions on Windows (and retain default visibility elsewhere). The test body then passed; **unknown:** that focused pytest process still returned 1 after printing `1 passed`, so session teardown remains unclassified. |
| Windows tool discovery recorded the executable suffix in a stable manifest identity. | The manifest's authored tool identity was `spirv-as.EXE` / `spirv-val.EXE`; the test and cross-platform schema require `spirv-as` / `spirv-val`. | **Observed:** only the spelling of the tool identity differed; execution and validation succeeded. | **Fixed** in `18b477b2`: pass the authored tool identity separately from the discovered executable path. Focused SPIR-V plus standalone BLAS tests: 5 passed. |
| The typed-corpus isolation harness requested the unavailable `fork` multiprocessing context. | `native_corpus_typed_check.py` hard-coded `fork`; Windows supports `spawn`. | **Observed:** the first Fortran corpus worker stopped before compilation with `ValueError: cannot find context for 'fork'`. | **Fixed** in `4a75a63f`: module-level worker entry, guarded main, `spawn` on Windows and `fork` elsewhere. |
| The whole-program bootstrap script was not directly importable on Windows when invoked by its documented path. | Script execution placed `tools/`, not the repository root, on `sys.path`; import of `src` failed. | **Observed:** exact documented command stopped with `ModuleNotFoundError: No module named 'src'`. | **Fixed** in `cc239e6f`: establish the repository root before importing `src`. |
| Three gate parts exceeded the authored 900-second per-test timeout before a part summary. | Windows timeout records named shared compiler stacks rather than a platform writer. | **Observed:** aa stopped in `shared_predicate_region__train` for `test_one_value_held_by_two_bindings_computes_the_authored_answer`; ac and rec stopped while `_synthetic_device_scalar_shell` grew `BitTensorMemory` for `test_declared_tuple_parameter_resolves_isinstance_schema_guard`. **Inferred:** Windows runtime cost exposes these shared paths sooner; no platform-specific writer was found. | **Handed off; not patched.** These stacks pass through shared planning/lowering code and are outside this lane. |

## Gate baseline

The gate was run from Git Bash in a second fresh worktree after the three
gate-launch fixes.  Part `ab` completed with 255 passed and seven failures;
the seven names are all present in the committed baseline, so its set
difference was empty.  Parts `aa`, `ac`, and `rec` did not reach summaries
because of the timeout records in the table.

Before the timeout, the only collected failure name outside the committed
baseline was:

```
tests/test_ssa_optional_values.py::test_a_bare_optional_field_read_stored_in_a_mapping_reads_its_presence
```

Its Windows DLL export disagreement is fixed.  The authored assertions now
print `1 passed`; the unexplained post-summary process status is kept unknown
rather than called green.

## Native corpus

With `TURING_ROOT` pointing at the worktree, the exact C corpus command
completed all 328 programs:

```
PASS=328
```

There were no C `FAIL`, `WRONG`, type, compile, or run records.

For PR #830, the corpus was then run with `TYPED_CHECK_BACKENDS=c,fortran`
using `C:\msys64\mingw64\bin\gfortran.exe`.  C remained 328/328.  Fortran
produced 327 PASS records and the already-authored c22 LUT-sine difference:

```
x=0.5:  got 3.0615908251258297, Python 3.0615908727406316
x=1.25: got 3.365176846666648,  Python 3.365176924612027
```

Owned-heap list programs c400-c403 all pass.  No Windows-only Fortran wrong
value, wrong type, compile failure, or run failure was observed.

## Recent cross-platform changes

### PR #849 — standalone BLAS C shell

`tests/test_glsl_backend.py`, `tests/test_glsl_blas_deployment.py`, and
`tests/test_shader_deployment.py` produced 206 passed, six skipped, and two
failures before the Windows tool-name fix.  The other failure is the shared
inventory record: the test expects 56 GLSL operations while the current
lowerable inventory has 58.  It is not a Windows-only result and was not
patched here.  After fixing the manifest identity, the focused deployable
SPIR-V test and the complete standalone BLAS test file pass (5 passed).

### PR #839 — whole-class artifact and private runtime

The exact catalogue command returned without a runnable whole-class artifact.
The direct class route refused at loop 164 because `self.trace(...)` is an
opaque state effect; the catalogue recorded eight failed and three partial
units.  The first shared stack is:

```
glsl_deployment_strategy._validation_control_blocks
  -> prepare_graph_precompile
  -> aot_compile
  -> CompilationSubdivisionRequired
```

Because no executable or staged private runtime was emitted, a different-
directory launch and loaded-DLL inspection were impossible.  **Observed:** no
artifact exists to run.  **Inferred:** the blocker is shared graph planning,
not Windows staging.  **Unknown:** which DLLs the eventual artifact will load
on Windows.  This is handed off without touching the shared pass.

### PR #830 — Fortran owned heap

See the corpus result above.  The owned-heap programs run under MinGW
gfortran; the only non-PASS Fortran record is c22's known, cross-platform
numerical law.

## Bootstrap and tire

The documented bootstrap command completed the current 358-unit plan before
correctly refusing incomplete enclosure assembly:

| Status | Count |
|---|---:|
| compiled-unverified (admitted) | 47 |
| partial | 72 |
| failed | 239 |

The admitted numerator matches the cloud receipt's 47.  The cloud receipt is
47/371 at an older main revision; this revision authors 358 units, so a
denominator or per-unit OS comparison is **unknown**, not an observed OS
difference.

The managed tire build reached the same shared stop recorded by the Linux
lane.  In `balloon_tire_membrane_face`, the posted `function_output` rows omit
the aliased names `pressure_force_{1,2}_{x,y,z}_n` and
`construction_force_{1,2}_{x,y,z}_n`, while `named_returns` retains those
names pointing to the same values as the index-0 names.  The writer is the
`function_output` page finalization in `precompile_to_ssa.py`; this Windows
lane did not edit it.  Since the build emits no manifest past that refusal,
the validator stops on the absent
`build/balloon-tire-managed-integrated-o0/balloon_tire_managed.manifest.json`.

## Commits

Each Windows repair is a separate signed-off commit with a receipt comment
at the repair site:

- `99575845` Make the regression gate runnable from Git Bash
- `ba061304` Declare the gate timeout dependency
- `5fcc70e9` Keep regression gate inputs LF on Windows
- `ef794189` Export the validation runtime ABI on Windows
- `18b477b2` Record stable SPIR-V tool identities on Windows
- `4a75a63f` Run backend corpus workers with Windows spawn
- `cc239e6f` Make the whole-program bootstrap directly runnable
