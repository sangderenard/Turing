2026-11-01 Bootstrap reconciliation: first repaired slice and measured frontier

Scope and environment
---------------------
Continues the cumulative threading/control/deployment work into bootstrap.
Worked in /workspace/Turing, work at 3eeab1bf, cloud counterpart of canonical
C:/dev/Powershell/turing. Preserved dirty/untracked work; no reset, commit,
installation or native-product activation. Read root/compiler AGENTS and latest
hazard first. External string-core arrangement remains excluded.

Used PYTHONPATH=/workspace/turing-deps, CC=gcc, CXX=g++,
CFLAGS=-Wno-error=incompatible-pointer-types. Installed llvmlite 0.50.0 into
that external dependency directory after two tests initially failed to import
it; no project version pin was found. Fortran regression used the existing
/tmp/turing-fortran-toolchain/usr/bin/gfortran-14. The optional nodus tensor
core is absent; tests warn and use existing Python tensor backends.

Baseline and attribution
------------------------
The 12-file bootstrap/provenance battery from the intention report initially
had 5 failures, 108 passes in 33.18s:
- Two actual source-pursuit cases failed MissingExceptionControlTopology on
  standalone Raise in pursued compiler helpers. Both pass on untouched
  3eeab1bf (/workspace/scratch/turing-thread-baseline): genuine regressions.
- Descriptor-adapter test supplied an obsolete Product stub without root.
  Also fails untouched baseline; current reader requires persisted native
  verification evidence.
- Two record writer tests initially failed importing llvmlite. With that
  dependency available, both current and untouched baseline fail on an
  untyped module catalogue node, integer ImportFrom.level=0.

Before edits, mapping audit: 28 rows, 2 functions, zero findings, zero unsourced
facts/identities, latch CLOSED, 1.2s. This was a measured current run, not a
historical receipt.

Repairs retained
----------------
1. source_exception_ingestion now converts standalone raises, even without a
   surrounding try, through the same exception identity/payload writer used by
   protected raises. Existing SOURCE_EXCEPTION_INGESTION conversion provenance
   precedes installation; generated operations flow through existing process,
   exception and control concordance records. No raw-Raise admission exemption.
   Existing SSA exception_raise already supplies cleanup, branch to fault Trap
   and unreachable continuation: no second error engine was added.
2. Nonliteral standalone raised messages survive as exact graph operands to
   ExceptionMake (signature zero or one input). They are not late AST payloads.
   C explicitly refuses a dynamic message pending an owned native-message ABI,
   and posts that refusal to C_EXCEPTION_OPERATION. Static messages retain the
   existing native error-packet and shell-channel route.
3. That dynamic-message case exposed a shared glyph formatter writer defect:
   the generated decimal_integer_glyph_tokens Call retained exact glyph origin
   and callee_ref but had no SOURCE_ACTIVITY_BINDING. The writer now derives
   binding from the exact source glyph activity, recording source identity,
   activity binding and boundary cells. The placement reader remains strict;
   it does not infer placement from operands, names or source positions.
   control_activity.binding accepts the writer's already declared scope before
   the function graph publishes lexical_read_scope. Ordinary readers retain
   the existing strict scope lookup.
4. The descriptor-adapter unit test uses the real ProjectCompilationProduct
   and persisted verification receipt layout. Its verifier remains mocked;
   this verifies routing only and does not claim native verification.

No dtype defaults were installed to conceal the record failures. An attempted
raw scalar/syntax classification patch merely advanced the catalogue failure
to actual self.a+self.b / multiplication nodes with no bound method ABI. Those
trial edits were removed. The remaining issue is source-catalogue versus
executable-body ownership, not permission to assign arbitrary arithmetic types.

Actual bootstrap attempt
------------------------
Plan-only command:
  python -m tools.bootstrap_compiler_whole_program --plan-only --output build/compiler-bootstrap-reconciliation
Recorded 10 boundaries, provenance-core first (initial plan fingerprint
502954e13f4f, before the repairs). The status tool expects exponential
bootstrap-state.json, not plan.json; the intention report now says so.

One bounded real stage attempt:
  python -u -m tools.bootstrap_compiler_concordance --output build/compiler-bootstrap-reconciliation/provenance-stage-first --jobs 1 --stage-units 1
Failed during ingestion before a source unit was compiled. Failure log:
artifacts/identity_logs/partitioned_source.20261101T192717.failed.log.xz
Error: native try: temporary has no declared basic scalar result type.
An ingestion-only trace of current identity_concordance.py identifies
IdentityBook.neighborhood.newest, lines 9114-9118:
  try: return self.stamp_of(item[0])
  except Exception: return -1
stamp_of has an int annotation, but this ingestion reader does not resolve an
exact bound-method result contract. newest has no declared return annotation.
The two protected return kinds therefore do not settle to one admitted scalar.
Additionally, ordinary protected calls need a declared exceptional continuation;
a guessed return dtype alone would not complete the semantics.

Progress is NOT a native bootstrap: no unit emitted, native compiled, verified,
installed or selected in a fresh process by this attempt. Plan and orchestration
unit tests must not be counted as successful native compiler execution.

Concordance audit after repairs
-------------------------------
Ran all seven existing audit cases and compared failing cases against untouched
3eeab1bf using the same dependencies:
- mapping: 28 rows / 2 functions, zero findings and unsourced identities, CLOSED.
- view: selected return lacks exact source call reachability authority; same
  failure on untouched baseline.
- energy: final concordance refuses two materialization dtype='int' versus
  exact integer domain='int64' facts; same on untouched baseline.
- controller: _restore_type input np lacks source_value_class/scalar dtype;
  same on untouched baseline.
- controller_untyped: pi_update dt_pen lacks authored/call-edge native ABI;
  same on untouched baseline.
- oscillator: root steps lacks settled dtype and incoming call record;
  same on untouched baseline.
- toplevel: current stops in exception ingestion (unknown temporary scalar
  type); untouched baseline gets farther and refuses _propose_dt_pen sqrt's
  dtype. This is a changed earlier failure frontier, NOT a fixed audit case.
The full audit exits 1. No audit acceptance or migration exemption was added.

Verification
------------
Final combined bootstrap/provenance + source C exception battery: 154 passed,
2 failed in 255.76s. Both failures remain the pre-existing record catalogue
cases above, on integer node 22317496 in the administrative module graph
(function_name=None). The original 113-case group had also been repeated
separately: 111 passed, 2 failed in 38.51s. These are overlapping gates, not
additional unique passing cases.
New standalone C tests cover both normal result and uncaught static raise
(return code 7, type/message on stderr, no final output), plus retained dynamic
message operands and explicit C refusal. These are public source-to-SSA and
separately compiled executable tests, not a constructed graph or in-process C
runner.

Retained capability gate: 44 passed in 111.64s across dual-IR sync,
shared-location threads, source C threads/synchronization, mixed LLVM/Fortran
C-shell sections and native OS thread counts. No strict xfail removed. Python compilation checks for changed compiler/test
files and git diff --check passed.

Exact next implementation boundary
----------------------------------
Carry exact callable/result and failure-continuation contracts through source
ingestion into concorded call/control nodes before protected-call lowering.
For newest, prove captured self's bound method and stamp_of's actual annotated
result through existing pursuit/binding identity, without guessing from method
spelling or the fallback return -1. Then use the existing exception channel and
cleanup edges for protected internal calls. Aggregate return/storage ownership
and dynamic message lifetime remain explicit ABI work; do not call arbitrary
calls safe or substitute opaque/None/default integer types.

Separately reconcile whole-source catalogue ownership with the existing
selected-unit path, which already uses extract_clean_process_subgraph for an
administrative root. A whole-source compile must preserve real module effects
and all method exports; simply emptying every root or skipping its errors would
be unsound. Keep both remaining record tests visible while resolving that
shared boundary through concordance. After these repairs, repeat the bounded
provenance stage, verification/install/fresh-process activation, and only then
advance the outer compiler partition. Wider audit failures above remain work.
