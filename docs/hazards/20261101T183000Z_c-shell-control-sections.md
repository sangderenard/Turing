2026-11-01 Explicit C orchestration around LLVM/Fortran AOT compute sections

Read compiler instructions and newest deployment inventory hazard. User asked
for an explicit option approving C-style control around non-C portions, with
native backend refinements kept distinct. Earlier JIT-adapter citations did not
establish the AOT route; implementation uses SSA emitters and standalone AOT
C artifact compilation only.

Implemented CShellControl(exact compute functions, optional build directory),
accepted by LLVM emit_ssa_function_to_llvm and Fortran emit_module. Defaults are
unchanged. Opt-in returns a CModuleArtifact using the existing C native-section
call ABI and existing thread/exception/synchronization materialization. Original
SSA bodies, function identities, control edges and storage are retained.
C_SHELL_SECTION_DEMAND derives from accepted SSA geometry before native emission;
C_SHELL_SECTION records exact snapshot/native ABI/source/object facts. C reads
and validates this cell and derives its call-emission unit from it. LLVM piece
source/build provenance is routed through ArtifactEmission.piece_sources.
Fortran uses actual ingested Function objects through the existing mapping
emitter, plus existing packed ABI source generator. Planned source/command and
completed object digest are separate revisions of C_SHELL_SECTION_BUILD, and
final C link checks native object digests.

Observed failures and repairs:
- Standalone LLVM native section compiled, but emission audit refused unrouted
  PIECE_FILE. No exemption: ArtifactEmission now accepts explicit source cells
  for linked pieces, populated from the selected section receipt. Absent source
  cells retain the original refusal.
- First test source called compute inside try: ingestion correctly refused the
  absent ordinary-call failure-return contract. This slice does not invent one.
  Final test has compute before the protected division inside the worker loop,
  with declared float-return type and typed local; try catches the checked
  division, finally waits at the authored barrier. Generic protected native
  compute calls remain a stated next boundary.
- A shallow one-function IRModule retained unrelated sequence tables and was
  refused by Fortran's reader. Removed that partial container: pass actual
  admitted scalar Function objects using the emitter's established Mapping
  entry, not a replacement or stand-in SSA body.

Initial native section admission is explicit scalar leaf numerical work;
Fortran float64 only. Unsupported operations/aggregates/extents/calls refuse.
No arbitrary block splitting, native coarrays, blanket native exception support
or source execution fallback claimed. Existing C control capabilities/refusals
remain authoritative. Hardware role is recorded by selection facts rather than
new duplicate control opcodes.

Verification toolchain: system GCC14.2.0-19; extracted matching Debian
14.2.0-19 gfortran and libgfortran-dev packages under /tmp/turing-fortran-toolchain
(no system installation), TURING_FC selects that compiler. Intermediate LLVM
source gate3passed/2skipped45.55s before Fortran toolchain, then Fortran2passed
30.31s after correct mapping emission. A final combined current-code gate and
measurements follow below.

Completed verification for the mixed section slice:
- Native barrier/thread/exception plus LLVM comparison and Fortran control
  regression gate87passed,1optional-nodus-warning351.51s. The extracted matching
  gfortran compiler executed the Fortran checks; no skip in that gate.
- Post-emission measurements (initial two-iteration source): both mixed targets
  complete=True/thread_runtime=True; selected actual numerical function
  root__compute__specialized_f42764097ead__planned_region_0. Each audit237rows
  across11functions,0findings,0unsourced,latchCLOSED. Fortran build receipts
  distinguish .f90 and packed .c object compilation.
- Follow-up deployment/count gate44passed97.80s, including existing outliner,
  deployment configuration and native thread regressions. Later strengthened
  wrapper tests use1000barrier phases and OS task counting, not just a result
  check: final combined thread/mixed-source tests11passed81.43s.
- Final passive policy refinement reads the exact ingested CONTROL_BLOCK loop
  schedule through its SSA header's recorded owner. Region keeps that ASAP/
  ALAP setting and source loop identity. Six native count tests passed8.38s
  after this refinement. Mixed section code is unchanged by that refinement.
- Mapping audit28rows/2functions,0findings,0unsourced,latchCLOSED before and
  after; py_compile and git diff --check pass. Warning is optional nodus absent,
  not Python execution of the standalone compiled product.

External string-core question was investigated separately: reduced native
regressions/source-identity tests do not establish full-core source compilation.
Bootstrap activation is early, before ingestion, but activates/re-verifies
registered products rather than automatically building string core. Selected
registry and sibling source are absent here. User identified the external
sangderenard/nogodsnomasters arrangement and explicitly excluded setting it up
from this task. No bootstrap or full-string-core success is claimed.

Final strengthened-source measurement: both LLVM and Fortran1000phase versions
retain the exact selected numerical function above, artifact complete, native
thread runtime required,237auditrows/11functions with0findings/0unsourced.
Linux OS observer receipts show exactly2native tasks in each mixed run.
Subsequent one-core/passive-count additions8passed9.98s; mixed code unchanged.
