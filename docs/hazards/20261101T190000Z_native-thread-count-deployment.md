2026-11-01 Native thread-count verification and general passive C deployment

Cumulative request: retain mixed C-control LLVM/Fortran work and add native OS
thread counts, including thread-free source that deployment makes two-threaded.
Compiler and C-backend instructions read. No synthetic stand-in SSA graph,
Python execution fallback, injected C executable or manager thread introduced.

Observed current gaps before this work:
General C SSA emission used a fixed seven parked workers. Automatic iteration
outlining was called by the vehicle-specific path; a simple typed source loop
had canonical SSA but no deployment region. Existing outliner rewrote source
SSA without posting its changed geometry. Existing synthetic tests did not
exercise this general source path. These are observations of current code,
not conclusions drawn from old run logs.

Implemented:
- A source-SSA discovery step in the existing deployment_outlining module.
  Narrow proof: ingested counted loop, start0/step1/fixed positive bound equal
  to one rank-one span's extent; actual call/formal edges lead only to scalar
  numerical leaf computations and stores at that exact induction. Shared
  reads, arbitrary calls, extra stores/aliases, unproved bounds and exceptional
  control do not qualify. Failure-capable integer arithmetic is not admitted
  as a supposedly pure floating computation.
- SSA_DEPLOYMENT_DISCOVERY derives from actual source/SSA geometry and the
  loop's exact CONTROL_BLOCK cell. It preserves the ingested schedule choice
  and source loop node. Only after this receipt are Deploy/Join and lane
  memberships applied to the actual source SSA.
- Existing _outline_one performs its existing checks on a proposed subgraph
  copy. SSA_DEPLOYMENT_OUTLINE records before/after topology and the complete
  actual outline record BEFORE applying the new functions/edges. Current
  geometry and block owners are then posted from that rewrite receipt. New
  formals retain original value identities under the parent's value scope;
  no fabricated worker body or spelling match is used.
- C emitter reads and validates that record AND current parent/outlined edges
  before using the outline. Its core-budget choice uses existing
  select_deployment_strategy, recorded by C_DEPLOYMENT_POLICY and consumed
  as a cause of C call emission. No one-off mutation exemption.
- General C emission calls discovery/outlining under existing work contract
  deployment='auto'. `deployment_cores=2` gives one parked pool worker plus
  caller. Serial baseline and shared-read loop remain serial. Budgets of one
  select the existing serial strategy; invalid budgets are refused before
  graph mutation. Legacy default budget remains seven parked workers.

Native tests observe ONLY each standalone child through /proc/<pid>/task,
without rewriting its binary, injecting callbacks or counting the test runner:
- two authored workers -> peak3 native tasks including main;
- no authored Thread operations, auto deployment/cores2 -> peak2, one worker
  plus main; identical serial source -> peak1;
- shared-read source under auto -> peak1, no pooled region;
- C wrapper + LLVM or Fortran numerical section -> peak2; native output matches
  successful division and caught zero-divisor results. Both schedules pass.
The observer saves expected/max counts, observed set and sample count beside
compiled products. Runtime count tests currently require Linux /proc; native
Windows thread-count observation is not claimed.

Verification: initial count-only3passed5.12s. Deployment/mixed/outlining/lowering/
thread combined gate42passed95.24s; adding both scheduling configurations gave
44passed97.80s. After strengthening mixed tests to1000phases with OS counting,
11native/mixed tests passed81.43s. After exact loop-policy/source identity
preservation,6native-count tests passed8.38s. Every run's warning was optional
nodus tensor core unavailable, not a product execution fallback.

Remaining boundary: discover broader independent loop/effect shapes only with
matching proof and transport contracts; do not broaden this unit-stride proof
into an alias analysis or communicating-worker serialization permission. Native
LLVM/Fortran lane/coarray mappings and richer numerical section ABIs remain
separate port work described in the shared operator reference.

Final measured identities and receipts:
- Passive source SHA99650523b622ae70ef3116aa5750b533dd256db3fc2d0d85d41b738310b64ca7.
  root source loop6, span1; actual control owner is CONTROL_BLOCK(loop,
  canonical_value(lexical-read-fork(scope,1),6)). Header SSA_BLOCK directly
  derives from that owner, so the schedule is read from its exact extra[5].
- Discovery row(root__root,loop_header), outline row(root__root,0), C policy
  row(root__root,0), each column0 in the measured fresh compile. Induction
  2305843010213693958; bound IDs3955/3956/3957 under the same230584301021369 prefix;
  outline root__root__deploy_region_0_lane0, formal actual IDs(induction,
  2305843010213693962,1). Current C policy('pool',2,1,'caller_participates').
- Root5blocks/18instructions; numerical callees each1block/3instructions;
  actual outlined function2blocks/6instructions. Artifact complete with pooled
  root region0. Module audit26rows/4functions,0findings,0unsourced,latchCLOSED.
- Final1000phase mixed source still audits237rows/11functions per requested
  numerical backend,0findings,0unsourced,latchCLOSED.
- OS observation receipts from the final mixed gate: eight LLVM/Fortran success/
  caught-error executions each peaked at2 native tasks (21–30samples each).
  Two automatic source schedules peaked at2 (56/59samples); serial schedules
  peaked at1 (7samples each); shared-read refusal peaked at1 (7samples);
  two-authored-worker test peaked at3 (39samples).
- Added one-core auto budgets to test the existing serial strategy explicitly,
  under both ASAP/ALAP. Final count suite8passed9.98s, including exact source
  loop/schedule assertions and all output values; one-core budget observed1.
  No implementation changed for this additional test.
- Final mapping audit28rows/2functions,0findings,0unsourced,latchCLOSED(1.2s).
  py_compile and git diff --check passed. Source/string-core external layout
  remains excluded per user's clarification, not a forgotten obligation.
