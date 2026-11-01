# Control IR before numerical carving

Process operations now cross the dual-IR boundary before structural folding
and numerical region carving. The existing source compiler entry exposes
`schedule_preference="asap" | "alap"`; it forwards to the existing deployment
scheduler. No native worker/lane selection is introduced.

At source conversion, `source_control_ingestion.py` records evaluation ports,
control ownership and flow/branch/loop-repeat edges. Its AST use is confined
to that ingestion writer. `control_activity.py` reads these concordance facts.
It does not recover ownership from AST, source lines, or method spelling.

Immediately after source topology reduction and thread/capture ingestion,
`source_control_ir.py` creates actual `ControlProgram` blocks. ProcessGraph
process calls become their registered semantic operators, including
MemoryBarrier, ThreadCreate, ThreadStart and ThreadJoin. Dispatch blocks carry
exact process identities, operand-graph cells, synchronization contracts and
thread targets. Structured `SourceRegionBlock` containers preserve branch and
loop ownership while header planning remains deferred. Their control blocks
and placements are posted through the incumbent control-program writer.
`SOURCE_CONTROL_IR` and `SOURCE_CONTROL_OPERATOR` retain those exact cells.

The carver consumes these control-IR operator identities. Its existing
partition mechanism cannot put an operator's input computation and work after
that operator into one indivisible numerical region. Reachability in the
source activity graph supplies the frontier; no source-coordinate comparison
or inferred synchronization from shared storage supplies it. Region creation
records the constraint and refuses a crossing. The audit checks that proof.

Later control placement projects current concorded operands onto the ingested
block. It records the projection as a control rewrite. It does not introduce
the operation or re-resolve a thread target from cached attributes. Control
facts now include operand-graph and ingestion-control references in their
provenance, including references carried in their extra fields.

A compilation-unit cut roots the activity/control/operand/literal contracts
explicitly. The acceptance writer no longer walks AST objects to discover
Ref roots. The reader's explicit IdentityBook is passed through the shared
node-identity lookup rather than silently consulting another ambient book.

## Measured

The two previously failing source assertions for composed fence-mask producer
placement and conditional/repeated synchronization now pass unchanged.
The focused C/thread/table/synchronization/sequence/shell/activity gate passes
60 cases. New acceptance cases compile the same source under ASAP and ALAP,
verify unchanged activity edges and separate worker-function graphs, and run
standalone C executables. Two workers return2341 under both configurations.
A sealed unit with real storage, a computed release-fence mask and a later
read compiles after graph AST metadata is removed and returns12 under both.
This does not establish compiler-wide AST independence.

## Remaining boundary

Some legacy conditional/loop header planners still read source AST. They must
consume the structured source control IR and exact dependency ports next.
Admitted exception forms are covered by the ingestion follow-up below.
Condition resource scopes are covered by the follow-up below; mutex/custom
With scopes remain unadmitted. The run-to-join region and paired Python Barrier.wait control/
memory projections remain unfinished. No new native rendezvous lowering or
LLVM exception/thread implementation is claimed.

The remaining fence/barrier policy must be keyed by exact source operation,
execution scope, memory scope, ordering/visibility bits, participant/generation
identity and storage-effect set. Control rendezvous and memory ordering remain
distinct projections of one source identity when combined. Existing C11 host
fence policies are the verified subset; unsupported scopes/visibility masks
remain refusals, not broadened global fences or automatic locks.

Exception ingestion follow-up (2026-11-01): admitted try syntax now converts at
ingestion into explicit native record operations and conditional/loop arms.
The temporary AST is consumed by normal graph ingestion; synthetic execution
positions are not restored. SOURCE_EXCEPTION_INGESTION/NODE/OPERATION join the
conversion, generated graph identities and early placed control blocks.
Break/continue capture and loop ownership use SOURCE_LOOP_CONTROL_SITE and
SOURCE_LOOP_SITE_EDGE, with SOURCE_LOOP_SITE_VIEW for local graph projections.
SSA branch writers publish SSA_CONTROL_EDGE, and SSA_ARM_CONTINUATION proves
whether an arm's current tip is reachable. An empty successor list on an
unreachable placeholder no longer causes an exited arm's bindings to escape.
The detailed measurements and remaining limits are in
`docs/hazards/20261101T013030Z_work.md`.


Condition resource follow-up (2026-11-01): authored With scopes now register
SOURCE_RESOURCE_CONVERSION/SCOPE at ingestion and materialize ResourceScopeBlock
in early control IR. SOURCE_RESOURCE_CONTROL records exact entry/cleanup
operations and the same captured receiver. Body-to-cleanup ordering is a
SOURCE_ACTIVITY_ARM_RELATION, not an edge between exclusive branches.
SOURCE_LOOP_CONTROL_WORK retains loops containing process/exit work even when
they have no numerical regions. The three handwritten IR tests now use authored
source; cleanup cases also pass accepted-graph resume and inspect actual CFG
cleanup order. Condition C emission remains unsupported and explicitly tested
as a refusal. See `docs/hazards/20261101T024049Z_work.md` for measurements.
