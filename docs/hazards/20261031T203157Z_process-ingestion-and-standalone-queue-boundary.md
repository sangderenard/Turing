# Process ingestion and standalone queue boundary (measured 2026-10-31)

Worktree /workspace/Turing, base 3eeab1bf plus ongoing uncommitted work.
The canonical Windows path is C:/dev/Powershell/turing; these measurements are
from its cloud clone. Root/compiler instructions and latest hazard were read.

## Correction to benchmark claims

The earlier handwritten native C queue workload was not a Python-to-SSA-to-C
benchmark. Its driver and C workload were removed. No speedup from that run is
evidence for source compilation. The unrelated PremixRing edit/test were also
removed; the independent source sequence-frame ownership fix remains.

The replacement input is examples/byte_queue/threaded_pipeline.py: actual
threading.Thread, Condition, Event and collections.deque, bounded channels,
multiple stages/lanes, copied byte payloads and tags, validation/checksum.
tools/benchmark_python_queue_c_shell.py uses lower_ast_source_to_ssa, the C emitter,
and compile_standalone, then launches the generated executable in a subprocess.
It compares the same source run by Python. There is no fallback on compile
failure. A scalar route smoke compiled/executed, but is not a queue benchmark.
The authored threaded queue program does NOT yet produce an executable.

## Observed ingestion defect and correction

The Python frontend generated turing_dispatch_* callable markers but left Call
nodes. Lexical reduction subsequently resolved the generated callee name as an
Input. Generic Dispatch SSA consequently obscured the semantic boundary. Also,
record_python_control_scopes treated enclosing calls as control scopes: the
actual threads.append(threading.Thread(...)) source acquired a Call/args scope
that the existing placement contract correctly refused.

The source frontend now selects declared ThreadCreate/Start/Join/IsAlive,
Condition*, Event*, and ByteQueueCreate graph operators from exact extraction
identity. Their operand schemas exclude func. Reduction resolves authored
operands only, records SOURCE_PROCESS_OPERATION from canonical source identity,
and keeps source control ancestry without treating calls as conditional owners.
Control placement retains this cell; SSA emits the corresponding declared opcode
and result contract, and the concordance audit checks that provenance, dtype,
and effect contract. Existing DispatchBlock is a control representation, not a
Python executor. The placement guard was not relaxed.

A second observed path resolved constructors inside source class initializers
only during static-reference reduction. Exact Python identity programs now route
these constructors through the same ingestion replacement. Handle field layout
is proven from the reduced producer and the existing field-declaration receipt,
not field spelling. A provisional process-result value must use produced_value
rather than replacing an already-bound source result with a fresh SSAValue.

Sequence record append also exposed premature physical-column conversion:
existing deferred record recipes now survive until their existing expansion
linker; physical argument arity is deferred only when the declared recipe width
exactly accounts for it. No heap-owner or immutable-storage refusal is disabled.

## Measurements

Focused source-driven thread ingestion/dual IR/shared storage/synchronization/
lock-box run: 39 passed, 1 strict xfailed, 16.32 seconds. The xfail still checks
missing worker-body SSA, not generic Dispatch spelling. Source lifecycle nodes
and SSA are ThreadCreate twice, ThreadStart twice, ThreadJoin twice; operands do
not include generated Python callee inputs. A new source Condition/Event test
also checks graph replacement and exact source-process receipts in SSA.

Existing threading-special-case file after adjusting opcode assertions:
7 passed, 3 failed, 3.31 seconds. Clean detached 3eeab1bf worktree:
7 passed, the same 3 failed, 2.87 seconds. They refuse unsettled root Add dtype,
unannotated root n and unannotated counter n. Node numbers differ (Add5 versus
Add9); the source failure and missing type contract are the same.

Mapping concordance audit still refuses before producing a report:
region value7, Add in root, no settled dtype (0.7 seconds). This agrees with the
before-change measurement in the preceding hazard; it is not a clean audit.
Focused successfully lowered source modules audit with zero findings.

Full authored queue compile command:
CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types PYTHONPATH=/workspace/turing-deps python tools/benchmark_python_queue_c_shell.py --inputs '{"lanes":2,"hops":1,"items":10,"capacity":2}' --directory /workspace/scratch/python-queue-source --rounds 1

Current actual refusal: incompatible physical call input root__root source4
(dtype int) versus root__root__ssa_sequence_42_fill formal2305843010213694575
(dtype int64), both exact builtins.int scalar/rank0 call-argument provenance.
This is before C emission. It must be traced at its dtype writers, not excused
by a mutation exemption or a handwritten C benchmark.

## Native foundations versus remaining source work

The bounded copied byte deque C runtime and its native-task tests are runtime
unit coverage only. Queue table uses an existing private condition monitor,
not a management thread. Logical worker IDs are endpoint metadata. Native
payload copy/publication, both deque ends, blocking/try/timed access, close/drain,
opaque record tags, zero-length records and buffer-size refusal are covered.
The prior focused native queue/pool/standalone-validation run passed34 tests.

Generated standalone C shells now service the existing validation ABI rather
than refusing checked arithmetic: reset, call, take fault, exit7 before outputs
or timings. Three actual source compile/subprocess tests cover success and
faults in warmup/timed calls. This establishes main-thread validation only.

Next exact boundaries: trace the sequence argument's int/int64 disagreement;
include the preserved exact thread target and closure in planned worker-body
SSA; then lower declared process operations and byte records through the C
runtime with explicit capture/lifetime ABI. The original worker-body strict
xfail remains. Neither native thread compilation nor barrier lowering is
claimed. Control rendezvous and memory ordering remain distinct concorded
operations; backend policy must select supported scope/order/storage combinations
and refuse unsupported ones rather than widen scopes.

Final source ingestion/sequence/standalone validation regression: 14 passed,
13.19 seconds. Includes all four actual source ingestion checks (including
Condition/Event), seven Python-to-native-C sequence fidelity cases, and three
standalone validation cases. git diff --check passes.
