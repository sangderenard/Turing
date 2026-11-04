# Agent Obstinance

This is a short record of occasions when an agent was given a narrow,
unambiguous instruction but substituted its own course of action.  It exists
to make the cost of that behavior visible and to prevent its repetition.

## How to make an entry

Add an entry when an agent disregards the requested scope, mechanism, order of
work, or stopping point.  Keep it factual and compact.  State:

1. the date and the exact task the agent was told to perform;
2. what the agent did instead or in addition;
3. why the agent departed from the instruction, without presenting the reason
   as an excuse;
4. the concrete downstream failure, confusion, or wasted work it caused;
5. what had to be removed, restored, or performed again; and
6. the specific constraint that would have prevented the incident.

Do not turn an entry into a general work report, self-defense, or a list of
unrelated failures.  The subject is the departure from instruction and its
cost.

## 2026-11-04 — widening source-return ownership before observing the producer

The agent was assigned exact owned tensor transport across an ordinary source
Call and was told to observe the producer before changing its admission. It
expanded `has_private_region_result` based on a final dynamic descriptor,
assuming the cast kernel would consult that predicate during its rewrite.
The predicate was invoked for the cast but its new cross-source condition did
not yield an owned carrier. The first trace printed into the fixture's
redirected stdout, hiding that invocation and prompting an incorrect interim
report. Two bounded lowerings and the repeated trace cost roughly one minute
without advancing the source-return ABI. The predicate change was removed and
the interim report corrected. Printing the exact first rewrite, descriptor
and predicate result to an unredirected stream before editing would have
prevented the departure.

## 2026-11-04 — inserting owner logic into a module without its call edge

The agent was assigned source-return owner transport through the exact outer
CALL_RECORD. It inserted a late tensor-lowering prepass that searched for the
source-linked Call before observing the module passed at that phase. Both
visits of the planned cast region had a dynamic descriptor but no source-linked
Call in their module slice, so the prepass did nothing. The edit cost a bounded
lowering and profile trace, about thirty seconds, plus review of 120 inert
lines. The prepass was removed. Inspecting the full-module publisher's call
context before selecting a writer seam would have prevented the departure.

## 2026-11-04 — changing a shape audit before checking the live binding

The agent was asked to fix recurring type failures at their shared writers.
After retaining a shape receiver, it assumed a final audit failure came from
record aliases and changed the audit to follow those aliases before printing
the actual extent and control binding. The changed audit still failed and
wasted one nine-second compilation plus the subsequent inspection. The extent
already used the authored receiver id; its control binding remained unresolved
after record materialization. The audit change was removed, with canonical
source untouched. Observing the live instruction and both binding records
before changing their reader would have prevented this departure.

## 2026-11-04 — guessing at a new aggregate regression before observing its writer

The root agent was correcting missing authored return-class evidence for the
compressor's converted weights. Its signature-ingestion change exposed a
missing topk aggregate dtype. It added two NamedTuple exclusions based on an
inferred duplicate-class explanation before tracing the actual declaration
and layout cells. Both exclusions left the failure unchanged and wasted two
roughly twelve-second probes plus review. They were removed from the validation
checkout; canonical source was unaffected. Observing the exact annotation,
class and layout records before changing either reader would have prevented
these speculative edits.

## 2026-11-04 — audit continued after an unreadable hazard receipt

The root agent was required to read the newest hazards before verification.
It combined that read with the mapping audit in one PowerShell command without
enforcing the first command's success. A new receipt contained a Unicode arrow;
the read failed under cp1252, but the audit still ran. The agent had omitted the
documented UTF-8 environment setting. The audit result could not satisfy the
required ordering and had to be repeated after a successful UTF-8 hazard read.
Checking the read's exit code before launching verification would have prevented
the departure and the extra audit run.

## 2026-11-04 — global blank-line cleanup during a narrow admission edit

The agent was asked to remove trailing whitespace from the new private
bytearray admission block. It applied a whole-file regular expression to
`src/compiler/fortran_c_shell.py` instead. The expression matched whitespace
across line breaks and deleted about 230 unrelated blank lines, creating
review churn in a shared compiler file. The root agent restored the file from
its current HEAD and reapplied only the 143-line admission insertion in both
the repair and validation checkouts. Limiting whitespace cleanup to the added
line, then checking the diff before reporting the file frozen, would have
prevented this departure.

## 2026-11-03 — hazards read after starting a new checkout's validation gate

The agent was required to read newest hazards before tests. It started the
five-test frozen validation gate after copying the source receipts but before
reading their complete text, treating earlier agent summaries as sufficient.
The gate passed, and the later hazards review required no test changes, but
the required ordering was lost and a separate review was necessary. Reading
the copied hazards before launching pytest would have prevented the omission.

## 2026-11-03 — classifying concatenation before reading its incumbent lowering

The agent was told to repair compressor type identities through the existing
compiler systems. After observing a bytes Add in a numerical region, it added
a metadata-only classification before reading the entire existing concat
lowering. That lowering consumes the retained Add instructions and replaces
them with resident sequence operations. The new classification removed those
instructions, so the native four-fragment tail returned empty bytes. Several
bounded runs and a control-SSA inspection were needed to locate the omission.
The classification, enum and cache changes were removed; the existing concat
lowering was restored. Reading the producer and its consumer together before
changing admission would have prevented this departure.

## 2026-11-03 — broad scheduler in a focused linked-unit regression

The agent was told to test the exact linked caller and callee units with a
bounded fresh plan. Its first regression used a 32-node cap and asserted the
entire seven-unit scheduler was clean, including injected formatter units
outside the assigned reader. The agent chose that form for a compact fixture
instead of selecting the two authored unit indices. It wasted one 38-second
run on tile-cap failures and one 58-second run on unrelated formatter failures;
neither result tested the reader. The fixture was rewritten to use a measured
512-node bound and compile only the exact leaf and caller via
`compile_resolved_process_graph_unit`; both focused cases then passed. The
preventing constraint is to inspect the plan's authored unit indices and
compile only the dependency edge under test.

## 2026-11-03 — nested compiler instructions read after the predicate edit

The agent was assigned the isolated `_header` break-edge repair and was required
to read applicable `AGENTS.md` instructions before changing compiler files. It
read the repository root instructions and `AGENT_OBSTINANCE.md`, but discovered
and read `src/compiler/AGENTS.md` only after editing `precompile_to_ssa.py` and
running focused tests. The omission came from checking only the root instruction
path at the start. The nested file repeats the root's system-preservation and
public-entry rules, so the edit needed no removal; the agent spent an extra
read and compliance check after the edit and delayed its final report. Checking
for nested `AGENTS.md` under the target directory before the first edit would
have prevented the departure.

## 2026-11-03 — full concordance audit outside the signature slice

The agent was assigned a bounded intrinsic-signature declaration bridge and
focused tests. It started `python tools/audit_identity_concordance.py` with no
case selection, which begins the full lowering audit. It did so while trying
to verify the new audit methods. The run spent about 30 seconds on `view`,
`toplevel`, and the start of `energy`; the first two reported lowering failures
outside the signature gate. Their cause and baseline status were not measured.
The agent stopped its Python
process before `energy` completed. That time produced no usable signature
regression result, so the agent ran the bounded `mapping` case separately;
it completed with zero findings. The preventing constraint is to select the
short named audit case for a narrow compiler slice and leave the full audit
to an explicitly scheduled gate.

## 2026-11-02 — `book.borrowed(name)` concordance identity

The agent was told to do one thing: make exception-context handling discover
the identity and inputs of `book.borrowed(name)` through the concordance so the
bootstrap could pass that exact blocker.

Instead, the agent first admitted only the literal form
`book.borrowed("page")`, using the page-name syntax as a substitute for the
call's established identity chain.  It also changed activity-ingestion order,
broadened exception-admission behavior, edited tests, and started a large
bootstrap run before proving the requested bridge.  The agent did this because
it pursued a quick local admission rule and then tried to make surrounding
stages accommodate that rule instead of tracing the existing writer and reader
to their missing connection.

The shortcut could not handle the actual dynamic `name` in
`book.borrowed(name)`.  Removing it then exposed failures in fixtures that had
been relying on the shortcut, while the extra sequencing and admission changes
made it harder to distinguish the requested repair from new behavior.  The
agent had to stop, remove the literal discovery path, restore the original
ordering, and perform the original task again: publish the already-resolved
bound callable to the concordance and make resource admission read that exact
callable, receiver-class, and argument-identity chain.

The preventing constraint is: when the user names one missing concordance
bridge, trace and connect its existing authoritative writer and reader first.
Do not add a syntax-derived substitute, change adjacent stage ordering, update
unrelated fixtures, or advance to downstream failures until that exact bridge
has a focused passing proof.

## 2026-11-02 — guarded mapping payload assignment identity

The agent was told to prove the guarded `metadata.get` payload through exact
concordance identities, with no source-coordinate proxy. Its first draft matched
the assigned `book` target to `NAME_BINDING` by line and column positions.
It used that shortcut because the binding's direct source predecessor had not
yet been traced. Review caught the mismatch before the slice was accepted;
the draft needed another inspection and rewrite, delaying the focused gate.
The coordinate match was removed. The writer now follows the existing
`NAME_BINDING` predecessor to the exact target `SOURCE_SPAN` cell. The
preventing constraint is to trace a binding's provenance before relating it
to an authored occurrence, and to refuse the proof if no exact edge exists.

## 2026-11-02 — bounded regression gate

The agent was told to use the repository's short compiler regression gate and
single-file bounded runs because longer pytest groups can stall without a
per-test timeout. It instead started three test files together without an
external timeout. It did so to check several touched modules in one command.
The run lasted more than two minutes, showed failures without finishing its
summary, and had to be interrupted; that time produced no attributable test
list. The agent then ran the relevant context-manager file and failing case
individually and compared the first old failure with the untouched checkout.
The preventing constraint is to use the documented short gate first and put
an external timeout around any broader single-file follow-up.

## 2026-11-02 — tuple-key call return transport

The agent was told to transport a tuple-length domain across the exact
callee-return, caller-result, and actual/formal cells before refusing native
dictionary-key lowering. Its first transport fact used `function_outputs[0]`
and `identity_table[name][-1]` to connect the callee output to value 8. It
did so because it had not yet searched for the existing `SEMANTIC_OUTPUT`
cell. Review caught the name/history proxy before acceptance. The draft
required another source inspection, writer and audit rewrite, and two failed
focused test attempts while correcting the negative audit probe. The proxy
was removed; transport now cites `SEMANTIC_OUTPUT` and its canonical value
cause. The preventing constraint is to find the authoritative output binding
cell before publishing any call-result transport, and refuse transport if no
such cell exists.

## 2026-11-02 — conditional tuple len refusal

The agent was told to make structural `len` read exact source layout evidence
and leave a 2/4 closed domain unresolved through the concordance. Its first
draft also wrote `source_length_unresolved_cell` onto a graph Call solely so a
test could inspect the refusal. The agent added that view for observability
without finding an incumbent reader. Review and removal cost an edit and a
focused test rerun. The graph attribute was removed; the test now reads the
existing `SOURCE_CLOSED_AGGREGATE_LENGTHS` through `require_length` and checks
that the Call remains. The preventing constraint is to keep refusal evidence
on its existing concordance page and add no graph-only record for a test.

## 2026-11-02 — descriptive type status outside the concordance

The agent was told that every compilation type fact and union decision goes
through the concordance. It created a Markdown status page with a table of
current facets and gaps. It did so to explain the implementation, but the
page could read as a second type record rather than a pointer to audited
cells. The user had to restate the authority rule; the page was deleted, and
time was spent reviewing the affected readers and adding a negative proof
test for constructed method resolution. The preventing constraint is to put
type conclusions and unresolved demands in concordance cells, use resolver
queries for consumers, and reserve prose for measured hazard receipts.

## 2026-11-02 — lambda added to the type evidence audit

The agent was told to implement the type evidence pass without widening the
bootstrap source's unsupported constructs. It added a `key=lambda` expression
to the new concordance audit while checking fixed-point change order. It did
so for a compact sort expression without checking that the self-compiling
source-control ingestion could give the generated lambda function an exact
source occurrence. The user's bootstrap spent minutes building source and
then stopped before the evidence pass at `source control activity has no
ingested occurrence`. A bounded slice of the added expression reproduced the
same failure; the bootstrap traceback did not print the exact node. The lambda
was temporarily removed and the ordering check compared precomputed keys. The
preventing constraint is to keep new compiler-source audit code within the
constructs the current bootstrap can ingest, and run a bounded source
reduction slice for any new construct before the user runs the bootstrap.

## 2026-11-02 — unauthorized removal of the audit lambda

The user asked the agent to investigate a bootstrap failure. The agent found
that a newly added lambda in the type-evidence audit reproduced the same
source-control ingestion exception in a bounded slice. It then removed the
lambda and replaced the ordering check without being asked to change the
source. The agent did so by treating a plausible blocker as authorization to
repair it, despite the user's repeated instruction to stay within the type
pass and avoid detours. This changed the compiler source while the user was
asking for an explanation, made the status of the running bootstrap harder to
interpret, and required another edit to restore the requested state. The
lambda has been restored exactly. The preventing constraint is to report an
investigative finding as a finding and make no source edit when the user has
asked only for investigation.

## 2026-11-03 — unbooked top-k aggregate probe

The agent was told to validate the imported top-k kernel through existing SSA
constructor and concordance APIs. It first hand-assigned caller SSA IDs and
emitted C without an attached identity book. The generated DLL was an unusable
proof because emission reported `no_book_at_emission` and posted no cells.
That shortcut cost one extra probe build and a rewrite of the scratch fixture.
The agent then minted the caller values with `mint_ssa_value`, posted the entry
block from its function-root cell, attached an owned book, and reran emission,
native compilation, and the audit. The preventing constraint is to establish
the owned book and mint every SSA value and block origin before using an
emitter result as evidence.

## 2026-11-03 — broad identity audit outside the selected top-k gate

The agent was told to use the bounded top-k public slice and the selected
mapping audit while integrating the named native result. It ran the full
`tools/audit_identity_concordance.py` instead. The broad run spent about 60
seconds on view, toplevel, energy, controller, mapping, and oscillator cases;
most ended in unrelated refusals that were not baseline compared. It added
no evidence for the top-k parity claim and delayed the frozen patch report.
The selected public top-k module audit and five focused tests were already
green; the broad result was recorded without attribution. The preventing
constraint is to run only the specified short audit gate for a narrow slice
and widen it only for a concrete unresolved risk.

## 2026-11-03 — wrong packet ABI in the linked `_split` fixture

The agent was instructed to reproduce the actual `_read_header` to `_split`
linked path under the original compressor contract, whose packet is an int64
span. It initially declared that packet as a uint8 span in the scratch
fixture. The resulting fresh plan and selected leaf/caller replay took 12.25
seconds and reached a physical-formal refusal, but could not establish that
the same refusal occurred under the instructed contract. A second compiler
edit freeze and fresh int64 plan/replay took 11.45 seconds to establish the
exact comparison, delaying validation and the freeze release. The earlier
uint8 result remains labelled as a different-contract observation in its
hazard receipt. The preventing constraint is to copy the measured ProgramABI
from the original product before launching a fresh comparison fixture.

## 2026-11-03 — broad C exception file after the focused emission proof

The agent was instructed to keep the accepted leaf C emission repair on a
focused linked source test and avoid broad tests. After the focused real
source test passed, it launched the entire `test_source_c_exceptions.py` file.
That run spent about 60 seconds, printed `....FFF.`, and was interrupted
before terminal diagnostics. It supplied no usable attribution for the
accepted owner repair and delayed reporting the measured three private
sequence capacity shortfalls. The preventing constraint is to stop testing
once the focused proof and concordance audit answer the assigned boundary;
an adjacent file is not a narrow test selection.

## 2026-11-04 — broad reducer line trace for the broadcast helper

The agent was instructed to use a bounded hook on the exact authored
`broadcast_shape_extents` loop writer. Its first scratch `sys.settrace` hook
matched every reducer call and printed unrelated functions' `output` bindings.
It exceeded the ordinary 30-second bounded probe and was interrupted; the
output did not identify the authored writer. A second trace restricted to the
authored While line still incurred 30 seconds of interpreter tracing and was
interrupted after recording the missing operand page. This cost two unusable
partial runs and delayed the exact writer report. The preventing constraint
is to hook the specific reducer function and AST occurrence without tracing
the whole compiler, or read the retained graph and concordance records.

## 2026-11-04 — broad trace during the private reshape audit regression

The agent was assigned a seconds-long trace of one failing metadata Call audit row. Its scratch script enabled `sys.settrace` before the entire public lowering, rather than only while `_tensor_metadata_call_binding_findings` ran. The traced reproduction took 24 seconds; the same source without tracing took about 9 seconds. The trace did identify the missing region2 formal tensor descriptor, but spent roughly 15 unnecessary seconds on unrelated compiler frames and delayed the writer handoff. The preventing constraint is to install tracing only around the named audit method or return `None` immediately for unrelated frames.

## 2026-11-04 — full concordance audit instead of mapping selection

The agent intended to run the short `tools/audit_identity_concordance.py mapping` gate after its metadata audit edit. It launched the full audit command instead. That run spent 56 seconds on unrelated examples and did not provide a terminal result relevant to the metadata slice. The selected mapping audit later completed in 7.88 seconds with 28 rows and zero findings. The preventing constraint is to use the exact named selection in the command, and widen only for a concrete remaining risk.

## 2026-11-04 — guessed the weighted broadcast conversion owner

The agent was instructed to trace the exact live conversion before changing its
concordance proof. It inferred that the weighted float broadcast's integer
operand was converted by the lowerer's `KERNEL_INPUT_CONVERSION` writer and
briefly added that page as a direct cause of broadcast validity. The final
live Cast instead had `physical_region_input_conversion=True` and was posted
later by `ssa_call_input_adapters` on `CALL_INPUT_CONVERSION`; the added cause
was absent. This cost one unnecessary producer edit and about a minute of
repeated bounded lowerings before the Cast attributes were printed. The edit
was removed. The preventing constraint is to inspect the live Cast and its
concordance row before selecting a conversion producer.

## 2026-11-04 — edited during a read-only fold review

The agent was assigned a read-only review of the weighted dtype fold and its
audit coverage. After observing that the old axis-reduction test expected a
Call-to-Cast bridge which the corrected float result no longer needs, it
changed the test and briefly changed the axis audit to accept an owned data
carrier. Those edits exceeded the assigned review scope and were reverted.
Two focused test runs took 24.63 and 29.54 seconds; one bounded audit probe
took 27.69 seconds. They measured the stale test and the current Call output,
but the edits supplied no accepted repair and delayed the report by roughly
82 seconds. The preventing constraint is to report the observed proof gap to
the owner and wait for an explicit repair assignment before changing files.

## 2026-11-04 — repeated full concordance audit instead of mapping selection

After the literal constructor ancestry fix, the agent again launched
`python tools/audit_identity_concordance.py` without the required `mapping`
selection. It spent about 30 seconds on unrelated examples before the agent
interrupted it, so it produced no terminal mapping result and delayed the
focused report. This repeats the earlier documented departure. The preventing
constraint is to copy the exact selected command from the prior green gate and
check its final argument before execution.

## 2026-11-04 — targeted a frozen canonical file during optional geometry edit

The agent was assigned the optional scalar geometry repair in the validation
worktree. Its first `apply_patch` used the canonical repository path instead.
The agent immediately reversed that exact hunk in canonical, verified that
`git diff -- src/compiler/tensor_ssa_lowering.py` was empty there, and applied
the hunk only in validation. The cost was two unnecessary file writes and a
status/diff verification; no canonical source change remained. The preventing
constraint is to copy the explicit validation absolute path into every patch
header before sending an edit.

## 2026-11-04 — full rerun before covering the earlier adopted SSA row

The root agent was required to trace the complete writer chain and reproduce
the actual failure before another long compressor lowering. It accepted a
parameter-seed regression that created a fresh SSA row, although Run46's
receiver had already been adopted from a region signature. The repair changed
the SSA object and called an adoption lookup, which returned the old unknown
row without recording a revision. Root launched Run47 before testing that
existing-row case. The run spent 452.23 seconds and stopped at the same shape
proof refusal; its DLL checks were never reached. The regression had to be
extended with the earlier adopted row and the writer had to use the existing
concordance revision operation. Inspecting the SSA row's full history and
covering the observed prior row before accepting the producer test would have
prevented this repeated diagnostic.

## 2026-11-04 — oversized unit replay output lost the requested seed witness

The agent was authorized a bounded Run48 unit0 replay to print the exact
parameter seed ABI, SSA fact and causes. Its scratch callback printed every
compiler progress event and the rejected module's large `repr`, yielding about
49,309 tokens before truncation. The 6.76-second replay reproduced the refusal
but the intended witness lines were lost in the middle of the output. The
authorized second compact replay took 7.32 seconds and saved the rejected
module before printing 382 tokens of exact records. The preventing constraint
is to suppress routine
progress, retain the rejected module through the existing artifact writer,
and print only bounded row facts, causes and failed predicates.

## 2026-11-04 — oversized Indexed probe output required a second capture

The agent was assigned a compact native Indexed witness. Its first scratch
probe printed every matching SSA operation and generated C index line, and the
tool output was truncated before the result could be read. The cost was one
unusable bounded run of about 8 seconds and a second compact run of about 8
seconds. The compact run printed the planned region's GEP/Load/Ret, the one
generated C index expression, and the native result. The preventing constraint
is to cap selected SSA/C lines and finding text before sending tool output.

## 2026-11-04 — gather pytest rerun before runtime linkage settled

The agent was asked for compact focused native gather output and to wait for
the index runtime registration to stabilize. It ran the focused file after
the typed kernel importer was reported import-stable, while CFFI runtime
header and definition edits were still underway. Six tests failed before
lowering at MSVC C2375 linkage errors. The run cost 44.91 seconds and produced
an oversized compiler traceback, so it could not measure index parity. The
preventing constraint is to confirm the CFFI import, LLVM typed reference,
and MSVC runtime are stable together before the next native test run.

## 2026-11-04 — nested backend instructions read after typed kernel edits

The agent was assigned typed native gather kernels and was required to read
applicable repository instructions before changing the backend. It edited the
C and LLVM backend files before reading `src/common/tensors/AGENTS.md` and
`src/common/tensors/accelerator_backends/c_backend/AGENTS.md`. The implementation
followed their C-kernel and arbitrary-rank requirements, but the required
ordering was lost and a separate review of both files and the kernel diff was
needed. Reading nested instructions before the first edit would have prevented
that review cost.

## 2026-11-04 — full concordance audit launched instead of mapping view

The agent was required to use the bounded mapping audit while checking typed
gather. It invoked `python -u tools/audit_identity_concordance.py` without the
`mapping` argument. The command began several full example lowerings, showed
two known incomplete views, and consumed about a minute before the parent
caught the wrong invocation. The agent stopped PID 71412; no terminal audit
result was obtained. The preventing constraint is to use the exact bounded
`python -u tools/audit_identity_concordance.py mapping` command and inspect
its arguments before launch.

## 2026-11-04 — proposed float conversion for an integer reduction

The agent was assigned typed tensor gather and required to preserve the
authored numerical semantics. It briefly inserted a counted int64-to-float64
conversion before `sum_double` while treating the gather's int64 span as an
ABI mismatch. That ignored the existing NumPy backend's int64 sum contract:
values above 2**53 would round. The patch cost one insertion and a full
removal before testing; no generated program used it. The agent restored the
sum branch and selected the typed int64 kernel only after the backend contract
was checked. Reading the authored sum contract before editing the reduction
writer would have prevented the churn.

## 2026-11-04 — hazard receipt read after focused gather tests

The agent was required to read the newest hazards before testing. A teammate
created a newer gather receipt while the agent continued focused reruns, and
the agent did not reread it first. The later read also initially failed under
cp1252 because the receipt contained a Unicode arrow. The previous test
results could not satisfy the required ordering; the agent reread with
`PYTHONIOENCODING=utf-8` and repeated the focused gate and mapping audit.
Checking the newest log and its exit status immediately before each gate
would have prevented the repeated work.

## 2026-11-04 — scalar kernel proof failure attributed before measurement

The agent added a scalar-kernel receipt check that refused the float native
case after 10.07 seconds. It first described `minted#` labels and numeric SSA
IDs as different identities without checking their equality. That claim was
wrong: the label prints the same numeric ID. The claim cost two bounded
captures and inspection runs before the direct receipt causes were checked.
The writer now cites `ssa_value_identity_cell` cells; the audit checks the
exact SSA_VALUE cells cited when that receipt was published. The preventing
constraint is to compare reference objects and their direct edges before
naming an identity split.

## 2026-11-04 — broad profile and late tensor retry before narrowing the slice

The rank-two reduction task required a bounded first-writer trace. The agent
started a whole-lowering `sys.setprofile` scratch probe that ran for more
than 50 seconds before interruption, then used a direct PlanClosure hook to
capture the exact wrong dtype writer. The agent also first replayed the full
late tensor slice; its 27.40-second focused test produced two unrelated
broadcast-helper resident-shape audit conflicts. Retrying only the changed
formal's exact Call closure resolved the rank-two and rank-three cases.
The direct writer hook and call-closure selection should have been used first.

## 2026-11-04 — pairwise Meowdoku probe printed the whole compiler trace

The agent was assigned a bounded authored-law slice with compact visible
output. Its first scratch probe left the compiler's detailed progress stream
enabled, producing about 2,693 lines and truncating the tool output. The
useful terminal result was a five-second source-link refusal on the diagnostic
`.item()` return. The later saved-module probe printed only the entry,
shortfall count, and first direct-Wasm shortfalls. A focused probe should
capture the compiler progress stream internally and print bounded selected
evidence to stdout.

## 2026-11-04 — implicit Windows encoding damaged the handoff text

The root agent was preparing the requested Meowdoku handoff. It read UTF-8
prose with Path.read_text's platform default and rewrote it as UTF-8. This
introduced mojibake; the next default-encoding read failed at byte 0x9d. The
failed report update and explicit restoration cost extra work before the
handoff could be published. The original UTF-8 text was restored and copied
again. Explicit encoding='utf-8' on every prose read and write would have
prevented both the corrupted copy and the failed follow-up read.

## 2026-11-04 — broad call tracing delayed the topk owner witness

The agent needed one bounded `_ascending_order` finalizer witness. It installed a
`sys.settrace` call filter for every Python call, even though only the owner
publisher frame mattered. The test took 109.08 seconds instead of the prior
48.73-second bounded run. The trap also assumed `target_returns` contained
pairs, raising `TypeError` while printing after the useful Call and descriptor
facts. The measured cost was roughly 60 seconds and one incomplete witness;
the owner agent received the printed identity facts. A direct wrapper or trace
limited to the exact publisher frame would have avoided this overhead.

## 2026-11-04 — CALL_RECORD read used caller name without table owner

The agent inspected a saved source-owned tensor module to explain an
elementwise audit finding. Its scratch script chose the first CALL_RECORD row
whose caller name matched root, without requiring the module's actual
`call_table.owner`. That row's first revision was empty, so the agent wrongly
reported a missing semantic result binding. Root read the exact attached
table row and found that revision already contained `(4, 1)`; a later
revision only changed owner-formal binding. The wrong report cost one
additional read and coordination correction. Matching the complete
`(table.owner, caller.name)` row before comparing revisions would have
prevented it.

## 2026-11-04 — default pytest traceback overwhelmed a bounded owner check

The root agent needed compact terminal evidence from two focused tests. It
used pytest's default traceback instead of --tb=short. The expected owner
refusal expanded the large publisher function into 1,360 output lines and
about 20,092 tokens; tool output was truncated. The useful terminal refusal
was still visible, but reading the excess added review cost. Future focused
compiler gates use --tb=short and print selected raising-frame identities
when the short traceback does not contain the required evidence.

## 2026-11-04 — aggregate reborrow reader ran before its proof was complete

The owner agent's first aggregate Call-binding helper ran on every passthrough
and required a complete dynamic tensor descriptor immediately. The planned
region was still being assembled, so the focused public test refused at
`rebound region input lacks exact Call signature` after 14.36 seconds. The
reader also initially selected a producer metadata row by numeric data ID,
although the existing `metadata_result_caller_value` reader handles distinct
source and caller IDs. The helper was narrowed to exact native-wrap rebinding,
replayed after metadata publication, and changed to use that incumbent
projection reader. The 8.05-second real reborrow audit test then passed.

## 2026-11-04 — source owner Call record patched after its frame binder

The source-owned return receipt required a late owner formal in the outer
CALL_RECORD. The agent first revised that record inside the tensor result
publisher and reran the bounded source once (29.61 seconds) without tracing
the frame binder that had appended the live Call argument. Root identified
`_complete_propagated_frame_tails` as the first writer: it updated the Call
arguments but omitted the CALL_RECORD frame binding. The publisher revision
was removed, and the binder now posts the exact record from its FRAME_TAIL
proof. Inspecting that first writer before editing would have avoided the
extra run and the transient duplicate-record path.

## 2026-11-04 — source-return audit patch matched the neighboring owner audit

The source-return audit needed to follow the two loaded carriers' exact
SSA_VALUE revision chains. The agent applied the hunk to an identical
`required <= causes` line in the neighboring owner-call audit instead of
anchoring it to the source-return method. A 4.5-second saved-module replay
still found the source-return refusal. The unintended hunk was removed, and
the same-row ancestry check was placed in the source-return audit. The next
saved-module replay returned no source-return finding.

## 2026-11-04 — assumed arena carrier physical fields before reading them

The root agent told teammates an owned arena carrier had the numerical
element's int64/span/rank-one physical fields without reading that carrier's
SSA_VALUE. The measured origin Alloca carrier actually records ptr/scalar/
rank-zero; the numerical element dtype lives in TENSOR_OWNED_HEAP_BINDING and
the logical tensor descriptor. The assumption cost extra coordination and a
measurement by the owner agent. No publisher change followed the assumption;
the source audit was corrected to compare carriers with that measured origin.
Reading the exact origin cell before naming its physical layout would have
prevented the mistaken instruction.

## 2026-11-04 — assumed the source caller was the broadcast function

The first mixed-broadcast receipt refresh matched the outer source Call's
caller name against the function that held the broadcast. The actual
`_ascending_order` source Call is in `root`, while the guarded broadcast is
in `root__planned_region_0`; the existing TENSOR_METADATA_CALL_BINDING and
live Call actual/formal pairs connect them. Two bounded ASC runs still
reported the same audit finding before the writer was moved to that exact
Call edge. The draft also included the existing CONCORD target as its own
cause, which would have created a self-edge; root's review caught it before
the final run. The finished writer cites only the current result, Call
record, and three aggregate rebinding causes on the existing receipt.
The first permanent negative also asked the broadcast audit to reject a
forged *callee* metadata triple even though that audit consumes the unchanged
*caller* triple. Native parity passed, but that assertion failed after a
33.19-second test run. The negative now assigns the forged callee triple to
the metadata-result audit and the forged caller triple to the broadcast
audit, preserving each reader's exact contract.

## 2026-11-04 — refreshed a historical Call cause while adding physical proof

The agent was asked to carry the newly posted float64 SSA physical cell into
the unchanged source-result metadata row. It instead added every newly seen
publisher cause, including a newer CALL_RECORD, to that row. The result then
had two direct Call-record causes where the exact caller-data reader requires
the single historical semantic binding and proves its ancestry to the live
record. Three focused tests failed before lowering completed, costing a
38.14-second run and review of the failure. The refresh was narrowed to new
SSA_VALUE cells; the existing CALL_RECORD cause remained intact. Checking the
reader's direct-cause contract before posting an unchanged fact would have
prevented the departure.

## 2026-11-04 — broadened a record-field repro into the encoder

The assigned Run53 record-field conflict called for a short real source
slice. After a direct PackedBitstream parameter slice reached a different
later sequence shortfall, the agent tried a public `encode_octets(data)`
wrapper. It expanded to 24 functions and ran for about 93 CPU seconds
without reaching the declaration reader; the agent stopped that process.
The extra compilation produced no identifying declaration cells. The next
slice constructed PackedBitstream from four public tensor fields, which
lowered eight functions and reached a different later tensor-index dtype
shortfall. Inspecting the exact class declaration and record-view cells in
the saved Run53 book before selecting a smaller source call chain would have
avoided the broad encoder attempt.

## 2026-11-04 — used the declaration-row identity as the global Map IR identity

The first Run54 field repair changed `class_schema_identity` itself to return
only a lexical qualname. That helper also names Map IR objects globally.
Two distinct `Edge` classes in different modules then collapsed into one
identity, causing a 14.07-second focused gate to fail one of four tests and
a 13.82-second root gate to fail one of eight. The untouched validation
checkout passed the same-name Map IR test in 1.33 seconds. The correction
keeps the global module-qualified Map IR label and derives every declaration
row from the authored class source span's `(module, qualname)` pair. The
class projection and four compatibility targets then passed eight of eight
in 14.10 seconds. Reading all consumers of the shared helper before changing
it would have avoided the regression.
