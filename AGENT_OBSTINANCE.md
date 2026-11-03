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
