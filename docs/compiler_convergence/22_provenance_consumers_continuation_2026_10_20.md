# Provenance continuation — 2026-10-20: consumers must read the book

Canonical repository `C:/dev/Powershell/turing`, branch `main`, HEAD
d5b43897. The tree is clean. This report follows report 21 (the sweep and
its addenda) and records the state after the tuple-mapping declaration was
reverted and the three consumer-side repairs were redone at their
generators.

## Measured state

The user's normal bootstrap on d5b43897 (fingerprint 87035e7b...) admitted
31 of 330 units: 30 compiled-unverified, 1 partial, 0 failed. The partial
unit is 034, `IdentityBook._virtual_edges_into`. No unit has been natively
verified in any run. The ten-file pre-commit gate is 457 passed and 9
failed; all nine failures were introduced by the checkpoint commit
2b94890e, before this work, and are listed in report 21.

## Two laws, stated by the user on 2026-10-20

1. Nothing may fix generated content after the fact. A record that is
   wrong is wrong at its generator, and the fix goes there.
2. All transformations are in the concordance, and nothing may plan
   without communicating it. Corollary: nothing may operate on a value
   without first checking the concordance for what it is about to use.

Three committed edits violated the first law and were reverted and redone
at their generators in 49df5d5e: the linker now binds payload positions by
reading the callee's receipts instead of editing its emitted Ret; a
conditional block is placed where its test ends, so hoisted predicate
items precede it, instead of re-orienting arena edges after scheduling;
and the sort contract for `sorted(<iterable>)` is posted once, by the
composer, with the final collection identity, instead of being revised
after the fact.

## What was reverted and why

Commit a0072423 declared a fixed-arity tuple mapping value as N value
columns at the reducer and the sequence schema, but none of the consumers
of that declaration existed. Unit 027 (`IdentityPage._fork_of_scope`)
went from compiling to a hard stop at the lookup helper, and the bootstrap
fell to 5 admitted units. An attempt to generate the consumers in one
agent run produced a reducer edit that referenced leaf nodes it had never
created, which killed whole-program topology reduction. Both are out of
the tree: d5b43897 reverts a0072423, and the working-tree edits are saved
as `agent_tuple_mapping_wip.patch` and `test_tuple_valued_mappings.py` in
the session scratchpad, not in the repository.

The lesson is recorded in the second law. A declaration without its
consumers is not a fix at the origin, and a feature that spans the reducer,
the keyed-field ABI and the sequence helpers lands whole or not at all.

## Why unit 034 is partial, by observation

The receipt names an unresolved required source value, the Tuple display
43 that contains the starred element `*ref.row[1:]`, and two unexplained
public inputs, `stamp` (13) and `origin_row` (19). Both come from
consumers that acted without reading the book.

The planner's classifier posts every node's verdict to the
`executable_node` page: executable, or dispatch metadata, with the rule.
The starred element is classified `PYTHON_SYNTAX`, which is dispatch
metadata. That is correct for a tuple display whose leaves do the work and
wrong for a starred element whose operand is a runtime slice of an opaque
reference, because the splice is work that only that node stands for. The
verdict is on the book. No consumer reads it. The return publication used
value 43 without reading the verdicts of its leaves; the formal collector
turned the unproduced operands 13 and 19 into public inputs without
reading whether any row produced them. Finalization then counted the
holes and assigned the partial status.

Partial is therefore a symptom of consumers that operate without reading,
and of a finalization that tallies holes instead of a generator that
refuses. Under the corollary the correction is at the consumers: return
slots, region feeds, display leaves and the formal collector read the
`executable_node` verdict and the producer rows of each operand before
acting, and refuse by name, with the authored span and the construct, when
the rows are absent or say the operand is not executable. The audit gets
the matching finding, so a book that contains such a derivation is
reported as discordant whether or not the compiler happened to stop.

Unit 034 also carries two deeper gaps that the refusal will name rather
than hide: `origin_row = (origin, *ref.row[1:])` constructs a new tuple
from an opaque reference, which no lane lowers, so the choice is a
structural declaration of `Ref.row` or a lowering that does not exist;
and `_fork_of_scope`'s `forks: dict[Any, tuple[Any, int]]` is lowered
with one value column, so its result has one leaf and the caller's unpack
cannot bind. The second is the tuple-mapping work above.

## The tuple-mapping value position, as design

A mapping's value position must consume the same leaf-layout record every
other value position consumes: scalars are columns, a fixed-arity
aggregate is its leaves recursively, a record is its row, a variable-length
sequence is a child pool. Today the value position is the one place that
does not, and it gets a single int64 handle instead. The whole feature is:
the reducer publishes the value's element schema; the sequence schema
declares its value-column count and derives the key split from it; the
keyed-field ABI decomposition posts N value parts; the lookup helper returns
the matched row index and the caller reads N leaves under the presence;
the store writes N columns from the stored value's leaves; and the lookup
result carries N leaf identities minted by the reducer so unpacks and
projections bind. It has to be general for any N, any element dtypes and
any key column count, verified by a parametrized test over the contract
and then by a bootstrap, and it has to land in one change.

## Open items, unchanged from report 21

The four audit-contract fixture tests stop on named gates; the variadic
tuple-display return lane is not built; the element dtype of sequence
returns is unknown because declared return annotations are not on the
book; two unannotated variadic functions remain; the whole-value use of an
optional aggregate in `shape_transformation_state` is unresolved; and the
raw-write inventory in report 21 still lists ten writers.

## Validation

Ten-file gate on d5b43897: 457 passed, 9 failed, the known ids. The user's
bootstrap on d5b43897: 31 admitted, 0 failed, 1 partial. No bootstrap or
native build was run while preparing this report.

## Prompt history

> anything being partial is a complete failure in that part, why would anything, deterministic as it is, allow that to happen

> all transformations are required to be in the concordance, nothing is allowed to plan something without communicating it

> okay, correlary to nothing can modify without recording in the concordance, nothing can operate without first checking the concordance on what they're going to use

> can you make a continuation report and commit and push
