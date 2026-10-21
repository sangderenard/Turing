# Continuation 2026-10-21 (later): the physical record is posted

Canonical repository `C:/dev/Powershell/turing`, branch `main`. This report
follows report 23 and covers the uncommitted wave that landed after it.
No bootstrap and no native build were run; every claim is a test gate, a
seconds-long audit on real source, or a classification against a clean
worktree of eb0477b2, the last pushed commit.

## Measured

The audit kind `ssa-physical-unposted` lists every SSA value whose
physical fields (dtype, storage, rank, arena, device) exist in an
accounting dict but were never posted on its row. At the start of this
wave it reported 382 rows across the thirteen probe modes (the four
real-program probes, the three aggregate-identity programs and the six
record and loop programs). On the quiet tree every mode reports zero
unposted rows and zero physical type disagreements.

Gate on the quiet tree, 33 files: 687 passed, 51 failed, 18 min 29 s.

Every failure in the gate was classified on a clean worktree of eb0477b2
and fails there identically. No id is new to this wave. Nine new test
files were added, all green: the `id(x)` lowering, the passthrough view,
the contract row for every value, keyed-descriptor member rows, nested-loop
shape transport, precompile contract rows, sequence helper rows, and the
structural-recovery output identity.

## The root, and what was done about it

Two fact-finders traced the whole worklist to one fact. The ProgramABI
contract, which declares each value's storage and rank, was never a page
row. It lived in `program_abi`, in graph attributes, and in a side table
keyed by graph id. The only page with storage, dtype, shape and rank per
arm, `VALUE_TYPE_ABI`, was posted only when the contract named a Python
type, so a primitive record field had no row for anything to derive from.
Every producer downstream copied the accounting dict instead, which is why
the structural view, the transport check and the result-type proof could
not be trusted to agree.

The ingestion site now posts the contract row for a primitive record field
by reading the class-field declaration that already held its type; no type
identity was invented. Values with neither a Python type nor an annotation
still post nothing, and none appeared in the probes.

With a row to derive from, every producer in the worklist was made to post
its physical fields Derived from the rows it reads, and to set accounting
from the posted fact. By writer: the planned-region builder and the
control-SSA value cell derive from the contract row; carried and merge Phis
derive from their operands and refuse by name if the operands disagree;
sequence descriptor members post arena and storage at their mint; the
result-storage lease, the linked-call output, the retained result and
recovered structural formals read the callee's row through the owner book;
the record-field formals of a linked function derive from the settled
contract; kernel pointer formals are posted by the caller that feeds them,
since the kernel's ABI for a pointer is the caller's declaration; a
materialized Load posts the storage its shape-concordance row records; the
tensor-op result, the shape-transfer loop and repository enrichment post on
the row instead of only in accounting; result-type settlement carries the
winner's storage with its dtype. One raw page write that bypassed the
closed latch was replaced with a Derived post. The accounting-from-fact
helper mirrors all five keys.

## Consumers that read a declaration before comparing

The aggregate passthrough now carries its declared output view through the
whole chain. Four consumers compared a view occurrence against the storage
it views and refused: the shape-transport predicate, the shell's canonical
shape loop, the boundary-transport producer endpoint and the structural
value page. Each now reads the occurrence's view declaration, and the
structural page posts a row for the view identity keyed by storage and
view shape. The refusals themselves are untouched.

Structural recovery compared a store-chain version of an output against
the reducer's declared root without resolving the version through the alias
ledger the book already holds; it now resolves first and still refuses any
other disagreement.

## Producers fixed instead of readers

The while-loop header Phi was minted with the authored dtype spelling of
its literal while the literal's posted row, after constant
materialization, said `int64`; the region call then copied the physical
dtype onto its argument occurrence without revising the row. One reader
patch that revised the authored spelling in enrichment was removed, and the
Phi and the backedge placeholder are minted from the posted row. The region
call's argument occurrence takes the posted row's dtype when one exists.
The remaining producer of the authored spelling is the planned-region
formal itself, built from the region signature at ~25590 and ~27027 in
precompile; it carries `int` where the row says `int64` and is the next
item on the dtype side.

A latent name error in the shell's structural-parameter marking made any
program calling `id(x)` a hard stop on clean HEAD; the name is bound.

## Open items

- Planned-region formals carry the authored dtype spelling; they must be
  minted from the posted rows of the values they are bound to.
- Values with no Python type and no annotation have no contract row; the
  contract validator rejects most of them already, and none appeared.
- The 82 wholesale accounting copies enumerated by the fact-finder remain
  as derived views; each is harmless only while its reader reads rows.
  The next audit should flag an accounting key that disagrees with its
  posted row, not only one that is missing.
- The pre-existing failures: the four audit-contract fixtures, the five
  checkpoint-introduced ids, the 17 and 9 shape-metadata ids, five
  sequence-contract ids and eight native memo membership ids, all
  identical on clean HEAD.
- The user's bootstrap has not been run on any commit since d5b43897.

## Prompt history

> using focused sonnet agents that are on rails for just the mechanical execution of fact finding or correction, orchestrate agents in making real progress to these ends

> I don't give a fucking shit about the tests I care if you have objectively deterministically finished repairing the compiler by first principles
