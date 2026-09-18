# PE-to-SSA staged and repeated-operand algorithm report

Date: 2026-08-12

## Purpose

The PE-to-SSA path would benefit from small, manually authored native routines
whose dependency structure is intentional. Ordinary compiler output often
hides the distinction this work needs to observe: a sequential accumulation, a
balanced reduction, several independent accumulators, and an addition chain can
compute the same value while presenting substantially different SSA graphs.

These routines should be treated as semantic and structural fixtures. Their
value is not merely that the lifted SSA returns the correct answer, but that it
retains repeated uses, shared definitions, dependency depth, flags, memory
effects, and control flow accurately enough for later analysis to recognize the
algorithm that was present in the PE.

## Principal opportunity

For an associative binary operator over `N` distinct operands, a left fold has
`N - 1` operations and dependency depth `N - 1`. A balanced tree still has
`N - 1` operations but reduces ideal dependency depth to `ceil(log2(N))`.

```text
left fold:                         balanced tree:

t0 = a + b                        t0 = a + b
t1 = t0 + c                       t1 = c + d
t2 = t1 + d                       t2 = t0 + t1
```

When the same operand occurs many times, an addition chain can reduce both
operation count and depth. For example, eight copies of `x` require seven adds
in a literal fold but only three doublings:

```text
t0 = x + x
t1 = t0 + t0
t2 = t1 + t1
```

For arbitrary multiplicity, binary decomposition supplies a simple initial
algorithm: build powers of two by doubling and combine the selected powers in
a balanced tree. Minimal addition-chain discovery can remain a later research
problem.

## Recommended hand-written PE fixture families

### 1. Sequential and balanced reductions

Implement matched routines for 4, 8, 16, and 32 integer operands:

- a single-accumulator left fold;
- a perfectly balanced tree where the arity permits it;
- a balanced tree with an odd tail;
- a four-accumulator loop followed by a final tree reduction.

Use addition first, then bitwise AND, OR, and XOR. Integer modular addition is a
clean first case because regrouping preserves the result when the observable
contract excludes intermediate flags and traps.

The SSA comparison should verify equal values but deliberately unequal critical
paths. It should not canonicalize both graphs before the structural assertion.

### 2. Repeated identical operands

Provide routines computing `N*x` for representative counts such as 3, 5, 8,
13, 31, and 64 in at least three forms:

- literal repeated addition;
- sequential accumulator loop;
- explicit addition chain.

These cases sharply test graph multiplicity. In `x + x`, both operand positions
must survive lifting as distinct uses of the same SSA definition. Deduplicating
the operand edges changes the program. Conversely, the definition of `x`
should remain shared rather than spuriously cloned.

Multiplication analogues can exercise exponentiation by squaring for repeated
factors, provided overflow semantics are fixed and tested.

### 3. Multi-accumulator loops

Write array sums and dot products with two, four, and eight independent partial
accumulators. After the loop, merge the accumulators with a balanced tree.
These fixtures expose instruction-level parallelism while retaining realistic
loads, pointer increments, loop-carried phi values, and a scalar remainder.

Useful variants include:

- aligned and deliberately unaligned inputs;
- lengths divisible and not divisible by the unroll factor;
- signed and unsigned integer elements;
- scalar and SIMD bodies;
- a dot product with separately accumulated multiply results.

### 4. SIMD and horizontal reductions

Use packed integer additions to consume several operands per instruction, then
perform an explicit horizontal reduction. Keep one fixture with lane extraction
and scalar combination and another with shuffle/add stages. This distinguishes
parallel lane computation from scalar dependency depth and exercises vector
register identity in the lifter.

### 5. Tournament and bitwise trees

Balanced min/max tournaments test comparisons and selects rather than only
arithmetic. AND/OR trees test idempotence (`x op x = x`), while XOR tests parity
(`x xor x = 0`). These identities are useful, but the raw lifted SSA should
preserve the original repeated uses before a separately authorized
optimization applies them.

### 6. Prefix/scan networks

A staged inclusive scan is a valuable second-wave fixture because it has both
fan-out and fan-in and produces every prefix, not just one final scalar. A
Hillis-Steele-style fixture is easy to write and gives the graph a recognizable
sequence of distance-1, distance-2, distance-4 stages. A work-efficient scan
can follow once the simpler structural test is stable.

### 7. Native repetition instructions

PE fixtures using REP MOVS/STOS belong beside the arithmetic cases but should
remain a distinct semantic family. They consume a repeated count through an
architectural loop with explicit RCX, pointer, direction-flag, and memory-state
effects. They should not be mistaken for pure associative reductions.

## Semantic gates

Tree balancing or repeated-operand compression is valid only when the operator
and observation boundary permit it. Record these properties explicitly rather
than infer them from an opcode name alone:

- associativity under the actual numeric semantics;
- commutativity, if operands may be reordered rather than merely regrouped;
- whether overflow wraps, saturates, traps, or is undefined;
- whether intermediate arithmetic flags are live;
- whether floating-point reassociation is allowed;
- whether exceptions, NaNs, signed zero, or rounding mode are observable;
- whether loads are volatile, atomic, trapping, or may alias writes;
- whether any operand-producing instruction has side effects;
- whether memory versions and control dependencies are preserved.

Floating-point fixtures should initially be fidelity tests, not candidates for
automatic balancing. A balanced floating-point sum generally differs from a
left fold. It becomes an allowed transformation only under an explicit relaxed
math/reassociation policy.

## Proposed SSA representation boundary

The most conservative design is to lift the PE exactly into ordinary SSA first
and derive reduction structure as analysis metadata. Candidate annotations are:

```text
ReduceTree(op, ordered_inputs, topology, semantic_policy)
RepeatReduce(op, shared_value, multiplicity, topology, semantic_policy)
```

These are descriptions of a proved subgraph, not replacements for it. Keeping
the exact SSA underneath provides provenance and allows the proof to be checked
or discarded. Only a later, proof-gated lowering or optimizer should expand,
rebalance, vectorize, or replace the region.

If a first-class reduction node is introduced earlier, it must retain:

- an ordered operand multiset, including duplicate positions;
- the original instruction-address span;
- the exact overflow/flag/floating-point policy;
- memory and control-effect boundaries;
- a reversible mapping back to the source SSA subgraph.

## Structural assertions for the lifter

For every fixture, test more than returned values:

1. Each machine instruction maps to the expected provenance address range.
2. `x + x` has two operand uses pointing to one shared definition.
3. Parallel outgoing uses are not collapsed by graph projection.
4. The sequential fixture has the expected long def-use chain.
5. The balanced fixture has the expected number of levels.
6. Multi-accumulator loops produce independent loop-carried phi families.
7. Loads remain ordered through explicit memory SSA where aliasing requires it.
8. Live arithmetic flags prevent an otherwise tempting regrouping.
9. Dead flags do not create false value dependencies.
10. The lifted routine executes equivalently to the native fixture over edge
    cases and randomized inputs.

Record at least instruction count, SSA instruction count, maximum value-DAG
depth, maximum fan-out, number of duplicate operand edges, phi count, memory
versions, and detected reduction regions. Runtime benchmarking is secondary;
the first goal is a trustworthy structural corpus.

## Suggested first implementation slice

Start with six leaf routines in one very small PE:

1. `sum8_sequential`
2. `sum8_balanced`
3. `sum32_four_accumulators`
4. `repeat13_literal`
5. `repeat13_addition_chain`
6. `xor8_balanced`

Compile with optimization disabled where exact source structure matters, then
inspect the disassembly and retain the accepted bytes or instruction tokens as
part of the fixture contract. For absolute stability, a small assembly source
is preferable to relying on a C compiler to preserve a chosen topology.

The initial success criterion is not that the PE-to-SSA layer invents the
balanced form. It is that it faithfully distinguishes the forms supplied to it,
preserves repeated operand multiplicity, and exposes sufficient graph facts for
a later pass to prove equivalence and choose a topology deliberately.

## Coordination note

The current PE-to-SSA and machine-semantic files are actively modified. This
report intentionally proposes an additive fixture and analysis direction and
does not prescribe changes inside those in-progress files. Before implementing,
coordinate ownership of the PE fixture location, exact-SSA graph projection,
and any first-class reduction annotation so current lifting work is not
silently reshaped underneath its author.
