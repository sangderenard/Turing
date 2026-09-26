# Compound nomenclature for algorithm specimens

**Date:** 2026-08-06
**Status:** design, nothing implemented
**Purpose:** give discovered subgraphs deterministic, decodable names, so
independent discoverers converge on the same name for the same thing.

## The problem this solves

The specimen idea -- lasso a region, name it, let the system surface similar
ones -- fails quietly if names are *chosen*. Two people, or two agents,
finding the same recurring pattern will name it differently, and the catalog
fragments into synonyms nobody can reconcile. The catalog only compounds in
value if the name is **computed from the structure**, not assigned to it.

Organic chemistry solved exactly this. IUPAC nomenclature is deterministic
(one structure, one name), compositional (name built from parts), and
decodable (the name reconstructs the structure). Two chemists who isolate
the same molecule independently write the same name. That is the property
we want, and the rules that produce it transfer more directly than the
analogy first suggests, because both domains are naming *labelled graphs*.

## What we already have that maps onto it

**The elements already exist.** `ssa_registry.Handler` is a closed set of
~40 canonical operations -- Add, Mul, Load, Store, Call, Phi, Select, Ret,
GetElementPtr, Cast... Every frontend spelling converges onto one of them
(`ast_ssa_equivalents`, `c_ssa_equivalents`). That closed, shared,
already-canonical vocabulary is the periodic table this naming system needs,
and it is the reason this is buildable rather than speculative. Elements ->
functional groups -> compounds.

## The rules

Applied in order. Each is a direct transposition of an IUPAC rule, named
here so the correspondence stays auditable.

### 1. Canonical ranking (IUPAC: canonical numbering; chemistry software:
Morgan / CANGEN)

Before naming, rank every node by iterative refinement on structural
invariants, so the name is invariant to node ids and discovery order:

    rank_0 = (handler, in_degree, out_degree, dtype)
    rank_n = (rank_{n-1}, sorted(rank_{n-1} of neighbours by edge role))

Iterate to a fixed point (Weisfeiler-Leman refinement). Remaining ties are
broken by a documented total order over handlers, then roles. **Ties must be
broken deterministically or the whole scheme collapses** -- an arbitrary
tie-break yields two names for one structure, which is precisely the failure
we are avoiding.

### 2. Principal chain (IUPAC: parent hydride / longest chain)

Select the principal dataflow path: longest path from a feed to an output.
Tie-break, in order: most operations, then highest total canonical rank.
This is the name's spine.

### 3. Locants (IUPAC: lowest-locant rule)

Number positions along the principal chain. Choose the direction giving the
lowest locant set to the principal functional group; if still tied, lowest
to the substituents. **Direction is not free here the way it is in
chemistry** -- dataflow is directed, so the "lowest locant" choice is
constrained by, and must be recorded against, the feed->output direction.

### 4. Functional groups (IUPAC: characteristic groups, suffix vs prefix)

Recognised recurring motifs, ranked by seniority so exactly one becomes the
suffix and the rest become prefixes. Starting set, deliberately small and
grown only by evidence:

| motif | shape | affix |
| --- | --- | --- |
| reduction | fold over an axis to lower rank | `-sum`, `-max` |
| predication | `where(mask, a, b)` | `-sel` |
| gather | read at a computed index | `-gath` |
| materialisation | scatter/assemble into a buffer | `-mat` |
| accumulation | loop-carried dependency | `-accum` |
| dispatch | call through a table/address | `-disp` |

Seniority order is a decision to record explicitly, exactly as IUPAC
publishes its table; it is arbitrary but must be *fixed*.

### 5. Substituents and multiplying prefixes

Side branches attached at locants, named **recursively by these same
rules**, alphabetised, with `di`/`tri`/`tetra` for repeats at distinct
locants. Recursion is what makes the scheme compositional rather than a flat
dictionary, and it is what lets a named specimen appear as a component of a
larger specimen without losing its identity.

### 6. Stereodescriptors (IUPAC: R/S, E/Z)

Properties that do not change connectivity but do change meaning:

- **operand order** on non-commutative operations (`sub`, `div`, `shl`)
- **direction**, which is real here in a way it is not in chemistry: the
  read head is bidirectional, and a region traversed forward is not the same
  specimen as the same region traversed backward.

## Layers, and why they carry the gamut

Borrowed from InChI, which stores formula / connectivity / hydrogen /
charge / stereo as separate layers so two structures can be compared at
whatever depth the question needs.

| layer | carries | matching at this depth means |
| --- | --- | --- |
| `S` skeleton | handlers + topology | same shape of computation |
| `T` types | dtypes, widths | same shape, same types |
| `V` values | baked constants | same shape, types, constants |
| `O` orientation | operand order, direction | fully specified behaviour |
| `E` evidence | observation grade | how well any of this was seen |

**This is where nomenclature meets the reversibility gamut.** Reversibility
is graded -- an edge may be natively executed and reversed, replayed from an
exact tape, completed by an external capability, or blocked. Evidence from
those is not equally strong, so specimen identity claimed from them is not
equally strong either. Layer depth is exactly the vocabulary for that: you
match at the depth your evidence supports, and the `E` layer states which
depth that was.

A specimen named to `S` from replayed evidence and one named to `O` from
natively-reversed evidence are *both legitimate* and are not the same claim.
Without layers there is one flat "is/isn't the same" question and every
answer overstates or understates. With them, "these agree at `S+T`, diverge
at `V`" is sayable, which is the sentence topological analysis actually
needs.

## Worked sketch

A read-head microstep region: gather an encoding row by opcode, select on a
phase mask, accumulate into a cursor.

    S: i64-2-gath-4-sel-accum
    T: .i64
    O: /fwd

    full: i64-2-gath-4-sel-accum.i64/fwd

Read back: an i64 principal chain, gather at locant 2, predication at
locant 4, accumulation as the senior group, all i64, forward direction.
The name reconstructs the shape.

## Honest limits

- **Canonical labelling is graph isomorphism-complete in general.** For the
  region sizes here (tens of nodes) refinement plus deterministic tie-break
  is fine, but this does not scale to arbitrary graphs and should not be
  claimed to. Bound the region size, or accept that pathological cases fall
  back to a rank-ordered serialisation rather than a pretty name.
- **The seniority table is a convention, not a discovery.** It must be
  published and versioned; changing it renames every existing specimen.
  Chemistry has the same problem and solves it with versioned rules, not by
  pretending the order is natural.
- **Names are not addresses.** `FunctionReference.address` addresses what
  the source *declared*; these name what the graph *exhibits*. Two
  namespaces, deliberately separate -- conflating an authored identity with
  a discovered one would let a rename in source silently invalidate a
  specimen catalog.
- **Numeric-region tokens are not identity tokens.** A named subgraph is an
  identity ("this is that"); a numeric region is an interval ("values here
  lie in this range"). Equality vs containment. If they share one token
  space, similarity search silently does the wrong thing for one of them.

## First implementable slice

Rules 1 and 2 alone (canonical ranking + principal chain) already yield a
stable `S`-layer key with no motif table and no seniority decisions. That is
enough to answer "have we seen this shape before?" across programs, which is
the question the catalog needs first, and it can be validated immediately
against the specimens the shortfall instrumentation is already surfacing --
the 32 static list-replication records are one specimen by any reasonable
definition, and a correct `S` key must give them one name.
