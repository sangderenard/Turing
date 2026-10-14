# Provenance activation and storage contracts

The whole-module activation probe now checks authored constants, class
declarations and methods, then runs `test_provenance_storage_contracts.py`
alongside the existing copy-on-read and compartment-operation tests. A genuine
Python extension initializer is necessary but does not establish a usable core.
Tests build real deliberately incomplete C extensions: the wrong initializer
fails Python import, and an importable empty module fails the authored-surface
check. Neither may replace an existing registry or leave a pending install.

The storage suite runs the canonical `IdentityBook.post` and page operations
over five representations: ordinary Python, compressed RAM, page spill, live
RAM tiles and live disk tiles. It checks:

- exact registry `Page`, `Stage` and `Transform` leaves in provenance edges;
- one clock tick per accepted post, including clocks wider than uint64;
- fork snapshots, materialization edges and revision history;
- negative and wider-than-uint64 identities and authored columns;
- heterogeneous facts, Unicode, binary data, and bool-versus-integer admission;
- shared book clocks and references after archive restoration;
- old dict-backed and new fixed-field audit-record archive states.

The tests are representation-neutral: a future ctypes or tensor-backed core
must preserve these results, or decline an unsupported representation before
changing state. A raw pointer address is not a replacement source identity.

## Implemented storage improvement

`Claim`, `ValueRow` and `Finding` now use dataclass slots. Their fields and
mutable list ownership remain the same; legacy dict-backed pickle states
remain readable. This removes one per-record dictionary without imposing an
integer width or changing the existing table writer. Over 10,000 objects per
class on this Python 3.12 runtime, shallow object-plus-dictionary bytes were:

| Record | Untouched baseline | Fixed fields |
| --- | ---: | ---: |
| `Claim` | 144 | 56 |
| `ValueRow` | 176 | 88 |
| `Finding` | 152 | 64 |

These measurements exclude the owned strings/lists and do not establish a
whole-core compile-time or RSS improvement.

The storage tests also exposed existing standalone regulator gaps. A regulator
without compile-wide live storage now compresses its declared cold pages
through `IdentityBook.compress_pages`, and pressure escalates already-compressed
tiles through `offload_tiles`. Compile-wide writable arenas keep their existing
adaptive ladder. `graph_function_socket` now has a lifecycle declaration: its
latest source-linking writes precede planning, with no later semantic reader.
Spill-policy tests explicitly disable RAM compression when selecting spill.

Forced-pressure source tests exposed another existing storage fault: ABI
settlement indexed wrappers by `id(graph)` without retaining the graph headers.
Cold header eviction could reload the graph under a different Python address,
making the index stale. Settlement now retains the exact indexed headers for
the index's lifetime; node and edge payloads remain tiled and pressure-regulated.
The existing call and loop tests now complete lowering, report zero structural
findings, and execute native results correctly after forced disk eviction.

## Span and struct opportunities

The existing storage already contains contiguous `array("Q")` routing/hash/
ordinal indexes, byte/uint32 tile metadata, and a packed struct disk directory.
These should remain the storage owners; another parallel ledger would duplicate
identity and memory accounting.

The next useful candidates, with the contracts that constrain them, are:

| Candidate | Possible representation | Required evidence before conversion |
| --- | --- | --- |
| Cell clock/stamp and revision indexes | Typed integer spans with an exact wide-value route | No uint64 wrap; authored column ordering and old archives remain exact |
| Homogeneous declared fact records | Internal ctypes structs or typed tensor columns | Registry field/type metadata selects the layout; opaque fields retain exact references |
| Audit claims and use/definition occurrences | Compact tag/ordinal spans within the existing table owner | Exact strings and ordering survive; report text and archived records remain compatible |
| Cell keys shared by cells/stamps/indexes | Existing arena handles in packed records | Handles stay storage coordinates; registry identity, aliasing, forks and relocation remain intact |

The clock, column and row domains are currently arbitrary Python integers.
Replacing them wholesale with `c_uint64`, `c_int64` or a float tensor would
change valid behavior. Heterogeneous facts also include objects whose exact
identity is significant. The new tests intentionally cover those cases before
any larger packing change. No universal packed replacement or new native ABI
boundary is introduced here.

Run the core contract without compiling the whole provenance module:

```sh
python -m pytest -q tests/test_native_provenance_bootstrap.py tests/test_provenance_storage_contracts.py tests/test_provenance_compartment_operations.py tests/test_copy_on_read_scopes.py
```

The staged SSA architecture is described in
[13_native_compilation_growth.md](13_native_compilation_growth.md). The full
provenance extension producer and full bootstrap remain unverified.

Validation covered 179 selected storage, copy-on-read, native-import, pressure
and scoped-composition checks. The broad run passed 177; both remaining
pressure failures reproduced on the untouched checkout, were repaired as
described above, and passed in the complete seven-case pressure suite. Actual
old record pickles from that checkout also load under the new layout. The
mapping audit retained zero structural findings and the same 21 unsourced facts.

Follow-up source-provenance repair: the mapping lowering now reports zero
structural findings, zero unsourced facts and zero unsourced identities. Region
formals are posted from their actual signature and control binding. Adaptation
and specialization rounds use distinct book-minted scopes and recorded inputs.
Loop reconciliation consumes ordered block geometry, including receipts from
the actual constant-hoisting producer. Scoped worker inputs retain the original
ingestion cells, name bindings and scope registration/origin cells that numeric
lookups consume. Accepted dependency bodies retain their original book owner.
This result covers the mapping repro; it is not a claim that every compiler
program has complete provenance. Call-containing enclosures still expose
separate unsourced call/frame/planning facts in the audit.
