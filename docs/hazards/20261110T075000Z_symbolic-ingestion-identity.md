# Symbolic scheduling identity repair

The symbolic equation compiler marked the SymPy ingestion graph with
`canonical_value_ids=True` without running canonical relabeling. The graph
has an existing, correctly minted `ingestion_value_scope`; it has neither
canonical value receipts nor a lexical read scope. During scheduling,
`ProcessGraph.finalize_graph_with_outputs` adds Store nodes derived from the
producer identities. The false flag sent those new nodes to the canonical
writer, which correctly refused the absent lexical scope.

Removed only that false canonical declaration from
`src/compiler/symbolic_equation_compiler.py`. No invented scope, guard
relaxation, alternate compiler, or change to the contact equations was made.
New regressions verify each scheduled Store remains in the exact ingestion
scope with its producer as a direct provenance cause, and a falsely marked
canonical graph still raises the original error.

Before: the existing named-equations test failed at topological_reducer840.
After: symbolic-equation suite has 8 passing tests and one failure: the fluid
fixture expects 30 inputs, while its current authored RHS expressions contain
28 distinct symbols and compilation produces 28. That expectation is left
unchanged. The actual contact WebGPU build and neighboring symbolic tests
are being checked separately; no site deployment is part of this repair.

Neighboring symbolic graph/structural tests: 13 passed, 4 failed. Failures
include reverse-translation predicate shape, an unadmitted Raise in JPEG
source, a separate aggressive-roundtrip Store missing identity, and a
function-address disagreement. These are not presented as verified baseline
failures or repaired by this change. The targeted provenance regression and
unchanged canonical-scope guard tests both pass.

The actual contact WebGPU test advances past the repaired identity error,
but its application adapter rejects 12 numerical regions (247 seconds).
An isolated capture confirms zero SSA lowering shortfalls and a control
function that calls all 12 regions; dropping all but one would lose work.

History identifies a supported planning control: Oct11 commit a4548f89
changed max_nodes_per_dispatch from 65536 to the work-contract limit,
whose default is 256. TURING_GRAPH_REGION_NODE_LIMIT exposes that existing
policy, and scoped_work_contract supports worker-local selection. This
postdates the vehicle adapter's Aug26/Sep3 history. A 65536 comparison is
being run before attributing the split to this policy or changing any code.
Comparison completed: the same contact test with the supported region limit
restored to 65536 still produces 12 regions (182 seconds). Therefore the
new default limit is not the cause of the observed split. FusionLevel.FUSED
is not a one-region promise and also changes emission; it is not substituted.

Planner capture further rules out the cap: 244 executable operations, all
admitted by the existing elementwise classifier, with one control partition.
There are zero direct executable-to-executable graph edges: generated
arithmetic reads lexical Load nodes and publishes Store nodes. The planner's
vertical-fusion identity requires direct numerical edges; separate lexical
routing remains in the complete control lowering. No source graph edges,
coordinator barriers, or fusion contracts were changed. The generated
control entry calls all 12 regions and must not be replaced by one region.

The current ssa_fusion_regions module documents downstream-only numerical
fusion and discovers candidates without rewriting. The native-only
inline_host_linear_source_regions identity has narrower source-integral
contracts, so it is not a generic license to flatten these WebGPU regions.
Next bounded investigation is ordinary WebGPU emission of the complete
lowered control entry, retaining its function/region ABI, rather than
assuming the historical single-region application adapter still applies.

Normal WebGPU emission of the complete captured control module was tested
without selecting guessed outputs: it returns complete=False with
'compute module has no named output'. The captured control has 40 formals,
zero Ret values and empty named_outputs, despite seven authored returns.
The legacy numerical capture therefore does not by itself preserve the
public output interface this adapter now needs.

The existing lower_vehicle_python_graph_ssa instead uses the sanctioned
lower_ast_source_to_ssa entry and vehicle_python_extraction_contract for
physical AbstractTensor spans. A scratch probe reuses that exact declaration
helper and the saved compiler-materialized contact source. No production
adapter, output selection, global planning policy, or shader was changed.

Canonical source probe completed using the existing vehicle extraction ABI
and ordinary source entry. All seven authored contact outputs survive as
float32[4], through 12 planned regions and one exported entry, with source
concordance accepted. Requesting ONLY that exported entry from ordinary
WebGPU emission returns complete=False: 711 float64 intermediate findings,
12 unsupported Cast operations, and 20 unresolved native tensor callsites
(binary_double, binary_scalar_double, unary_double), 743 reported symptoms.
This is a tensor/native-kernel physical-width/backend lowering frontier,
not evidence that source regions must be discarded or forced into one.
No production adapter or backend edits were made for these experiments.
Probe summary: build/contact-canonical-probe.json. All probe processes ended.
