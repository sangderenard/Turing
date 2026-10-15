# SSA compilation growth

Activation/storage conformance and fixed-field Python storage improvements are
documented in [14_provenance_storage_contracts.md](14_provenance_storage_contracts.md).

Compilation can now stop after a bounded set of interior dependency enclosures and
resume in a fresh process. Provenance bootstrap uses this path instead of
deploying the complete core as its first indivisible job. The compiler still
ingests and resolves the source once; the retained planning archive and scoped
inputs let later stages avoid repeating that work.

## Complexity hierarchy

The initial triage uses the existing authored-call catalogue, dependency
resolver, lexical containment and SCC contraction. AST node counts estimate
source complexity without running the costly lowering. They are estimates,
not timing measurements or proof that a boundary is compilable.

After ingestion, the resolved FunctionTable owns the actual cuts. Recursive
components and specialization coupling stay indivisible. Each unit has:

- its own source-node count;
- the remaining source nodes in its dependency cone, counting shared units once;
- the largest enclosing cone that contains it;
- the source work its replacement could avoid across distinct enclosing units;
- the enclosing-to-interior dependency chain used to select it.

Admission chooses only dependency-ready units, prioritizing the largest
remaining enclosure, then shared reuse, then interior cone size. Completed
SSA enclosures remove their source weight from subsequent estimates.
Partial and failed products remove no weight. Existing memory admission,
compression, spill, worker budgets and elastic-front policies remain active.
Stage limits count admitted units, wait for their workers to finish, and never
terminate a running compile. The source-node bound still exposes oversized
indivisible cuts through the existing subdivision receipts; this change does
not claim an unsupported intra-SCC ABI is safe.

## Provenance candidates

Source triage of `identity_concordance.py` found 230 SCC units. The largest
estimated enclosing cones are:

| Enclosure | Own AST nodes | Cone AST nodes |
| --- | ---: | ---: |
| `CorrelationTable.findings` | 633 | 15,592 |
| `publish_program_abi_graph_identities` | 2,024 | 7,914 |
| `publish_projected_iterable_layouts` | 1,792 | 7,538 |
| `record_shape_transformation` | 517 | 5,483 |
| `concord_loop_scope_latch_residents` | 427 | 5,092 |

Promising interiors under the first enclosure include:

```text
CorrelationTable.findings
├── _function_findings → _dominance_findings → _dominators (307 nodes)
├── _native_interface_findings (891 nodes)
├── _layout_table_findings (807 nodes)
└── _loop_scope_findings → _blocks_between (277 nodes)
```

The shared `_PageTable.__init__` also ranks highly by estimated reuse, through
`identity_book → current_identity_book → IdentityBook.__init__`. Its object
state and initialization ABI must be representable before it can supply a
standalone binary interface. SSA composition instead retains its implementation and
state semantics within the eventual full unit. These are source-triage candidates, not verified native
provenance products. The stage planner may contract or refuse their boundaries.

## SSA composition and restart

Provenance growth uses `ssa_composition=True`. Each scoped input retains its
own SCC bodies and composes completed dependency SSA, preserving the original
FunctionTable references and component identity books. SSA call signatures
carry exact formal/result identities; they are not native ABI boundaries.
Recursive and specialization-coupled components remain together. The final
loader joins maximal dependency enclosures, checks their frozen canonical
function ownership and audits composition before native emission.

`compile_partitioned_source(..., ssa_composition=True, plan_only=True)` returns
frozen graph/plan paths after releasing planning state. Use
`compile_resolved_process_graph_plan(..., max_stage_units=1)` to admit one new
unit per invocation. Repeating the invocation resumes completed SSA receipts;
source/plan digests and the compiler fingerprint reject stale inputs. The unit
receipt retains its admission reason and measured time/RSS. The product
manifest records the updated hierarchy and pending unit indices.
The public `lower_ast_source_to_ssa` partitioned route also accepts
`partitioned_options={"ssa_composition": True}`. The catalogue CLI exposes
`--compose-ssa` with `--partitioned-source`.

Native interface tiles remain a separate supported compilation mode for
programs that require independently linked binaries. Provenance growth does
not create or install one binary per interior edge.

The ordinary catalogue CLI exposes the same stage boundary:

```sh
python -u -m tools.compile_project_catalogue --compile-resolved-plan --resolved-process-graph build/program/resolved-process-graph.pkl --process-graph-plan build/program/inputs/plan.json --output build/program/product --stage-units 1
```

`tools.bootstrap_compiler_concordance` defaults to a growth stage admitting up
to 12 units in up to 12 isolated workers. `--jobs` changes the worker ceiling;
`--stage-units` independently changes the admission limit. Ready units start
as slots and resident-memory reservations allow. Completed dependencies supply
SSA to newly ready enclosures within the same stage. The whole-program
bootstrap forwards its worker count and all resident-memory controls to this
first element; no outer workers start before complete core activation. Each
stage drains its workers before requesting a restart. It writes
the cheap triage to `growth/source-complexity.json`, freezes the resolved plan
once, then resumes `growth/product`. A `native-growth-stage-persisted` receipt
causes the existing whole-program coordinator to restart before the next
SSA enclosure stage. The outer compiler-source barrier stays closed.
`--whole-module` retains the original indivisible lowering for diagnostics.
`--lower-only` stops after final SSA composition in the growth path.

The whole-program bootstrap command uses these stages automatically:

```sh
PYTHONPATH=/workspace/turing-deps:/workspace/turing-toolchain:. CC=gcc CXX=g++ ZIG_GLOBAL_CACHE_DIR=/workspace/zig-cache python -u -m tools.bootstrap_compiler_whole_program --output build/bootstrap-native-growth --jobs 2 --min-workers 1
```

## General source trees and initialization

For ordinary source products, `compiler_source_tree` records exact authored
lexical containment and definition hashes. `publish_native_interface_product`
can publish verified supported scalar exports.
`initialize_source_tree(product_paths)` installs maximal available cuts and
rebuilds Python enclosures to call them. A completed outer definition
supersedes its lexical interiors. Initialization prepares verification before
mutation and restores the previous installation if a splice fails. Published
bootstrap registries retain these trees for initialization in fresh processes.

The native callable also carries the existing declared compiler piece, so
subsequent source compilation imports its binary implementation. Capture-free
nested functions and ordinary scalar functions are covered; captured closure
cells, dynamic defaults and unsupported object interfaces require their own
explicit representation. Definition hashes prevent changed source from
silently reusing an old cut.

Provenance bootstrap excludes these partial installations. Its first complete
unit stays inactive until the existing whole-module activation gate passes.
Restarts resume compilation; partial stages do not replace Python provenance.

## Validation and remaining frontier

124 focused source-tree, bootstrap, compilation-unit, project-product and
scoped-input tests passed. A four-stage SSA program retained dependency bodies,
composed maximal enclosures and emitted one native program with no linked
interior objects. Execution matched expected results for -3, 0 and 21.
A separate native-source-tree test installed a nested binary, initialized it
in a fresh process, replaced it with its compiled outer definition, and reused
the resulting binary from a subsequently compiled caller. Failed splicing
restored the previous installation. Public partitioned SSA integration and
backend-contract preservation tests also passed. Four broader controller,
energy and top-level concordance audit failures reproduced unchanged on the
untouched baseline checkout (three undefined-value audits and one exact source
identity conflict).

The mapping concordance audit before and after reported zero structural
findings and the same 21 unsourced facts. Growth scores are derived planning
metadata; they create no new value identity or ABI authority.

No complete provenance bootstrap was run, and no whole-core speedup is claimed.
The existing whole-module activation gate requires a Python extension module
initialization ABI and a matching semantic verification receipt. The final
bind(C) emitter does not supply those artifacts. This change implements staged
SSA growth and general verified source-tree installation; it does not complete
that separate whole-module Python-extension producer. Unsupported object,
aggregate and text-sink binary interfaces remain explicit frontiers.
