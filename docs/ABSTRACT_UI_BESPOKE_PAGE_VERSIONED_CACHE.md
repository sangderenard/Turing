# Versioned cache for the bespoke MechanicalCreature page

## Purpose

The generated living-data-map page is a bespoke product, not a generic HTML
template. Its current publication path starts with:

```python
from src.compiler.abstract_ui_div_map import project_class_to_div_map
from src.compiler.mechanical_creature import MechanicalCreature

page = project_class_to_div_map(MechanicalCreature)
```

That call currently rebuilds and reassembles symbolic vehicle equations,
AbstractTensor/SSA products, WebAssembly and WGSL kernels, the physics worker,
the world model, shaders, controls, and the final self-contained HTML document.
The expensive compiler products are mostly unchanged when a body-shell shader,
control label, terrain material, or worker containment rule changes. They need
durable, content-addressed caching behind explicit versioned interfaces.

The goal is not to cache one opaque HTML file. The goal is to cache the slow,
referentially transparent layers and cheaply relink them into the bespoke page.

## Current problem

`abstract_ui_vehicles.py` uses process-local `lru_cache` instances. They avoid
duplicate work during one Python process, but a fresh page bake starts cold.
Consequently, a presentation-only change can incur the same roughly 20–30
minute build as a change to the 76-equation vehicle transition system.

At the time this document was written, the vehicle model reported:

- 80 configuration defaults;
- 161 total symbolic names, including state and control inputs;
- 76 vehicle transition equations;
- fixed-default and parametric WebGPU stage variants;
- compiled wheel-contact, wrench-reduction, vehicle-integration, and Wasm
  fallback products.

These counts are diagnostic observations, not cache keys. The normalized
content of each layer is authoritative.

## Required artifact graph

The build should be an artifact DAG. A node may be reused only when its own key
and every declared dependency key match.

| Layer | Product | Expected rebuild trigger |
|---|---|---|
| L0 | Canonical vehicle configuration and MechanicalCreature schema | Configuration or schema content |
| L1 | Normalized SymPy equation sets and symbol tables | Equation-builder or symbolic ABI change |
| L2 | Repository SSA / AbstractTensor IR | L1 key, lowering version, precision policy |
| L3 | Fixed-default and parametric WGSL stages | L2 key, specialization defaults, WebGPU target ABI |
| L4 | Scalar Wasm fallback and arena ABI | L2 key, Wasm backend ABI/toolchain |
| L5 | Contact and wrench-reduction kernels | Their independent equation/shape/backend keys |
| L6 | Resident graph manifest and common-buffer layout | Referenced L3–L5 keys or buffer ABI change |
| L7 | Physics worker, page JavaScript, CSS, and shader presentation source | Their own source and declared ABI keys |
| L8 | Introspective world/model JSON | Subject class, map schema, authored world data |
| L9 | Final self-contained HTML | Referenced L6–L8 artifacts and document assembler |

L1–L6 are the long-compiling region. L7–L9 should normally relink in seconds.
The wheel-contact, wrench-reduction, and vehicle-transition branches must be
independently addressable; changing one must not invalidate its siblings.

## Cache identity

Every artifact key should be the SHA-256 digest of a canonical manifest. The
manifest must contain only stable semantic inputs:

```text
artifact schema and schema version
producer identity and producer version
normalized source/IR digest
ordered dependency artifact keys
target backend and target feature set
numeric precision and approximation policy
entrypoint and complete input/output ABI
specialization mode and specialization-value digest
compiler/lowering version fingerprints
```

Do not key artifacts by modification time, output path, Python object identity,
process ID, random seed that has no semantic role, or a runtime GPU/Wasm memory
address. Common-buffer offsets belong in the versioned ABI manifest; live
buffer objects and addresses never belong in a durable artifact.

Canonicalization must preserve ordered ABIs while normalizing unordered maps,
numeric spelling, and source newlines. A build must be reproducible from its
manifest without relying on ambient process state.

## Version boundaries

Use separate monotonically versioned schemas instead of one global cache
version:

- `mechanical-creature-schema-vN`
- `vehicle-symbolic-abi-vN`
- `vehicle-ssa-lowering-vN`
- `vehicle-resident-buffer-abi-vN`
- `vehicle-wgsl-stage-abi-vN`
- `vehicle-wasm-arena-abi-vN`
- `abstract-ui-worker-abi-vN`
- `abstract-ui-document-assembly-vN`

A version changes only when compatibility at that boundary changes. Editing a
CSS rule must not bump the vehicle symbolic ABI. Adding a SymPy equation must
not invalidate an unrelated scene shader unless its consumed interface changes.

Compiler implementation commits may be recorded as provenance, but the key
should prefer the digest of the actual normalized producer inputs. A broad Git
commit hash used as the sole invalidator would recreate the current all-or-
nothing build behavior.

## Fixed-default and parametric variants

The eager fixed-default vehicle and the lazy parametric vehicle are two products
of the same normalized symbolic program. Cache them as sibling variants:

```text
vehicle-transition/<symbolic-key>/fixed/<defaults-digest>/<backend-key>
vehicle-transition/<symbolic-key>/parametric/<backend-key>
```

The fixed variant key includes the folded default values. The parametric key
does not. Both variants publish the same resident state/output ABI, allowing a
pipeline swap without moving the common state buffer. A shock-control change
should select the cached parametric program; it should not invoke SymPy or
rebuild WGSL in the browser.

If every mutable value returns exactly to the fixed variant's defaults, runtime
policy may switch back to the cached fixed program after an ABI-safe barrier.
That reverse switch is an optimization, not a requirement for correctness.

## Artifact format

Each cache entry should be an immutable directory containing:

```text
manifest.json       canonical identity, dependencies, ABI, provenance
payload.*           WGSL, Wasm, serialized IR, or model fragment
payload.sha256      digest checked before use
validation.json     parser/compiler/test result and validator versions
```

Large source strings should be stored once and referenced by digest. The final
HTML assembler may inline them for the browser, but the cache should not copy
the same kernel source into several intermediate entries.

Write entries to a temporary sibling directory, validate them, then rename
atomically to the content-addressed destination. Concurrent builders may share
completed immutable entries. They must not read another process's partial
entry. A per-key lock or atomic create protocol is sufficient; a global build
lock would discard useful parallelism.

Corrupt, truncated, or ABI-incompatible entries are cache misses, never
best-effort inputs.

## Proposed on-disk layout

The exact root should be configurable and ignored by Git. One workable layout
is:

```text
.cache/abstract-ui-page/v1/
  objects/<first-two-digest-bytes>/<full-digest>/...
  refs/mechanical-creature/default.json
  locks/<digest>.lock
  tmp/<unique-build-id>/...
```

`objects` owns immutable content. `refs` contains replaceable human-facing
aliases and is not trusted as artifact identity. CI may restore or publish the
object store independently from the generated page.

## Invalidation examples

| Change | Must invalidate | Must remain reusable |
|---|---|---|
| Cosmetic body-shell geometry or color | Presentation/model and HTML | SymPy, SSA, WGSL physics, Wasm physics |
| Shock-control DOM layout | Page JavaScript/CSS and HTML | All compiled kernels |
| Worker support-latch policy | Worker and HTML | Symbolic vehicle kernels unless their ABI changes |
| Terrain height samples | World/model and terrain upload data | Vehicle equation and backend products |
| Spring equation | Vehicle L1 and dependent L2–L6 products | Unrelated presentation and navigation artifacts |
| One fixed default value | That fixed specialization and final manifest | Parametric kernel and unrelated kernels |
| Buffer field order or width | Resident buffer manifest and all consumers | Symbolic equation normalization when unchanged |
| WGSL backend lowering bug fix | Affected WGSL products | Wasm fallback and source world model |

## Build and publication flow

The desired build is:

```text
canonical inputs
  -> compute artifact keys
  -> load and validate cache hits
  -> build only missing DAG nodes
  -> link resident graph from immutable artifacts
  -> assemble model + worker + shaders + controls
  -> emit candidate HTML
  -> parse embedded scripts and validate manifests
  -> browser smoke test candidate on localhost
  -> atomically publish the verified bytes
```

The known publication targets are:

- `docs/generated/abstract_ui_object_map.html`
- the local site's `index.html`
- the local site's `demos/living-data-map/index.html`

All targets should receive byte-identical content from one verified candidate.
Publication must never read a partially written candidate and must not mutate a
cached object.

## Proposed API

The page projection should accept a cache service rather than importing a
global cache implicitly:

```python
cache = ArtifactCache(root=cache_root, policy="read-write-verified")
page = project_class_to_div_map(
    MechanicalCreature,
    artifact_cache=cache,
)
```

Compiler functions should return typed artifact records with keys and
dependencies. They should not decide where the final HTML is written. Useful
command-line operations would include:

```text
build-page --subject MechanicalCreature --cache read-write --output candidate.html
build-page --subject MechanicalCreature --cache read-only --explain-misses
cache inspect <artifact-key>
cache verify <artifact-key>
cache gc --keep-ref mechanical-creature/default
```

`--explain-misses` is important: it should report the first changed manifest
field and the dependency path that forced each rebuild.

## Validation requirements

A cache hit is accepted only after:

1. manifest schema and producer compatibility checks;
2. payload digest verification;
3. dependency-key verification;
4. ABI validation against the consumer;
5. WGSL/Wasm/source parsing appropriate to the artifact;
6. optional golden-vector execution for high-risk compiler changes.

Tests should prove both reuse and invalidation. At minimum:

- two identical cold-process builds produce identical artifact keys and HTML;
- a CSS-only edit produces zero L1–L6 misses;
- a worker-only edit produces zero symbolic/backend misses;
- a fixed default edit rebuilds only the fixed specialization branch;
- a symbolic equation edit rebuilds the dependent vehicle branch;
- changing a buffer ABI rejects stale producers and consumers;
- a corrupt payload is rebuilt and never linked;
- concurrent builders converge on one valid immutable object;
- cached and uncached kernels agree on representative vehicle/contact vectors.

Record per-layer hit/miss counts and elapsed time in the page build report. Do
not put cache polling, compilation, or status DOM work into the 120 Hz physics
loop.

## Migration plan

1. Introduce artifact and manifest dataclasses without changing compilation.
2. Serialize and reload the normalized SymPy equation/symbol products.
3. Cache contact, reduction, fixed vehicle, and parametric vehicle backend
   products independently.
4. Make the resident graph consume artifact records and verify common-buffer
   ABIs.
5. Split page linking from kernel construction.
6. Add miss explanations and deterministic cold-process tests.
7. Enable read-write caching locally, then in CI.
8. Add garbage collection only after dependency traversal and pinned refs are
   tested.

Until this migration is complete, a successful long bake remains authoritative.
Do not substitute ad hoc string surgery on an older HTML file for a correctly
linked build. The cache exists to preserve compiler work while keeping the page
coherent, versioned, and reproducible.
