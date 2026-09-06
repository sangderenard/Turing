# AbstractUI and Pluck world convergence

AbstractUI owns a language-neutral world contract; Pluck supplies the first
rich game vocabulary and adapter precedent.  This is inheritance by contract,
not a browser import of Pluck's Python runtime.

## Three retained authorities

1. A `WorldObject` owns conceptual identity, containment, transform, form
   recipe, material-role bindings, capabilities, physics intent, persistence,
   and namespaced source extensions.
2. A mesh packet realizes those recipes as triangles while retaining parallel
   object and semantic-part span tables. GPU buffers are disposable products.
3. A bake artifact may cache a mesh, image, sprite, or light field using the
   recipe and render conditions as content identity. It never replaces the
   object or its authored recipe.

The shape parallels Pluck's `PlacedObject`/`RoomWorkspace`, procedural mesh
builders, and `RenderAssetCatalog` layers. `pluck_placed_object()` accepts the
dictionary produced by Pluck's plain data classes. Recognized fields populate
the neutral surface and the complete source record is retained below the
`pluck.placed_object` extension, including fields unknown to the adapter.

## Mesh identity

The living data map publishes two parallel tables:

- object spans correlate a variable-length vertex interval with a world-object
  identity and living-document revision;
- semantic-part spans identify floors, individual boundary walls, opening
  lintels, and ceilings inside that interval.

For hot paths, a deterministic specialization table assigns dense 1-based
`u32` IDs to both tables. Zero means unresolved. The JavaScript world registry
supports lookup in both directions, while authored strings remain the
persistence and editing authority.

This is the missing low-level correspondence in Pluck's current scene-order
compiler, which validates authored object IDs but merges compiled subject
triangles into one renderer `object` group. AbstractUI can consequently send a
crosshair, tool action, collision, or material edit to a particular wall while
remaining compatible with a coarser backend.

## WebAssembly plugins

The world registry carries bounded plugins with Python source, compiled WASM,
entry point, exact parameter ordering, memory requirements, and capability.
The initial plugin set is:

- parametric box-vertex construction;
- Pluck-style position plus yaw transformation;
- perspective projection for a software presentation backend.

The host owns scheduling and linear-memory allocation. Plugins are numerical
helpers and cannot mutate world authority implicitly. Later enclosure, camera,
room, broad-phase, contact, and integration plugins can use the same boundary.

The page carries each WASM binary once in a content-addressed module table.
Plugin records and the scene/software descriptors reference that key. The
emitter-supplied WASM registry lazily instantiates a module and shares the same
pending or completed instance across consumers.

Performance and inline labels are realized as loose observation objects inside
method domains. They are deliberately not structural members: the observation
belongs to the method without becoming part of its source definition.

## Permanence and emergence

Containment is authority: an object's `parent` says which region or object is
responsible for it. Form edits increment the same living revision published to
the DOM and mesh packet. Present browser cookie/local-storage data is a
representation override; permanent edits must become typed LivingDocument and
DualIR transactions.

Composite structures should emerge by adding identified objects, parts,
openings, and relationships. A renderer may combine them into an optimized
solid or batch, but the composite must preserve the contributing identities
and must be reproducible from the recipes.
