# AbstractUI scene mesh round trip

The scene mesh is a representation of document geometry, not a WebGL-owned
asset. Its contract lives in `abstract_ui_scene_mesh.py` and is embedded in an
AbstractUI model under `scene_mesh`.

The broader Pluck-compatible object registry and WASM plugin boundary are
defined in `abstract_ui_world.py`; see
`ABSTRACT_UI_PLUCK_WORLD_CONVERGENCE.md`. Object spans now contain semantic
part spans for floors, named walls, opening lintels, and ceilings.

## One identity across representations

Each object in `document_geometry.boxes` retains its AbstractUI identity. A
compiled mesh records a parallel identity span with the first vertex, vertex
count, object kind, and revision. The initial box topology uses 36 ordered
triangle vertices per instance. Ordering follows document geometry order.

The browser realizes this contract in four stages:

1. Expand each box into the published corner and face topology.
2. Run the Python-authored `instantiate_box_vertex` kernel as WebAssembly.
3. Interleave the resulting positions with neutral normals and palette roles,
   then upload that mesh to WebGL or supply it to Canvas projection.
4. Publish the identity, revision, and form parameters onto the DOM element
   carrying the same `data-node-id`.

This is deliberately a parallel identity table rather than an identity encoded
in floating-point vertex data. Future C++, Java, SDL, and native renderers can
retain the table without inheriting a browser representation.

## Crosshair and Form

The viewer camera casts a ray through its crosshair and intersects the neutral
axis-aligned boxes. Secondary action (right mouse or mapped gamepad B1) opens a
context menu for the selected identity. Its `Form` submenu is generated from
the model's parametric instruction vocabulary:

- scale height;
- scale either planar half-extent;
- restore the object's baseline form.

Applying an instruction changes the shared geometry parameters, increments the
scene revision, reruns the WebAssembly constructor, updates the GPU buffer, and
republishes the same revision to the DOM. It also emits an `apply-form` action
edge, so the edit is observable by the system-root timer and entity mezzanine.

## Present boundary

This establishes a live round trip, but it is not yet durable source editing.
The browser model owns the current revision in memory. The next layer should
express each Form instruction as a LivingDocument edit transaction, validate it
against an archetype parameter schema, and choose whether to serialize it back
to Python, SSA, or another source representation. The current contract keeps
that future authority out of WebGL and the DOM.

## Border walls and future openings

Document borders now have a neutral interpretation as object walls. The
document-geometry model publishes `dom-border` as the boundary source and the
box `height` parameter as wall height. Runtime publication places the same
value in mesh geometry, `data-wall-height`, `--wall-height`, and the focus
tooltip. Form height edits consequently revise both spatial and document
presentations through one identity.

The boundary contract reserves ordered `door`, `window`, and `portal` opening
kinds and names the future composition operation
`boundary-union-minus-openings`. No boolean solid operation is claimed yet.
This gives later archetypes a stable place to attach openings without making
DOM layout, CSS borders, or GPU buffers authoritative.

The live layout realization now implements the rectangular subset of that
contract. Every layout object owns a mandatory floor slab and four narrow wall
prisms around a hollow interior. Ordered openings subtract intervals from their
named wall; an opening shorter than the wall retains a lintel. A ceiling slab
is emitted only when wall height reaches the declared absolute maximum of
`4.0`. Courtyards default to low fencing with a full-height gate, buildings to
an enclosing entry, and rooms to their own door-bearing walls.

One document identity therefore owns a variable-length vertex span composed of
multiple 36-vertex primitives. Floor and wall colors are independent palette
roles. Radius is published as the future bevel parameter but is still marked
unimplemented in mesh geometry.

## Edit persistence

Runtime Form and aesthetic edits serialize only changed object identities,
including height, appearance, and openings. The browser attempts a one-year
cookie first and verifies the exact encoded value. It also mirrors the payload
to local storage, which is the practical fallback when the generated page is
opened through `file://` and cookies are unavailable. Saved edits restore
before the first mesh realization.
