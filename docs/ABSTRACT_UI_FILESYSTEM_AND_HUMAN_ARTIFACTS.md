# AbstractUI filesystems and human artifacts

AbstractUI treats annotations, readmes, scratch notes, source files, and tests
as human-facing program objects. A human artifact has one stable identity and
three independent relations:

- `owned-by` says which semantic object is responsible for it.
- `contained-by` places its file node in a logical filesystem.
- `placed-in` says where its current representation appears in the world.

Moving a source-file token across the map therefore does not rename it or
transfer it to another class. Renaming a path does not move its world body.
Changing ownership is an explicit graph edit.

## The filesystem root

The root is a logical, revisioned `FileSystemGraph`, not an ambient operating
system directory. Its paths use canonical POSIX spelling so the graph remains
portable. Directory and file nodes form the containment graph; file nodes point
to human artifacts; human artifacts point separately to their semantic owners
and world placements.

The same graph has four realization contracts:

- The internal runtime uses a revisioned virtual file graph and inline or
  content-addressed blobs.
- C and C++ receive an explicitly rooted host-filesystem adapter. Translation
  units and symbol tables explain structural ownership; ambient path access is
  not implied.
- The web receives a manifest plus an optional OPFS or IndexedDB adapter.
  Module URLs, source maps, and bundle manifests explain structural ownership;
  the browser origin remains the security boundary.
- WebAssembly receives host-imported operations and dense node identifiers with
  a reversible logical-path table.

This gives the game an internal filesystem even when the host exposes none.

## Physical realization and welding

An unwelded human artifact is a small dynamic solid box. Its body carries the
same identity as the graph artifact. Attachment follows a deterministic state
machine:

`loose -> settling -> welded`

The transition advances only while distance is within `connection_radius` and
relative speed is below `maximum_connection_speed`, continuously for
`required_settle_time`. Disturbance resets a settling artifact to loose. A
welded artifact becomes a compound child of its owner, but its filesystem path
and authored identity remain unchanged. Detachment will be an explicit edit,
not an accidental consequence of world motion.

## Invariants

1. Every artifact is referenced by exactly one file node.
2. Filesystem node identities and canonical paths are unique within a graph.
3. Every non-root filesystem node names an existing parent node.
4. Ownership, filesystem containment, and representation placement are never
   inferred from one another.
5. Host backends may realize storage differently, but must preserve graph
   identity and relationship meaning.

The current living-data-map projection includes five seed artifacts so these
rules are inspectable rather than merely documentary. Content editing,
revision history, host adapters, dragging, and explicit attach/detach tools are
the next realization layers over this contract.
