# Nodus console world projection

`src.compiler.nodus_console_world` converts the existing Nodus GRAPH response
into an AbstractUIWorld. It then uses the existing Living Map first-person
projection and its Pluck shader selection. It creates no renderer or scheduler.

Run with an authorized GRAPH export (not credentials):

    python -m src.compiler.nodus_console_world graph.json console-world.html

For any selected subgraph, pass `ConsoleSelection(identity, module_ids,
label, panel)` to `graph_to_world` or `project_graph`. `panel` accepts the
existing Pluck `controls.Panel` dataclass or its serializable fields. Without
one, row descriptors generate a procedural inspection panel. Unselected tables
remain visible so cross-boundary edges are retained. Overlapping selections
are views, not cloned execution state. Stable uint64 identities become exact
strings in spatial identity; descriptors retain original source values.

Each console has an existing paired Living Map door to its interior region;
that interior contains the selected modules and their real row descriptors.
The schema and source records are exposed through the existing inspector.
This is a read-only view. UI actions are not permission grants, and no scripts
from metadata execute. The supplied schema is not yet rendered as editable
Pluck knobs. Bindings must use the authenticated host's real row mechanisms.

## Remaining integration, explicitly not completed

- Actual native Pluck console interaction and authenticated parameter writes.
- Live revision/telemetry subscriptions and colored relaxed wire geometry.
- Native Pluck portal traversal: its PlacedPortalFrame currently is a visual
  stub. Living Map's browser door runtime is the implemented path used here.
- Dial/treadmill scroll selection. Existing doorway declarations carry these
  records, but the current browser runtime does not consume their selectors.
- Cross-document world export/linking and browser visual acceptance.

No alternate portal engine, synthetic telemetry, or graph execution semantics
are introduced to conceal those missing pieces.
