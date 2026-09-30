# Continuation: viewer core pin + cell_ref edges (lane V)

File: `tools/view_identity_concordance.py` only.  Not committed.

## What changed

- **Core pinned to the shell.** `extract_graph` returns its `(page, row) -> row node`
  map under `_row_node`; `load_graph` pops it (never saved) and calls
  `core_identity_rows(pg, book, row_node)`: per ProcessGraph node, the
  `node_identity_cell` lookup order read only (`canonical_value
  (lexical_read_scope, id)` when `canonical_value_ids`, else `ingestion_value`
  in `operand_position_scope`, then `ingestion_value_scope`), row existence via
  `book.pages[name].history(row)`.  Nothing is posted.  Result `core_row`
  (int64, -1 = none) is saved in the npz; `ensure_core_row` defaults older npz
  / `--book` graphs to -1s.
- Pins drawn as straight chords core point -> shell row point (`PIN_RGB`,
  dim), with the core (K); bright for the picked core node or for the pins onto
  the picked shell row.  Click picking now also hits core points (when the core
  is shown and no nearer shell point); picking a core node sets `pick` to its
  row, so C / D focus/diffuse from the identity cell.  HUD: `core pinned N/M`
  (also in the focus/diffusion HUD, plus "seeded from core#N").
- `--focus core#N` resolves to that core node's row (batch; pin bright, file
  `diffuse_core<N>_<row>_...png`).  `--list TEXT` also lists matching core
  nodes with their pin.
- **Refs are edges.** `_atoms(obj, out, refs)` descends dataclass facts field
  by field, renders Enum members by name, and collects `Ref`-shaped objects
  (`.page.name`, `.row`, `.column`) into `refs` without descending them.
  `extract_graph` walks every cell of every row; each Ref becomes an
  `EDGE_CELL_REF` (kind 3, `cell_ref`) edge referenced row -> holding row,
  tagged with the holding page, timed by the holding cell's stamp, deduped per
  (source, target) keeping the earliest cell.  `KIND_WEIGHT`/`KIND_TINT`
  extended; it is REAL (diffusion blocking and world springs treat it like
  derived).  `causal_summary` prints `cell_ref N`.

## Verified

- `_atoms` on a synthetic dataclass/Enum/Ref fact: atoms `['A', 7, 'hi']`,
  one Ref collected, its row ints not read as ids.
- `--case mapping --save-graph shots/mapping_pins.npz`: core 25 nodes,
  **pinned 23/25** (all to `ingestion_value`; the resolved graph handed to the
  sink is the AST ingestion graph -- the two unpinned are `Store` nodes with no
  row); edge kinds derived 290, mint 16, heuristic 1106, **cell_ref 50**
  (hierarchy_global_value 24, control_value_binding 9, return_site_slot 8,
  reducer_field_state 4, return_site_container 4, loop_region_membership 1).
  `_row_node` absent from the npz.  `--graph` of an older npz lists fine.

## Not done

- `--case energy`: lowering currently refuses inside another lane's edit
  (`ConcordanceRefusal: identity_transition: fact ('retire', ...,
  'optional_presence_detach') is not a OperandTransition`).  No energy counts.
- No PNGs: the snapshot and `--focus core#18 --diffuse` runs were stopped from
  outside before writing.  The GL path (pins VBO, core picking, HUD) is
  therefore unobserved.

## Next edit

Run, then Read the PNGs:

    python tools/view_identity_concordance.py --graph shots/mapping_pins.npz --snapshot shots/mapping_core_pins.png --exit-after 3
    python tools/view_identity_concordance.py --graph shots/mapping_pins.npz --diffuse --focus "core#18" --settle 60 --out shots/
    python tools/view_identity_concordance.py --case energy --save-graph shots/energy_pins.npz --snapshot shots/energy_core_pins.png --exit-after 3

Check energy: `core pinned N/M` with N > 0, `cell_ref` > 0 (reducer_field_state).
