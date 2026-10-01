"""Offline viewer for the identity concordance: every page row and every
identity it names, as one coloured point-and-line graph in a pygame OpenGL
window (the repository's viewer standard: core-profile context via
``pygame.OPENGL | DOUBLEBUF``, PyOpenGL ``compileProgram``, VAO/VBO, a
shader that draws the geometry, orbit/zoom mouse camera, ``--snapshot`` and
``--exit-after`` like ``examples/chamber_raincloud_view.py``).

The graph is read straight off the ``IdentityBook`` (``IdentityPage`` rows,
their latest fact and revision history), with nothing recomputed:

  ROW point    one per (page, row); coloured by page; height = revisions
  ID  point    one per (scope, integer) named by any row or fact; the
               scope is the first string in the row (function / scope label),
               else the page name
  time         every point carries its construction time: the book's write
               clock (``IdentityPage.stamps``) at the row's first cell, and for
               an identity the earliest row naming it.  The background is the
               kernel-smoothed average of that time over the sphere (contours =
               equal construction time).
  physics      one ``ComputationalWorld`` (``src/computational_world``: the
               dt-managed BoundSpring) holds TWO spring networks and advances
               both on one admitted dt per frame through ``WorldTickLease``:
               the SHELL is this graph, its network pinned ON the unit sphere
               (``surface=True``); the CORE is the compiler's own resolved
               ``ProcessGraph`` of the same lowering, contained INSIDE a small
               sphere at the centre, its compile order (asap levels) top to
               bottom.  Space runs the lease.  The order field is a force layer
               before the integrator (``spring_external_force``): each shell
               point is pushed down the gradient of smoothed time minus its
               own, from the same field the contours are drawn from (O toggles).
  line         row -> every id its key names (bright) or its latest fact
               names (dim), coloured by the row's page
  causal edge  a directed edge the book itself recorded: DERIVED (source
               cell -> target cell, tagged with its Stage), MINT (operand
               refs -> the minted row, tagged with its Transform); rows the
               book admitted with no source are UNSOURCED.  Read through the
               book's post api (``book.registry.pages``, ``book.edges_into``,
               ``book.mint_of``, ``book.unsourced_rows``, ``book.latch``)
               when the book has it; otherwise from the exact edge pages the
               book already keeps (``shape_transformation_concordance`` and
               its state/dependents pages, ``scope_registry`` mints, ids
               carrying the MINTED flag) plus write-order inference
               (``causal_edges``) as a weaker HEURISTIC class.  The row->id
               lines stay as a second, dimmer layer.
  cell_ref     a fact that holds a ``Ref(page, row, column)`` (FieldState
               value/effect, Ref facts, ``Unresolved.read``) is an edge from
               the referenced row to the holding row, timed by the holding
               cell's stamp; REAL like DERIVED.  A Ref's row ints are never
               read as loose ids.
  pin          each core (ProcessGraph) node is pinned to the shell row of its
               identity cell -- ``node_identity_cell``'s lookup, read only:
               ``canonical_value (lexical_read_scope, id)`` after the canonical
               relabel, else ``ingestion_value`` in ``operand_position_scope``
               / ``ingestion_value_scope``.  Drawn dim with the core (K), the
               picked node's pin bright; picking a core node selects its row
               (so D diffuses the book from that node's identity cell).
               ``--focus core#N`` does the same in batch.

The layout starts as a plan: pages on a ring, each page's rows a disc around
its slot, identities pulled to the centroid of the rows that name them.  That
plan is the (u, v) map of a sphere: u is longitude (periodic, so no seam),
v is height on the Lambert equal-area mapping (sin latitude = 1 - 2v), so
equal map area is equal sphere area.  Edges are drawn as arcs between their
points on the sphere, so an edge across the old map seam is drawn whole.  The
sphere is translucent: the far side shows through dimmer and softer.

Sources (one of):

    python tools/view_identity_concordance.py --case mapping
    python tools/view_identity_concordance.py --book module_or_book.pkl
    python tools/view_identity_concordance.py --graph saved.npz
    python tools/view_identity_concordance.py --case oscillator --backend torch --device cuda --drift --anim flow

``--backend torch --device cuda`` runs the world physics (both spring
networks) on the torch backend on the GPU; the lowering itself stays as it
is.  The oscillator case (8881 shell nodes, 19k springs) needs it: the
sanctioned force assembly is a dense node-by-edge incidence, 60-100 s per
frame on NumPy, and its matmul on a GPU.

``--case`` lowers one of ``audit_identity_concordance``'s seconds-long cases;
``--book`` reads a pickled ``IdentityBook`` or SSA module (what
``audit_identity_concordance.py --pickle`` takes); ``--save-graph out.npz``
writes the extracted graph so ``--graph`` reopens it with no compiler import.

    left-drag: turn the sphere (release to coast)   right/middle-drag or
    shift+left-drag: pan   wheel: zoom   arrows: pan
    O: order-field force on/off   K: core (process graph) on/off
    F: reset zoom/pan   T: reset rotation   click: pick a point
    M: mass (toggle; ``--mass``)  a released spin no longer decays to a stop:
        the damping drops out as the speed approaches an inertial floor
        (``MASS_FLOOR`` radians per frame), only the excess over the floor
        decays, and the sphere keeps turning at the floor until dragged again
    1 page  2 scope  3 revisions  4 degree  5 time   (colour mode)
    Space: run/stop the world   R: reset layout   B: time background
        ``--drift`` starts with the world running (after ``--settle``),
        so the points drift and the contour background evolves live from the
        first frame; Space still stops and restarts it
    G: animation  off -> build -> flow      , .: slower / faster
        build  the graph is constructed in its recorded order: a point appears
               with a cyan border when its row is written, an identity gets a
               pink border each time a later row attaches to it, and the very
               front of the compilation is white-hot
        flow   the finished graph, everything visible; the world's own
               activation cycle sweeps FLOW_GROUPS construction-order groups
               through BOTH networks: the active group's edges contract their
               rest length (the yank), its nodes glow larger, then relax --
               read straight off the spring state (glow, rest lengths), so
               the twitch is the compile order and it perturbs the physics
    C: causal focus on the picked point (again to leave)
    D: colormap diffusion from the picked/focused point (toggle; ``--diffuse``)
        heat spreads from the point over the causal edges, forward along them
        (consequence, warm) and backward (history, cool), decaying per hop
        (``--diffuse-decay``) and per clock distance, for ``--diffuse-steps``
        hops; neutral elsewhere.  UNSOURCED rows are a hard red with no heat
        (flow dies there); MINT rows and minted ids wear a green ring.

Diagnostic view (batch, one image per node):

    python tools/view_identity_concordance.py --case mapping --list scope_registry
    python tools/view_identity_concordance.py --case mapping \
        --focus "#57" --focus "transformation_decision" --depth 5 --out shots/

``--list TEXT`` prints the nodes whose label contains TEXT (with their ``#index``).
``--focus`` takes ``#index`` or a label substring that names exactly one node
(otherwise the candidates are listed and nothing is drawn).  Each image turns
the sphere so that node faces you.  The node is the middle of a colour
gradient: its past sources go negative (cool, fading with distance back,
``--depth`` hops), the nodes built on it go positive (warm, fading forward);
everything else recedes.  Every coloured node is labelled with its signed hop.
Direction comes from the book's recorded causal edges when it has them;
``causal_edges`` infers the rest from write order (an identity that already
existed is a source of the row that attaches to it; a row is the source of the
identity it births).  With ``--diffuse`` each image is the diffusion view
instead of the hop gradient.

    L: lines   P: points   [ ]: point size   - =: line alpha
    PgUp/PgDn: isolate a page   A: all pages   H: HUD   Esc: quit
"""

from __future__ import annotations

import argparse
import colorsys
import ctypes
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_ATOMS = 32          # ids taken from one row + fact, so a fat fact cannot fan out
ID_SCALE = 1.9          # identity points draw larger than row points

# causal edge classes (graph["cedge_kind"]) and node provenance (graph["node_prov"])
EDGE_DERIVED, EDGE_MINT, EDGE_HEURISTIC, EDGE_CELL_REF = 0, 1, 2, 3
EDGE_KIND_NAMES = ("derived", "mint", "heuristic", "cell_ref")   # cell_ref: a fact holds a Ref to another cell
PROV_NONE, PROV_MINT, PROV_UNSOURCED = 0, 1, 2
EDGE_PATH_API = "book-api"                       # book.registry / edges_into / mint_of / unsourced_rows
EDGE_PATH_PAGES = "book-pages+write-order"       # exact edge pages the book keeps today + causal_edges


# -- graph extraction ---------------------------------------------------------

def _named(obj):
    """A registered Stage / Transform / Latch / Page as text."""
    return str(getattr(obj, "name", obj))


def _api_causal(book, module, row_node, lookup_id, cell_time, out):
    """Causal edges through the book's post api (``identity_concordance``):

        book.registry.pages            {name: Page}; ``registry.private_pages``
                                       are the edge/mint/unsourced pages
        book.edges_into(Ref)           ((source Ref, Stage), ...)
        book.mint_of(Ref)              (Transform, (operand Ref, ...)) | None
        book.unsourced_rows()          ((Page | name, row, Reason, Stage), ...)
        book.latch                     Latch.OPEN | Latch.CLOSED

    ``None`` if the book does not expose it (then the caller reads the exact
    pages it has).  A row listed unsourced that also has an inbound DERIVED
    or MINT edge is sourced (the edge is the record); one with neither is
    UNSOURCED, where the book's flow dies."""
    registry = getattr(book, "registry", None)
    Ref = getattr(module, "Ref", None)
    if registry is None or Ref is None or not all(hasattr(book, name) for name in
                                                   ("edges_into", "mint_of", "unsourced_rows", "latch")):
        return None
    private = set(getattr(registry, "private_pages", ()))
    sourced = set()
    for page_name, page_obj in dict(registry.pages).items():
        page = book.pages.get(page_name)
        if page is None or page_name in private:
            continue
        for row in page.rows():
            target = row_node.get((page_name, row))
            if target is None:
                continue
            for column, _fact in page.history(row):
                ref = Ref(page_obj, row, column)
                when = cell_time(page_name, row, column)
                for source_ref, stage in book.edges_into(ref):
                    source = row_node.get((_named(source_ref.page), source_ref.row))
                    if source is not None:
                        out.edge(source, target, EDGE_DERIVED, _named(stage), when)
                        sourced.add(target)
                mint = book.mint_of(ref)
                if mint is not None:
                    transform, operands = mint
                    out.prov[target] = PROV_MINT
                    out.tag_of_node[target] = _named(transform)
                    sourced.add(target)
                    for operand in operands:
                        source = row_node.get((_named(operand.page), operand.row))
                        if source is not None:
                            out.edge(source, target, EDGE_MINT, _named(transform), when)
    listed = 0
    for page_obj, row, reason, stage in book.unsourced_rows():
        node = row_node.get((_named(page_obj), row))
        if node is None:
            continue
        listed += 1
        if node not in sourced:
            out.prov[node] = PROV_UNSOURCED
            out.tag_of_node[node] = f"{_named(reason)}@{_named(stage)}"
    out.unsourced_listed = listed
    out.latch = _named(book.latch)
    _flag_minted_ids(out)
    return EDGE_PATH_API


def _flag_minted_ids(out):
    """Identity nodes whose id carries the MINTED flag are novel identities;
    they wear the mint ring even before a mint edge names their operands."""
    from src.compiler.id_space import MINTED, has_flag
    for (_scope_name, value), node in list(out.id_nodes.items()):
        if has_flag(value, MINTED) and out.prov[node] == PROV_NONE:
            out.prov[node] = PROV_MINT
            out.tag_of_node[node] = "MINTED"


def _page_causal(book, row_node, lookup_id, cell_time, out):
    """Causal edges from the exact edge pages the book keeps before the post
    api: ``record_shape_transformation``'s three pages, ``mint_scope``'s
    ``scope_registry`` rows (the design's NOVEL post), and ids carrying the
    MINTED flag (novel identities that have no mint edge yet)."""
    edge_page = book.pages.get("shape_transformation_concordance")
    if edge_page is not None:
        for row in edge_page.rows():
            if not (isinstance(row, tuple) and len(row) >= 5):
                continue
            target_scope, target_id, source_scope, source_id, stage = row[:5]
            if not (isinstance(target_id, int) and isinstance(source_id, int)):
                continue
            when = cell_time("shape_transformation_concordance", row, edge_page.history(row)[0][0])
            out.edge(lookup_id(str(source_scope), source_id), lookup_id(str(target_scope), target_id),
                     EDGE_DERIVED, str(stage), when)
    state_page = book.pages.get("shape_transformation_state")
    if state_page is not None:
        for row in state_page.rows():
            for column, fact in state_page.history(row):
                if isinstance(fact, tuple) and len(fact) == 3 and fact[0] == "resolved":
                    source = row_node.get(("shape_transformation_concordance", fact[2]))
                    target = row_node.get(("shape_transformation_state", row))
                    if source is not None and target is not None:
                        stage = fact[2][4] if isinstance(fact[2], tuple) and len(fact[2]) > 4 else "shape_transformation_state"
                        out.edge(source, target, EDGE_DERIVED, str(stage),
                                 cell_time("shape_transformation_state", row, column))
    dependents = book.pages.get("shape_transformation_dependents")
    if dependents is not None:
        for row in dependents.rows():
            if isinstance(row, tuple) and len(row) == 2:
                source = row_node.get(("shape_transformation_concordance", row[1]))
                target = row_node.get(("shape_transformation_dependents", row))
                if source is not None and target is not None:
                    stage = row[1][4] if isinstance(row[1], tuple) and len(row[1]) > 4 else "shape_transformation_dependents"
                    out.edge(source, target, EDGE_DERIVED, str(stage),
                             cell_time("shape_transformation_dependents", row, dependents.history(row)[0][0]))
    scope_registry = book.pages.get("scope_registry")
    if scope_registry is not None:
        for row in scope_registry.rows():
            node = row_node.get(("scope_registry", row))
            if node is not None:
                out.prov[node] = PROV_MINT
                out.tag_of_node[node] = "mint_scope"
    _flag_minted_ids(out)
    out.latch = "n/a"
    return EDGE_PATH_PAGES


class _CausalSink:
    """Collects causal edges while ``extract_graph`` reads the book."""

    def __init__(self, n, id_nodes):
        self.src, self.dst, self.kind, self.tag, self.t = [], [], [], [], []
        self.tags: dict[str, int] = {}
        self.prov = [PROV_NONE] * n
        self.tag_of_node: dict[int, str] = {}
        self.id_nodes = id_nodes          # (scope name, id value) -> node
        self.latch = "n/a"
        self.unsourced_listed = 0         # rows the book lists unsourced (sourced or not)

    def tag_index(self, name):
        return self.tags.setdefault(str(name), len(self.tags))

    def edge(self, source, target, kind, tag, when):
        if source is None or target is None or source == target:
            return
        self.src.append(int(source)); self.dst.append(int(target))
        self.kind.append(int(kind)); self.tag.append(self.tag_index(tag))
        self.t.append(float(when) if when is not None else math.nan)

    def grow(self, n):
        while len(self.prov) < n:
            self.prov.append(PROV_NONE)

def _is_ref(obj):
    """A concordance cell reference (``identity_concordance.Ref(page, row,
    column)``), duck-typed so a pickled book reads without the class."""
    page = getattr(obj, "page", None)
    return (page is not None and hasattr(page, "name") and hasattr(obj, "row")
            and hasattr(obj, "column") and not isinstance(obj, type))


def _atoms(obj, out, refs=None, depth=0):
    """Integers (ids) and strings inside a row or fact, in order.

    Dataclass facts (NodeFact, SpanFact, BindingFact, FieldState, ...) are
    descended field by field; an Enum member is its name.  A ``Ref`` is a
    cell reference, not a set of ids: it is appended to ``refs`` (when given)
    and NOT descended, so its row ints are never read as loose identities."""
    import dataclasses
    import enum
    if depth > 5 or obj is None or isinstance(obj, bool) or (refs is None and len(out) >= MAX_ATOMS):
        return
    if _is_ref(obj):
        if refs is not None:
            refs.append(obj)
        return
    if isinstance(obj, enum.Enum):
        if len(out) < MAX_ATOMS:
            out.append(str(obj.name)[:80])
    elif isinstance(obj, int):
        if len(out) < MAX_ATOMS:
            out.append(int(obj))
    elif isinstance(obj, str):
        if len(out) < MAX_ATOMS:
            out.append(obj[:80])
    elif isinstance(obj, dict):
        for value in obj.values():
            _atoms(value, out, refs, depth + 1)
    elif isinstance(obj, (tuple, list, set, frozenset)):
        for item in obj:
            _atoms(item, out, refs, depth + 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _atoms(getattr(obj, f.name, None), out, refs, depth + 1)


def extract_graph(book, infer_edges="auto") -> dict:
    """``infer_edges``: add ``causal_edges``' write-order inference as the
    HEURISTIC class -- ``on``, ``off``, or ``auto`` (on unless the book's
    post api is the source and its latch is CLOSED, i.e. everything is
    sourced and inference has nothing left to say)."""
    import src.compiler.identity_concordance as concordance
    from src.compiler.identity_concordance import render_row
    from src.compiler.id_space import label as id_label

    private = set(getattr(getattr(book, "registry", None), "private_pages", ()))   # the edges themselves
    pages = [name for name in sorted(book.pages) if book.pages[name].rows() and name not in private]
    scopes: dict[str, int] = {}
    id_nodes: dict[tuple[int, int], int] = {}
    kind, page_of, scope_of, rev, label, detail = [], [], [], [], [], []
    born: list[float] = []                          # first clock stamp of each row
    id_keys: list[tuple[int, int]] = []
    edges: list[tuple[int, int, float]] = []       # (row node, id key slot, weight)
    row_node: dict[tuple[str, object], int] = {}   # (page name, row) -> row node
    held_refs: list[tuple] = []                    # (holding node, page, row, column, Ref)

    def scope_index(name):
        return scopes.setdefault(name, len(scopes))

    for page_index, page_name in enumerate(pages):
        page = book.pages[page_name]
        for row in page.rows():
            node = len(kind)
            row_node[(page_name, row)] = node
            key_atoms, fact_atoms = [], []
            history = page.history(row)
            key_refs = []
            _atoms(row, key_atoms, key_refs)
            _atoms(history[-1][1] if history else None, fact_atoms)
            # every cell's Refs (FieldState.value/.effect, Ref facts, Unresolved.read):
            # each is a cell_ref edge into this row at that cell's clock stamp
            for c_index, (column, fact) in enumerate(history):
                refs = list(key_refs) if c_index == 0 else []
                _atoms(fact, [], refs)
                for ref in refs:
                    held_refs.append((node, page_name, row, column, ref))
            scope = next((a for a in key_atoms if isinstance(a, str)), page_name)
            scope_id = scope_index(scope)
            kind.append(0); page_of.append(page_index); scope_of.append(scope_id)
            rev.append(len(history))
            stamps = getattr(page, "stamps", {})    # absent on books pickled before the clock
            first = [stamps[(row, c)] for c, _ in history if (row, c) in stamps]
            born.append(float(min(first)) if first else math.nan)
            label.append(f"[{page_name}] {render_row(row)}")
            detail.append(repr(history[-1][1])[:300] if history else "")
            for atoms, weight in ((key_atoms, 1.0), (fact_atoms, 0.45)):
                for atom in atoms:
                    if isinstance(atom, str):
                        continue
                    key = (scope_id, atom)
                    slot = id_nodes.get(key)
                    if slot is None:
                        slot = id_nodes[key] = len(id_keys)
                        id_keys.append(key)
                    edges.append((node, slot, weight))

    n_rows = len(kind)

    # -- the book's own causal edges (post api, else the exact pages it keeps)
    scope_names_now = {index: name for name, index in scopes.items()}
    id_node_by_name = {(scope_names_now[s], v): n_rows + slot for (s, v), slot in id_nodes.items()}

    def lookup_id(scope_name, value):
        """Node of identity ``value`` in scope ``scope_name``; made if no row named it."""
        key = (scope_index(scope_name), int(value))
        slot = id_nodes.get(key)
        if slot is None:
            slot = id_nodes[key] = len(id_keys)
            id_keys.append(key)
            id_node_by_name[(scope_name, int(value))] = n_rows + slot
        return n_rows + slot

    def cell_time(page_name, row, column):
        stamps = getattr(book.pages[page_name], "stamps", {})
        return stamps.get((row, column), math.nan)

    sink = _CausalSink(n_rows + len(id_keys), id_node_by_name)
    edge_path = _api_causal(book, concordance, row_node, lookup_id, cell_time, sink)
    if edge_path is None:
        edge_path = _page_causal(book, row_node, lookup_id, cell_time, sink)
    # Refs held inside facts: the referenced row -> the holding row (REAL edges)
    seen_refs = set()
    for target, page_name, row, column, ref in held_refs:
        try:
            source = row_node.get((_named(ref.page), ref.row))
        except TypeError:                         # an unhashable row cannot be a book row
            source = None
        if source is None or (source, target) in seen_refs:
            continue
        seen_refs.add((source, target))
        sink.edge(source, target, EDGE_CELL_REF, page_name, cell_time(page_name, row, column))
    sink.grow(n_rows + len(id_keys))

    scope_names = [None] * len(scopes)
    for name, index in scopes.items():
        scope_names[index] = name
    for scope_id, value in id_keys:
        kind.append(1); page_of.append(-1); scope_of.append(scope_id); rev.append(0)
        label.append(f"id {id_label(value)}  scope={scope_names[scope_id]}")
        detail.append("")
    edge = np.asarray(edges, dtype=np.float64).reshape((-1, 3))
    time_raw = np.full(len(kind), math.inf)
    time_raw[:n_rows] = born
    np.minimum.at(time_raw, edge[:, 1].astype(np.int64) + n_rows,
                  np.asarray(born, np.float64)[edge[:, 0].astype(np.int64)])
    known = np.isfinite(time_raw)
    t_min = time_raw[known].min() if known.any() else 0.0
    span = (time_raw[known].max() - t_min) if known.any() else 0.0
    t = np.where(known, (time_raw - t_min) / max(span, 1.0), np.nan)
    cedge_t = (np.asarray(sink.t, np.float64) - t_min) / max(span, 1.0)
    graph = {
        "pages": np.asarray(pages, dtype="U"),
        "scopes": np.asarray(scope_names, dtype="U"),
        "kind": np.asarray(kind, np.int8),
        "page_of": np.asarray(page_of, np.int32),
        "scope_of": np.asarray(scope_of, np.int32),
        "rev": np.asarray(rev, np.int32),
        "t": t.astype(np.float32),                  # construction time 0..1, NaN = unstamped
        "label": np.asarray(label, dtype="U"),
        "detail": np.asarray(detail, dtype="U"),
        "edge_row": edge[:, 0].astype(np.int64),
        "edge_id": (edge[:, 1].astype(np.int64) + n_rows),
        "edge_weight": edge[:, 2].astype(np.float32),
        # causal edges the book recorded (see ``ensure_causal`` for the arrays)
        "cedge_src": np.asarray(sink.src, np.int64),
        "cedge_dst": np.asarray(sink.dst, np.int64),
        "cedge_kind": np.asarray(sink.kind, np.int8),
        "cedge_tag": np.asarray(sink.tag, np.int32),
        "cedge_t": cedge_t.astype(np.float32),
        "tags": np.asarray(list(sink.tags), dtype="U"),
        "node_prov": np.asarray(sink.prov, np.int8),
        "node_tag": np.asarray([sink.tag_of_node.get(i, "") for i in range(len(kind))], dtype="U"),
        "edge_path": np.asarray(edge_path, dtype="U"),
        "latch": np.asarray(sink.latch, dtype="U"),
        "unsourced_listed": np.asarray(sink.unsourced_listed, np.int64),
    }
    if infer_edges == "on" or (infer_edges == "auto" and not (edge_path == EDGE_PATH_API and sink.latch == "CLOSED")):
        add_heuristic_edges(graph)
    graph["_row_node"] = row_node                  # private: popped by load_graph (not saved)
    return graph


def core_identity_rows(pg, book, row_node) -> np.ndarray:
    """Per ProcessGraph node (``process_graph_arrays`` order): the shell row
    node of its identity cell, -1 if the book has none.

    The lookup order is ``node_identity_cell``'s (topological_reducer), read
    only: after the canonical relabel the ``canonical_value`` row
    ``(lexical_read_scope, node_id)``; else the ``ingestion_value`` row in
    ``operand_position_scope`` then ``ingestion_value_scope``.  Nothing is
    posted: a node with no row stays unpinned."""
    from src.compiler.concordance_declarations import CANONICAL_VALUE, INGESTION_VALUE
    metadata = getattr(pg.G, "graph", {}) or {}
    canonical = CANONICAL_VALUE.name
    ingestion = INGESTION_VALUE.name
    lexical = metadata.get("lexical_read_scope") if metadata.get("canonical_value_ids") else None
    scopes = tuple(s for s in (metadata.get("operand_position_scope"),
                               metadata.get("ingestion_value_scope")) if s is not None)

    def has_row(page_name, row):
        page = book.pages.get(page_name)
        try:
            return page is not None and bool(page.history(row))
        except (KeyError, TypeError):
            return False

    out = np.full(len(pg.G.nodes), -1, np.int64)
    for i, node in enumerate(pg.G.nodes):
        try:
            node_id = int(node)
        except (TypeError, ValueError):
            continue
        candidates = ([(canonical, (lexical, node_id))] if lexical is not None else []) + \
                     [(ingestion, (scope, node_id)) for scope in scopes]
        for page_name, row in candidates:
            if has_row(page_name, row):
                out[i] = row_node.get((page_name, row), -1)
                break
    return out


def ensure_core_row(graph) -> None:
    """A graph without pins (``--book``, or an npz saved before them) gets
    ``core_row`` = -1 for every core node."""
    if "core_row" not in graph:
        graph["core_row"] = np.full(len(graph.get("core_t", ())), -1, np.int64)


def add_heuristic_edges(graph) -> None:
    """Append ``causal_edges``' write-order inference as HEURISTIC causal
    edges, tagged ``write-order``, so a book without the post api still has
    a flow to follow (dropped once the api path is the source)."""
    src, dst = causal_edges(graph)
    if not len(src):
        return
    tags = list(graph["tags"])
    if "write-order" not in tags:
        tags.append("write-order")
    tag = tags.index("write-order")
    graph["tags"] = np.asarray(tags, dtype="U")
    graph["cedge_src"] = np.concatenate([graph["cedge_src"], src.astype(np.int64)])
    graph["cedge_dst"] = np.concatenate([graph["cedge_dst"], dst.astype(np.int64)])
    graph["cedge_kind"] = np.concatenate([graph["cedge_kind"], np.full(len(src), EDGE_HEURISTIC, np.int8)])
    graph["cedge_tag"] = np.concatenate([graph["cedge_tag"], np.full(len(src), tag, np.int32)])
    graph["cedge_t"] = np.concatenate([graph["cedge_t"], graph["t"][dst].astype(np.float32)])


def ensure_causal(graph) -> None:
    """A graph saved before the causal arrays existed (``--graph`` of an
    older ``--save-graph``) gets them from write order alone."""
    n = len(graph["kind"])
    if "cedge_src" in graph:
        return
    graph.update({
        "cedge_src": np.zeros(0, np.int64), "cedge_dst": np.zeros(0, np.int64),
        "cedge_kind": np.zeros(0, np.int8), "cedge_tag": np.zeros(0, np.int32),
        "cedge_t": np.zeros(0, np.float32), "tags": np.asarray([], dtype="U"),
        "node_prov": np.zeros(n, np.int8), "node_tag": np.asarray([""] * n, dtype="U"),
        "edge_path": np.asarray("write-order (saved graph)", dtype="U"),
        "latch": np.asarray("n/a", dtype="U"),
        "unsourced_listed": np.asarray(0, np.int64),
    })
    add_heuristic_edges(graph)


def causal_summary(graph) -> str:
    """The HUD/stdout line: edge path, latch, edge-class counts, stages."""
    kinds = graph["cedge_kind"]
    prov = graph["node_prov"]
    counts = {name: int((kinds == k).sum()) for k, name in enumerate(EDGE_KIND_NAMES)}
    tags = [str(t) for t in graph["tags"]]
    stage_tags = sorted({tags[int(i)] for i in graph["cedge_tag"][kinds == EDGE_DERIVED]}) if len(tags) else []
    transform_tags = sorted({tags[int(i)] for i in graph["cedge_tag"][kinds == EDGE_MINT]}) if len(tags) else []
    transform_tags = sorted(set(transform_tags) | {str(t) for t in graph["node_tag"][prov == PROV_MINT] if str(t)})
    listed = int(graph.get("unsourced_listed", 0))
    return (f"edges: {graph['edge_path']}   latch {graph['latch']}   "
            f"derived {counts['derived']}  mint {counts['mint']} ({int((prov == PROV_MINT).sum())} mint nodes)  "
            f"unsourced {int((prov == PROV_UNSOURCED).sum())} ({listed} listed)  cell_ref {counts['cell_ref']}  "
            f"heuristic {counts['heuristic']}   "
            f"stages: {', '.join(stage_tags) or '-'}   transforms: {', '.join(transform_tags) or '-'}")


def layout(graph) -> np.ndarray:
    """Plan layout: page discs on a ring, identities at their rows' centroid."""
    kind, page_of = graph["kind"], graph["page_of"]
    n = len(kind)
    pages = len(graph["pages"])
    rows = np.flatnonzero(kind == 0)
    counts = np.bincount(page_of[rows], minlength=pages)
    spacing = 1.0
    disc = spacing * np.sqrt(np.maximum(counts, 1))
    ring = max(disc.max() * 1.6, disc.sum() / (2 * math.pi) * 1.3) if pages > 1 else 0.0
    home = np.zeros((n, 3), np.float64)
    seen = np.zeros(pages, np.int64)
    for node in rows:
        page = page_of[node]
        i = seen[page]; seen[page] += 1
        angle = 2 * math.pi * page / max(pages, 1)
        r, theta = spacing * math.sqrt(i + 0.5), i * 2.399963
        home[node] = (ring * math.cos(angle) + r * math.cos(theta),
                      0.0,
                      ring * math.sin(angle) + r * math.sin(theta))
    home[rows, 1] = 0.35 * np.log1p(graph["rev"][rows])
    pos = home.copy()
    er, ei, ew = graph["edge_row"], graph["edge_id"], graph["edge_weight"]
    id_nodes = np.flatnonzero(kind == 1)
    for _ in range(24):
        acc = np.zeros((n, 3)); wsum = np.zeros(n)
        np.add.at(acc, ei, pos[er] * ew[:, None]); np.add.at(wsum, ei, ew)
        np.add.at(acc, er, pos[ei] * ew[:, None]); np.add.at(wsum, er, ew)
        mean = acc / np.maximum(wsum, 1e-9)[:, None]
        pos[id_nodes] = np.where(wsum[id_nodes, None] > 0, mean[id_nodes], pos[id_nodes])
        pos[rows] = 0.55 * home[rows] + 0.45 * np.where(wsum[rows, None] > 0, mean[rows], home[rows])
    rng = np.random.default_rng(0)                   # deterministic de-overlap
    pos[id_nodes] += rng.normal(0.0, 0.15, (len(id_nodes), 3)) * np.array([1, 0.4, 1])
    return pos.astype(np.float32)


def process_graph_arrays(pg) -> dict:
    """The compiler's resolved ``ProcessGraph`` as plain arrays for the core:
    nodes in graph order, its own edges, labels from the semantic ``type``,
    and compile order as the asap level normalized to 0..1 (the graph's
    ``levels`` when the lowering computed them, else the scheduler's asap)."""
    G = pg.G
    nodes = list(G.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    levels = dict(pg.levels) if getattr(pg, "levels", None) else pg.scheduler.compute_asap_levels()
    level = np.array([float(levels.get(node, 0) or 0) for node in nodes], np.float64)
    span = (level.max() - level.min()) if len(level) else 0.0
    core_t = ((level - level.min()) / span if span > 0 else np.zeros_like(level)).astype(np.float32)
    edges = np.array([(index[a], index[b]) for a, b in G.edges if a in index and b in index], np.int64).reshape(-1, 2)
    label = [str(G.nodes[node].get("type") or G.nodes[node].get("label") or node)[:48] for node in nodes]
    return {
        "core_t": core_t,
        "core_src": edges[:, 0].copy(),
        "core_dst": edges[:, 1].copy(),
        "core_label": np.asarray(label, dtype="U"),
    }


def load_graph(args) -> dict:
    if args.graph:
        with np.load(args.graph, allow_pickle=False) as data:
            graph = {key: data[key] for key in data.files}
        ensure_causal(graph)
        ensure_core_row(graph)
        return graph
    from src.compiler.identity_concordance import IdentityBook, identity_book
    graphs = []
    if args.book:
        import pickle
        obj = pickle.loads(Path(args.book).read_bytes())
        book = obj if isinstance(obj, IdentityBook) else identity_book(obj)
    else:
        import audit_identity_concordance as audit
        book = identity_book(audit.CASES[args.case](process_graph_sink=graphs.append))
    graph = extract_graph(book, args.infer_edges)
    row_node = graph.pop("_row_node")
    if graphs:
        graph.update(process_graph_arrays(graphs[-1]))
        graph["core_row"] = core_identity_rows(graphs[-1], book, row_node)
    ensure_core_row(graph)
    return graph


# -- colour ------------------------------------------------------------------

def _hue_table(count: int, sat=0.72, val=1.0) -> np.ndarray:
    return np.asarray([colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, sat, val)
                       for i in range(max(count, 1))], np.float32)


def _heat(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)[:, None]
    cold, mid, hot = (np.array(c, np.float32) for c in
                      ((0.15, 0.25, 0.9), (0.2, 0.9, 0.55), (1.0, 0.85, 0.15)))
    return np.where(t < 0.5, cold + (mid - cold) * (t * 2), mid + (hot - mid) * (t * 2 - 1))


MODES = ("page", "scope", "revisions", "degree", "time")
ID_NEUTRAL = np.array([0.92, 0.92, 0.95], np.float32)


def _time_ramp(t: np.ndarray) -> np.ndarray:
    """Construction time 0..1: indigo (first written) -> teal -> amber (last)."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    a, b, c = (np.array(x, np.float32) for x in ((0.25, 0.20, 0.85), (0.10, 0.80, 0.75), (1.0, 0.80, 0.20)))
    return np.where(t < 0.5, a + (b - a) * (t * 2), b + (c - b) * (t * 2 - 1))


def node_colors(graph, mode: int, degree: np.ndarray, isolate: int) -> np.ndarray:
    kind, page_of, scope_of = graph["kind"], graph["page_of"], graph["scope_of"]
    n = len(kind)
    rgb = np.zeros((n, 3), np.float32)
    is_row = kind == 0
    if mode == 0:
        table = _hue_table(len(graph["pages"]))
        rgb[is_row] = table[page_of[is_row]]
        rgb[~is_row] = ID_NEUTRAL
    elif mode == 1:
        rgb[:] = _hue_table(len(graph["scopes"]), 0.6)[scope_of]
    elif mode == 2:
        top = max(int(graph["rev"].max()), 2)
        rgb[is_row] = _heat(np.log1p(graph["rev"][is_row]) / math.log1p(top))
        rgb[~is_row] = ID_NEUTRAL
    elif mode == 3:
        rgb[:] = _heat(np.log1p(degree) / math.log1p(max(degree.max(), 2)))
    else:
        rgb[:] = _time_ramp(np.nan_to_num(graph["t"], nan=0.0))
    alpha = np.full(n, 0.95, np.float32)
    if isolate >= 0:
        alpha[:] = 0.0
        alpha[is_row & (page_of == isolate)] = 0.95
        touched = np.zeros(n, bool)
        er, ei = graph["edge_row"], graph["edge_id"]
        keep = page_of[er] == isolate
        touched[ei[keep]] = True
        alpha[touched] = 0.95
    return np.concatenate([rgb, alpha[:, None]], axis=1)


# -- time field and integrator -----------------------------------------------

class TimeField:
    """The smoothed occurrence-time map over the plan, and its gradient.

    Every point splats its own construction time onto a grid; a Gaussian blur
    of (time x count) over a blur of count is the time a point *here* would
    be expected to have.  Coordinates are ``u`` in [0, 1]^2 = (longitude,
    equal-area height).  Longitude is periodic; the poles are the two ends of
    the v axis, handled by blurring over the grid mirrored at both ends, so the
    smoothing, gradients and integrator see a sphere with no seam and no edge.
    """

    def __init__(self, res=192, sigma=0.09, sigma_repel=0.035, sigma_far=0.05):
        self.res = res
        f = np.fft.fftfreq(2 * res)[:, None] ** 2 + np.fft.rfftfreq(res)[None, :] ** 2   # v mirrored: 2*res rows
        self._blur_time = np.exp(-2 * math.pi ** 2 * (sigma * res) ** 2 * f)
        self._blur_repel = np.exp(-2 * math.pi ** 2 * (sigma_repel * res) ** 2 * f)
        self._blur_far = np.exp(-2 * math.pi ** 2 * (sigma_far * res) ** 2 * f)
        self.time = np.zeros((res, res), np.float32)       # T[z, x]
        self.density = np.zeros((res, res), np.float32)
        self.grad_time = np.zeros((2, res, res), np.float32)   # d/du (x, z)
        self.grad_repel = np.zeros((2, res, res), np.float32)
        self.confidence = np.zeros((res, res), np.float32)
        self.time_far = np.zeros((res, res), np.float32)       # the same map, blurred again for the far side
        self.density_far = np.zeros((res, res), np.float32)

    def _blur(self, grid, kernel):
        mirrored = np.concatenate([grid, grid[::-1]], axis=0)
        blurred = np.fft.irfft2(np.fft.rfft2(mirrored) * kernel, s=mirrored.shape)
        return blurred[: self.res]

    def _cells(self, u):
        x = np.floor(u[:, 0] * self.res).astype(np.int64) % self.res
        z = np.clip(np.floor(u[:, 1] * self.res).astype(np.int64), 0, self.res - 1)
        return np.stack([x, z], axis=1)

    def update(self, u, t):
        known = np.isfinite(t)
        cell = self._cells(u[known])
        flat = cell[:, 1] * self.res + cell[:, 0]
        size = self.res * self.res
        count = np.bincount(flat, minlength=size).reshape(self.res, self.res).astype(np.float64)
        weighted = np.bincount(flat, weights=t[known], minlength=size).reshape(self.res, self.res)
        w, wt = self._blur(count, self._blur_time), self._blur(weighted, self._blur_time)
        eps = 1e-3 * max(float(w.max()), 1e-9)
        self.time = (wt / (w + eps)).astype(np.float32)
        self.confidence = (w / (w + 8 * eps)).astype(np.float32)
        self.density = (w / max(float(w.max()), 1e-9)).astype(np.float32)
        # The far side's extra blur is made here, on the same periodic grid, so it wraps
        # at the longitude seam exactly like the near map (mip levels clamp there).
        self.time_far = self._blur(self.time, self._blur_far).astype(np.float32)
        self.density_far = self._blur(self.density, self._blur_far).astype(np.float32)
        repel = self._blur(count, self._blur_repel)
        repel /= max(float(repel.max()), 1e-9)
        self.grad_time = self._gradient(self.time)
        self.grad_repel = self._gradient(repel)

    def _gradient(self, grid):
        """Central difference per unit u, stacked (x, z): periodic in x,
        mirrored (zero slope) across the poles in z."""
        gx = (np.roll(grid, -1, axis=1) - np.roll(grid, 1, axis=1)) * (self.res / 2)
        padded = np.concatenate([grid[:1], grid, grid[-1:]], axis=0)
        gz = (padded[2:] - padded[:-2]) * (self.res / 2)
        return np.stack([gx, gz]).astype(np.float32)

    def sample(self, grid, u):
        cell = self._cells(u)
        return grid[..., cell[:, 1], cell[:, 0]]


class SphereMap:
    """The plan's (u, v) map of the unit sphere and its geometry helpers.
    u is longitude (periodic), v is height on the Lambert equal-area mapping
    (sin latitude = 1 - 2v), so equal map area is equal sphere area."""

    def __init__(self, pos):
        lo, hi = pos[:, [0, 2]].min(axis=0), pos[:, [0, 2]].max(axis=0)
        size = float(max((hi - lo).max(), 1.0) * 1.35)
        origin = (lo + hi) / 2 - size / 2
        self.u0 = ((pos[:, [0, 2]] - origin) / size).astype(np.float64)
        self.u0[:, 0] %= 1.0
        self.u0[:, 1] = np.clip(self.u0[:, 1], 0.02, 0.98)

    @staticmethod
    def sphere(u):
        """(u, v) -> unit sphere: longitude 2 pi (u - 1/2), sin(latitude) = 1 - 2v."""
        lon = 2 * math.pi * (u[:, 0] - 0.5)
        s = 1.0 - 2.0 * u[:, 1]
        c = np.sqrt(np.maximum(1.0 - s * s, 0.0))
        return np.stack([c * np.sin(lon), s, c * np.cos(lon)], axis=1)

    @staticmethod
    def unmap(p):
        """Points -> (u, v) in [0, 1]^2 by direction (the inverse of ``sphere``)."""
        r = np.linalg.norm(p, axis=1)
        q = p / np.maximum(r, 1e-9)[:, None]
        u = (np.arctan2(q[:, 0], q[:, 2]) / (2 * math.pi) + 0.5) % 1.0
        v = np.clip((1.0 - q[:, 1]) / 2.0, 0.0, 1.0)
        return np.stack([u, v], axis=1)

    @staticmethod
    def segments(pos, a, b):
        """Line vertices (4 per edge): an arc from node ``a`` to node ``b``
        through a midpoint lifted back to the endpoints' mean radius, so an
        edge across the map seam is drawn whole and stays near its surface."""
        pa, pb = pos[a].astype(np.float64), pos[b].astype(np.float64)
        mid = (pa + pb) / 2
        length = np.linalg.norm(mid, axis=1, keepdims=True)
        reach = (np.linalg.norm(pa, axis=1) + np.linalg.norm(pb, axis=1))[:, None] / 2
        mid = np.where(length > 0.25, mid / np.maximum(length, 1e-9) * reach, mid)
        out = np.empty((len(a), 4, 3), np.float32)
        out[:, 0], out[:, 1], out[:, 2], out[:, 3] = pa, mid, mid, pb
        return out


def order_force(field, positions, t, gain):
    """The order field as a force on the shell: splat construction time from
    where the points ARE, smooth it, and push each point down the gradient of
    (smoothed time - its own time), in 3-d tangent to the sphere.  Returns the
    (n, 3) force and the mean |T - t| (the HUD's settle measure)."""
    u = SphereMap.unmap(positions)
    field.update(u, t)
    known = np.isfinite(t)
    error = np.where(known, field.sample(field.time, u) - np.nan_to_num(t), 0.0)
    conf = field.sample(field.confidence, u)
    f_map = -(gain * error * conf)[:, None] * field.sample(field.grad_time, u).T   # per unit u
    lon = 2 * math.pi * (u[:, 0] - 0.5)
    s = 1.0 - 2.0 * u[:, 1]
    c2 = np.maximum(1.0 - s * s, 0.05)
    c = np.sqrt(c2)
    d_lon = 2 * math.pi * np.stack([c * np.cos(lon), np.zeros_like(c), -c * np.sin(lon)], axis=1)
    d_lat = np.stack([2 * s / c * np.sin(lon), -2 * np.ones_like(c), 2 * s / c * np.cos(lon)], axis=1)
    # a map-space force to 3-d: each basis vector over its own squared length
    force = (f_map[:, :1] * d_lon / (4 * math.pi ** 2 * c2)[:, None]
             + f_map[:, 1:] * d_lat * (c2 / 4)[:, None])
    mean_error = float(np.abs(error[known]).mean()) if known.any() else 0.0
    return force.astype(np.float32), mean_error


def core_layout(core_t, src, dst) -> np.ndarray:
    """Initial placement of the process graph inside the core sphere: compile
    order (asap level, ``core_t`` 0..1) runs top to bottom, each level's nodes
    on a disc, then a few rounds toward neighbours' means."""
    n = len(core_t)
    pos = np.zeros((n, 3), np.float64)
    if n == 0:
        return pos.astype(np.float32)
    levels = np.round(core_t * 64).astype(np.int64)
    pos[:, 1] = (0.5 - core_t) * 1.5 * CORE_RADIUS
    for level in np.unique(levels):
        members = np.flatnonzero(levels == level)
        for i, node in enumerate(members):
            r = 0.55 * CORE_RADIUS * math.sqrt((i + 0.5) / len(members))
            theta = i * 2.399963 + level * 0.7
            pos[node, 0], pos[node, 2] = r * math.cos(theta), r * math.sin(theta)
    for _ in range(12):
        acc = np.zeros((n, 3)); cnt = np.zeros(n)
        np.add.at(acc, src, pos[dst]); np.add.at(cnt, src, 1)
        np.add.at(acc, dst, pos[src]); np.add.at(cnt, dst, 1)
        mean = acc / np.maximum(cnt, 1)[:, None]
        pull = np.where(cnt[:, None] > 0, mean, pos)
        pos[:, [0, 2]] = 0.6 * pos[:, [0, 2]] + 0.4 * pull[:, [0, 2]]
    radius = np.linalg.norm(pos, axis=1)
    pos *= np.minimum(1.0, 0.85 * CORE_RADIUS / np.maximum(radius, 1e-9))[:, None]
    return pos.astype(np.float32)


def group_masks(node_group, edge_group, groups):
    """(groups, n) and (groups, e) boolean membership from per-item group ids (-1 = none)."""
    ids = np.arange(groups)[:, None]
    return (node_group[None, :] == ids), (edge_group[None, :] == ids)


class World:
    """The one physics: ``ComputationalWorld`` with two BoundSpring networks
    (core = the process graph, contained; shell = the concordance graph, on the
    surface), advanced through ``WorldTickLease`` on one admitted dt per frame.

    Groups: FLOW_GROUPS construction-order bins.  A shell node's group is its
    construction time; a line's is its row's, a causal edge's its target's.
    A core node's group is its asap level scaled to the same bins.  The active
    group's edges contract (level mask: lines, mild; role mask: the book's
    causal edges and the core's edges, strong) and its nodes glow.
    """

    def __init__(self, graph, shell_pos, flow_seconds=16.0, tile_bytes=512 * 2 ** 20):
        from src.computational_world.spring import BoundSpringParameters
        self._params_cls = BoundSpringParameters
        self.graph = graph
        self.n_shell = len(shell_pos)
        self.shell_pos0 = shell_pos.astype(np.float32)
        er, ei = graph["edge_row"], graph["edge_id"]
        csrc, cdst, ckind = graph["cedge_src"], graph["cedge_dst"], graph["cedge_kind"]
        real = ckind != EDGE_HEURISTIC                    # heuristic edges coincide with the lines
        self.shell_edges = np.concatenate([np.stack([er, ei], axis=1),
                                           np.stack([csrc[real], cdst[real]], axis=1)]).astype(np.int64)
        t = graph["t"]
        known = np.isfinite(t)
        node_group = np.where(known, np.floor(np.clip(np.nan_to_num(t), 0, 1) * (FLOW_GROUPS - 1e-6)).astype(np.int64), -1)
        line_group = node_group[er]
        causal_group = node_group[cdst[real]]
        has_causal = np.zeros(self.n_shell, bool)
        has_causal[csrc[real]] = True
        has_causal[cdst[real]] = True
        none_lines, none_causal = np.full(len(er), -1, np.int64), np.full(int(real.sum()), -1, np.int64)
        self.shell_masks = {
            "level": group_masks(node_group, np.concatenate([line_group, none_causal]), FLOW_GROUPS),
            "role": group_masks(np.where(has_causal, node_group, -1), np.concatenate([none_lines, causal_group]), FLOW_GROUPS),
        }
        self.shell_group = node_group
        self.core_t = np.asarray(graph.get("core_t", np.zeros(0, np.float32)), np.float32)
        self.core_src = np.asarray(graph.get("core_src", np.zeros(0, np.int64)), np.int64)
        self.core_dst = np.asarray(graph.get("core_dst", np.zeros(0, np.int64)), np.int64)
        self.n_core = len(self.core_t)
        self.core_pos0 = core_layout(self.core_t, self.core_src, self.core_dst)
        core_group = np.round(np.clip(self.core_t, 0, 1) * (FLOW_GROUPS - 1)).astype(np.int64)
        self.core_masks = group_masks(core_group, core_group[self.core_dst], FLOW_GROUPS)
        self.core_group = core_group
        self.cycle_period = flow_seconds / FLOW_GROUPS
        shared = dict(k_stretch=SPRING_K, c_repulse=SPRING_REPULSE, damping=SPRING_DAMPING, growth_rate=0.0,
                      relax_rate=0.12, cycle_period=self.cycle_period, nominal_dt=1.0 / 60.0,
                      glow_rise=0.5, glow_decay=0.08, force_tile_bytes=int(tile_bytes))
        # Contraction is legacy per-nominal-step: each step the active group's rest
        # length loses base * (1 - target) and relaxes back toward base by
        # relax_rate, so a group active for many frames settles at
        # rest = base * (1 - (1 - target) / relax_rate).  Targets below give
        # 0.85 * base for the lines and 0.5 * base for the causal/core edges.
        self.yank = BoundSpringParameters(level_target=1.0 - 0.15 * 0.12, type_target=1.0, role_target=1.0 - 0.5 * 0.12,
                                          glow_peak_alpha=1.0, glow_floor_alpha=0.0,
                                          glow_peak_radius=1.0, glow_floor_radius=0.0, **shared)
        self.quiet = BoundSpringParameters(level_target=1.0, type_target=1.0, role_target=1.0,
                                           glow_peak_alpha=0.0, glow_floor_alpha=0.0,
                                           glow_peak_radius=0.0, glow_floor_radius=0.0, **shared)
        self.accepted = self.rejected = 0
        self.ext_mean = 0.0
        self.reset()

    def _install(self):
        from src.common.tensors.abstraction import AbstractTensor as AT
        from src.computational_world.state import ComputationalWorldState
        from src.computational_world.spring import install_bound_spring, append_bound_spring
        state = ComputationalWorldState.empty()
        if self.n_core:
            nmask, emask = self.core_masks
            install_bound_spring(
                state, self.core_pos0.tolist(),
                [tuple(e) for e in np.stack([self.core_src, self.core_dst], axis=1).tolist()],
                edge_level_mask=emask.tolist(), node_level_mask=nmask.tolist(),
                edge_type_mask=(emask & False).tolist(), node_type_mask=(nmask & False).tolist(),
                edge_role_mask=emask.tolist(), node_role_mask=nmask.tolist(),
                parameters=self._params_cls(boundary_radius=CORE_RADIUS, cycle_period=self.cycle_period))
        ln, le = self.shell_masks["level"]
        rn, re_ = self.shell_masks["role"]
        append_bound_spring(
            state, self.shell_pos0.tolist(), [tuple(e) for e in self.shell_edges.tolist()],
            edge_level_mask=le.tolist(), node_level_mask=ln.tolist(),
            edge_type_mask=(le & False).tolist(), node_type_mask=(ln & False).tolist(),
            edge_role_mask=re_.tolist(), node_role_mask=rn.tolist(),
            parameters=self._params_cls(boundary_radius=SHELL_RADIUS, cycle_period=self.cycle_period),
            surface=True)
        # both spheres sit at the origin: the core inside, the shell on the unit sphere
        networks = int(state.spring_boundary_center.shape[0])
        state.spring_boundary_center = AT.tensor([[0.0, 0.0, 0.0]] * networks, dtype="float32")
        state.validate_sparse_shapes()
        return state

    def reset(self):
        from src.computational_world.engine import ComputationalWorld, WorldTickLease
        from src.common.dt_system.state_table import StateTable
        self.state = self._install()
        self.world = ComputationalWorld(self.state, spring_parameters=self.quiet)
        self.lease = WorldTickLease(self.world, self.state, StateTable())
        self.lease.set_active(True)
        self.t = 0.0
        self.request = 0

    def set_flow(self, flow: bool):
        self.world.spring_parameters = self.yank if flow else self.quiet

    def step(self, dt, external=None):
        """One frame: the host force layer, then the managed window [t, t + dt]."""
        from src.common.tensors.abstraction import AbstractTensor as AT
        from src.computational_world.engine import WorldStatusBatch
        from src.common.dt_system.time_runtime import TimeWindowRequest
        if external is not None:
            self.state.spring_external_force = AT.tensor(np.ascontiguousarray(external, np.float32), dtype="float32")
            self.ext_mean = float(np.linalg.norm(external, axis=1).mean())
        self.request += 1
        start = float(self.state.managed_time.item())          # the record, not a running sum
        report = self.lease.advance_from_shell(
            TimeWindowRequest(self.request, 0, start, start + dt, dt), WorldStatusBatch)
        self.t = float(self.state.managed_time.item())
        self.accepted = len(report.result.accepted_dts)
        self.rejected = int(report.result.rejected_attempts)

    def all_positions(self):
        """Every node's position in world order (core, then shell): one read-back."""
        return np.asarray(self.state.spring_position.numpy(), np.float32).reshape(-1, 3)

    def positions(self):
        p = self.all_positions()
        return p[self.n_core:], p[:self.n_core]

    def glow(self):
        """Per node (core then shell): border strength 0..1 and size boost 0..1 from the spring state."""
        alpha = np.asarray(self.state.spring_glow_alpha.numpy(), np.float32).reshape(-1)
        radius = np.asarray(self.state.spring_glow_radius.numpy(), np.float32).reshape(-1)
        return np.clip(alpha, 0, 1), np.clip(radius, 0, 1)

    def backend_name(self):
        data = self.state.spring_position.data
        device = getattr(data, "device", None)
        return type(self.state.spring_position).__name__.replace("TensorOperations", "") + (f":{device}" if device is not None else "")

    def active_group(self):
        return int(self.state.spring_group_index.item()) % FLOW_GROUPS


def spring_effects(world, er, ei):
    """The flow animation read off the spring state: glow -> border and size,
    the active group's nodes white-hot, lit lines between glowing nodes."""
    strength, boost = world.glow()
    strength, boost = strength[world.n_core:], boost[world.n_core:]
    n = world.n_shell
    active = world.shell_group == world.active_group()
    rgb = np.where(active[:, None], FRONT_RGB, ATTACH_RGB).astype(np.float32)
    g_edge = np.maximum(strength[er], strength[ei]).astype(np.float32)
    return {
        "border": np.concatenate([rgb, strength[:, None]], axis=1).astype(np.float32),
        "size": (1.0 + 1.6 * boost).astype(np.float32),
        "node_alpha": np.ones(n, np.float32),
        "edge_alpha": (1.0 + 2.0 * g_edge).astype(np.float32),
        "edge_glow": g_edge,
        "edge_tint": np.broadcast_to(ATTACH_RGB, (len(er), 3)).astype(np.float32),
        "front": -1,
    }


# -- construction animation -----------------------------------------------------

BIRTH_RGB = np.array([0.55, 0.95, 1.00], np.float32)     # border when a point is instantiated
ATTACH_RGB = np.array([1.00, 0.45, 0.80], np.float32)    # border when a later row attaches to it
FRONT_RGB = np.array([1.00, 1.00, 1.00], np.float32)     # the very front
FLOW_FRONTS = 3                                           # waves in the legacy border sweep
FLOW_GROUPS = 24                                          # construction-order groups the world cycles through
ANIMATIONS = ("off", "build", "flow")
CORE_RADIUS = 0.42      # the process graph's boundary sphere, inside the unit shell
CORE_RING_RGB = np.array([0.95, 0.95, 1.0], np.float32)   # the core's resting ring: pale, always on
SHELL_RADIUS = 1.0
SPRING_K, SPRING_REPULSE, SPRING_DAMPING = 8.0, 0.005, 0.9  # BoundSpring parameters shared by both networks
ORDER_GAIN = 30.0       # the order field's force gain (``--order-gain``); measured: 20-40 settles |T-t| 0.136 -> 0.06 in 150 frames


def animation_effects(graph, mode: str, clock: float) -> dict:
    """Per-node border/visibility and per-edge glow at construction time ``clock``.

    Times are the book's write clock, normalized (``graph["t"]``).  An edge
    happens when its row is written; if its identity already existed (born
    earlier) the edge is an *attach* to that identity, else it is the birth of
    both.  ``build`` hides whatever has not happened yet (clock > time) and
    lets every event glow from the moment it happens; ``flow`` shows everything
    and lets the same glow run around a repeating phase, FLOW_FRONTS waves.
    """
    t, er, ei = graph["t"], graph["edge_row"], graph["edge_id"]
    n = len(t)
    known = np.isfinite(t)
    tt = np.where(known, t, -1.0)
    et = tt[er]
    attach = known[er] & known[ei] & (tt[ei] < et - 1e-9)
    if mode == "build":
        node_vis, edge_vis = tt <= clock, et <= clock
        age_n, age_e = clock - tt, clock - et
        tau, head = 0.05, 0.02
    else:
        period = 1.0 / FLOW_FRONTS
        node_vis, edge_vis = np.ones(n, bool), np.ones(len(er), bool)
        age_n, age_e = np.mod(clock - tt, period), np.mod(clock - et, period)
        tau, head = 0.03, 0.012
    g_node = np.where(node_vis & known, np.exp(-np.maximum(age_n, 0.0) / tau), 0.0)
    g_edge = np.where(edge_vis & known[er], np.exp(-np.maximum(age_e, 0.0) / tau), 0.0)
    g_attach = np.zeros(n)
    hit = attach & edge_vis
    np.maximum.at(g_attach, ei[hit], g_edge[hit])            # the older end is what is attached to
    front = np.clip(1.0 - np.maximum(age_n, 0.0) / head, 0.0, 1.0) * (g_node > 0)
    weight = g_node + g_attach + 1e-9
    rgb = (BIRTH_RGB * g_node[:, None] + ATTACH_RGB * g_attach[:, None]) / weight[:, None]
    rgb = rgb * (1.0 - front[:, None]) + FRONT_RGB * front[:, None]
    strength = np.clip(np.maximum(g_node, g_attach), 0.0, 1.0)
    fx = {
        "border": np.concatenate([rgb, strength[:, None]], axis=1).astype(np.float32),
        "size": (1.0 + 1.0 * strength).astype(np.float32),
        "node_alpha": np.where(node_vis, 1.0, 0.0).astype(np.float32),
        "edge_alpha": np.where(edge_vis, 1.0 + 2.0 * g_edge, 0.0).astype(np.float32),
        "edge_glow": g_edge.astype(np.float32),
        "edge_tint": np.where(attach[:, None], ATTACH_RGB, BIRTH_RGB).astype(np.float32),
        "front": -1,
    }
    if mode == "build":
        rows = np.flatnonzero((graph["kind"] == 0) & known & node_vis)
        if len(rows):
            fx["front"] = int(rows[np.argmax(tt[rows])])
    return fx


# -- causal focus ---------------------------------------------------------------

COOL_NEAR = np.array([0.35, 0.85, 1.00], np.float32)     # sources, one hop back
COOL_FAR = np.array([0.20, 0.28, 0.75], np.float32)      # sources, deep history
WARM_NEAR = np.array([1.00, 0.75, 0.25], np.float32)     # built on it, one hop forward
WARM_FAR = np.array([0.75, 0.25, 0.15], np.float32)      # built on it, far downstream


def causal_edges(graph):
    """(source, built) node pairs: what each node was made from.

    From the book as it stands, direction comes from write order.  A row is
    written at time ``t_row``; an identity it names that already existed
    (``t_id < t_row``) is a source of the row, and one the row created
    (``t_id == t_row``) is built by it.  This is the one place that decides
    direction: explicit source/builder links recorded on the book belong here.
    """
    t, er, ei = graph["t"], graph["edge_row"], graph["edge_id"]
    ok = np.isfinite(t[er]) & np.isfinite(t[ei])
    attach = ok & (t[ei] < t[er] - 1e-9)
    birth = ok & ~attach
    src = np.concatenate([ei[attach], er[birth]])
    dst = np.concatenate([er[attach], ei[birth]])
    return src, dst


def causal_reach(src, dst, n, focus, depth):
    """Signed hops from ``focus``: negative = sources back in time, positive =
    built on it, 0 = the node itself, NaN = not causally connected (within
    ``depth`` hops)."""
    hops = np.full(n, np.nan)
    hops[focus] = 0.0
    for direction, (a, b) in ((+1, (src, dst)), (-1, (dst, src))):
        seen = np.zeros(n, bool); seen[focus] = True
        frontier = np.zeros(n, bool); frontier[focus] = True
        for hop in range(1, depth + 1):
            step = np.zeros(n, bool)
            step[b[frontier[a]]] = True
            step &= ~seen
            if not step.any():
                break
            hops[step] = direction * hop
            seen |= step
            frontier = step
    return hops


def resolve_focus(graph, spec):
    """Node indices a ``--focus`` / ``--list`` selector names."""
    n = len(graph["kind"])
    if spec.startswith("core#"):                    # a process-graph node: its pinned identity row
        core_row = graph.get("core_row", np.zeros(0, np.int64))
        index = int(spec[5:])
        if not 0 <= index < len(core_row):
            raise SystemExit(f"no core node {spec}: the core has {len(core_row)} nodes")
        if core_row[index] < 0:
            raise SystemExit(f"core node {spec} ({graph['core_label'][index]}) has no identity row on the shell")
        return [int(core_row[index])]
    if spec.startswith("#"):
        index = int(spec[1:])
        if not 0 <= index < n:
            raise SystemExit(f"no node {spec}: the graph has {n} nodes (0..{n - 1})")
        return [index]
    needle = spec.lower()
    return [i for i in range(n) if needle in str(graph["label"][i]).lower()]


def describe_node(graph, i):
    t = graph["t"][i]
    when = "   t=--" if not np.isfinite(t) else f"   t={t:.3f}"
    return f"#{i:<5d}{when}  {graph['label'][i]}"


def focus_effects(graph, hops, depth, er, ei):
    """Colours for a causal focus: the node white at the middle of a gradient,
    sources cool and fading back, builders warm and fading forward, all else
    receded."""
    n = len(hops)
    mine = np.isfinite(hops)
    rgba = np.zeros((n, 4), np.float32)
    rgba[:] = (0.55, 0.55, 0.60, 0.10)                       # the rest of the graph recedes
    fade = np.where(mine, 1.0 - (np.abs(np.nan_to_num(hops)) - 1.0) / max(depth, 1), 0.0)
    fade = np.clip(fade, 0.0, 1.0)[:, None].astype(np.float32)
    cool = COOL_FAR + (COOL_NEAR - COOL_FAR) * fade
    warm = WARM_FAR + (WARM_NEAR - WARM_FAR) * fade
    past, future = mine & (hops < 0), mine & (hops > 0)
    rgba[past, :3], rgba[future, :3] = cool[past], warm[future]
    rgba[mine, 3] = (0.30 + 0.70 * fade[mine, 0])
    centre = mine & (hops == 0)
    rgba[centre] = (1.0, 1.0, 1.0, 1.0)
    size = np.where(mine, 2.4, 0.8).astype(np.float32)
    size[centre] = 3.6
    border = np.zeros((n, 4), np.float32)
    border[centre] = (1.0, 1.0, 1.0, 1.0)
    both = mine[er] & mine[ei]
    edge_alpha = np.where(both, 2.6, 0.12).astype(np.float32)
    edge_rgb = (rgba[er, :3] + rgba[ei, :3]) / 2
    return {"rgba": rgba, "size": size, "border": border,
            "edge_alpha": edge_alpha, "edge_rgb": edge_rgb.astype(np.float32)}


# -- colormap diffusion ---------------------------------------------------------

HISTORY_RGB = np.array([0.30, 0.72, 1.00], np.float32)       # where it came from
CONSEQUENCE_RGB = np.array([1.00, 0.58, 0.12], np.float32)   # what it caused
NEUTRAL_RGB = np.array([0.40, 0.40, 0.46], np.float32)       # no flow reaches it
UNSOURCED_RGB = np.array([1.00, 0.10, 0.16], np.float32)     # hard: flow dies here
MINT_RING_RGB = np.array([0.35, 1.00, 0.45], np.float32)     # ring on a minted row / id
KIND_WEIGHT = np.array([1.0, 1.0, 0.6, 1.0], np.float32)     # derived, mint, heuristic, cell_ref
KIND_TINT = np.array([(0.85, 0.92, 1.00), (0.35, 1.00, 0.45), (0.5, 0.5, 0.55), (1.00, 0.80, 0.35)], np.float32)
PIN_RGB = np.array([0.80, 0.55, 1.00], np.float32)          # core node -> its identity row on the shell
CLOCK_TAU = 0.25          # heat falls by 1/e across a quarter of the compilation clock
DIFFUSE_STEPS, DIFFUSE_DECAY = 6, 0.7


def diffuse_heat(graph, seeds, steps, decay):
    """Backward (history) and forward (consequence) heat from ``seeds`` over
    the causal edges.  Each hop multiplies by ``decay``, by the edge class
    weight and by exp(-clock distance / CLOCK_TAU); heat combines by max, so
    a node's heat is its strongest chain to a seed (a decaying flood, not a
    sum that saturates at hubs).  UNSOURCED nodes are where the book's flow
    dies: heat arriving over DERIVED / MINT edges stops there.  HEURISTIC
    (write-order) edges are inference made for exactly the rows the book has
    not sourced, so they carry heat through an unsourced row; the row itself
    still shows no heat (hard colour), only that it was reached.  Returns
    (history, consequence, reached unsourced)."""
    n = len(graph["kind"])
    src, dst, kind = graph["cedge_src"], graph["cedge_dst"], graph["cedge_kind"]
    t = graph["t"]
    both = np.isfinite(t[src]) & np.isfinite(t[dst])
    clock_gap = np.where(both, np.abs(np.nan_to_num(t[dst]) - np.nan_to_num(t[src])), 0.0)
    w = decay * KIND_WEIGHT[kind] * np.exp(-clock_gap / CLOCK_TAU)
    real = kind != EDGE_HEURISTIC
    blocked = graph["node_prov"] == PROV_UNSOURCED
    reached = np.zeros(n, bool)
    out = []
    for a, b in ((dst, src), (src, dst)):              # backward first, then forward
        heat = np.zeros(n)
        heat[seeds] = 1.0
        for _ in range(max(int(steps), 0)):
            carried = heat[a] * w
            new = np.zeros(n)
            np.maximum.at(new, b[real], carried[real])
            reached |= blocked & (new > 1e-3)
            new[blocked] = 0.0                            # the book's flow dies here ...
            passed = np.zeros(n)
            np.maximum.at(passed, b[~real], carried[~real])
            reached |= blocked & (passed > 1e-3)
            new = np.maximum(new, passed)                 # ... inference walks on
            grown = np.maximum(heat, new)
            if np.array_equal(grown, heat):
                break
            heat = grown
        heat[seeds] = 1.0
        out.append(heat)
    return out[0], out[1], reached


def diffuse_effects(graph, seeds, steps, decay, er, ei):
    """Colours for the diffusion view: a divergent map, history cool and
    consequence warm fading to neutral, UNSOURCED hard red with no heat, MINT
    nodes ringed green; the causal edges lit by the heat they carry and the
    row->id lines a dimmer second layer."""
    n = len(graph["kind"])
    history, consequence, reached = diffuse_heat(graph, seeds, steps, decay)
    heat = np.maximum(history, consequence)
    vis = np.clip(heat, 0.0, 1.0) ** 0.6
    hue = np.where((history >= consequence)[:, None], HISTORY_RGB, CONSEQUENCE_RGB)
    rgb = NEUTRAL_RGB * (1.0 - vis[:, None]) + hue * vis[:, None]
    alpha = 0.10 + 0.90 * vis
    prov = graph["node_prov"]
    unsourced = prov == PROV_UNSOURCED
    rgb[unsourced] = UNSOURCED_RGB
    alpha[unsourced] = np.where(reached[unsourced], 0.95, 0.40)
    size = 0.8 + 1.6 * vis
    size[unsourced & reached] = 2.2
    border = np.zeros((n, 4), np.float32)
    minted = prov == PROV_MINT
    border[minted, :3] = MINT_RING_RGB
    border[minted, 3] = np.where(heat[minted] > 0, 1.0, 0.55)
    border[unsourced] = (0.35, 0.0, 0.05, 1.0)
    seed_mask = np.zeros(n, bool)
    seed_mask[seeds] = True
    rgb[seed_mask], alpha[seed_mask], size[seed_mask] = (1.0, 1.0, 1.0), 1.0, 3.6
    border[seed_mask] = (1.0, 1.0, 1.0, 1.0)
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1).astype(np.float32)
    lit = np.minimum(vis[er], vis[ei])
    edge_alpha = (0.06 + 0.6 * lit).astype(np.float32)
    edge_rgb = ((rgba[er, :3] + rgba[ei, :3]) / 2).astype(np.float32)
    cs, cd, ck = graph["cedge_src"], graph["cedge_dst"], graph["cedge_kind"]
    flow = np.maximum(np.minimum(vis[cs], vis[cd]), 0.0) * KIND_WEIGHT[ck]
    cedge_alpha = np.where(ck == EDGE_HEURISTIC, 0.02 + 1.2 * flow, 0.12 + 2.6 * flow).astype(np.float32)
    cedge_rgb = ((rgba[cs, :3] + rgba[cd, :3]) / 2).astype(np.float32)
    return {"rgba": rgba, "size": size.astype(np.float32), "border": border,
            "edge_alpha": edge_alpha, "edge_rgb": edge_rgb,
            "cedge_alpha": cedge_alpha, "cedge_rgb": cedge_rgb,
            "history": history.astype(np.float32), "consequence": consequence.astype(np.float32),
            "heat": heat.astype(np.float32), "reached_unsourced": reached}


# -- GL ----------------------------------------------------------------------

# Every node lives in one GPU buffer in world order (core nodes, then shell
# nodes): positions are written once per physics frame, colours only when the
# mode / focus / pick changes, the spring glow once per physics frame.  The
# shaders fetch by index (``texelFetch`` on texture buffers); nothing is
# assembled per vertex on the CPU.
NODE_VERT = """#version 330 core
uniform samplerBuffer uPos;      // xyz per node
uniform samplerBuffer uColor;    // rgba per node
uniform samplerBuffer uBorder;   // rgba per node: the border outside flow
uniform samplerBuffer uScale;    // r per node
uniform samplerBuffer uGlow;     // rg per node: spring glow alpha, glow radius
uniform isamplerBuffer uMeta;    // r: activation group, g: flags (1 core, 2 minted)
uniform int uOffset;             // first node of this draw
uniform int uFlow;               // 1: border and size from the spring glow
uniform int uActive;             // the spring's active group
uniform int uOverride;           // 1: a pick mark, drawn with the colours below
uniform vec4 uOverrideColor;
uniform vec4 uOverrideBorder;
uniform float uOverrideScale;
uniform mat4 uMVP;
uniform mat3 uRot;
uniform float uPointSize;
uniform float uRef;
out vec4 vCol;
out vec4 vBorder;
out float vBack;
const vec3 FRONT = @FRONT@;
const vec3 ATTACH = @ATTACH@;
const vec3 BIRTH = @BIRTH@;
const vec3 RING = @RING@;
const vec3 MINT = @MINT@;
void main() {
  int i = uOffset + gl_VertexID;
  vec3 p = texelFetch(uPos, i).xyz;
  vec4 col = texelFetch(uColor, i);
  vec4 border = texelFetch(uBorder, i);
  float scale = texelFetch(uScale, i).r;
  if (uFlow == 1) {
    vec2 g = clamp(texelFetch(uGlow, i).rg, 0.0, 1.0);
    ivec2 meta = texelFetch(uMeta, i).rg;
    bool isActive = meta.r == uActive;
    if ((meta.g & 1) != 0) {
      border = vec4(isActive ? FRONT : (g.r > 0.02 ? BIRTH : RING), max(g.r, 0.45));
    } else {
      border = vec4(isActive ? FRONT : ATTACH, g.r);
      if ((meta.g & 2) != 0 && border.a < 0.05) border = vec4(MINT, 0.6);
    }
    scale *= 1.0 + 1.6 * g.g;
  }
  if (uOverride == 1) { col = uOverrideColor; border = uOverrideBorder; scale = uOverrideScale; }
  gl_Position = uMVP * vec4(p, 1.0);
  float facing = (uRot * p).z / max(length(p), 1e-6);
  float front = smoothstep(-0.25, 0.25, facing);
  vBack = 1.0 - front;
  vBorder = vec4(border.rgb, border.a * mix(0.4, 1.0, front));
  vCol = vec4(col.rgb * mix(0.55, 1.0, front), col.a * mix(0.32, 1.0, front));
  gl_PointSize = uPointSize * scale * mix(1.7, 1.0, front)
               * clamp(uRef / max(gl_Position.w, 1e-3), 0.5, 5.0);
}
"""
# An edge is four vertices (two segments: a -> mid -> b); the shader takes the
# edge from ``gl_VertexID / 4``, its endpoints from the static pair buffer and
# their positions from the node buffer, and lifts the midpoint back to the
# endpoints' mean radius (an arc on the sphere) unless uArc is 0.
LINE_VERT = """#version 330 core
uniform samplerBuffer uPos;
uniform isamplerBuffer uPairs;    // rg: the edge's two node indices
uniform samplerBuffer uEdgeColor; // rgba per edge
uniform samplerBuffer uGlow;
uniform int uGlowMode;            // 0 none, 1 alpha * (1 + 2g), 2 alpha + 2g
uniform int uGlowEdges;           // only the first uGlowEdges edges glow
uniform int uArc;
uniform mat4 uMVP;
uniform mat3 uRot;
out vec4 vCol;
out float vBack;
const vec3 ATTACH = @ATTACH@;
void main() {
  int e = gl_VertexID / 4;
  int k = gl_VertexID - 4 * e;
  ivec2 ab = texelFetch(uPairs, e).rg;
  vec3 pa = texelFetch(uPos, ab.x).xyz;
  vec3 pb = texelFetch(uPos, ab.y).xyz;
  vec3 mid = 0.5 * (pa + pb);
  float len = length(mid);
  if (uArc == 1 && len > 0.25) mid *= 0.5 * (length(pa) + length(pb)) / len;
  vec3 p = (k == 0) ? pa : ((k == 3) ? pb : mid);
  vec4 c = texelFetch(uEdgeColor, e);
  if (uGlowMode != 0 && e < uGlowEdges) {
    float g = max(clamp(texelFetch(uGlow, ab.x).r, 0.0, 1.0), clamp(texelFetch(uGlow, ab.y).r, 0.0, 1.0));
    c.rgb = mix(c.rgb, ATTACH, g);
    c.a = (uGlowMode == 1) ? c.a * (1.0 + 2.0 * g) : c.a + 2.0 * g;
  }
  gl_Position = uMVP * vec4(p, 1.0);
  float facing = (uRot * p).z / max(length(p), 1e-6);
  float front = smoothstep(-0.25, 0.25, facing);
  vBack = 1.0 - front;
  vCol = vec4(c.rgb * mix(0.55, 1.0, front), c.a * mix(0.32, 1.0, front));
}
"""
for _name, _value in (("@FRONT@", FRONT_RGB), ("@ATTACH@", ATTACH_RGB), ("@BIRTH@", BIRTH_RGB),
                      ("@RING@", CORE_RING_RGB), ("@MINT@", MINT_RING_RGB)):
    _glsl = "vec3(%s)" % ", ".join(f"{float(x):.4f}" for x in _value)
    NODE_VERT = NODE_VERT.replace(_name, _glsl)
    LINE_VERT = LINE_VERT.replace(_name, _glsl)
FRAG_POINT = """#version 330 core
in vec4 vCol;
in vec4 vBorder;
in float vBack;
out vec4 fragColor;
void main() {
  float d = length(gl_PointCoord - vec2(0.5));
  if (d > 0.5 || vCol.a < 0.02) discard;
  float inner = mix(mix(0.36, 0.0, vBack), 0.46, vBorder.a);   // far side: soft, blurred; a border keeps the rim crisp
  float edge = smoothstep(0.5, inner, d);
  vec3 fill = vCol.rgb * (1.0 - 0.35 * d);
  float ring = smoothstep(0.27, 0.36, d) * vBorder.a;    // border band: the outer rim of the dot
  fragColor = vec4(mix(fill, vBorder.rgb, ring), max(vCol.a, ring * 0.95) * edge);
}
"""
FRAG_LINE = """#version 330 core
in vec4 vCol;
in float vBack;
uniform float uLineAlpha;
out vec4 fragColor;
void main() {
  if (vCol.a < 0.02) discard;
  fragColor = vec4(vCol.rgb, vCol.a * uLineAlpha);
}
"""
BG_VERT = """#version 330 core
layout(location=0) in vec2 aUv;
uniform mat4 uMVP;
uniform mat3 uRot;
out vec2 vUv;
out float vBack;
out float vZ;
void main() {
  float lon = 6.28318530718 * (aUv.x - 0.5);
  float s = 1.0 - 2.0 * aUv.y;
  float c = sqrt(max(1.0 - s * s, 0.0));
  vec3 p = vec3(c * sin(lon), s, c * cos(lon));
  vUv = aUv;
  vZ = (uRot * p).z;
  vBack = 1.0 - smoothstep(-0.2, 0.2, vZ);
  gl_Position = uMVP * vec4(p, 1.0);
}
"""
BG_FRAG = """#version 330 core
in vec2 vUv;
in float vBack;
in float vZ;
uniform int uPass;             // 0: far hemisphere, 1: near hemisphere (drawn in that order)
uniform sampler2D uField;      // rg = smoothed construction time, density; ba = the same, blurred more
uniform float uContours;
out vec4 fragColor;
vec3 ramp(float t) {
  vec3 a = vec3(0.25, 0.20, 0.85), b = vec3(0.10, 0.80, 0.75), c = vec3(1.0, 0.80, 0.20);
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, t * 2.0 - 1.0);
}
void main() {
  if ((uPass == 0) == (vZ > 0.0)) discard;   // one layer per pass: a hemisphere never overlaps itself
  vec4 t = texture(uField, vUv);
  vec2 f = mix(t.rg, t.ba, vBack);                          // far side: blurrier ...
  float band = abs(fract(f.x * uContours) - 0.5);
  float line = (1.0 - vBack) * (1.0 - smoothstep(0.0, 0.06, band - 0.44));
  vec3 col = ramp(clamp(f.x, 0.0, 1.0)) * (0.55 + 0.45 * sqrt(f.y)) * mix(1.0, 0.6, vBack)   // ... and dimmer
           + vec3(line * 0.30 * (0.4 + 0.6 * f.y));
  fragColor = vec4(col, mix(0.70, 0.24, vBack));
}
"""
HUD_VERT = """#version 330 core
layout(location=0) in vec2 pos;
layout(location=1) in vec2 uv;
out vec2 vUv;
void main() { gl_Position = vec4(pos, 0.0, 1.0); vUv = uv; }
"""
HUD_FRAG = """#version 330 core
in vec2 vUv;
uniform sampler2D uTex;
out vec4 fragColor;
void main() { fragColor = texture(uTex, vUv); }
"""


def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(fov / 2)
    m = np.zeros((4, 4), np.float64)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


COAST_DAMPING = 0.94    # spin kept per frame while coasting
MASS_FLOOR = 0.0012     # radians per frame the sphere keeps turning at once mass is on (~4 deg/s at 60 fps)


class Camera:
    """The camera stays put; dragging turns the sphere itself (trackball, no
    gimbal), and a released drag coasts.  With ``mass`` on the coast never
    stops: the damping acts only on the speed above ``MASS_FLOOR``, so it
    drops out as the spin approaches the floor and the sphere keeps turning."""

    def __init__(self, mass=False):
        self.mass = bool(mass)
        self.reset_view()

    def reset_view(self):
        self.rot = np.eye(3)
        self.pan_xy = np.zeros(2)
        self.dist = 3.4
        self.spin = None

    def front(self):
        self.rot = np.eye(3)
        self.spin = None

    def rotate(self, axis, angle):
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        self.rot = (np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)) @ self.rot
        u, _, vt = np.linalg.svd(self.rot)               # keep it a rotation over long drags
        self.rot = u @ vt

    def drag(self, dx, dy):
        norm = math.hypot(dx, dy)
        if norm > 0:
            axis = np.array([dy, dx, 0.0]) / norm
            angle = norm * 0.008
            self.rotate(axis, angle)
            self.spin = (axis, angle)

    def coast(self):
        if self.spin is None:
            return
        axis, angle = self.spin
        self.rotate(axis, angle)
        if self.mass:
            # only the excess over the floor decays; at or under the floor the spin is kept as it is
            self.spin = (axis, MASS_FLOOR + (angle - MASS_FLOOR) * COAST_DAMPING if angle > MASS_FLOOR else angle)
        else:
            self.spin = (axis, angle * COAST_DAMPING) if angle > 1e-4 else None

    def face(self, point):
        """Turn the sphere so ``point`` (a position on it) looks straight at the
        camera with the poles still vertical: longitude about y, then latitude
        about x."""
        p = np.asarray(point, np.float64)
        p = p / np.linalg.norm(p)
        self.rot = np.eye(3)
        self.spin = None
        self.rotate(np.array([0.0, 1.0, 0.0]), -math.atan2(p[0], p[2]))
        self.rotate(np.array([1.0, 0.0, 0.0]), math.asin(np.clip(p[1], -1.0, 1.0)))

    def pan(self, dx, dy):
        self.pan_xy += np.array([-dx, dy]) * self.dist * 0.0016

    def mvp(self, aspect):
        view = np.eye(4)
        view[:3, 3] = (-self.pan_xy[0], -self.pan_xy[1], -self.dist)
        model = np.eye(4)
        model[:3, :3] = self.rot
        return perspective(math.radians(40), aspect, 0.1, 60.0) @ view @ model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--case", default="mapping", help="audit_identity_concordance case to lower")
    source.add_argument("--book", help="pickled IdentityBook or SSA module")
    source.add_argument("--graph", help="graph saved by --save-graph")
    ap.add_argument("--save-graph", help="write the extracted graph (.npz) and continue")
    ap.add_argument("--settle", type=int, default=None, help="world frames (1/60 s each) to run before the first frame")
    ap.add_argument("--order-gain", type=float, default=ORDER_GAIN, help="order-field force gain (0 disables the layer)")
    ap.add_argument("--backend", choices=("numpy", "torch", "jax"), default=None,
                    help="AbstractTensor backend for the world physics (AbstractTensor.set_default_backend); "
                         "set after the lowering so the compile itself is unchanged")
    ap.add_argument("--device", default=None, help="device for --backend, e.g. cuda or cuda:0")
    ap.add_argument("--tile-mb", type=float, default=512.0,
                    help="memory one tile of the spring force assembly may hold (BoundSpringParameters."
                         "force_tile_bytes); larger tiles are faster, smaller ones fit smaller devices")
    ap.add_argument("--flow-seconds", type=float, default=16.0, help="one sweep of the activation cycle through all groups")
    ap.add_argument("--drift", action="store_true",
                    help="start with the world running (Space toggles it): the points drift and the "
                         "contour background is recomputed live every frame")
    ap.add_argument("--mass", action="store_true",
                    help="start with mass on (M toggles it): a released spin keeps an inertial floor instead of stopping")
    ap.add_argument("--turn", type=float, default=0.0, help="start with the sphere turned this many degrees about its axis")
    ap.add_argument("--tilt", type=float, default=0.0, help="start with the sphere tipped this many degrees about the horizontal axis")
    ap.add_argument("--list", metavar="TEXT", help="print the nodes whose label contains TEXT, then exit")
    ap.add_argument("--focus", action="append", metavar="#N|TEXT",
                    help="draw a causal-focus image of this node (repeatable; one image each)")
    ap.add_argument("--depth", type=int, default=5, help="hops of history/consequence to colour")
    ap.add_argument("--out", default=".", help="directory for --focus images")
    ap.add_argument("--diffuse", action="store_true",
                    help="colormap diffusion from the focused/picked node over the book's causal edges (key D)")
    ap.add_argument("--diffuse-steps", type=int, default=DIFFUSE_STEPS, help="hops the heat travels")
    ap.add_argument("--diffuse-decay", type=float, default=DIFFUSE_DECAY, help="heat kept per hop (0..1)")
    ap.add_argument("--infer-edges", choices=("auto", "on", "off"), default="auto",
                    help="add write-order inference (causal_edges) as the weak HEURISTIC class; "
                         "auto = unless the book's post api is the source and its latch is CLOSED")
    ap.add_argument("--max-labels", type=int, default=45)
    ap.add_argument("--anim", choices=ANIMATIONS, default="off", help="start in this construction animation")
    ap.add_argument("--anim-at", type=float, help="freeze the animation clock at this construction time (0..1)")
    ap.add_argument("--bare", action="store_true", help="start with points and lines hidden (background only)")
    ap.add_argument("--no-bg", action="store_true", help="start with the time background hidden (B toggles it)")
    ap.add_argument("--size", type=int, nargs=2, default=(1400, 900))
    ap.add_argument("--snapshot", help="write a PNG of the first frame and continue")
    ap.add_argument("--exit-after", type=float, help="quit after this many seconds")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    graph = load_graph(args)
    if args.save_graph:
        np.savez_compressed(args.save_graph, **graph)
    kind, page_of = graph["kind"], graph["page_of"]
    n = len(kind)
    ensure_core_row(graph)
    core_row = np.asarray(graph["core_row"], np.int64)
    core_label = graph.get("core_label", np.zeros(0, dtype="U"))
    if args.list is not None:
        found = resolve_focus(graph, args.list)
        for i in found[:200]:
            print(describe_node(graph, i))
        print(f"{len(found)} node(s)" + (" (first 200 shown)" if len(found) > 200 else ""), flush=True)
        needle = args.list.lower()
        core_found = [i for i in range(len(core_label)) if needle in str(core_label[i]).lower()]
        for i in core_found[:200]:
            pin = f"-> #{int(core_row[i])}  {graph['label'][core_row[i]]}" if core_row[i] >= 0 else "-> (unpinned)"
            print(f"core#{i:<5d} {str(core_label[i])[:40]:40s} {pin}")
        if core_found:
            print(f"{len(core_found)} core node(s)" + (" (first 200 shown)" if len(core_found) > 200 else ""), flush=True)
        return
    focus_jobs, focus_core = [], {}
    for spec in args.focus or ():
        found = resolve_focus(graph, spec)
        if len(found) != 1:
            print(f"--focus {spec!r} names {len(found)} nodes; pick one with #index:", flush=True)
            for i in found[:15]:
                print("  " + describe_node(graph, i))
            raise SystemExit(2)
        if spec.startswith("core#"):
            focus_core[len(focus_jobs)] = int(spec[5:])
        focus_jobs.append(found[0])
    if args.settle is None:
        args.settle = 600 if focus_jobs else 0
    if "t" not in graph or not np.isfinite(graph["t"]).any():
        print("no construction stamps in this book (pickled before the clock); time map is empty", flush=True)
        graph["t"] = np.full(n, np.nan, np.float32)
    er, ei, ew = graph["edge_row"], graph["edge_id"], graph["edge_weight"]
    ensure_causal(graph)
    csrc, cdst, ckind = graph["cedge_src"], graph["cedge_dst"], graph["cedge_kind"]
    field = TimeField()
    plan = SphereMap(layout(graph))
    if args.backend or args.device:
        from src.common.tensors.abstraction import AbstractTensor
        AbstractTensor.set_default_backend(args.backend or "numpy", args.device)
    world = World(graph, SphereMap.sphere(plan.u0) * SHELL_RADIUS, flow_seconds=args.flow_seconds,
                  tile_bytes=int(args.tile_mb * 2 ** 20))
    print(f"world physics on {world.backend_name()}", flush=True)
    pos, core_pos = world.positions()
    pos, core_pos = pos.copy(), core_pos.copy()
    t_shell = graph["t"].astype(np.float64)
    FRAME_DT = 1.0 / 60.0

    def world_frame(order_on: bool):
        """One frame of the one physics: the order-field layer (if on), then the lease."""
        external = None
        if order_on and args.order_gain > 0:
            force, err = order_force(field, pos, t_shell, args.order_gain)
            external = np.zeros((world.n_core + world.n_shell, 3), np.float32)
            external[world.n_core:] = force
        else:
            u = SphereMap.unmap(pos)
            field.update(u, t_shell)
            known = np.isfinite(t_shell)
            err = float(np.abs(field.sample(field.time, u)[known] - t_shell[known]).mean()) if known.any() else 0.0
            if world.ext_mean:
                external = np.zeros((world.n_core + world.n_shell, 3), np.float32)
        world.step(FRAME_DT, external)
        if external is None or not external.any():
            world.ext_mean = 0.0
        shell, core = world.positions()
        pos[:] = shell
        core_pos[:] = core
        return err

    settle_started = time.perf_counter()
    for _ in range(args.settle):
        world_frame(True)
    field.update(SphereMap.unmap(pos), t_shell)
    if args.settle:
        print(f"settled {args.settle} world frames in {time.perf_counter() - settle_started:.1f}s "
              f"(world time {world.t:.2f}s)", flush=True)
    degree = (np.bincount(er, minlength=n) + np.bincount(ei, minlength=n)).astype(np.float32)
    if len(core_row) != world.n_core:                    # a saved graph whose core and pins disagree
        core_row = np.full(world.n_core, -1, np.int64)
    pinned = np.flatnonzero((core_row >= 0) & (core_row < n))
    n_pinned = len(pinned)
    summary = causal_summary(graph)
    print(f"graph: {int((kind == 0).sum())} rows, {int((kind == 1).sum())} ids, "
          f"{len(er)} lines, {len(csrc)} causal edges, {len(graph['pages'])} pages; "
          f"core: {world.n_core} process-graph nodes, {len(world.core_src)} edges, "
          f"pinned {n_pinned}/{world.n_core} "
          f"({time.perf_counter() - started:.1f}s)", flush=True)
    print(summary, flush=True)

    import pygame
    from pygame.locals import (OPENGL, DOUBLEBUF, RESIZABLE, QUIT, KEYDOWN, MOUSEBUTTONDOWN,
                               MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL, VIDEORESIZE)
    from OpenGL import GL as gl
    from OpenGL.GL.shaders import compileProgram, compileShader

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    W, H = args.size
    pygame.display.set_mode((W, H), OPENGL | DOUBLEBUF | RESIZABLE | (pygame.HIDDEN if focus_jobs else 0))
    pygame.display.set_caption("identity concordance")

    def program(vs, fs):
        return compileProgram(compileShader(vs, gl.GL_VERTEX_SHADER),
                              compileShader(fs, gl.GL_FRAGMENT_SHADER))

    prog_point, prog_line, prog_hud = program(NODE_VERT, FRAG_POINT), program(LINE_VERT, FRAG_LINE), program(HUD_VERT, HUD_FRAG)
    prog_bg = program(BG_VERT, BG_FRAG)
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    class TexBuf:
        """A GPU buffer read by the shaders as a texture buffer.  ``set`` writes
        in place (``glBufferSubData``) while the size is unchanged."""

        def __init__(self, internal, comps, dtype):
            self.buf, self.tex = gl.glGenBuffers(1), gl.glGenTextures(1)
            self.internal, self.comps, self.dtype, self.nbytes = internal, comps, dtype, -1
            self.set(np.zeros((1, comps), dtype))

        def set(self, data):
            data = np.ascontiguousarray(data, self.dtype).reshape(-1, self.comps)
            if not len(data):
                data = np.zeros((1, self.comps), self.dtype)
            gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, self.buf)
            if data.nbytes != self.nbytes:
                gl.glBufferData(gl.GL_TEXTURE_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
                self.nbytes = data.nbytes
                gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.tex)
                gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, self.internal, self.buf)
            else:
                gl.glBufferSubData(gl.GL_TEXTURE_BUFFER, 0, data.nbytes, data)

        def bind(self, unit):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.tex)

    def vec4s(data):
        data = np.asarray(data, np.float32).reshape(-1, data.shape[-1] if np.ndim(data) > 1 else 1)
        out = np.zeros((len(data), 4), np.float32)
        out[:, : data.shape[1]] = data
        return out

    empty_vao = gl.glGenVertexArrays(1)                  # attributeless draws read only buffers
    nc, M = world.n_core, world.n_core + n               # node index: core j -> j, shell i -> nc + i
    pos_tb = TexBuf(gl.GL_RGBA32F, 4, np.float32)
    glow_tb = TexBuf(gl.GL_RG32F, 2, np.float32)
    color_tb = TexBuf(gl.GL_RGBA32F, 4, np.float32)
    border_tb = TexBuf(gl.GL_RGBA32F, 4, np.float32)
    scale_tb = TexBuf(gl.GL_R32F, 1, np.float32)
    meta_tb = TexBuf(gl.GL_RG32I, 2, np.int32)
    shell_pairs_tb, shell_color_tb = TexBuf(gl.GL_RG32I, 2, np.int32), TexBuf(gl.GL_RGBA32F, 4, np.float32)
    core_pairs_tb, core_color_tb = TexBuf(gl.GL_RG32I, 2, np.int32), TexBuf(gl.GL_RGBA32F, 4, np.float32)
    pin_pairs_tb, pin_color_tb = TexBuf(gl.GL_RG32I, 2, np.int32), TexBuf(gl.GL_RGBA32F, 4, np.float32)
    pick_pairs_tb, pick_color_tb = TexBuf(gl.GL_RG32I, 2, np.int32), TexBuf(gl.GL_RGBA32F, 4, np.float32)
    scale = np.where(kind == 1, ID_SCALE, 1.0).astype(np.float32)

    # static: topology and identity of every node and edge, uploaded once
    node_color = np.zeros((M, 4), np.float32)
    node_border = np.zeros((M, 4), np.float32)
    node_scale = np.zeros((M, 1), np.float32)
    meta = np.zeros((M, 2), np.int32)
    meta[:nc, 0], meta[:nc, 1] = world.core_group, 1
    meta[nc:, 0] = world.shell_group
    meta[nc:, 1] = np.where(graph["node_prov"] == PROV_MINT, 2, 0)
    meta_tb.set(meta)
    core_rgb = _time_ramp(world.core_t) if nc else np.zeros((0, 3), np.float32)
    node_color[:nc, :3], node_color[:nc, 3] = core_rgb, 0.9
    node_border[:nc, :3], node_border[:nc, 3] = CORE_RING_RGB, 0.45
    node_scale[:nc, 0] = 1.15
    shell_pairs = np.concatenate([np.stack([er, ei], axis=1), np.stack([csrc, cdst], axis=1)]).astype(np.int32) + nc
    shell_pairs_tb.set(shell_pairs)
    core_pairs_tb.set(np.stack([world.core_src, world.core_dst], axis=1).astype(np.int32))
    core_color_tb.set(vec4s(np.concatenate([
        (core_rgb[world.core_src] + core_rgb[world.core_dst]) / 2 if nc else np.zeros((0, 3), np.float32),
        np.full((len(world.core_src), 1), 1.6, np.float32)], axis=1)))
    pin_pairs_tb.set(np.stack([pinned, core_row[pinned] + nc], axis=1).astype(np.int32) if n_pinned else np.zeros((0, 2), np.int32))

    def upload_positions():
        """The one per-frame geometry write: every node's position, in place."""
        pos_tb.set(vec4s(np.concatenate([core_pos, pos])))

    def upload_glow():
        alpha, radius = world.glow()
        glow_tb.set(np.stack([alpha, radius], axis=1))

    # HUD: a pygame-rendered surface on a screen quad, as SpeciesHud does
    hud_vao = gl.glGenVertexArrays(1); hud_vbo = gl.glGenBuffers(1)
    quad = np.array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1], np.float32)
    gl.glBindVertexArray(hud_vao); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, hud_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(1); gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(8))
    gl.glBindVertexArray(0)
    hud_tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, hud_tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    font = pygame.font.Font(None, 22)

    # background: the time field on a (u, v) sphere mesh, translucent both faces
    nu, nv = 128, 64
    gu, gv = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1))
    corner = np.stack([gu, gv], axis=-1)
    a, b, c, d = corner[:-1, :-1], corner[:-1, 1:], corner[1:, 1:], corner[1:, :-1]
    bg_quad = np.ascontiguousarray(np.stack([a, b, c, a, c, d], axis=2).reshape((-1, 2)), np.float32)
    bg_count = len(bg_quad)
    bg_vao = gl.glGenVertexArrays(1); bg_vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(bg_vao); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, bg_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, bg_quad.nbytes, bg_quad, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 8, ctypes.c_void_p(0))
    gl.glBindVertexArray(0)
    bg_tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, bg_tex)
    for name, value in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                        (gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
        gl.glTexParameteri(gl.GL_TEXTURE_2D, name, value)

    field_allocated = [False]

    def upload_field():
        data = np.ascontiguousarray(
            np.stack([field.time, field.density, field.time_far, field.density_far], axis=-1), np.float32)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, bg_tex)
        if field_allocated[0]:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, field.res, field.res, gl.GL_RGBA, gl.GL_FLOAT, data)
        else:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, field.res, field.res, 0, gl.GL_RGBA, gl.GL_FLOAT, data)
            field_allocated[0] = True

    camera = Camera(mass=args.mass)
    if args.turn:
        camera.rotate(np.array([0.0, 1.0, 0.0]), math.radians(args.turn))
    if args.tilt:
        camera.rotate(np.array([1.0, 0.0, 0.0]), math.radians(args.tilt))
    state = dict(mode=0, isolate=-1, lines=True, points=True, psize=7.0, lalpha=0.35,
                 hud=True, bg=True, physics=bool(args.drift), error=0.0, pick=-1, dirty=True, hud_dirty=True, drag=None, moved=False, last_motion=0.0,
                 anim=args.anim, anim_t0=time.time(), speed=1.0, front=-1, focus=None, diffuse=bool(args.diffuse),
                 order=True, core=True, pick_core=-1)
    world.set_flow(args.anim == "flow")
    if args.bare:
        state["lines"] = state["points"] = False
    if args.no_bg:
        state["bg"] = False

    def anim_clock():
        if args.anim_at is not None:
            return float(args.anim_at)
        cycle = 16.0 / state["speed"]
        hold = 0.15                                        # build: linger after the last write, then repeat
        return ((time.time() - state["anim_t0"]) / cycle) % (1.0 + hold)

    def write_shell_edges(rgb, alpha, cedge_rgb, cedge_alpha):
        """Edge colours: row->id lines first, then the book's causal edges."""
        shell_color_tb.set(np.concatenate([
            np.concatenate([rgb, alpha[:, None]], axis=1),
            np.concatenate([cedge_rgb, cedge_alpha[:, None]], axis=1)]))

    def rebuild():
        """Colours only (positions and glow are their own buffers).  Runs when
        the mode, isolation, focus or pick changes, and per frame only for the
        ``build`` animation, whose colours follow the construction clock."""
        colors = node_colors(graph, state["mode"], degree, state["isolate"])
        focus = state["focus"]
        fx = animation_effects(graph, "build", anim_clock()) if state["anim"] == "build" and not focus else None
        if focus:
            colors = focus["fx"]["rgba"].copy()
            border, size = focus["fx"]["border"], scale * focus["fx"]["size"]
            alpha = focus["fx"]["edge_alpha"] * ew
            if "cedge_alpha" in focus["fx"]:
                cedge_rgb, cedge_alpha = focus["fx"]["cedge_rgb"], focus["fx"]["cedge_alpha"]
            else:                                         # hop gradient: causal edges lit between coloured nodes
                both = np.isfinite(focus["hops"][csrc]) & np.isfinite(focus["hops"][cdst])
                cedge_rgb = (colors[csrc, :3] + colors[cdst, :3]) / 2
                cedge_alpha = np.where(both, 2.6, 0.08) * np.where(ckind == EDGE_HEURISTIC, 0.35, 1.0)
            edge_rgb = focus["fx"]["edge_rgb"]
            state["front"] = -1
        else:
            if fx is None:
                border, size, state["front"] = np.zeros((n, 4), np.float32), scale, -1
            else:
                colors[:, 3] *= fx["node_alpha"]
                border, size, state["front"] = fx["border"], scale * fx["size"], fx["front"]
            border = border.copy()
            ring = (graph["node_prov"] == PROV_MINT) & (border[:, 3] < 0.05)   # minted rows / ids keep their ring
            border[ring, :3], border[ring, 3] = MINT_RING_RGB, 0.6
            row_color = colors[er]
            alpha = row_color[:, 3] * colors[ei][:, 3] * ew
            edge_rgb = row_color[:, :3]
            if fx is not None:
                alpha = alpha * fx["edge_alpha"]
                edge_rgb = edge_rgb * (1.0 - fx["edge_glow"][:, None]) + fx["edge_tint"] * fx["edge_glow"][:, None]
            # the book's own edges: class tint, heuristic ones hidden (they coincide with the lines above)
            cedge_rgb = KIND_TINT[ckind]
            cedge_alpha = colors[csrc, 3] * colors[cdst, 3] * np.where(ckind == EDGE_HEURISTIC, 0.0, 0.7)
        node_color[nc:], node_border[nc:], node_scale[nc:, 0] = colors, border, size
        color_tb.set(node_color)
        border_tb.set(node_border)
        scale_tb.set(node_scale)
        write_shell_edges(edge_rgb, alpha.astype(np.float32), cedge_rgb, np.asarray(cedge_alpha, np.float32))
        state["dirty"] = False

    def rebuild_pins():
        """Pin colours: the picked core node's pin (or the pins onto the picked row) bright."""
        if not n_pinned:
            return
        bright = (pinned == state["pick_core"]) | ((state["pick"] >= 0) & (core_row[pinned] == state["pick"]))
        rgb = np.where(bright[:, None], np.array([1.0, 0.92, 1.0], np.float32), PIN_RGB)
        pin_color_tb.set(np.concatenate([rgb, np.where(bright, 3.0, 0.22).astype(np.float32)[:, None]], axis=1))

    def rebuild_pick():
        """The picked row's lines: their pairs and colours (positions come from the node buffer)."""
        rebuild_pins()
        p = state["pick"]
        if p < 0:
            return 0
        mask = (er == p) | (ei == p)
        cmask = ((csrc == p) | (cdst == p)) & (ckind != EDGE_HEURISTIC)
        pick_pairs_tb.set(np.concatenate([np.stack([er[mask], ei[mask]], axis=1),
                                          np.stack([csrc[cmask], cdst[cmask]], axis=1)]).astype(np.int32) + nc)
        pick_color_tb.set(np.concatenate([
            np.ones((int(mask.sum()), 4), np.float32),
            np.concatenate([KIND_TINT[ckind[cmask]], np.ones((int(cmask.sum()), 1), np.float32)], axis=1)]))
        return int(mask.sum() + cmask.sum())

    causal_src, causal_dst = csrc, cdst                 # the book's edges (plus write-order when no api)

    def set_focus(node):
        if node is None or node < 0:
            state["focus"] = None
        else:
            hops = causal_reach(causal_src, causal_dst, n, node, args.depth)
            if state["diffuse"]:
                fx = diffuse_effects(graph, [int(node)], args.diffuse_steps, args.diffuse_decay, er, ei)
            else:
                fx = focus_effects(graph, hops, args.depth, er, ei)
            state["focus"] = {"node": int(node), "hops": hops, "fx": fx}
            camera.face(pos[node])
            camera.dist = 3.0
        state["dirty"] = state["hud_dirty"] = True

    pick_edges = 0
    pages = list(graph["pages"])
    page_rgb = _hue_table(len(pages))

    def draw_hud(w, h):
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        lines = [(f"identity concordance   color: {MODES[state['mode']]}   "
                  f"{int((kind == 0).sum())} rows  {int((kind == 1).sum())} ids  {len(er)} lines   "
                  f"physics {'ON' if state['physics'] else 'off'}  |T-t| {state['error']:.3f}   "
                  f"mass {'ON' if camera.mass else 'off'}   "
                  f"diffusion {'ON' if state['diffuse'] else 'off'}",
                  (235, 235, 240)),
                 (f"world [{world.backend_name()}]: t {world.t:6.2f}s   last frame {world.accepted} admitted dt, {world.rejected} rejected   "
                  f"order force {'ON' if state['order'] and args.order_gain > 0 else 'off'} (gain {args.order_gain:g}, mean |F| {world.ext_mean:.3f})   "
                  f"core {world.n_core} nodes {len(world.core_src)} edges {'shown' if state['core'] else 'hidden'}, "
                  f"core pinned {n_pinned}/{world.n_core}   "
                  f"group {world.active_group()}/{FLOW_GROUPS}",
                  (200, 225, 235)),
                 (summary[:200], (190, 200, 215))]
        if state["anim"] != "off":
            clock = anim_clock()
            front = state["front"]
            note = f"animation: {state['anim']}   speed x{state['speed']:.2g}   time {min(clock, 1.0) * 100:3.0f}%"
            if front >= 0:
                note += f"   front: {str(graph['label'][front])[:90]}"
            lines.append((note, (170, 235, 255)))
        for i, name in enumerate(pages):
            dim = state["isolate"] >= 0 and state["isolate"] != i
            rgb = tuple(int(c * 255 * (0.35 if dim else 1)) for c in page_rgb[i])
            count = int((page_of == i).sum())
            lines.append((f"  {name}  ({count})", rgb))
        if 0 <= state["pick_core"] < world.n_core:
            c = state["pick_core"]
            pin = "pinned to its identity row" if core_row[c] >= 0 else "no identity row on the shell"
            lines.append(("", (0, 0, 0)))
            lines.append((f"core#{c}  {str(core_label[c])[:90]}   {pin}", tuple(int(255 * v) for v in PIN_RGB)))
        if state["pick"] >= 0:
            p = state["pick"]
            lines.append(("", (0, 0, 0)))
            lines.append((str(graph["label"][p])[:170], (255, 255, 255)))
            if graph["detail"][p]:
                lines.append(("  fact: " + str(graph["detail"][p])[:170], (200, 200, 210)))
            lines.append((f"  degree {int(degree[p])}   lines lit {pick_edges}", (200, 200, 210)))
        focus = state["focus"]
        if focus:
            hops = focus["hops"]
            fx = focus["fx"]
            if "heat" in fx:
                mode_name = f"colormap diffusion  ({args.diffuse_steps} hops, decay {args.diffuse_decay:g}, clock tau {CLOCK_TAU:g})"
                counts = (f"history {int((fx['history'] > 1e-3).sum() - 1)} nodes   "
                          f"consequence {int((fx['consequence'] > 1e-3).sum() - 1)} nodes   "
                          f"flow died at {int(fx['reached_unsourced'].sum())} unsourced   "
                          f"t={graph['t'][focus['node']]:.3f}")
            else:
                mode_name = f"causal focus  (hop gradient, depth {args.depth})"
                counts = (f"{int((hops < 0).sum())} sources back {args.depth} hops   "
                          f"{int((hops > 0).sum())} built on it   t={graph['t'][focus['node']]:.3f}")
            lines = [(f"{mode_name}   {graph['label'][focus['node']][:100]}", (255, 255, 255)),
                     (counts, (200, 205, 215)),
                     (summary[:200], (190, 200, 215)),
                     (f"core pinned {n_pinned}/{world.n_core}", (200, 225, 235))]
            c = state["pick_core"]
            if 0 <= c < world.n_core and core_row[c] == focus["node"]:
                lines.append((f"seeded from core#{c}  {str(core_label[c])[:90]}  (its identity cell)",
                              tuple(int(255 * v) for v in PIN_RGB)))
        y = 6
        for text, color in lines:
            if not text:
                y += 8; continue
            img = font.render(text, True, color)
            surf.blit(img, (8, y)); y += img.get_height() + 1
        if focus:
            draw_focus_labels(surf, w, h, focus)
        if not focus:
         surf.blit(font.render("drag turns sphere | rmb pan | wheel zoom | F view | T front | M mass | O order | K core | 1-5 color | SPACE physics | R reset | B bg | G animate | L P lines/points | PgUp/Dn page | C focus | D diffuse | H hud", True, (150, 150, 160)), (8, h - 22))
        data = pygame.image.tostring(surf, "RGBA", True)
        gl.glBindTexture(gl.GL_TEXTURE_2D, hud_tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)

    def draw_focus_labels(surf, w, h, focus):
        hops = focus["hops"]
        rgba = focus["fx"]["rgba"]
        mvp = state["mvp"]
        clip = np.concatenate([pos, np.ones((n, 1), np.float32)], axis=1) @ mvp.T.astype(np.float32)
        facing = (pos @ camera.rot.T)[:, 2] / np.maximum(np.linalg.norm(pos, axis=1), 1e-9)
        ok = (clip[:, 3] > 1e-6)
        sx = (clip[:, 0] / np.where(ok, clip[:, 3], 1.0) * 0.5 + 0.5) * w
        sy = (1 - (clip[:, 1] / np.where(ok, clip[:, 3], 1.0) * 0.5 + 0.5)) * h
        heat = focus["fx"].get("heat")
        if heat is None:
            order = [i for i in np.argsort(np.abs(np.nan_to_num(hops, nan=1e9)), kind="stable")
                     if np.isfinite(hops[i]) and ok[i] and facing[i] > -0.1][: args.max_labels]
        else:                                                 # hottest first; dead unsourced ends too
            score = heat + np.where(focus["fx"]["reached_unsourced"], 0.5, 0.0)
            order = [i for i in np.argsort(-score, kind="stable")
                     if score[i] > 0.02 and ok[i] and facing[i] > -0.1][: args.max_labels]
        small = pygame.font.Font(None, 19)
        taken = [pygame.Rect(0, 0, 330, 60)]                  # the title block
        for i in order:
            if np.isfinite(hops[i]):
                hop = int(hops[i])
                tag = "0" if hop == 0 else f"{hop:+d}"
            else:
                tag = f"~{heat[i]:.2f}"
            if heat is not None and graph["node_prov"][i] == PROV_UNSOURCED:
                tag += " UNSOURCED"
            elif heat is not None and graph["node_prov"][i] == PROV_MINT:
                tag += " MINT"
            text = f"{tag} {str(graph['label'][i])[:56]}"
            color = tuple(int(255 * c) for c in rgba[i, :3])
            width, height = small.size(text)
            for dx, dy in ((9, -height - 3), (9, 3), (-width - 9, -height - 3), (-width - 9, 3),
                           (9, -height // 2), (-width - 9, -height // 2), (-width // 2, -height - 12),
                           (-width // 2, 10)):
                rect = pygame.Rect(int(sx[i] + dx), int(sy[i] + dy), width, height)
                if rect.left >= 0 and rect.right <= w and rect.top >= 0 and rect.bottom <= h - 60 \
                        and rect.collidelist(taken) < 0:
                    taken.append(rect.inflate(4, 2))
                    surf.blit(small.render(text, True, (8, 10, 14)), (rect.x + 1, rect.y + 1))
                    surf.blit(small.render(text, True, color), rect.topleft)
                    break
        # legend: the gradient the colours are read against
        bar = pygame.Rect(w // 2 - 260, h - 46, 520, 12)
        for k in range(bar.width):
            v = (k / (bar.width - 1)) * 2 - 1                # -1 sources ... +1 builders
            if heat is not None:                             # divergent: history | neutral | consequence
                f = min(1.0, abs(v)) ** 0.6
                col = NEUTRAL_RGB * (1.0 - f) + (HISTORY_RGB if v < 0 else CONSEQUENCE_RGB) * f
                if abs(v) < 0.02:
                    col = (1.0, 1.0, 1.0)
            else:
                f = 1.0 - (abs(v) * args.depth - 1.0) / max(args.depth, 1) if abs(v) > 0 else 1.0
                f = max(0.0, min(1.0, f))
                if abs(v) < 0.02:
                    col = (1.0, 1.0, 1.0)
                elif v < 0:
                    col = COOL_FAR + (COOL_NEAR - COOL_FAR) * f
                else:
                    col = WARM_FAR + (WARM_NEAR - WARM_FAR) * f
            pygame.draw.line(surf, tuple(int(255 * c) for c in col), (bar.x + k, bar.y), (bar.x + k, bar.bottom))
        if heat is not None:
            surf.blit(small.render("history (hot = near)", True, (150, 200, 235)), (bar.x, bar.bottom + 3))
            surf.blit(small.render("this node", True, (255, 255, 255)), (bar.centerx - 26, bar.bottom + 3))
            surf.blit(small.render("consequence (hot = near)", True, (255, 190, 120)), (bar.right - 160, bar.bottom + 3))
            swatch = pygame.Rect(bar.right + 24, bar.y, 12, 12)
            pygame.draw.rect(surf, tuple(int(255 * c) for c in UNSOURCED_RGB), swatch)
            surf.blit(small.render("unsourced (flow dies)", True, (235, 150, 150)), (swatch.right + 4, swatch.y - 2))
            ring = pygame.Rect(bar.right + 24, bar.y + 16, 12, 12)
            pygame.draw.ellipse(surf, tuple(int(255 * c) for c in MINT_RING_RGB), ring, 2)
            surf.blit(small.render("mint (ring)", True, (170, 235, 180)), (ring.right + 4, ring.y - 2))
        else:
            surf.blit(small.render(f"sources (past)  -{args.depth}", True, (150, 200, 235)), (bar.x, bar.bottom + 3))
            surf.blit(small.render("this node", True, (255, 255, 255)), (bar.centerx - 26, bar.bottom + 3))
            surf.blit(small.render(f"+{args.depth}  built on it (future)", True, (255, 190, 120)),
                      (bar.right - 150, bar.bottom + 3))

    def pick_at(mx, my, mvp, w, h):
        clip = np.concatenate([pos, np.ones((n, 1), np.float32)], axis=1) @ mvp.T.astype(np.float32)
        ok = clip[:, 3] > 1e-6
        ndc = clip[:, :2] / np.where(ok, clip[:, 3], 1.0)[:, None]
        sx, sy = (ndc[:, 0] * 0.5 + 0.5) * w, (1 - (ndc[:, 1] * 0.5 + 0.5)) * h
        dist2 = (sx - mx) ** 2 + (sy - my) ** 2
        colors_alpha = node_colors(graph, 0, degree, state["isolate"])[:, 3]
        dist2 = np.where(ok & (colors_alpha > 0.02), dist2, np.inf)
        near = np.flatnonzero(dist2 < 16 ** 2)
        best, best_w = -1, math.inf
        if len(near):
            best = int(near[np.argmin(clip[near, 3])])        # the one nearest the camera
            best_w = float(clip[best, 3])
        if state["core"] and world.n_core:                    # a core node: select its identity row
            cclip = np.concatenate([core_pos, np.ones((world.n_core, 1), np.float32)], axis=1) @ mvp.T.astype(np.float32)
            cok = cclip[:, 3] > 1e-6
            cndc = cclip[:, :2] / np.where(cok, cclip[:, 3], 1.0)[:, None]
            csx, csy = (cndc[:, 0] * 0.5 + 0.5) * w, (1 - (cndc[:, 1] * 0.5 + 0.5)) * h
            cd2 = np.where(cok, (csx - mx) ** 2 + (csy - my) ** 2, np.inf)
            cnear = np.flatnonzero(cd2 < 16 ** 2)
            if len(cnear):
                c = int(cnear[np.argmin(cclip[cnear, 3])])
                if float(cclip[c, 3]) < best_w or best < 0:
                    return int(core_row[c]), c
        return best, -1

    clock = pygame.time.Clock()
    frame_count = 0
    t0 = time.time()
    snapshot_pending = args.snapshot

    def update_scene():
        nonlocal pick_edges, frame_count
        stepped = bool(state["physics"])
        if stepped:
            state["error"] = world_frame(state["order"])
            state["hud_dirty"] = True
        if stepped or state["dirty"]:
            upload_positions()
            upload_glow()
            upload_field()
        if state["anim"] != "off":
            frame_count += 1
            if frame_count % 6 == 0:
                state["hud_dirty"] = True
        was_dirty = state["dirty"]
        if was_dirty or (state["anim"] == "build" and not state["focus"]):
            rebuild()
        if was_dirty:
            pick_edges = rebuild_pick()

    def draw_scene(w, h):
        mvp = camera.mvp(w / max(h, 1))
        state["mvp"] = mvp
        mvp32 = np.ascontiguousarray(mvp.T, np.float32)
        rot32 = np.ascontiguousarray(camera.rot, np.float32)
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.035, 0.04, 0.055, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ref = float(camera.dist)

        locations = {}

        def loc(prog, name):
            key = (int(prog), name)
            if key not in locations:
                locations[key] = gl.glGetUniformLocation(prog, name)
            return locations[key]

        flow = 1 if (state["anim"] == "flow" and not state["focus"]) else 0
        active = world.active_group()

        def use(prog, psize=None, lalpha=None):
            gl.glUseProgram(prog)
            gl.glUniformMatrix4fv(loc(prog, "uMVP"), 1, gl.GL_FALSE, mvp32)
            gl.glUniformMatrix3fv(loc(prog, "uRot"), 1, gl.GL_TRUE, rot32)
            if psize is not None:
                gl.glUniform1f(loc(prog, "uPointSize"), psize)
                gl.glUniform1f(loc(prog, "uRef"), ref)
            if lalpha is not None:
                gl.glUniform1f(loc(prog, "uLineAlpha"), lalpha)

        def draw_nodes(offset, count, override=None):
            use(prog_point, psize=state["psize"])
            for unit, (buf, name) in enumerate(((pos_tb, "uPos"), (color_tb, "uColor"), (border_tb, "uBorder"),
                                                (scale_tb, "uScale"), (glow_tb, "uGlow"), (meta_tb, "uMeta")), start=1):
                buf.bind(unit)
                gl.glUniform1i(loc(prog_point, name), unit)
            gl.glUniform1i(loc(prog_point, "uOffset"), int(offset))
            gl.glUniform1i(loc(prog_point, "uFlow"), flow)
            gl.glUniform1i(loc(prog_point, "uActive"), active)
            gl.glUniform1i(loc(prog_point, "uOverride"), 0 if override is None else 1)
            if override is not None:
                gl.glUniform4f(loc(prog_point, "uOverrideColor"), *override[0])
                gl.glUniform4f(loc(prog_point, "uOverrideBorder"), *override[1])
                gl.glUniform1f(loc(prog_point, "uOverrideScale"), override[2])
            gl.glBindVertexArray(empty_vao)
            gl.glDrawArrays(gl.GL_POINTS, 0, int(count))

        def draw_edges(pairs, colors, count, lalpha, glow_mode=0, glow_edges=0, arc=1):
            use(prog_line, lalpha=lalpha)
            for unit, (buf, name) in enumerate(((pos_tb, "uPos"), (pairs, "uPairs"), (colors, "uEdgeColor"),
                                                (glow_tb, "uGlow")), start=1):
                buf.bind(unit)
                gl.glUniform1i(loc(prog_line, name), unit)
            gl.glUniform1i(loc(prog_line, "uGlowMode"), glow_mode if flow else 0)
            gl.glUniform1i(loc(prog_line, "uGlowEdges"), int(glow_edges))
            gl.glUniform1i(loc(prog_line, "uArc"), int(arc))
            gl.glBindVertexArray(empty_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, 4 * int(count))

        if state["bg"]:
            gl.glUseProgram(prog_bg)
            gl.glUniformMatrix4fv(loc(prog_bg, "uMVP"), 1, gl.GL_FALSE, mvp32)
            gl.glUniformMatrix3fv(loc(prog_bg, "uRot"), 1, gl.GL_TRUE, rot32)
            gl.glUniform1f(loc(prog_bg, "uContours"), 12.0)
            gl.glUniform1i(loc(prog_bg, "uField"), 0)
            gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, bg_tex)
            gl.glBindVertexArray(bg_vao)
            for hemisphere in (0, 1):       # far side first, then the near side over it
                gl.glUniform1i(loc(prog_bg, "uPass"), hemisphere)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, bg_count)
        if state["core"] and nc:
            # the process graph inside: over the translucent near hemisphere, under the shell
            draw_edges(core_pairs_tb, core_color_tb, len(world.core_src), min(1.0, state["lalpha"] * 1.4),
                       glow_mode=2, glow_edges=len(world.core_src))
            draw_nodes(0, nc)
            if n_pinned:                                # core -> shell identity pins
                draw_edges(pin_pairs_tb, pin_color_tb, n_pinned, 1.0, arc=0)
        if state["lines"]:
            draw_edges(shell_pairs_tb, shell_color_tb, len(er) + len(csrc), state["lalpha"],
                       glow_mode=1, glow_edges=len(er))
        if state["pick"] >= 0 and pick_edges:
            draw_edges(pick_pairs_tb, pick_color_tb, pick_edges, 0.9)
        if state["points"]:
            draw_nodes(nc, n)
        if state["pick"] >= 0:
            draw_nodes(nc + state["pick"], 1, ((1, 1, 1, 1), (0, 0, 0, 0), 2.2))
        if 0 <= state["pick_core"] < nc and state["core"]:
            draw_nodes(state["pick_core"], 1, ((1, 0.92, 1, 1), (*PIN_RGB, 1.0), 2.2))
        if state["hud"]:
            if state["hud_dirty"]:
                draw_hud(w, h); state["hud_dirty"] = False
            gl.glUseProgram(prog_hud)
            gl.glUniform1i(loc(prog_hud, "uTex"), 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, hud_tex)
            gl.glBindVertexArray(hud_vao)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)
    if focus_jobs:                                  # batch: one image per node, then done
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        camera.spin = None
        for job, node in enumerate(focus_jobs):
            state["pick_core"] = focus_core.get(job, -1)   # a core# job: its pin drawn bright
            set_focus(node)
            update_scene()
            w, h = pygame.display.get_window_size()
            draw_scene(w, h)                        # first pass builds the HUD; draw again with it current
            state["hud_dirty"] = True
            draw_scene(w, h)
            slug = re.sub(r"[^A-Za-z0-9]+", "_", str(graph["label"][node]))[:48].strip("_")
            core_tag = f"core{focus_core[job]}_" if job in focus_core else ""
            path = out_dir / f"{'diffuse' if state['diffuse'] else 'focus'}_{core_tag}{node}_{slug}.png"
            buf = gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            image = pygame.image.frombuffer(buf, (w, h), "RGBA")
            pygame.image.save(pygame.transform.flip(image, False, True), str(path))
            print(f"{describe_node(graph, node)}\n  -> {path}", flush=True)
        pygame.quit()
        return

    while True:
        w, h = pygame.display.get_window_size()
        mvp = camera.mvp(w / max(h, 1))                       # the matrix events pick against
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit(); return
            if ev.type == VIDEORESIZE:
                state["hud_dirty"] = True
            if ev.type == KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    pygame.quit(); return
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    state["mode"] = k - pygame.K_1; state["dirty"] = state["hud_dirty"] = True
                elif k == pygame.K_SPACE: state["physics"] = not state["physics"]; state["hud_dirty"] = True
                elif k == pygame.K_r:
                    world.reset(); world.set_flow(state["anim"] == "flow")
                    shell, core = world.positions(); pos[:] = shell; core_pos[:] = core
                    field.update(SphereMap.unmap(pos), t_shell); state["dirty"] = True
                elif k == pygame.K_o: state["order"] = not state["order"]; state["hud_dirty"] = True
                elif k == pygame.K_k: state["core"] = not state["core"]; state["hud_dirty"] = True
                elif k == pygame.K_b: state["bg"] = not state["bg"]
                elif k == pygame.K_g:
                    state["anim"] = ANIMATIONS[(ANIMATIONS.index(state["anim"]) + 1) % len(ANIMATIONS)]
                    state["anim_t0"] = time.time(); state["dirty"] = state["hud_dirty"] = True
                    world.set_flow(state["anim"] == "flow")
                    if state["anim"] == "flow":
                        state["physics"] = True                     # the twitch is the world's cycle
                elif k == pygame.K_COMMA: state["speed"] = max(0.1, state["speed"] / 1.5); state["hud_dirty"] = True
                elif k == pygame.K_PERIOD: state["speed"] = min(20.0, state["speed"] * 1.5); state["hud_dirty"] = True
                elif k == pygame.K_l: state["lines"] = not state["lines"]
                elif k == pygame.K_p: state["points"] = not state["points"]
                elif k == pygame.K_LEFTBRACKET: state["psize"] = max(1.0, state["psize"] - 1)
                elif k == pygame.K_RIGHTBRACKET: state["psize"] += 1
                elif k == pygame.K_MINUS: state["lalpha"] = max(0.02, state["lalpha"] * 0.7)
                elif k == pygame.K_EQUALS: state["lalpha"] = min(1.0, state["lalpha"] / 0.7)
                elif k == pygame.K_f: camera.reset_view()
                elif k == pygame.K_t: camera.front()
                elif k == pygame.K_m: camera.mass = not camera.mass; state["hud_dirty"] = True
                elif k == pygame.K_h: state["hud"] = not state["hud"]
                elif k == pygame.K_c:
                    set_focus(None if state["focus"] else (state["pick"] if state["pick"] >= 0 else None))
                elif k == pygame.K_d:
                    state["diffuse"] = not state["diffuse"]
                    focus = state["focus"]
                    set_focus(focus["node"] if focus else (state["pick"] if state["pick"] >= 0 else None))
                elif k == pygame.K_a: state["isolate"] = -1; state["dirty"] = state["hud_dirty"] = True
                elif k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN) and pages:
                    step = 1 if k == pygame.K_PAGEUP else -1
                    state["isolate"] = (state["isolate"] + step) % len(pages)
                    state["dirty"] = state["hud_dirty"] = True
            if ev.type == MOUSEBUTTONDOWN and ev.button in (1, 2, 3):
                shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                state["drag"] = "pan" if (ev.button in (2, 3) or shift) else "orbit"
                state["moved"] = False
                camera.spin = None
            if ev.type == MOUSEBUTTONUP and ev.button in (1, 2, 3):
                if ev.button == 1 and state["drag"] == "orbit" and not state["moved"]:
                    state["pick"], state["pick_core"] = pick_at(ev.pos[0], ev.pos[1], mvp, w, h)
                    pick_edges = rebuild_pick(); state["hud_dirty"] = True
                if time.time() - state["last_motion"] > 0.06:
                    camera.spin = None                        # released while still: no coast
                state["drag"] = None
            if ev.type == MOUSEMOTION and state["drag"]:
                if abs(ev.rel[0]) + abs(ev.rel[1]) > 0:
                    state["moved"] = True
                    state["last_motion"] = time.time()
                if state["drag"] == "orbit":
                    camera.drag(ev.rel[0], ev.rel[1])
                else:
                    camera.pan(ev.rel[0], ev.rel[1])
            if ev.type == MOUSEWHEEL:
                camera.dist = float(np.clip(camera.dist * (0.88 if ev.y > 0 else 1.14), 0.5, 1e6))
        if state["drag"] is None:
            camera.coast()
        keys = pygame.key.get_pressed()
        for key_a, key_b, dx, dy in ((pygame.K_LEFT, pygame.K_RIGHT, 1, 0), (pygame.K_UP, pygame.K_DOWN, 0, 1)):
            delta = keys[key_b] - keys[key_a]
            if delta:
                camera.pan(-dx * delta * 6.0, dy * delta * 6.0)

        update_scene()
        draw_scene(w, h)

        if snapshot_pending:
            buf = gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            image = pygame.image.frombuffer(buf, (w, h), "RGBA")
            pygame.image.save(pygame.transform.flip(image, False, True), snapshot_pending)
            print(f"snapshot -> {snapshot_pending}", flush=True)
            snapshot_pending = None
        pygame.display.flip()
        clock.tick(60)
        if args.exit_after is not None and time.time() - t0 > args.exit_after:
            pygame.quit(); return


if __name__ == "__main__":
    main()
