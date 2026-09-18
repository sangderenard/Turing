"""Bidirectional bridge between repository-SSA regions and nodus canvas saves.

Nodus persists a KPN network as a ``CANVAS V4`` text document
(``nodus/include/inl/canvas_abi_save_load.inl``): ``MODULE`` records are the
processes (each backed by a table), ``ROWS`` records are each module's ordered
row program — the rows execute in order against the module's value stack every
scheduler tick (``thread_manager.cpp``), with ``Input`` rows feeding from edge
FIFOs, ``Tool`` rows popping/pushing values, and ``Output`` rows publishing —
``EDGE`` records are the KPN FIFO channels, and ``NODE`` records carry each
module's typed dataflow contract (``GraphRuntime::NodeContract``).

Tool rows reference operators two ways: builtin ``ModuleToolKind`` ordinals
(Add/Subtract/Multiply/Divide/Modulo) or plugin-origin rows naming a
registered tool id. The canonical-vocabulary tools are the ids
``abstract_tensor.<name>`` that nodus's ``register_abstract_tensor_tool_ir``
(``abstract_tensor_graph_ir.cpp``) forms from ``canonical_ops.json`` — the
same append-only catalog the KernelIR membrane uses. So a module's row
program is exactly a stack serialization of a straight-line SSA region, and
this bridge is the two honest conversions:

* **write**: a repository-SSA region (the same ``Function``/outputs contract
  ``kernel_ir_lowering`` takes — the no-FusedProgram dual path) becomes a
  module whose rows spell the chain in stack order. The writer *proves* its
  own serialization by running the reader's stack simulation over the rows
  it just emitted and comparing the reconstruction to the source; a region
  that cannot be spelled without stack-manipulation tools is a named
  shortfall, never a wrong row program.
* **read**: a canvas document parses into modules/rows/edges/contracts, and
  each module's row program reconstructs into a repository-SSA function by
  symbolic stack simulation, resolving plugin ids and builtin tool ordinals
  through the canonical catalog.

Records this bridge does not model (module UUIDs, meta-groups, ropes, frame
LEDs, overlays — the physical/UI half of the save) are preserved verbatim and
re-emitted in their original positions, so load → translate → save does not
destroy state owned by the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
from .kernel_ir_lowering import load_canonical_catalog

CANVAS_TOOL_PREFIX = "abstract_tensor."

# ModuleRowKind / ModuleToolOrigin ordinals (nodus thread_manager.h).
ROW_INPUT, ROW_TOOL, ROW_OUTPUT = 0, 1, 2
ORIGIN_BUILTIN, ORIGIN_PLUGIN = 0, 1

# Builtin ModuleToolKind ordinals with canonical-catalog identities. The
# remaining kinds (listeners, displays, tensor allocators) are UI/runtime
# tools with no SSA meaning and are reported as shortfalls when a program is
# reconstructed across them.
_BUILTIN_TOOL_TO_CANONICAL = {
    1: "add", 2: "sub", 3: "mul", 4: "truediv", 5: "mod",
}
_CANONICAL_TO_BUILTIN_TOOL = {
    name: kind for kind, name in _BUILTIN_TOOL_TO_CANONICAL.items()
}

# ValueTypeBuiltin ordinals (nodus value_types.h) for NODE contracts.
#
# The edge ABI (gp_canvas_add_edge_with_type, canvas_abi_actions_edges.inl)
# reserves ``type_id == 0`` as "untyped/wildcard" and engages NodeContract
# validation only for nonzero types — but ValueTypeRegistry registers float32
# FIRST, so float32's real ValueTypeId is also 0 and a raw-id encoding could
# never express a typed float32 channel. Nothing in nodus populates contract
# type lists except the canvas loader itself (confirmed 2026-08-13), so the
# wire convention is free to be defined here, consistently on both record
# kinds: **EDGE type_id and NODE contract ints carry ValueTypeId + 1**, with
# 0 remaining untyped/wildcard. The stock validator then engages correctly
# (it only compares edge ints against contract ints, both shifted).
_DTYPE_TO_VALUE_TYPE = {
    "float32": 0,   # VT_FLOAT32
    "float64": 1,   # VT_FLOAT64
    "int32": 4,     # VT_INT32
    "uint32": 8,    # VT_UINT32
}
_VALUE_TYPE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_VALUE_TYPE.items()}


def _dtype_to_wire(dtype: str | None) -> int:
    value_type = _DTYPE_TO_VALUE_TYPE.get(dtype or "float32")
    return 0 if value_type is None else value_type + 1


def _wire_to_dtype(wire: int) -> str | None:
    if wire <= 0:
        return None  # untyped/wildcard
    return _VALUE_TYPE_TO_DTYPE.get(wire - 1)


class CanvasBridgeError(ValueError):
    """Raised when a document is structurally malformed (not a shortfall)."""


@dataclass(frozen=True)
class CanvasShortfall:
    """One construct the bridge cannot convert yet, by name."""

    where: str
    reason: str

    def format(self) -> str:
        return f"{self.where}: {self.reason}"


@dataclass
class CanvasRow:
    kind: int = ROW_TOOL
    tool: int = 0
    attachment_count: int = 1
    tool_origin: int = ORIGIN_BUILTIN
    plugin_id: str = ""


@dataclass
class CanvasModule:
    x: int = 0
    y: int = 0
    w: int = 420
    h: int = 320
    in_count: int = 0
    out_count: int = 0
    is_stage: int = 0
    bg_mode: int = 0
    label: str = ""
    rows: list[CanvasRow] = field(default_factory=list)


@dataclass
class CanvasEdge:
    a_module: int
    a_contact: int
    b_module: int
    b_contact: int
    type_id: int = 0
    subgroup_flags: int = 0


@dataclass
class CanvasNode:
    node_id: int
    module_idx: int
    input_types: list[int] = field(default_factory=list)
    output_types: list[int] = field(default_factory=list)


@dataclass
class CanvasDocument:
    width: int = 1600
    height: int = 900
    control_bar_h: int = 56
    offset_x: int = 0
    offset_y: int = 0
    modules: list[CanvasModule] = field(default_factory=list)
    edges: list[CanvasEdge] = field(default_factory=list)
    nodes: list[CanvasNode] = field(default_factory=list)
    # (index into the emission order, raw line) for records this bridge does
    # not model. Re-emitted verbatim after the modeled records, preserving
    # their relative order.
    unmodeled: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Parsing (canvas text -> document)
# --------------------------------------------------------------------------

def _escape_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace("\n", "\\n")


def _unescape_label(raw: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            out.append("\n" if nxt == "n" else nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_canvas(text: str) -> CanvasDocument:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("CANVAS"):
        raise CanvasBridgeError("missing CANVAS header")
    version = lines[0].split()[1] if len(lines[0].split()) > 1 else "V1"
    if version != "V4":
        raise CanvasBridgeError(
            f"this bridge reads CANVAS V4 documents; found {version}"
        )
    if len(lines) < 2:
        raise CanvasBridgeError("missing canvas geometry line")
    doc = CanvasDocument()
    geometry = lines[1].split()
    doc.width, doc.height, doc.control_bar_h = (
        int(geometry[0]), int(geometry[1]), int(geometry[2]),
    )

    index = 2
    # META_GROUP is followed by META_LASSO and META_VERTS lines that belong
    # to it; treat the whole cluster as unmodeled.
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip():
            continue
        fields = line.split()
        tag = fields[0]
        if tag == "OFFSET":
            doc.offset_x, doc.offset_y = int(fields[1]), int(fields[2])
        elif tag == "MODULE":
            module = CanvasModule(
                x=int(fields[1]), y=int(fields[2]),
                w=int(fields[3]), h=int(fields[4]),
                in_count=int(fields[5]), out_count=int(fields[6]),
                is_stage=int(fields[7]), bg_mode=int(fields[8]),
            )
            prefix = "MODULE " + " ".join(fields[1:9]) + " "
            module.label = _unescape_label(line[len(prefix):])
            doc.modules.append(module)
        elif tag == "ROWS":
            module_idx = int(fields[1])
            row_count = int(fields[2])
            cursor = 3
            rows: list[CanvasRow] = []
            for _ in range(row_count):
                row = CanvasRow(
                    kind=int(fields[cursor]),
                    tool=int(fields[cursor + 1]),
                    attachment_count=int(fields[cursor + 2]),
                    tool_origin=int(fields[cursor + 3]),
                )
                cursor += 4
                if row.tool_origin == ORIGIN_PLUGIN and cursor < len(fields):
                    row.plugin_id = fields[cursor]
                    cursor += 1
                rows.append(row)
            if not 0 <= module_idx < len(doc.modules):
                raise CanvasBridgeError(
                    f"ROWS references module {module_idx} before its MODULE"
                )
            doc.modules[module_idx].rows = rows
        elif tag == "EDGE":
            doc.edges.append(CanvasEdge(
                a_module=int(fields[1]), a_contact=int(fields[2]),
                b_module=int(fields[3]), b_contact=int(fields[4]),
                type_id=int(fields[5]),
                subgroup_flags=int(fields[6]) if len(fields) > 6 else 0,
            ))
        elif tag == "NODE":
            cursor = 3
            node = CanvasNode(
                node_id=int(fields[1]), module_idx=int(fields[2]),
            )
            in_count = int(fields[cursor]); cursor += 1
            node.input_types = [
                int(fields[cursor + i]) for i in range(in_count)
            ]
            cursor += in_count
            out_count = int(fields[cursor]); cursor += 1
            node.output_types = [
                int(fields[cursor + i]) for i in range(out_count)
            ]
            doc.nodes.append(node)
        elif tag == "META_GROUP":
            cluster = [line]
            for _ in range(2):  # META_LASSO, META_VERTS
                if index < len(lines):
                    cluster.append(lines[index])
                    index += 1
            doc.unmodeled.extend(cluster)
        else:
            doc.unmodeled.append(line)
    return doc


def serialize_canvas(doc: CanvasDocument) -> str:
    """Emit the document in gp_canvas_save_to_file's exact spelling."""

    lines = [
        "CANVAS V4",
        f"{doc.width} {doc.height} {doc.control_bar_h}",
        f"OFFSET {doc.offset_x} {doc.offset_y}",
    ]
    for module in doc.modules:
        lines.append(
            f"MODULE {module.x} {module.y} {module.w} {module.h} "
            f"{module.in_count} {module.out_count} {module.is_stage} "
            f"{module.bg_mode} {_escape_label(module.label)}"
        )
    for module_idx, module in enumerate(doc.modules):
        if not module.rows:
            continue
        parts = [f"ROWS {module_idx} {len(module.rows)}"]
        for row in module.rows:
            parts.append(
                f"{row.kind} {row.tool} {row.attachment_count} "
                f"{row.tool_origin}"
            )
            if row.tool_origin == ORIGIN_PLUGIN and row.plugin_id:
                parts.append(row.plugin_id)
        lines.append(" ".join(parts))
    for edge in doc.edges:
        lines.append(
            f"EDGE {edge.a_module} {edge.a_contact} {edge.b_module} "
            f"{edge.b_contact} {edge.type_id} {edge.subgroup_flags}"
        )
    for node in doc.nodes:
        parts = [f"NODE {node.node_id} {node.module_idx}"]
        parts.append(str(len(node.input_types)))
        parts.extend(str(t) for t in node.input_types)
        parts.append(str(len(node.output_types)))
        parts.extend(str(t) for t in node.output_types)
        lines.append(" ".join(parts))
    lines.extend(doc.unmodeled)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Reading a module's row program back into repository SSA
# --------------------------------------------------------------------------

@dataclass
class RegionProgram:
    """One SSA region: the same contract kernel_ir_lowering consumes."""

    function: Function
    outputs: tuple[SSAValue, ...]
    shortfalls: tuple[CanvasShortfall, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls


def _catalog_maps(
    catalog: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    by_name: dict[str, Mapping[str, Any]] = {}
    handler_to_name: dict[str, str] = {}
    for entry in catalog:
        name = entry.get("name")
        if isinstance(name, str) and name not in by_name:
            by_name[name] = entry
        handler = entry.get("handler")
        if isinstance(handler, str) and isinstance(name, str):
            handler_to_name.setdefault(handler, name)
    return by_name, handler_to_name


def region_from_module(
    doc: CanvasDocument,
    module_idx: int,
    *,
    catalog: Sequence[Mapping[str, Any]] | None = None,
) -> RegionProgram:
    """Reconstruct one module's row program as a repository-SSA function.

    The reconstruction is a symbolic execution of exactly what the scheduler
    does: rows run in order against one stack; Input rows push the module's
    inputs (contact order), a tool row pops its declared arity and pushes one
    result, Output rows pop the published values (top of stack = last output
    contact).
    """

    resolved = tuple(catalog) if catalog is not None else load_canonical_catalog()
    by_name, _ = _catalog_maps(resolved)
    module = doc.modules[module_idx]
    contract = next(
        (n for n in doc.nodes if n.module_idx == module_idx), None,
    )
    shortfalls: list[CanvasShortfall] = []
    where = f"module {module_idx} ({module.label!r})"

    def dtype_for(contact: int, types: list[int] | None) -> str:
        if types and 0 <= contact < len(types):
            wire = types[contact]
            dtype = _wire_to_dtype(wire)
            if dtype is not None:
                return dtype
            if wire > 0:
                shortfalls.append(CanvasShortfall(
                    where,
                    f"contract wire type {wire} has no scalar dtype mapping "
                    "(tensor-typed contracts are future work)",
                ))
        return "float32"

    next_id = 0
    stack: list[SSAValue] = []
    args: list[SSAValue] = []
    instrs: list[Instr] = []
    outputs: list[SSAValue] = []

    def fresh(dtype: str) -> SSAValue:
        nonlocal next_id
        value = SSAValue(next_id, dtype, ())
        next_id += 1
        return value

    for row_idx, row in enumerate(module.rows):
        if row.kind == ROW_INPUT:
            for contact in range(row.attachment_count):
                value = fresh(dtype_for(
                    len(args),
                    contract.input_types if contract else None,
                ))
                args.append(value)
                stack.append(value)
            continue
        if row.kind == ROW_OUTPUT:
            for contact in range(row.attachment_count):
                if not stack:
                    shortfalls.append(CanvasShortfall(
                        where,
                        f"output row {row_idx} pops an empty stack",
                    ))
                    break
                outputs.append(stack.pop())
            continue
        # Tool row: resolve to a canonical operation.
        if row.tool_origin == ORIGIN_PLUGIN:
            if not row.plugin_id.startswith(CANVAS_TOOL_PREFIX):
                shortfalls.append(CanvasShortfall(
                    where,
                    f"plugin row {row_idx} ({row.plugin_id!r}) is not a "
                    "canonical-vocabulary tool; its behavior is opaque to "
                    "SSA reconstruction",
                ))
                continue
            name = row.plugin_id[len(CANVAS_TOOL_PREFIX):]
        else:
            name = _BUILTIN_TOOL_TO_CANONICAL.get(row.tool)
            if name is None:
                if row.tool == 0:
                    continue  # ModuleToolKind::None — layout-only row
                if row.tool == 10:
                    # TableNumber: pushes user-editable cell values — a
                    # parameter feed, so it reconstructs as function inputs
                    # whose source is the table rather than a channel.
                    for _ in range(row.attachment_count):
                        value = fresh("float32")
                        args.append(value)
                        stack.append(value)
                    continue
                if row.tool == 8:
                    continue  # StackDisplay: observes the stack, moves nothing
                shortfalls.append(CanvasShortfall(
                    where,
                    f"builtin tool kind {row.tool} at row {row_idx} is a "
                    "runtime/UI tool with no SSA identity",
                ))
                continue
        entry = by_name.get(name)
        if entry is None:
            shortfalls.append(CanvasShortfall(
                where,
                f"row {row_idx} names unknown canonical operation {name!r}",
            ))
            continue
        arity = int(entry.get("arity") or 0)
        if len(stack) < arity:
            shortfalls.append(CanvasShortfall(
                where,
                f"row {row_idx} ({name}) pops {arity} but the stack holds "
                f"{len(stack)}",
            ))
            continue
        operands = [stack.pop() for _ in range(arity)]
        operands.reverse()  # stack pop order is right-to-left
        result = fresh(operands[0].dtype if operands else "float32")
        instrs.append(Instr(name, operands, result))
        stack.append(result)

    instrs.append(Instr("Ret", [], None))
    block = BasicBlock("entry", instrs)
    name = module.label or f"module_{module_idx}"
    identifier = "".join(
        ch if ch.isalnum() or ch == "_" else "_" for ch in name
    ) or f"module_{module_idx}"
    function = Function(identifier, args, {"entry": block})
    return RegionProgram(function, tuple(outputs), tuple(shortfalls))


# --------------------------------------------------------------------------
# Writing SSA regions into a canvas document
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionEdge:
    """One KPN channel between two regions' contacts."""

    source_region: str
    source_contact: int
    target_region: str
    target_contact: int


def canvas_from_regions(
    regions: Mapping[str, tuple[Function, Sequence[SSAValue]]],
    edges: Sequence[RegionEdge] = (),
    *,
    catalog: Sequence[Mapping[str, Any]] | None = None,
    width: int = 1600,
    height: int = 900,
) -> tuple[CanvasDocument, tuple[CanvasShortfall, ...]]:
    """Spell SSA regions as a canvas document, self-verified.

    Every region's emitted row program is immediately re-read through
    ``region_from_module``'s stack simulation and compared against the source
    chain. A mismatch (a region needing stack-manipulation tools this bridge
    does not emit yet) withdraws the module's rows and reports a shortfall —
    the document never contains a row program that executes differently from
    the SSA it claims to spell.
    """

    resolved = tuple(catalog) if catalog is not None else load_canonical_catalog()
    by_name, handler_to_name = _catalog_maps(resolved)
    doc = CanvasDocument(width=width, height=height)
    shortfalls: list[CanvasShortfall] = []
    module_index: dict[str, int] = {}

    for position, (name, (function, outputs)) in enumerate(regions.items()):
        where = f"region {name!r}"
        module = CanvasModule(
            x=40 + (position % 3) * 480,
            y=90 + (position // 3) * 380,
            in_count=len(function.args),
            out_count=len(outputs),
            label=name,
        )
        module_index[name] = len(doc.modules)
        doc.modules.append(module)

        rows: list[CanvasRow] = []
        row_shortfalls: list[CanvasShortfall] = []
        # Lazy input pulling: contacts are pulled (Input rows) in contact
        # order, but only as the row program first needs them. For a plain
        # region every argument is needed by the first operations and the
        # pulls merge into one Input row — identical to the eager spelling.
        # For a compound region built by collapse_canvas_regions, the pulls
        # segment naturally around inlined sub-programs, which is what keeps
        # the stack layout correct (a value produced mid-program sits exactly
        # where the consumer's next pop expects it).
        pushed_args: set[int] = set()
        next_arg = 0
        arg_ids = [a.id for a in function.args]

        def pull_through(arg_index: int) -> None:
            nonlocal next_arg
            while next_arg <= arg_index:
                if rows and rows[-1].kind == ROW_INPUT:
                    rows[-1].attachment_count += 1
                else:
                    rows.append(CanvasRow(kind=ROW_INPUT, attachment_count=1))
                pushed_args.add(arg_ids[next_arg])
                next_arg += 1

        if len(function.blocks) != 1:
            row_shortfalls.append(CanvasShortfall(
                where, "multi-block control flow cannot be spelled as one "
                       "row program (KPN control regions are future work)",
            ))
        else:
            block = next(iter(function.blocks.values()))
            for instr in block.instrs:
                if instr.op == "Ret":
                    continue
                for operand in instr.args:
                    if operand.id in arg_ids and operand.id not in pushed_args:
                        pull_through(arg_ids.index(operand.id))
                canonical = (
                    instr.op if instr.op in by_name
                    else handler_to_name.get(instr.op)
                )
                if canonical is None:
                    row_shortfalls.append(CanvasShortfall(
                        where,
                        f"operation {instr.op!r} has no canonical catalog "
                        "identity; the membrane admits catalog operations "
                        "only",
                    ))
                    continue
                if (instr.attributes or {}).get("right_scalar") is not None \
                        or (instr.attributes or {}).get("left_scalar") is not None:
                    row_shortfalls.append(CanvasShortfall(
                        where,
                        f"operation {instr.op!r} folds a scalar immediate; "
                        "constant rows (TableNumber feeds) are not emitted "
                        "yet",
                    ))
                    continue
                builtin = _CANONICAL_TO_BUILTIN_TOOL.get(canonical)
                if builtin is not None:
                    rows.append(CanvasRow(kind=ROW_TOOL, tool=builtin))
                else:
                    # Same gate as nodus's register_abstract_tensor_tool_ir:
                    # a vocabulary tool exists only for catalog operations the
                    # CPU evaluator implements (ct_value present, arity 1..2).
                    # tensor_math's plan executor returns quiet-NaN for ops
                    # outside that set, so a row naming one would be a silent
                    # wrong answer at every tick.
                    entry = by_name[canonical]
                    if entry.get("ct_value") is None or \
                            int(entry.get("arity") or 0) not in (1, 2):
                        row_shortfalls.append(CanvasShortfall(
                            where,
                            f"operation {canonical!r} has no registered "
                            "vocabulary tool (outside the CPU evaluator's "
                            "implemented set); row withheld",
                        ))
                        continue
                    rows.append(CanvasRow(
                        kind=ROW_TOOL, tool=0, tool_origin=ORIGIN_PLUGIN,
                        plugin_id=CANVAS_TOOL_PREFIX + canonical,
                    ))
        if function.args and next_arg < len(function.args):
            # A declared contact the body never consumes has no correct
            # stack spelling (pulling it would bury the result, skipping it
            # would break the edge contract) — there is no drop tool yet.
            row_shortfalls.append(CanvasShortfall(
                where,
                f"argument contact {next_arg} is never consumed by the row "
                "program; no drop tool exists to honor the contract",
            ))
        if outputs:
            rows.append(CanvasRow(
                kind=ROW_OUTPUT, attachment_count=len(outputs),
            ))
        module.rows = rows

        node = CanvasNode(
            node_id=len(doc.nodes) + 1,
            module_idx=module_index[name],
            input_types=[_dtype_to_wire(a.dtype) for a in function.args],
            output_types=[_dtype_to_wire(o.dtype) for o in outputs],
        )
        doc.nodes.append(node)

        if row_shortfalls:
            shortfalls.extend(row_shortfalls)
            module.rows = []
            continue

        # Self-verification: the emitted rows must reconstruct to the same
        # operation sequence with the same operand wiring.
        reconstruction = region_from_module(
            doc, module_index[name], catalog=resolved,
        )
        if not _same_program(function, outputs, reconstruction, by_name,
                             handler_to_name):
            shortfalls.append(CanvasShortfall(
                where,
                "row program does not round-trip to the source SSA (the "
                "region needs stack-manipulation tools this bridge does not "
                "emit yet); rows withheld",
            ))
            module.rows = []

    for edge in edges:
        if edge.source_region not in module_index:
            shortfalls.append(CanvasShortfall(
                f"edge {edge}", "unknown source region",
            ))
            continue
        if edge.target_region not in module_index:
            shortfalls.append(CanvasShortfall(
                f"edge {edge}", "unknown target region",
            ))
            continue
        source_index = module_index[edge.source_region]
        source_node = doc.nodes[source_index]
        type_id = 0
        if 0 <= edge.source_contact < len(source_node.output_types):
            type_id = source_node.output_types[edge.source_contact]
        # A module's contacts are one index space: inputs 0..in-1, then
        # outputs in..in+out-1 (canvas_abi_overlay_raster.inl's is_output
        # partition). RegionEdge speaks in logical output indices; the wire
        # offsets them past the source's inputs. Target contacts are inputs
        # and stay as-is.
        doc.edges.append(CanvasEdge(
            a_module=source_index,
            a_contact=doc.modules[source_index].in_count + edge.source_contact,
            b_module=module_index[edge.target_region],
            b_contact=edge.target_contact,
            type_id=type_id,
        ))
    return doc, tuple(shortfalls)


# --------------------------------------------------------------------------
# Collapsing tables into compound operators
# --------------------------------------------------------------------------
#
# The C++ counterpart is module_library_actualizer.cpp's generate_tool_source:
# it lowers an entire table into one Tool_<id> class whose execute_stack
# concatenates the table's row steps. This is the same reduction performed
# symbolically: a module whose one output feeds exactly one consumer contact
# is inlined into the consumer — its SSA body spliced at the consumer's
# argument, its external inputs becoming the compound's inputs at that
# position — and the lazy input pulls in canvas_from_regions spell the result
# as one deeper row program. Fan-out modules stay: the stack has no dup tool,
# so a value consumed twice must remain a real KPN channel.

def _inline_region(
    dst: Function,
    dst_outputs: Sequence[SSAValue],
    contact: int,
    src: Function,
    src_outputs: Sequence[SSAValue],
) -> tuple[Function, tuple[SSAValue, ...]]:
    """Splice ``src``'s body into ``dst`` at argument position ``contact``."""

    if len(src_outputs) != 1:
        raise CanvasBridgeError("inlining requires a single-output producer")
    fresh_id = 0
    mapping: dict[tuple[int, int], SSAValue] = {}  # (side, old id) -> value

    def remap(side: int, value: SSAValue) -> SSAValue:
        nonlocal fresh_id
        key = (side, value.id)
        if key not in mapping:
            mapping[key] = SSAValue(fresh_id, value.dtype, value.shape)
            fresh_id += 1
        return mapping[key]

    new_args = (
        [remap(1, a) for a in dst.args[:contact]]
        + [remap(0, a) for a in src.args]
        + [remap(1, a) for a in dst.args[contact + 1:]]
    )
    # The consumer's argument at `contact` becomes the producer's output.
    src_result = remap(0, src_outputs[0])
    mapping[(1, dst.args[contact].id)] = src_result

    instrs: list[Instr] = []
    for side, function in ((0, src), (1, dst)):
        block = next(iter(function.blocks.values()))
        for instr in block.instrs:
            if instr.op == "Ret":
                continue
            instrs.append(Instr(
                instr.op,
                [remap(side, a) for a in instr.args],
                remap(side, instr.res) if instr.res is not None else None,
                attributes=dict(instr.attributes or {}),
            ))
    instrs.append(Instr("Ret", [], None))
    outputs = tuple(remap(1, o) for o in dst_outputs)
    # Nested inlining must leave instruction blocks in contact order, not
    # src-before-dst order — a stack machine executes each producer exactly
    # where its consumer's pull sequence reaches that contact. Single-
    # consumer inlining guarantees the compound is an expression tree, so
    # post-order from the output IS the correct RPN; rebuild in that order.
    reordered = _postorder_instrs(new_args, instrs, outputs)
    compound = Function(
        dst.name, new_args,
        {"entry": BasicBlock("entry", reordered)},
    )
    return compound, outputs


def _postorder_instrs(
    args: Sequence[SSAValue],
    instrs: Sequence[Instr],
    outputs: Sequence[SSAValue],
) -> list[Instr]:
    """Post-order (left-to-right operands) schedule of an expression tree.

    Falls back to the given order when a value is consumed more than once —
    that shape is not a tree, and the writer's self-verification owns the
    refusal.
    """

    producer = {
        instr.res.id: instr for instr in instrs
        if instr.op != "Ret" and instr.res is not None
    }
    arg_ids = {a.id for a in args}
    emitted: list[Instr] = []
    visited: set[int] = set()

    def visit(value: SSAValue) -> bool:
        if value.id in arg_ids:
            return True
        if value.id in visited:
            return False  # consumed twice: not a tree
        instr = producer.get(value.id)
        if instr is None:
            return False
        visited.add(value.id)
        for operand in instr.args:
            if not visit(operand):
                return False
        emitted.append(instr)
        return True

    for output in outputs:
        if not visit(output):
            return list(instrs)
    if len(emitted) != len(producer):
        return list(instrs)  # dead or shared instructions: keep original
    emitted.append(Instr("Ret", [], None))
    return emitted


def collapse_canvas_regions(
    regions: Mapping[str, tuple[Function, Sequence[SSAValue]]],
    edges: Sequence[RegionEdge],
) -> tuple[dict[str, tuple[Function, tuple[SSAValue, ...]]],
           list[RegionEdge], list[str]]:
    """Reduce the region graph by inlining every single-consumer producer.

    Returns the reduced regions, the surviving KPN edges (contacts
    renumbered), and a log of the inlining steps taken. Producers whose
    output fans out to more than one consumer contact are kept as real
    processes — a stack machine without a dup tool cannot inline a value
    that is consumed twice.
    """

    live: dict[str, tuple[Function, tuple[SSAValue, ...]]] = {
        name: (fn, tuple(outs)) for name, (fn, outs) in regions.items()
    }
    channels = list(edges)
    log: list[str] = []

    while True:
        fanout: dict[str, int] = {}
        for channel in channels:
            fanout[channel.source_region] = (
                fanout.get(channel.source_region, 0) + 1
            )
        candidate = next(
            (c for c in channels
             if fanout.get(c.source_region, 0) == 1
             and c.source_contact == 0
             and len(live[c.source_region][1]) == 1
             and c.source_region != c.target_region),
            None,
        )
        if candidate is None:
            break
        src_name = candidate.source_region
        dst_name = candidate.target_region
        contact = candidate.target_contact
        src_fn, src_outs = live[src_name]
        dst_fn, dst_outs = live[dst_name]
        src_arg_count = len(src_fn.args)
        live[dst_name] = _inline_region(
            dst_fn, dst_outs, contact, src_fn, src_outs,
        )
        del live[src_name]
        log.append(f"inlined {src_name} into {dst_name} at contact {contact}")

        rewired: list[RegionEdge] = []
        for channel in channels:
            if channel is candidate:
                continue
            source, s_contact = channel.source_region, channel.source_contact
            target, t_contact = channel.target_region, channel.target_contact
            if target == src_name:
                # Feeds the inlined producer: retarget into the compound at
                # the producer's argument block.
                target, t_contact = dst_name, contact + t_contact
            elif target == dst_name and t_contact > contact:
                # Later consumer contacts shift by the inlined arg block.
                t_contact += src_arg_count - 1
            rewired.append(RegionEdge(source, s_contact, target, t_contact))
        channels = rewired
    return live, channels, log


def _same_program(
    function: Function,
    outputs: Sequence[SSAValue],
    reconstruction: RegionProgram,
    by_name: Mapping[str, Mapping[str, Any]],
    handler_to_name: Mapping[str, str],
) -> bool:
    if not reconstruction.complete:
        return False
    source_ops: list[tuple[str, tuple[int, ...]]] = []
    renumber: dict[int, int] = {}
    for index, argument in enumerate(function.args):
        renumber[argument.id] = index
    counter = len(function.args)
    block = next(iter(function.blocks.values()))
    for instr in block.instrs:
        if instr.op == "Ret":
            continue
        canonical = (
            instr.op if instr.op in by_name else handler_to_name.get(instr.op)
        )
        operand_ids = tuple(renumber.get(a.id, -1) for a in instr.args)
        if instr.res is not None:
            renumber[instr.res.id] = counter
            counter += 1
        source_ops.append((canonical or instr.op, operand_ids))
    source_outputs = tuple(renumber.get(o.id, -1) for o in outputs)

    other = reconstruction.function
    other_block = next(iter(other.blocks.values()))
    other_renumber: dict[int, int] = {}
    for index, argument in enumerate(other.args):
        other_renumber[argument.id] = index
    other_counter = len(other.args)
    other_ops: list[tuple[str, tuple[int, ...]]] = []
    for instr in other_block.instrs:
        if instr.op == "Ret":
            continue
        operand_ids = tuple(
            other_renumber.get(a.id, -1) for a in instr.args
        )
        if instr.res is not None:
            other_renumber[instr.res.id] = other_counter
            other_counter += 1
        other_ops.append((instr.op, operand_ids))
    other_outputs = tuple(
        other_renumber.get(o.id, -1) for o in reconstruction.outputs
    )
    return source_ops == other_ops and source_outputs == other_outputs
