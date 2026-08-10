"""Auto-segment a fully-reduced program into WASM "class" modules.

This is **runtime segmentation**, not an OOP presentation of the source. The
two are deliberately at odds and this file picks a side: a compiler wants to
reduce first -- constant-fold, fuse, flatten control flow -- and by the time
that is done, the AST's own function boundaries (``helper`` called from
``kernel``, say) are gone; a call is fully inlined into one flat, linear
``FusedProgram`` (see ``fused_program_wasm_backend.py``'s own docstring: "a
flat, topologically ordered list of ``OpStep`` with no branches at all").
So segmentation here has nothing left to preserve semantic boundaries with --
it takes whatever pieces come out of cutting the reduced graph into
roughly-equal-sized, connected chunks, named mechanically (``chunk0``,
``chunk1``, ...), not meaningfully. A *meaningful*, class-named presentation
of the same program is a separate, later concern for a shell to plan and
display (it can still draw on ``hierarchical_plan.PlanClosure`` for that),
but it does not get to decide where the actual module boundaries fall here.

**How the cut is made.** ``process_graph_fusion.py`` already has the pieces:
``fused_program_to_process_graph`` projects a reduced ``FusedProgram`` into a
real ``ProcessGraph`` (nodes are the program's own SSA-like values); the
program's own topological order gives a stable node sequence; and
``DispatchRegion``/``dispatch_region_to_fused_program`` already know how to
carve a node set into a self-contained, independently-lowerable
``FusedProgram`` with its ``input_ids``/``outputs`` correctly accounted for
at the cut. That accounting *is* the module's calling contract -- once a
region is turned back into a ``FusedProgram``, ``emit_wasm_module`` derives
its ``CompiledProgramAPI`` from it exactly the way it already does for any
other program, no separate contract-description step needed.

What is new in this file is only the *cutting rule*: neither of
``process_graph_fusion.py``'s two existing planners does plain equal-sized
contiguous chunking (``plan_process_graph_dispatches`` accepts or entirely
rejects one whole connected component; ``serialize_scheduled_operator_dispatches``
batches by schedule-level and operator, for GPU dispatch shaping, not module
count). ``partition_reduced_program`` below does the missing thing: slice the
program's topological node order into contiguous windows of ``chunk_size``,
which is always dependency-safe for a contiguous slice of a topological
order, then reuse ``DispatchRegion``'s own boundary accounting for each
window.

The chunk containing the program's own declared outputs is the root -- the
last thing to run, level 0 (see ``schedule_module_levels``) -- and is named
after the entrypoint itself (``owner_name``), since (as agreed) whatever was
submitted as the AST entry point owns the whole process graph. Every other
chunk is a numbered prerequisite of it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..common.tensors.fused_ir import FusedProgram, ordered_feed_ids
from .process_graph_fusion import (
    DispatchRegion, dispatch_region_to_fused_program,
    fused_program_to_process_graph,
    reduce_scheduled_shader_regions,
)


# These operations change an invocation-wide tensor extent into shared state.
# Element tiling may only cross them with an explicit partial-reduction Join;
# silently evaluating them once per tile changes one world into many worlds.
COLLECTIVE_FUSED_OPERATIONS = frozenset({
    "sum", "mean", "prod", "min", "max", "any", "all", "argmin", "argmax",
})


def fused_program_extent_effect(program: FusedProgram) -> str:
    """Classify whether a fused program is safe to split by element extent."""

    program = getattr(program, "program", program)
    for step in program.steps:
        if (
            str(step.op_name) in COLLECTIVE_FUSED_OPERATIONS
            or "reduce_op" in dict(step.attrs or {})
        ):
            return "collective"
    return "pointwise"


def _diagnose_region(program, region, module_name) -> str:
    """A self-explaining diagnosis appended to a region emission failure.

    An 'operand was never produced' shortfall is opaque on its own. This walks
    the region and classifies every DANGLING operand -- a value an op reads that
    is neither a region feed nor produced by an earlier step -- reporting what
    the IR still knows about it: a capture origin means an undeclared feed; being
    in ``state_in`` means unwired state; surviving metadata means its producer
    was pruned; nothing at all means the producer was eliminated during
    capture/partitioning while a consumer kept the reference. Naming the
    distinction in the error turns a blind rebuild-and-guess loop into a direct
    fix -- for this program and every future one.
    """

    produced = set(program.feeds) | {s.result_id for s in program.steps}
    meta = program.meta or {}
    origins = (program.extras or {}).get("capture_feed_origins", {})
    state_in = program.state_in or set()
    dangling: dict[int, list] = {}
    for step in program.steps:
        for operand in step.input_ids:
            if operand in produced:
                continue
            dangling.setdefault(operand, []).append((step.step_id, step.op_name))
    if not dangling:
        return ""  # the shortfall is something else; no dangling operands
    lines = [f"\n\nregion {region} ({module_name}) dangling operands "
             f"(read but never produced and not a feed):"]
    for operand in sorted(dangling):
        entry = meta.get(operand)
        origin = origins.get(operand) or origins.get(str(operand))
        readers = ", ".join(f"{op}#{sid}" for sid, op in dangling[operand])
        if origin is not None:
            what = f"UNDECLARED FEED (capture origin {origin})"
        elif operand in state_in:
            what = "STATE value not wired as a feed"
        elif entry is not None and getattr(entry, "shape", None) is not None:
            what = (f"lost producer (meta shape={tuple(entry.shape)} "
                    f"dtype={getattr(entry, 'dtype', None)})")
        else:
            what = "producer eliminated during capture/partitioning (no metadata)"
        lines.append(f"  value {operand}: {what}; read by {readers}")
    lines.append(f"  region feeds={sorted(program.feeds)} "
                 f"outputs={dict(program.outputs)}")
    return "\n".join(lines)


def partition_threaded_wasm_program(
    program: FusedProgram,
    *,
    max_nodes_per_region: int = 64,
    schedule_preference: str = "alap",
):
    """Cut a numerical DAG into vertically fused, parallel Wasm waves.

    This consumes the existing fixed-point ProcessGraph reducer. Each emitted
    region is therefore a maximal compatible DAG up to the explicit Wasm
    region cap, including vertical-fusion rewrites; regions at the same
    dependency level become lanes of one ``ParallelDeployment``.
    """

    import networkx as nx

    from .control_source import (
        ControlProgram,
        ParallelDeployment,
        SequenceBlock,
        StatementBlock,
    )
    from .fused_program_wasm_backend import required_steps

    preference = str(schedule_preference).lower()
    if preference not in {"asap", "alap"}:
        raise ValueError("Wasm thread schedule must be asap or alap")
    live = required_steps(program)
    if len(live) <= int(max_nodes_per_region):
        return None
    pruned = FusedProgram(
        version=program.version,
        feeds=set(program.feeds),
        steps=list(live),
        outputs=dict(program.outputs),
        state_in=program.state_in,
        meta=program.meta,
        extras=program.extras,
    )
    graph = fused_program_to_process_graph(pruned)
    executable = {
        int(step.result_id)
        for step in live
        if graph.G.nodes[int(step.result_id)].get("type")
        not in {"input", "const", "return"}
    }
    numerical_graph = graph.G.subgraph(executable)
    topological = tuple(
        nx.lexicographical_topological_sort(
            numerical_graph, key=lambda node_id: int(node_id)
        )
    )
    visited: set[int] = set()
    columns: list[tuple[int, ...]] = []
    for start in topological:
        if int(start) in visited:
            continue
        predecessors = tuple(numerical_graph.predecessors(start))
        if (
            len(predecessors) == 1
            and numerical_graph.out_degree(predecessors[0]) == 1
        ):
            continue
        column = [int(start)]
        current = start
        while numerical_graph.out_degree(current) == 1:
            successor = next(iter(numerical_graph.successors(current)))
            if numerical_graph.in_degree(successor) != 1:
                break
            column.append(int(successor))
            current = successor
        visited.update(column)
        columns.append(tuple(column))
    for node_id in topological:
        if int(node_id) not in visited:
            columns.append((int(node_id),))
    lane_partitions = min(4, max(1, len(columns)))
    partition_keys = {
        node_id: column_index % lane_partitions
        for column_index, column in enumerate(columns)
        for node_id in column
    }
    reduced = reduce_scheduled_shader_regions(
        graph,
        executable,
        max_nodes_per_region=int(max_nodes_per_region),
        max_bindings_per_region=256,
        partition_keys=partition_keys,
        schedule=preference,
    )
    if len(reduced.dispatches) < 2:
        return None
    region_of = {
        int(node_id): index
        for index, dispatch in enumerate(reduced.dispatches)
        for node_id in dispatch.node_ids
    }
    dependency = nx.DiGraph()
    dependency.add_nodes_from(range(len(reduced.dispatches)))
    for parent, child in graph.G.edges:
        left = region_of.get(int(parent))
        right = region_of.get(int(child))
        if left is not None and right is not None and left != right:
            dependency.add_edge(left, right)
    if not nx.is_directed_acyclic_graph(dependency):
        raise ValueError("reduced Wasm deployment topology contains a cycle")

    if preference == "asap":
        waves = [tuple(map(int, wave)) for wave in nx.topological_generations(dependency)]
    else:
        levels: dict[int, int] = {}
        for node in reversed(tuple(nx.topological_sort(dependency))):
            levels[int(node)] = max(
                (levels[int(child)] + 1 for child in dependency.successors(node)),
                default=0,
            )
        waves = [
            tuple(sorted(node for node, value in levels.items() if value == level))
            for level in sorted(set(levels.values()), reverse=True)
        ]
    if not any(len(wave) > 1 for wave in waves):
        return None

    origins = dict((program.extras or {}).get("capture_feed_origins", {}) or {})
    region_programs: dict[int, FusedProgram] = {}
    for index, dispatch in enumerate(reduced.dispatches):
        boundary = _boundary_region(
            graph, tuple(map(int, dispatch.node_ids)), tuple(graph.G.nodes)
        )
        member = dispatch_region_to_fused_program(graph, boundary)
        member.extras = {
            **dict(member.extras or {}),
            "capture_feed_origins": {
                value_id: origins[value_id]
                for value_id in member.feeds
                if value_id in origins
            },
            "wasm_vertical_fusion": tuple(dispatch.rewrite_history),
        }
        region_programs[index] = member

    blocks = []
    for wave in waves:
        lanes = tuple(
            StatementBlock((f"__scheduled_region_{index}__",))
            for index in wave
        )
        blocks.append(
            lanes[0]
            if len(lanes) == 1
            else ParallelDeployment(lanes, preference)
        )
    control = ControlProgram(
        SequenceBlock(tuple(blocks)),
        region_indices=tuple(
            index for wave in waves for index in wave
        ),
    )
    summary = {
        "abi": "turing.wasm-vertical-fusion.v1",
        "source_operations": len(live),
        "regions": len(reduced.dispatches),
        "waves": len(waves),
        "parallel_waves": sum(len(wave) > 1 for wave in waves),
        "max_wave_width": max(map(len, waves)),
        "lane_partitions": lane_partitions,
        "region_sizes": [len(dispatch.node_ids) for dispatch in reduced.dispatches],
        "vertical_fused_regions": sum(
            "vertical-fusion" in dispatch.rewrite_history
            for dispatch in reduced.dispatches
        ),
        "extent_effect": fused_program_extent_effect(pruned),
        "collective_regions": [
            index for index, member in region_programs.items()
            if fused_program_extent_effect(member) == "collective"
        ],
        "rewrite_history": [
            list(dispatch.rewrite_history) for dispatch in reduced.dispatches
        ],
    }
    return control, region_programs, summary


@dataclass(frozen=True)
class ClassModuleCallSite:
    """One dependency edge: this module needs a value only ``callee``
    produces. Unlike a real function call there is no argument/result
    *binding* to carry -- ``DispatchRegion`` accounting already resolved
    exactly which value crosses the cut and what it is named on each side,
    so there is nothing left to correlate."""

    callsite_id: int
    callee_name: str
    callee_index: int

    @property
    def callee_module_name(self) -> str:
        return f"{self.callee_name}__{self.callee_index}"


@dataclass(frozen=True)
class ClassModuleSpec:
    """One chunk's worth of structure: its already-carved region (node ids
    plus the boundary ``DispatchRegion`` already accounted), the
    already-lowered ``FusedProgram`` for it, and which other chunks it
    depends on."""

    name: str
    index: int
    region: DispatchRegion
    program: FusedProgram
    calls: tuple[ClassModuleCallSite, ...]
    is_root: bool = False

    @property
    def module_name(self) -> str:
        return f"{self.name}__{self.index}"


def _region_reduction_digest(program, module_name, dtype, static_offset) -> str:
    """Content key for one lowered region.

    ``emit_wasm_module`` is a pure function of the region program plus these
    emission parameters, so hashing exactly those makes the cached kernel valid
    to reuse whenever they are unchanged. ``static_offset`` is included on
    purpose: two builds that place a region's static data at different offsets
    are genuinely different kernels and must not share a cache entry.
    """

    from joblib.externals import cloudpickle

    digest = hashlib.sha256()
    digest.update(cloudpickle.dumps(program))
    digest.update(
        f"|{module_name}|{dtype}|{int(static_offset)}".encode("utf-8")
    )
    return digest.hexdigest()


def emit_control_region_modules(
    control,
    region_programs: Mapping[int, FusedProgram],
    *,
    owner_name: str,
    module_dir: str,
    dtype: str = "float64",
    reduction_cache=None,
    progress=None,
) -> tuple[dict[int, object], dict]:
    """Emit planner regions without flattening their controlling program.

    Each ``ControlProgram`` region remains one callable WebAssembly kernel.
    Values crossing region boundaries are assigned shared-memory identities;
    the companion control coordinator invokes these exact kernels according
    to the planner-owned loop/state-machine structure.

    ``reduction_cache`` (a ``ReductionArtifactStore``), when given, persists each
    lowered region under its content key so an interrupted or repeated bake
    reloads regions it already lowered instead of re-emitting them. ``progress``,
    if given, is called ``progress(region, was_cached)`` as each region resolves.
    """

    from .fused_program_wasm_backend import (
        emit_wasm_module,
        program_feed_order,
        required_steps,
    )
    # Container recognition is shared, backend-neutral IR analysis -- take it from
    # the central module, not from another backend.
    from .ir_container_ops import (
        pure_container_store as _pure_container_store,
        pure_container_read as _pure_container_read,
    )
    from .wasm_binary import WasmImport
    from .wasm_container import (
        HEAP_CURSOR_ADDR,
        HEAP_RESERVED_BYTES,
        DEFAULT_MAP_CAPACITY,
        MAP_HEADER_BYTES,
        map_block_bytes,
    )

    ordered_regions = tuple(dict.fromkeys(map(int, control.region_indices)))
    missing = set(ordered_regions) - set(map(int, region_programs))
    if missing:
        raise ValueError(
            "control references missing WebAssembly regions: "
            + ", ".join(map(str, sorted(missing)))
        )
    programs = {
        region: getattr(region_programs[region], "program", region_programs[region])
        for region in ordered_regions
    }
    producer: dict[int, tuple[str, str]] = {}
    module_names = {
        region: f"{owner_name}_region_{region}" for region in ordered_regions
    }
    for region, program in programs.items():
        for output_name, value_id in program.outputs.items():
            producer[int(value_id)] = (
                module_names[region], str(output_name)
            )

    modules = {}
    entries = []
    # Byte-identical kernels (same topology + constants + dtype) share one file:
    # the module binary is independent of the region's value-ids and name, so a
    # program that repeats an operation over different data collapses to a
    # handful of distinct kernels. Per-region method-card wiring is unaffected --
    # only the emitted ``.wasm`` file dedups. ``kernel_files`` maps a binary
    # hash to its shared kernel name.
    kernel_files: dict[str, str] = {}
    edges = []
    logical_inputs: dict[str, list[tuple[str, str]]] = {}
    value_bindings: dict[int, str] = {
        value_id: f"out::{module_name}::{output_name}"
        for value_id, (module_name, output_name) in producer.items()
    }
    # Reserve the heap-control bytes at the start of linear memory so the fixed
    # HEAP_CURSOR_ADDR the container kernels bake never collides with static
    # data. Region static data (and everything else) starts past it.
    static_offset = HEAP_RESERVED_BYTES
    for region in ordered_regions:
        program = programs[region]
        module_name = module_names[region]

        def _lower_region(program=program, module_name=module_name,
                          static_offset=static_offset):
            module = emit_wasm_module(
                program,
                name=module_name,
                dtype=dtype,
                imports=(WasmImport(
                    module="env",
                    field="memory",
                    kind="memory",
                    memory_pages=1,
                ),),
                static_data_offset=static_offset,
            )
            # Raise *inside* the compute closure so an incomplete lowering
            # never reaches the cache: a shortfall must be retried on the next
            # pass, not persisted as if it were a finished region.
            if not module.complete:
                raise RuntimeError(
                    module.shortfall_report()
                    + _diagnose_region(program, region, module_name)
                )
            return module

        if reduction_cache is not None:
            module, was_cached = reduction_cache.get_or_compute(
                _region_reduction_digest(
                    program, module_name, dtype, static_offset
                ),
                _lower_region,
            )
        else:
            module, was_cached = _lower_region(), False
        if progress is not None:
            progress(int(region), was_cached)
        modules[region] = module
        kernel_hash = hashlib.sha256(module.binary).hexdigest()
        kernel_name = kernel_files.setdefault(
            kernel_hash, f"{owner_name}_k{kernel_hash[:12]}"
        )
        api_entry = module.api.entry_points[0]
        inputs = [
            parameter.name for parameter in api_entry.parameters
            if parameter.role == "input"
        ]
        outputs = [
            parameter.name for parameter in api_entry.parameters
            if parameter.role == "output"
        ]
        feed_ids = tuple(map(int, program_feed_order(program)))
        if len(feed_ids) != len(inputs):
            raise ValueError(
                f"region {region} WebAssembly feed ABI is inconsistent"
            )
        origins = dict((program.extras or {}).get("capture_feed_origins", {}))
        for value_id, input_name in zip(feed_ids, inputs):
            source = producer.get(value_id)
            if source is not None:
                edges.append({
                    "from": {
                        "module": source[0], "output": source[1]
                    },
                    "to": {
                        "module": module_name, "input": input_name
                    },
                })
                continue
            origin = origins.get(value_id, origins.get(str(value_id), {}))
            logical_name = str(
                origin.get("binding_name") or f"input_{value_id}"
            )
            logical_inputs.setdefault(logical_name, []).append(
                (module_name, input_name)
            )
            value_bindings.setdefault(value_id, f"in::{logical_name}")
        entries.append({
            "name": module_name,
            # The kernel file is shared by byte-identical regions; the method
            # card keeps its own per-region ``name`` for field-slot wiring.
            "kernel": kernel_name,
            "url": f"{module_dir}/{kernel_name}.wasm",
            "entry": module.api.entry,
            "inputs": inputs,
            "outputs": outputs,
            "value_type": module.api.metadata.get("value_type", "f64"),
            "element_bytes": module.api.metadata.get("element_bytes", 8),
            "memory_export": module.api.metadata.get("memory_export", "memory"),
            "reserved_bytes": module.api.metadata.get("reserved_bytes", 0),
            "static_data_offset": module.api.metadata.get("static_data_offset", 0),
            "shared_memory_import": module.api.metadata.get(
                "shared_memory_import", {"module": "env", "field": "memory"}
            ),
            "operation_count": len(program.steps),
            "extent_effect": fused_program_extent_effect(program),
            "node_ids": [int(step.result_id) for step in program.steps],
            "is_root": False,
            "region_index": region,
        })
        static_offset = max(
            static_offset,
            int(module.api.metadata.get("reserved_bytes", 0)),
        )
    # A field write (``index_set``) mutates its object in place: the updated
    # buffer IS the source buffer. Redirect the region's output storage onto the
    # ``data`` input's storage so the coordinator hands both the same resident
    # slot -- the scatter store then writes the live field, not a fresh copy.
    # (``build_class_inventory`` merges redirected keys into one field slot.)
    storage_redirects: dict[str, str] = {}
    for region in ordered_regions:
        for step in getattr(programs[region], "steps", ()):
            # Both subscript-store conventions put the target buffer first.
            if step.op_name not in ("index_set", "IndexedStore"):
                continue
            out_key = value_bindings.get(step.result_id)
            data_key = value_bindings.get(step.input_ids[0])
            if out_key and data_key and out_key != data_key:
                storage_redirects[str(out_key)] = str(data_key)

    # A container field (a dict/list keyed by RVAs or string names) needs a heap
    # map seeded by the coordinator, not a plain count-sized array. Detect them
    # with the same predicates the backend uses to lower container stores/reads,
    # and record the resident field key so the coordinator allocates a map there.
    container_field_keys: set[str] = set()
    for region in ordered_regions:
        program = programs[region]
        live = required_steps(program)
        descriptor = (
            _pure_container_store(program, live)
            or _pure_container_read(program, live)
        )
        if descriptor is None:
            continue
        key = value_bindings.get(descriptor[0])
        if key:
            container_field_keys.add(str(key))

    return modules, {
        "modules": entries,
        "edges": edges,
        "logical_inputs": logical_inputs,
        "shared_memory": True,
        "shared_static_bytes": static_offset,
        "control_regions": list(ordered_regions),
        "value_bindings": {
            str(value_id): key for value_id, key in value_bindings.items()
        },
        "storage_redirects": storage_redirects,
        "container_fields": sorted(container_field_keys),
        "heap": {
            "cursor_addr": HEAP_CURSOR_ADDR,
            "reserved_bytes": HEAP_RESERVED_BYTES,
            "map_capacity": DEFAULT_MAP_CAPACITY,
            "map_block_bytes": map_block_bytes(DEFAULT_MAP_CAPACITY),
            "map_header_bytes": MAP_HEADER_BYTES,
        },
        "region_count": len(ordered_regions),
        "unique_kernels": len(kernel_files),
    }


def _boundary_region(
    graph, window: tuple[int, ...], all_nodes: Sequence[int]
) -> DispatchRegion:
    """The same accounting ``process_graph_fusion.plan_process_graph_dispatches``
    uses for one connected component, applied to one contiguous window of
    the topological order instead: ``input_ids`` are outside-window
    predecessors (excluding constants, which get inlined again by
    ``dispatch_region_to_fused_program``); ``outputs`` are window members
    consumed outside the window, named after a ``return`` node's declared
    name when one is the consumer, or after the value id otherwise; a window
    member with no consumer at all (a graph root) is an output too, by the
    same reasoning ``plan_process_graph_dispatches`` uses.
    """

    node_set = set(window)
    input_ids = tuple(
        node_id
        for node_id in all_nodes
        if node_id not in node_set
        and graph.G.nodes[node_id].get("type") != "const"
        and any(child in node_set for child in graph.G.successors(node_id))
    )
    output_names: dict[int, str] = {}
    for node_id in window:
        for child in graph.G.successors(node_id):
            if child in node_set:
                continue
            child_data = graph.G.nodes[child]
            if child_data.get("type") == "return":
                name = str(
                    (child_data.get("attributes") or {}).get(
                        "name", f"result_{len(output_names)}"
                    )
                )
            else:
                name = f"value_{node_id}"
            output_names.setdefault(node_id, name)
        if node_id in graph.roots and node_id not in output_names:
            output_names[node_id] = f"value_{node_id}"
    outputs = tuple((name, node_id) for node_id, name in output_names.items())
    return DispatchRegion(window, input_ids, outputs, score=0.0)


def partition_reduced_program(
    program: FusedProgram, *, chunk_size: int, owner_name: str
) -> list[ClassModuleSpec]:
    """Cut a fully-reduced ``FusedProgram`` into class modules of at most
    ``chunk_size`` operations each, contiguous in the program's own
    topological order.

    ``owner_name`` names the whole process graph -- the chunk containing the
    program's declared outputs (its root, level 0) is named exactly
    ``owner_name``; every other chunk is ``f"{owner_name}_chunk{i}"``, a
    mechanical label, not a claim about what it means (see the module
    docstring).

    Returns specs in dependency order (earliest chunk first, root last), the
    same contract ``emit_class_modules``/``build_module_process_graph``
    already expect.
    """

    import networkx as nx

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    # A capture may retain observed values after the declared result. Those
    # nodes are not part of the executable program and must not become a
    # trailing deployment region that consumes live tensors and produces
    # nothing. Prune once, before graph projection, so every later boundary
    # calculation sees only the output-reachable calculation.
    from .fused_program_wasm_backend import required_steps

    program = FusedProgram(
        version=program.version,
        feeds=set(program.feeds),
        steps=required_steps(program),
        outputs=dict(program.outputs),
        state_in=program.state_in,
        meta=program.meta,
        extras=program.extras,
    )
    graph = fused_program_to_process_graph(program)
    topological = tuple(
        nx.lexicographical_topological_sort(graph.G, key=lambda n: int(n))
    )
    op_nodes = tuple(
        node_id
        for node_id in topological
        if graph.G.nodes[node_id].get("type") not in ("input", "const", "return")
    )

    windows = [
        op_nodes[start:start + chunk_size]
        for start in range(0, len(op_nodes), chunk_size)
    ] or [()]

    chunk_of: dict[int, int] = {
        node_id: index
        for index, window in enumerate(windows)
        for node_id in window
    }

    output_value_ids = set(program.outputs.values())
    root_index = next(
        (index for index, window in enumerate(windows)
         if any(node_id in output_value_ids for node_id in window)),
        len(windows) - 1,
    )

    specs: list[ClassModuleSpec] = []
    for index, window in enumerate(windows):
        region = _boundary_region(graph, window, topological)
        chunk_program = dispatch_region_to_fused_program(graph, region)
        depends_on = sorted({
            chunk_of[value_id]
            for value_id in region.input_ids
            if value_id in chunk_of and chunk_of[value_id] != index
        })
        calls = tuple(
            ClassModuleCallSite(
                callsite_id=callee_index,
                callee_name=(
                    owner_name if callee_index == root_index
                    else f"{owner_name}_chunk{callee_index}"
                ),
                callee_index=callee_index,
            )
            for callee_index in depends_on
        )
        name = owner_name if index == root_index else f"{owner_name}_chunk{index}"
        specs.append(ClassModuleSpec(
            name=name,
            index=index,
            region=region,
            program=chunk_program,
            calls=calls,
            is_root=(index == root_index),
        ))
    return specs


def emit_class_modules(
    specs: Sequence[ClassModuleSpec], *, dtype: str | None = None,
    link_calls: bool = True, shared_memory: bool = False,
):
    """Emit one ``WasmModule`` per spec, optionally import-linked.

    Each spec already carries its own lowered ``FusedProgram``
    (``partition_reduced_program`` produced it directly from the
    ``DispatchRegion`` cut), so unlike the closure-based approach this
    replaces, there is no separate numeric-lowering step to hand in.

    ``shared_memory`` emits independently downloadable host-scheduled punch
    cards that all import ``env.memory``. Their static data receives disjoint
    absolute addresses and the browser passes graph-derived offsets, so no
    tensor payload crosses JavaScript at a region seam.

    ``link_calls`` is the older experimental mode controlling whether a dependency gets a real WASM
    import/export declaration (``WasmImport``, proven end to end in
    ``test_wasm_binary.py``) or none at all, for a page that instead carries
    values between independently owned memories itself. See
    ``test_wasm_binary.py``'s ``_main_module`` for how a caller's body would
    actually invoke a linked import -- nothing in this file writes that call
    for the caller yet.
    """

    from .fused_program_wasm_backend import emit_wasm_module
    from .wasm_binary import WasmImport

    if shared_memory and link_calls:
        raise ValueError(
            "shared-memory punch cards are host-scheduled; link_calls must be false"
        )

    modules: dict[int, object] = {}
    static_cursor = 0
    for spec in specs:
        imports: list[WasmImport] = []
        static_data_offset = 0
        if shared_memory:
            # Discover this module's private static payload, then place that
            # payload at a unique address in the one imported memory. The
            # final emission bakes absolute table/constant addresses into the
            # punch card, so no instantiation can overwrite a peer's data.
            probe = emit_wasm_module(spec.program, name=spec.module_name, dtype=dtype)
            alignment = max(8, int(probe.api.metadata.get("element_bytes", 8)))
            static_cursor = ((static_cursor + alignment - 1) // alignment) * alignment
            static_data_offset = static_cursor
            local_static_bytes = int(probe.api.metadata.get("reserved_bytes", 0))
            imports.append(WasmImport(
                module="env", field="memory", kind="memory", memory_pages=1,
            ))
            static_cursor += local_static_bytes
        elif link_calls:
            for call in spec.calls:
                callee_module = modules[call.callee_index]
                imports.append(WasmImport(
                    module=call.callee_module_name,
                    field=_entry_field(callee_module),
                    kind="func",
                    parameter_types=_parameter_value_types(callee_module),
                ))
                imports.append(WasmImport(
                    module=call.callee_module_name,
                    field="memory",
                    kind="memory",
                    memory_pages=1,
                ))
        modules[spec.index] = emit_wasm_module(
            spec.program,
            name=spec.module_name,
            dtype=dtype,
            imports=imports,
            static_data_offset=static_data_offset,
        )
    return modules


def build_module_process_graph(specs: Sequence[ClassModuleSpec]):
    """Build a real ``transmogrifier.graph.graph_express2.ProcessGraph`` --
    the same class ``glsl_deployment_strategy.py`` builds while compiling,
    with its own ``ILPScheduler`` attached -- with one node per emitted
    class module and one edge per dependency (callee before caller).

    This is the thing to schedule, not a hand-rolled ordering: a bespoke
    FIFO/topological sort only ever gives a single flat sequence, and cannot
    answer "which modules could run at the same level" the way
    ``ILPScheduler.compute_asap_levels``/``compute_alap_levels`` can -- which
    is exactly the grouping a schedule visualization (modules as nodes that
    light up together, level by level) needs.

    Nodes are keyed by ``ClassModuleSpec.module_name``.
    """

    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    for spec in specs:
        graph.G.add_node(
            spec.module_name, label=spec.module_name, type="WasmClassModule",
            parents=[], children=[],
        )
    for spec in specs:
        for call in spec.calls:
            graph.connect(
                call.callee_module_name, spec.module_name,
                "result", f"call#{call.callsite_id}",
            )
    return graph


def schedule_module_levels(
    specs: Sequence[ClassModuleSpec], *, method: str = "asap"
) -> dict[str, int]:
    """The real ``ILPScheduler``'s level for each module, shifted so each
    process graph's own root -- its owner, the entrypoint itself -- sits at
    level 0.

    ``ILPScheduler`` numbers from its own leaves upward (0 at the earliest
    thing that can run), which puts prerequisite work at small positive
    numbers and the entrypoint wherever the deepest chain happens to land.
    That is backwards for what this is for: the entrypoint is the thing the
    whole graph is named after, so it is the reference point, and anything
    that has to happen before it -- data it depends on, a chunk that has to
    run first -- is *behind* it, at a negative level, not counted up from
    some unrelated zero. The shift is exactly the root's own raw level:
    subtract it from every level in the root's own weakly-connected
    component, and the root becomes 0 while everything upstream of it lands
    at -1, -2, and so on.

    ``specs`` may describe more than one disjoint process graph at once
    (several unrelated compiled programs shown together) -- each root's
    component is shifted independently, against its own root, not against
    some other program's.
    """

    import networkx as nx

    specs_by_name = {spec.module_name: spec for spec in specs}
    graph = build_module_process_graph(specs)
    raw_levels = dict(graph.scheduler.compute_levels(method, "dependency"))

    shift_by_node: dict[str, int] = {}
    for component in nx.weakly_connected_components(graph.G):
        roots = [
            name for name in component if specs_by_name[name].is_root
        ]
        if len(roots) != 1:
            raise ValueError(
                f"expected exactly one root module per connected component, "
                f"found {roots!r} in {sorted(component)!r}"
            )
        shift = raw_levels[roots[0]]
        for name in component:
            shift_by_node[name] = shift

    return {
        name: level - shift_by_node[name] for name, level in raw_levels.items()
    }


def build_manifest(
    specs: Sequence[ClassModuleSpec],
    modules,
    *,
    module_dir: str = "modules",
) -> dict:
    """A JSON-able manifest for ``process_graph_runner.js``: one entry per
    emitted module (name, its own ``.wasm`` file, entry symbol, and its
    input/output parameter names in declared order) plus the edges wiring
    one module's output to another's input.

    Each module's ``inputs``/``outputs``/``entry``/``value_type``/
    ``element_bytes``/``memory_export`` come straight from its own
    ``CompiledProgramAPI`` (``WasmModule.api``) -- the manifest describes
    exactly the calling contract the artifact itself declares.

    Edges are derived here, not stated by hand and not reconstructed from
    ambiguous bindings the way the closure-based approach this file replaced
    had to: a spec's ``region.input_ids`` are exactly the value ids it needs
    from outside itself, and ``DispatchRegion``/``partition_reduced_program``
    already guarantee each one is produced by exactly one other chunk (or is
    a genuine program feed, in which case it is not a graph edge at all --
    the caller supplies it directly, see ``graph_inputs`` below).
    """

    from .fused_program_wasm_backend import program_feed_order

    specs_by_index = {spec.index: spec for spec in specs}
    producer_of_value: dict[int, tuple[str, str]] = {}
    for spec in specs:
        for name, value_id in spec.region.outputs:
            producer_of_value[value_id] = (spec.module_name, name)

    module_entries = []
    edges = []
    graph_input_value_ids: dict[str, list[tuple[int, str]]] = {}
    for spec in specs:
        module = modules[spec.index]
        entry_point = module.api.entry_points[0]
        inputs = [p.name for p in entry_point.parameters if p.role == "input"]
        outputs = [p.name for p in entry_point.parameters if p.role == "output"]
        module_entries.append({
            "name": spec.module_name,
            "url": f"{module_dir}/{spec.module_name}.wasm",
            "entry": module.api.entry,
            "inputs": inputs,
            "outputs": outputs,
            "value_type": module.api.metadata.get("value_type", "f64"),
            "element_bytes": module.api.metadata.get("element_bytes", 8),
            "memory_export": module.api.metadata.get("memory_export", "memory"),
            "reserved_bytes": module.api.metadata.get("reserved_bytes", 0),
            "static_data_offset": module.api.metadata.get("static_data_offset", 0),
            "shared_memory_import": module.api.metadata.get("shared_memory_import", {}),
            "operation_count": len(spec.region.node_ids),
            "node_ids": list(spec.region.node_ids),
            "is_root": spec.is_root,
        })
        # Pair names with the exact order used by emit_wasm_module. Region
        # input_ids are boundary-set order, while the module ABI follows
        # first use inside the lowered program; confusing them silently
        # routes correctly-shaped arrays to the wrong operands.
        emitted_feed_ids = program_feed_order(spec.program)
        if len(emitted_feed_ids) != len(inputs):
            raise ValueError(
                f"{spec.module_name} declares {len(inputs)} inputs for "
                f"{len(emitted_feed_ids)} program feeds"
            )
        for value_id, input_name in zip(emitted_feed_ids, inputs):
            producer = producer_of_value.get(value_id)
            if producer is not None:
                edges.append({
                    "from": {"module": producer[0], "output": producer[1]},
                    "to": {"module": spec.module_name, "input": input_name},
                })
            else:
                graph_input_value_ids.setdefault(spec.module_name, []).append(
                    (value_id, input_name)
                )

    shared_memory = all(
        bool(entry.get("shared_memory_import")) for entry in module_entries
    ) if module_entries else False
    return {
        "modules": module_entries,
        "edges": edges,
        "graph_input_value_ids": graph_input_value_ids,
        "shared_memory": shared_memory,
        "shared_static_bytes": max(
            (int(entry.get("reserved_bytes", 0)) for entry in module_entries),
            default=0,
        ),
    }


def build_embedded_class_graph(
    specs: Sequence[ClassModuleSpec],
    modules,
    program: FusedProgram,
    *,
    entrypoint: str,
    embed_binaries: bool = True,
    module_dir: str = "modules",
    storage_redirects: Mapping[str, str] | None = None,
) -> dict:
    """Adapt ``build_manifest`` for a logical program's browser shell.

    By default the historical self-contained representation is retained for
    callers that need one file.  A deployed site should set
    ``embed_binaries=False`` so the manifest keeps relative module URLs and
    the runner fetches each private region lazily.

    Also includes ``logical_inputs`` -- ``{parameter_name: [[module,
    input], ...]}`` -- resolving every value nothing in the graph produces
    back to the same real source-parameter name
    ``describe_process_graph_api`` uses, so a shell's input row (one per
    logical parameter) knows every ``(module, input)`` pair its value has
    to be delivered to, even when more than one chunk needs it directly.
    """

    import base64

    manifest = build_manifest(specs, modules, module_dir=module_dir)
    origins = (program.extras or {}).get("capture_feed_origins", {})

    module_entries = []
    for entry in manifest["modules"]:
        spec = next(s for s in specs if s.module_name == entry["name"])
        module = modules[spec.index]
        if embed_binaries:
            module_entries.append({
                **{k: v for k, v in entry.items() if k != "url"},
                "wasm_base64": base64.b64encode(module.binary).decode("ascii"),
            })
        else:
            module_entries.append(dict(entry))

    logical_inputs: dict[str, list[tuple[str, str]]] = {}
    for module_name, value_entries in manifest["graph_input_value_ids"].items():
        for value_id, input_name in value_entries:
            origin = origins.get(value_id)
            binding_name = origin.get("binding_name") if origin else None
            logical_name = binding_name or f"input_{value_id}"
            logical_inputs.setdefault(logical_name, []).append(
                (module_name, input_name)
            )

    root_spec = next(spec for spec in specs if spec.is_root)
    root_entry = next(
        m for m in manifest["modules"] if m["name"] == root_spec.module_name
    )
    producer_of_value = {
        value_id: (spec.module_name, output_name)
        for spec in specs
        for output_name, value_id in spec.region.outputs
    }
    logical_outputs = {}
    for output_name, value_id in program.outputs.items():
        producer = producer_of_value.get(value_id)
        if producer is None:
            raise ValueError(
                f"segmented program output {output_name!r} value {value_id} "
                "has no producing module"
            )
        logical_outputs[output_name] = list(producer)

    resolved_redirects = {}
    for input_name, output_name in dict(storage_redirects or {}).items():
        if input_name not in logical_inputs:
            raise ValueError(f"storage redirect names unknown input {input_name!r}")
        producer = logical_outputs.get(output_name)
        if producer is None:
            raise ValueError(f"storage redirect names unknown output {output_name!r}")
        resolved_redirects[f"in::{input_name}"] = (
            f"out::{producer[0]}::{producer[1]}"
        )

    from .process_graph_shell import schedule_table

    return {
        "modules": module_entries,
        "edges": manifest["edges"],
        "logical_inputs": {
            name: [list(pair) for pair in pairs]
            for name, pairs in logical_inputs.items()
        },
        "root_module": root_spec.module_name,
        "root_outputs": root_entry["outputs"],
        "logical_outputs": logical_outputs,
        "storage_redirects": resolved_redirects,
        "schedule": schedule_table(specs),
        "shared_memory": manifest["shared_memory"],
        "shared_static_bytes": manifest["shared_static_bytes"],
    }


def describe_process_graph_api(
    specs: Sequence[ClassModuleSpec],
    modules,
    program: FusedProgram,
    *,
    entrypoint: str,
):
    """The whole segmented kernel's own calling contract -- what a caller
    outside the graph needs to know: its real external inputs, resolved
    back to the source parameter each one actually is (via
    ``program.extras["capture_feed_origins"]``, wherever any chunk needed
    one directly), and the root chunk's own declared outputs.

    Reuses ``compiled_program_api.CompiledProgramAPI``/``EntryPoint``/
    ``Parameter`` unchanged -- the same descriptor format
    ``wasm_html_shell.emit_html_shell`` already renders for a single
    module's inputs/outputs, so this can be handed to it directly with no
    new rendering code, only a new way of actually calling the kernel it
    describes (which is a shell concern, not this function's).
    """

    from .compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter

    manifest = build_manifest(specs, modules)
    origins = (program.extras or {}).get("capture_feed_origins", {})

    root_spec = next(spec for spec in specs if spec.is_root)
    root_module = modules[root_spec.index]
    value_type = root_module.api.metadata.get("value_type", "f64")
    c_type = "float" if value_type == "f32" else "double"
    ctypes_name = "c_float" if value_type == "f32" else "c_double"

    # Every value nothing in the graph produces, across every chunk that
    # needs one directly, resolved back to its real source parameter --
    # several chunks needing the same page-level value collapse to the one
    # logical kernel input it actually is. Iterated in spec (dependency)
    # order, so the result is deterministic for a given compile.
    external_value_ids = {
        value_id
        for entries in manifest["graph_input_value_ids"].values()
        for value_id, _input_name in entries
    }
    ordered_external_ids = [
        value_id
        for value_id in ordered_feed_ids(program)
        if value_id in external_value_ids
    ]
    ordered_external_ids.extend(sorted(
        external_value_ids - set(ordered_external_ids)
    ))
    input_names = []
    used_names = set()
    for index, value_id in enumerate(ordered_external_ids):
        origin = origins.get(value_id)
        binding_name = origin.get("binding_name") if origin else None
        name = str(binding_name or f"input_{value_id}")
        if not name.isidentifier() or name in used_names:
            name = f"input_{index}"
        used_names.add(name)
        input_names.append(name)

    parameters = [Parameter(
        name="count", role="extent", dtype="int32", c_type="int32_t",
        ctypes_name="c_int32", passing="value",
    )]
    for name in input_names:
        parameters.append(Parameter(
            name=name, role="input", dtype=value_type, c_type=c_type,
            ctypes_name=ctypes_name, passing="value", extent="count",
        ))
    for name in program.outputs:
        parameters.append(Parameter(
            name=str(name), role="output", dtype=value_type, c_type=c_type,
            ctypes_name=ctypes_name, passing="value", extent="count",
        ))

    return CompiledProgramAPI(
        module=entrypoint,
        language="wasm",
        entry=entrypoint,
        entry_points=(
            EntryPoint(
                name=entrypoint,
                symbol=entrypoint,
                kind="numerical",
                parameters=tuple(parameters),
                note=(
                    "one logical compiled program; its browser deployment is "
                    f"privately segmented into {len(specs)} WASM regions"
                ),
            ),
        ),
        metadata={
            "value_type": value_type,
            "element_bytes": root_module.api.metadata.get("element_bytes", 8),
            "memory_export": root_module.api.metadata.get("memory_export", "memory"),
            "reserved_bytes": 0,
            "execution_mode": "segmented",
            "class_graph_module_count": len(specs),
        },
    )


def build_hued_process_graph_views(
    original_graph,
    program: FusedProgram,
    specs: Sequence[ClassModuleSpec],
) -> dict:
    """Describe original and reduced schedules with provenance hue identities.

    Hues identify concepts and deployment regions; they are not final pixel
    colours. The web shell mixes every identity reaching a node and applies
    its rolling phosphor/decay function at draw time, so profiling pulses can
    accumulate even when execution is faster than the display refresh.
    """

    import networkx as nx
    from .fused_program_wasm_backend import required_steps

    identities: dict[str, dict] = {}

    def stable_hue(label: str) -> float:
        return float(sum((i + 1) * ord(ch) for i, ch in enumerate(label)) % 360)

    def add_identity(identity: str, label: str, kind: str, hue: float | None = None):
        identities.setdefault(identity, {
            "label": label,
            "kind": kind,
            "hue": stable_hue(identity) if hue is None else float(hue) % 360.0,
        })

    def levels_of(nx_graph) -> dict:
        if nx.is_directed_acyclic_graph(nx_graph):
            return {
                node_id: level
                for level, generation in enumerate(nx.topological_generations(nx_graph))
                for node_id in generation
            }
        return {node_id: 0 for node_id in nx_graph.nodes}

    def node_payload(nx_graph, contributors, groups, regions=None):
        levels = levels_of(nx_graph)
        nodes = []
        for node_id, data in nx_graph.nodes(data=True):
            nodes.append({
                "id": str(node_id),
                "label": str(data.get("label") or data.get("op") or data.get("type") or node_id)[:96],
                "type": str(data.get("type") or data.get("op") or "?"),
                "level": int(levels.get(node_id, 0)),
                "group": int(groups.get(node_id, 0)),
                "region": None if regions is None else regions.get(node_id),
                "contributors": sorted(contributors.get(node_id, ())),
            })
        return {
            "nodes": nodes,
            "edges": [[str(left), str(right)] for left, right in nx_graph.edges],
            "level_min": min(levels.values(), default=0),
            "level_max": max(levels.values(), default=0),
            "groups": max(groups.values(), default=-1) + 1,
        }

    # Original AST ProcessGraph: top-level functions are the conceptual
    # identities. Every structural node in a function's ancestry keeps that
    # identity, while the Module/root naturally mixes all four.
    original_nx = getattr(original_graph, "G", original_graph)
    original_contributors: dict[object, set[str]] = {
        node_id: set() for node_id in original_nx.nodes
    }
    original_groups: dict[object, int] = {}
    functions = [
        node_id for node_id, data in original_nx.nodes(data=True)
        if str(data.get("type")) == "FunctionDef"
    ]
    for group, function_id in enumerate(functions):
        data = original_nx.nodes[function_id]
        name = next((
            str(original_nx.nodes[parent].get("label"))
            for parent, role in data.get("parents", ())
            if role == "name" and parent in original_nx
        ), f"function_{group}")
        identity = f"concept:{name}"
        add_identity(identity, name, "concept")
        members = nx.ancestors(original_nx, function_id) | {function_id}
        for node_id in members:
            original_contributors[node_id].add(identity)
            original_groups.setdefault(node_id, group)
    structural = "concept:program-structure"
    add_identity(structural, "program structure", "concept", 210.0)
    for node_id in original_nx.nodes:
        if not original_contributors[node_id]:
            inherited = set().union(*(
                original_contributors.get(parent, set())
                for parent in original_nx.predecessors(node_id)
            ))
            original_contributors[node_id] = inherited or {structural}
        original_groups.setdefault(node_id, len(functions))

    # Reduced graph: every region is a procedural identity. Identities flow
    # through dependencies, so an operation after a seam carries the mixed
    # colours of every region that materially contributes to it.
    live_program = FusedProgram(
        version=program.version, feeds=set(program.feeds),
        steps=required_steps(program), outputs=dict(program.outputs),
        state_in=program.state_in, meta=program.meta, extras=program.extras,
    )
    reduced = fused_program_to_process_graph(live_program)
    reduced_nx = reduced.G
    region_of = {
        node_id: spec.index
        for spec in specs for node_id in spec.region.node_ids
    }
    for spec in specs:
        add_identity(
            f"region:{spec.index}", spec.module_name, "region",
            360.0 * spec.index / max(1, len(specs)),
        )
    origins = (program.extras or {}).get("capture_feed_origins", {}) or {}
    for feed_id in program.feeds:
        name = str((origins.get(feed_id) or {}).get("binding_name") or f"feed_{feed_id}")
        add_identity(f"feed:{name}", name, "feed")

    reduced_contributors: dict[object, set[str]] = {}
    reduced_groups: dict[object, int] = {}
    order = (
        list(nx.topological_sort(reduced_nx))
        if nx.is_directed_acyclic_graph(reduced_nx) else list(reduced_nx.nodes)
    )
    for node_id in order:
        contributors = set().union(*(
            reduced_contributors.get(parent, set())
            for parent in reduced_nx.predecessors(node_id)
        ))
        if node_id in program.feeds:
            name = str((origins.get(node_id) or {}).get("binding_name") or f"feed_{node_id}")
            contributors.add(f"feed:{name}")
        region = region_of.get(node_id)
        if region is not None:
            contributors.add(f"region:{region}")
            reduced_groups[node_id] = region
        else:
            parent_groups = [
                reduced_groups[parent] for parent in reduced_nx.predecessors(node_id)
                if parent in reduced_groups
            ]
            reduced_groups[node_id] = parent_groups[0] if parent_groups else 0
        reduced_contributors[node_id] = contributors or {structural}

    return {
        "schema": "turing-process-graph-hues-v1",
        "identities": identities,
        "views": {
            "original": node_payload(
                original_nx, original_contributors, original_groups,
            ),
            "reduced": node_payload(
                reduced_nx, reduced_contributors, reduced_groups, region_of,
            ),
        },
    }


def _entry_field(module) -> str:
    """The exported entry-point name a callee module's ``run`` function is
    reachable under -- ``emit_wasm_module`` always names it from its own
    ``function_name`` default, recorded on the returned ``WasmModule``'s
    API descriptor."""

    return module.api.entry


def _parameter_value_types(module) -> tuple[str, ...]:
    """Every parameter to an emitted class-module function is an ``i32``
    byte offset (or the ``count``) -- see ``fused_program_wasm_backend``'s
    module docstring. One entry per declared parameter."""

    return tuple("i32" for _ in module.parameters)


__all__ = [
    "ClassModuleCallSite",
    "ClassModuleSpec",
    "build_embedded_class_graph",
    "build_hued_process_graph_views",
    "build_manifest",
    "build_module_process_graph",
    "describe_process_graph_api",
    "emit_class_modules",
    "emit_control_region_modules",
    "partition_reduced_program",
    "fused_program_extent_effect",
    "schedule_module_levels",
]
