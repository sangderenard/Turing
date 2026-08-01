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

from dataclasses import dataclass
from typing import Sequence

from ..common.tensors.fused_ir import FusedProgram
from .process_graph_fusion import (
    DispatchRegion, dispatch_region_to_fused_program,
    fused_program_to_process_graph,
)


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
    link_calls: bool = True,
):
    """Emit one ``WasmModule`` per spec, optionally import-linked.

    Each spec already carries its own lowered ``FusedProgram``
    (``partition_reduced_program`` produced it directly from the
    ``DispatchRegion`` cut), so unlike the closure-based approach this
    replaces, there is no separate numeric-lowering step to hand in.

    ``link_calls`` controls whether a dependency gets a real WASM
    import/export declaration (``WasmImport``, proven end to end in
    ``test_wasm_binary.py``) or none at all, for a page that instead carries
    values between modules itself (``process_graph_runner.js``). See
    ``test_wasm_binary.py``'s ``_main_module`` for how a caller's body would
    actually invoke a linked import -- nothing in this file writes that call
    for the caller yet.
    """

    from .fused_program_wasm_backend import emit_wasm_module
    from .wasm_binary import WasmImport

    modules: dict[int, object] = {}
    for spec in specs:
        imports: list[WasmImport] = []
        if link_calls:
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
            "count": 1,
            "inputs": inputs,
            "outputs": outputs,
            "value_type": module.api.metadata.get("value_type", "f64"),
            "element_bytes": module.api.metadata.get("element_bytes", 8),
            "memory_export": module.api.metadata.get("memory_export", "memory"),
        })
        for value_id, input_name in zip(spec.region.input_ids, inputs[:len(spec.region.input_ids)]):
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

    return {
        "modules": module_entries,
        "edges": edges,
        "graph_input_value_ids": graph_input_value_ids,
    }


def build_embedded_class_graph(
    specs: Sequence[ClassModuleSpec],
    modules,
    program: FusedProgram,
    *,
    entrypoint: str,
) -> dict:
    """The same manifest ``build_manifest`` produces, adapted for a
    self-contained page: each module's bytes are embedded as base64
    (``wasm_base64``) instead of a fetchable ``url``, since a shell page
    (``wasm_html_shell.py``) is one file, not a page plus a ``modules/``
    directory the way ``wasm-gallery/`` pages are.

    Also includes ``logical_inputs`` -- ``{parameter_name: [[module,
    input], ...]}`` -- resolving every value nothing in the graph produces
    back to the same real source-parameter name
    ``describe_process_graph_api`` uses, so a shell's input row (one per
    logical parameter) knows every ``(module, input)`` pair its value has
    to be delivered to, even when more than one chunk needs it directly.
    """

    import base64

    manifest = build_manifest(specs, modules)
    origins = (program.extras or {}).get("capture_feed_origins", {})

    module_entries = []
    for entry in manifest["modules"]:
        spec = next(s for s in specs if s.module_name == entry["name"])
        module = modules[spec.index]
        module_entries.append({
            **{k: v for k, v in entry.items() if k != "url"},
            "wasm_base64": base64.b64encode(module.binary).decode("ascii"),
        })

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

    return {
        "modules": module_entries,
        "edges": manifest["edges"],
        "logical_inputs": {
            name: [list(pair) for pair in pairs]
            for name, pairs in logical_inputs.items()
        },
        "root_module": root_spec.module_name,
        "root_outputs": root_entry["outputs"],
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
    root_output_parameters = tuple(
        p for p in root_module.api.entry_points[0].parameters if p.role == "output"
    )
    value_type = root_module.api.metadata.get("value_type", "f64")

    # Every value nothing in the graph produces, across every chunk that
    # needs one directly, resolved back to its real source parameter --
    # several chunks needing the same page-level value collapse to the one
    # logical kernel input it actually is. Iterated in spec (dependency)
    # order, so the result is deterministic for a given compile.
    seen: dict[int, str] = {}
    for spec in specs:
        for value_id, _input_name in manifest["graph_input_value_ids"].get(
            spec.module_name, ()
        ):
            if value_id in seen:
                continue
            origin = origins.get(value_id)
            binding_name = origin.get("binding_name") if origin else None
            seen[value_id] = binding_name or f"input_{value_id}"

    parameters = [Parameter(
        name="count", role="extent", dtype="int32", c_type="int32_t",
        ctypes_name="c_int32", passing="value",
    )]
    for name in seen.values():
        parameters.append(Parameter(
            name=name, role="input", dtype=value_type, c_type="int32_t",
            ctypes_name="c_int32", passing="value", extent="count",
        ))
    parameters.extend(root_output_parameters)

    return CompiledProgramAPI(
        module=entrypoint,
        language="wasm-class-graph",
        entry=entrypoint,
        entry_points=(
            EntryPoint(
                name=entrypoint,
                symbol=root_spec.module_name,
                kind="numerical",
                parameters=tuple(parameters),
                note=(
                    f"auto-segmented into {len(specs)} WASM class module(s) "
                    f"run by process_graph_runner.js; {root_spec.module_name} "
                    "is the module holding this kernel's own declared outputs"
                ),
            ),
        ),
        metadata={
            "value_type": value_type,
            "element_bytes": root_module.api.metadata.get("element_bytes", 8),
            "memory_export": root_module.api.metadata.get("memory_export", "memory"),
            "reserved_bytes": 0,
            "class_graph_module_count": len(specs),
        },
    )


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
    "build_manifest",
    "build_module_process_graph",
    "describe_process_graph_api",
    "emit_class_modules",
    "partition_reduced_program",
    "schedule_module_levels",
]
