"""GLSL deployment-strategy stage for ProcessGraphs."""

from __future__ import annotations

import ast
import copy
from typing import Any

from .deployment_fifo import DeploymentFIFO
from .process_graph_fusion import serialize_scheduled_operator_dispatches
from ..common.tensors.abstraction import AbstractTensor
from ..common.tensors.accelerator_backends.glsl_fused_network import (
    GLSLFusedProgramNetwork,
)
from ..common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
    compile_recorded_elementwise_tape,
)
from ..common.tensors.accelerator_backends.glsl_backend import (
    emit_multi_output_program_source,
)
from ..common.tensors.autograd import autograd
from ..common.tensors.fused_ir import FusedProgram, OpStep
from ..transmogrifier.graph.graph_deep_compiler import GraphDeepCompiler
from ..transmogrifier.operator_defs import (
    abstract_tensor_funcs,
    abstract_tensor_sigs,
)


def _is_ast_metadata_node(graph: Any, node_id: int) -> bool:
    """Return whether a structural AST node carries no runtime value."""

    data = graph.G.nodes[node_id]
    node_type = data.get("type")
    return (
        node_type in {"str", "NoneType"}
        or isinstance(
            data.get("expr_obj"),
            (
                ast.Module,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.arguments,
                ast.arg,
            ),
        )
    )


def _is_dispatch_metadata_node(graph: Any, node_id: int) -> bool:
    """Return whether a node routes syntax but performs no computation."""

    data = graph.G.nodes[node_id]
    return (
        bool(
            (data.get("attributes") or {}).get(
                "contextual_requirement"
            )
        )
        or
        _is_ast_metadata_node(graph, node_id)
        or isinstance(
            data.get("expr_obj"),
            (
                ast.expr_context,
                ast.operator,
                ast.unaryop,
                ast.boolop,
                ast.cmpop,
            ),
        )
        or (
            isinstance(data.get("expr_obj"), ast.Attribute)
            and graph.G.out_degree(node_id) == 0
        )
        or (
            data.get("type") == "Load"
            and (data.get("attributes") or {}).get("source_type") == "Name"
        )
    )


def _dispatch_subgraph(graph: Any, node_ids: tuple[int, ...]) -> Any:
    """Return the planned dispatch as an independent ProcessGraph subgraph."""

    selected = set(node_ids)
    boundary = {
        parent
        for node_id in node_ids
        for parent in graph.G.predecessors(node_id)
        if parent not in selected
        and not _is_ast_metadata_node(graph, parent)
    }
    included = selected | boundary
    subgraph = copy.copy(graph)
    subgraph.G = graph.G.subgraph(included).copy()

    for node_id in boundary:
        data = subgraph.G.nodes[node_id]
        data["type"] = "Input"
        data["op"] = "input"
        data["label"] = f"value_{node_id}"
        data["parents"] = []
        for parent in tuple(subgraph.G.predecessors(node_id)):
            subgraph.G.remove_edge(parent, node_id)

    for node_id in subgraph.G:
        data = subgraph.G.nodes[node_id]
        data["parents"] = [
            (parent, role)
            for parent, role in data.get("parents", ())
            if parent in included
        ]
        data["children"] = [
            (child, role)
            for child, role in data.get("children", ())
            if child in included
        ]

    source_levels = graph.levels
    minimum_level = min(
        (int(source_levels.get(node_id, 0)) for node_id in included),
        default=0,
    )
    subgraph.levels = {
        node_id: int(source_levels.get(node_id, 0)) - minimum_level
        for node_id in included
    }
    subgraph.roots = [
        node_id
        for node_id in node_ids
        if not any(child in selected for child in graph.G.successors(node_id))
    ]
    deployment_outputs = tuple(subgraph.roots)
    next_node_id = max(
        (node_id for node_id in graph.G if isinstance(node_id, int)),
        default=0,
    ) + 1
    output_level = max(subgraph.levels.values(), default=0) + 1
    store_nodes = []
    for output_id in deployment_outputs:
        while next_node_id in subgraph.G:
            next_node_id += 1
        store_id = next_node_id
        next_node_id += 1
        subgraph.G.add_node(
            store_id,
            type="Store",
            op="store",
            label=f"value_{output_id}",
            parents=[(output_id, "value")],
            children=[],
        )
        subgraph.G.add_edge(output_id, store_id)
        subgraph.G.nodes[output_id].setdefault("children", []).append(
            (store_id, "value")
        )
        subgraph.levels[store_id] = output_level
        store_nodes.append(store_id)
    subgraph.roots = store_nodes
    subgraph.G.graph["deployment_inputs"] = tuple(
        node_id
        for node_id, data in subgraph.G.nodes(data=True)
        if data.get("type") == "Input"
    )
    subgraph.G.graph["deployment_outputs"] = deployment_outputs
    return subgraph


def _compiler_input_name(label: str) -> str:
    """Mirror the established GraphDeepCompiler input naming convention."""

    import re

    name = str(label).strip()
    lowered = name.lower()
    match = re.match(r"([a-zA-Z]+)[_\d]*$", lowered)
    root = match.group(1) if match else lowered
    if root in {"i", "j", "k", "l", "m", "n"}:
        prefix = "int"
    elif (
        lowered.startswith("num")
        or lowered.endswith("idx")
        or lowered.isdigit()
    ):
        prefix = "int"
    elif lowered.startswith("is_") or lowered.startswith("has_"):
        prefix = "bool"
    else:
        prefix = "float"
    return f"{prefix}{name}"


def _remap_captured_program(
    captured: CapturedFusedProgram,
    *,
    feed_ids: dict[int, int],
    output_ids: tuple[int, ...],
) -> CapturedFusedProgram:
    """Replace transient tensor identities with ProcessGraph boundary IDs."""

    program = captured.program
    captured_outputs = tuple(program.outputs.values())
    if len(captured_outputs) != len(output_ids):
        raise ValueError(
            "captured terminal count does not match planned outputs"
        )
    id_map = dict(feed_ids)
    id_map.update(zip(captured_outputs, output_ids))
    remap = lambda value_id: id_map.get(value_id, value_id)
    remapped = FusedProgram(
        version=program.version,
        feeds={remap(value_id) for value_id in program.feeds},
        steps=[
            OpStep(
                step_id=step.step_id,
                op_name=step.op_name,
                input_ids=[remap(value_id) for value_id in step.input_ids],
                attrs=dict(step.attrs),
                result_id=remap(step.result_id),
                mode_sensitive=step.mode_sensitive,
                level=step.level,
            )
            for step in program.steps
        ],
        outputs={
            f"value_{output_id}": output_id
            for output_id in output_ids
        },
        state_in=(
            None
            if program.state_in is None
            else {remap(value_id) for value_id in program.state_in}
        ),
        meta={
            remap(value_id): meta
            for value_id, meta in (program.meta or {}).items()
        },
        extras=program.extras,
    )
    return CapturedFusedProgram(
        remapped,
        {
            remap(value_id): value
            for value_id, value in captured.feeds.items()
        },
    )


def strategize_glsl_deployment(
    graph: Any,
    *,
    max_nodes_per_dispatch: int = 256,
) -> type:
    """Build a stateful shell around the graph's flat dispatch schedule."""

    dispatch_plan = serialize_scheduled_operator_dispatches(
        graph,
        max_nodes_per_dispatch=max_nodes_per_dispatch,
    )
    executable_dispatch_nodes = tuple(
        tuple(
            node_id
            for node_id in dispatch.node_ids
            if not _is_dispatch_metadata_node(graph, node_id)
        )
        for dispatch in dispatch_plan.dispatches
    )
    dispatch_subgraphs = tuple(
        _dispatch_subgraph(graph, node_ids)
        for node_ids in executable_dispatch_nodes
        if node_ids
    )
    deep_compilers = tuple(
        GraphDeepCompiler(
            subgraph,
            dict(abstract_tensor_funcs),
            abstract_tensor_sigs,
        )
        for subgraph in dispatch_subgraphs
    )

    class_ast = ast.parse(
        """
class ProcessGraphGLSLDeployment:
    process_graph = __process_graph__
    dispatch_plan = __dispatch_plan__
    dispatch_subgraphs = __dispatch_subgraphs__
    deep_compilers = __deep_compilers__
    source_node_count = __source_node_count__
    primitive_count = __primitive_count__
    loop_count = __loop_count__
    dispatch_count = __dispatch_count__

    def __init__(
        self,
        *,
        batch_size=None,
        profiling=False,
        input_slots=64,
        output_slots=64,
        **tuning,
    ):
        self.state = {}
        self.tuning = dict(tuning)
        self.batch_size = dict(
            __deployment_batches__
            if batch_size is None
            else batch_size
        )
        self.profiling = profiling
        self.profile = {}
        self.inputs = DeploymentFIFO(input_slots)
        self.outputs = DeploymentFIFO(output_slots)
        self.fused_network = None
        self.forward_tapes = ()
        self.forward_feed_ids = ()
        self.compiled_tapes = ()
        self.glsl_sources = ()

    def capture_forward_tapes(
        self,
        dispatch_inputs,
        *,
        backend="numpy",
        device=None,
    ):
        if len(dispatch_inputs) != len(self.deep_compilers):
            raise ValueError(
                "dispatch_inputs must provide one input mapping per "
                "planned subgraph"
            )
        tapes = []
        feed_maps = []
        with AbstractTensor.use_backend(backend, device):
            for compiler, subgraph, inputs in zip(
                self.deep_compilers,
                self.dispatch_subgraphs,
                dispatch_inputs,
            ):
                function = compiler.build_function(device=device)
                with autograd.forward_capture() as tape:
                    function(**dict(inputs))
                tapes.append(tape)
                feed_maps.append({
                    id(inputs[
                        _compiler_input_name(
                            subgraph.G.nodes[node_id]["label"]
                        )
                    ]): node_id
                    for node_id in subgraph.G.graph["deployment_inputs"]
                })
        self.forward_tapes = tuple(tapes)
        self.forward_feed_ids = tuple(feed_maps)
        return self.forward_tapes

    def compile_forward_tapes(self, *, dynamic_scalar_ids=()):
        if not self.forward_tapes:
            raise RuntimeError("capture forward tapes before compiling them")
        if dynamic_scalar_ids and (
            len(dynamic_scalar_ids) != len(self.forward_tapes)
        ):
            raise ValueError(
                "dynamic_scalar_ids must provide one sequence per tape"
            )
        scalar_ids = (
            dynamic_scalar_ids
            if dynamic_scalar_ids
            else ((),) * len(self.forward_tapes)
        )
        compiled = []
        sources = []
        for tape, feed_ids, subgraph, dynamic_ids in zip(
            self.forward_tapes,
            self.forward_feed_ids,
            self.dispatch_subgraphs,
            scalar_ids,
        ):
            captured = compile_recorded_elementwise_tape(
                tape,
                dynamic_scalar_ids=tuple(dynamic_ids),
            )
            captured = _remap_captured_program(
                captured,
                feed_ids=feed_ids,
                output_ids=tuple(
                    subgraph.G.graph["deployment_outputs"]
                ),
            )
            compiled.append(captured)
            sources.append(
                emit_multi_output_program_source(captured.program)
            )
        self.compiled_tapes = tuple(compiled)
        self.glsl_sources = tuple(sources)
        return self.glsl_sources

    def install_fused_programs(self, programs, *, fifo_slots=2):
        if self.fused_network is not None:
            self.fused_network.release()
        self.fused_network = GLSLFusedProgramNetwork(
            programs,
            fifo_slots=fifo_slots,
        )
        return self

    def execute(self, feeds):
        if self.fused_network is None:
            raise RuntimeError(
                "install the vertical fused programs before execution"
            )
        return self.fused_network.execute(feeds)

    def __call__(self, feeds):
        return self.execute(feeds)

    def release(self):
        if self.fused_network is not None:
            self.fused_network.release()
            self.fused_network = None
"""
    )
    executable_node_ids = {
        node_id
        for node_ids in executable_dispatch_nodes
        for node_id in node_ids
    }
    node_locations = {
        node_id: location
        for node_id, location in dispatch_plan.node_locations.items()
        if node_id in executable_node_ids
    }
    namespace = {
        "__process_graph__": graph,
        "__dispatch_plan__": dispatch_plan,
        "__dispatch_subgraphs__": dispatch_subgraphs,
        "__deep_compilers__": deep_compilers,
        "__source_node_count__": graph.G.number_of_nodes(),
        "__primitive_count__": sum(
            len(node_ids)
            for node_ids in executable_dispatch_nodes
        ),
        "__loop_count__": sum(
            1
            for _node_id, data in graph.G.nodes(data=True)
            if str(data.get("type")) in {"For", "AsyncFor", "While"}
        ),
        "__dispatch_count__": len(dispatch_subgraphs),
        "__deployment_batches__": node_locations,
        "DeploymentFIFO": DeploymentFIFO,
        "GLSLFusedProgramNetwork": GLSLFusedProgramNetwork,
        "compile_recorded_elementwise_tape": (
            compile_recorded_elementwise_tape
        ),
        "emit_multi_output_program_source": (
            emit_multi_output_program_source
        ),
        "_compiler_input_name": _compiler_input_name,
        "_remap_captured_program": _remap_captured_program,
        "AbstractTensor": AbstractTensor,
        "autograd": autograd,
    }
    exec(
        compile(
            class_ast,
            filename="<glsl-deployment-strategy>",
            mode="exec",
        ),
        namespace,
    )
    deployment_class = namespace["ProcessGraphGLSLDeployment"]
    deployment_class.generated_ast = class_ast
    return deployment_class


__all__ = ["strategize_glsl_deployment"]
