"""Reverse-mode differentiation of semantic :class:`ProcessGraph` objects.

The forward ``ProcessGraph`` is the authority.  This module constructs a new
``ProcessGraph`` whose nodes are ordinary, inspectable numerical operations;
it does not execute Python backward callables and it does not discover a
backward program by observing a tape traversal.

Numerical dataflow is differentiated directly as graph dataflow.  Logical
control is differentiated through the planner-owned ``ControlProgram``
adjoint, which retains predicates and loop history explicitly; a cyclic raw
``ProcessGraph`` is still rejected because silently differentiating one
observed traversal would be a false whole-program result.
"""

from __future__ import annotations

import ast
import copy
import contextlib
from functools import lru_cache
import inspect
import io
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx

from ..common.tensors.backward_registry import BACKWARD_RULES
from ..transmogrifier.graph.graph_express2 import ProcessGraph


class ProcessGraphAutogradError(ValueError):
    """The semantic forward graph cannot yet be differentiated faithfully."""


_GRAPH_ADJOINT_RULE_ALIASES = {
    "truediv": "div",
    "mm": "matmul",
    "select": "where",
    "flatten": "reshape",
    "identity": "clone",
}


def graph_adjoint_rule_name(operation: str) -> str:
    """Return the canonical graph-native backward-registry spelling."""

    name = str(operation)
    return _GRAPH_ADJOINT_RULE_ALIASES.get(name, name)


@dataclass(frozen=True)
class SavedValueContract:
    """Forward value or descriptor crossing into the backward graph."""

    forward_value_id: int
    backward_input_id: int
    shape: tuple[Any, ...]
    dtype: str
    storage: str = "resident"
    lifetime: str = "forward_through_backward"


@dataclass(frozen=True)
class GradientValueContract:
    """One accumulated adjoint result and its authored binding identity."""

    forward_value_id: int
    backward_value_id: int
    binding_kind: str
    binding_name: str
    accumulation: str = "sum"
    mutable: bool = False


@dataclass(frozen=True)
class AdjointBindingGraph:
    """Non-executable value ledger joining forward and backward graphs.

    Nodes retain the authoritative forward value id.  Their attributes state
    whether the crossing is a numerical product, structural descriptor,
    predicate, parameter, alias/version, or mutable state, and enumerate the
    exact backward call operands that consume it.  Edges preserve dependencies
    among retained forward values.  This graph caches identities and facts; it
    is not another numerical program and owns no differentiation rules.
    """

    graph: nx.DiGraph
    forward_to_backward_input: Mapping[int, int]

    def binding(self, forward_value_id: int) -> Mapping[str, Any]:
        return self.graph.nodes[int(forward_value_id)]


@dataclass(frozen=True)
class ProcessGraphAdjoint:
    """A parametric backward graph and its explicit forward-value contract."""

    forward: ProcessGraph
    backward: ProcessGraph
    output_value_ids: tuple[int, ...]
    wrt_value_ids: tuple[int, ...]
    seed_value_ids: Mapping[int, int]
    saved_value_ids: Mapping[int, int]
    gradient_value_ids: Mapping[int, int]
    saved_value_contracts: Mapping[int, SavedValueContract]
    gradient_contracts: Mapping[int, GradientValueContract]
    binding_graph: AdjointBindingGraph


@dataclass(frozen=True)
class ForwardLossBackwardMotion:
    """One semantic graph containing forward, loss, and reverse motion."""

    graph: ProcessGraph
    loss_value_ids: tuple[int, ...]
    gradient_value_ids: Mapping[int, int]
    seed_value_ids: Mapping[int, int]
    binding_graph: AdjointBindingGraph


@dataclass(frozen=True)
class ProcessGraphBackwardProduct:
    """Uniform result of an independent or combined backward request.

    ``graph`` is the executable semantic graph selected by ``packaging``.
    ``adjoint`` always preserves the independent backward graph, and
    ``binding_graph`` is always the same graph-owned forward/backward ledger,
    regardless of packaging.  A caller therefore never has to opt into the
    saved-product/predicate/state contract as a separate diagnostic feature.
    """

    graph: ProcessGraph
    packaging: str
    adjoint: ProcessGraphAdjoint
    motion: ForwardLossBackwardMotion | None
    binding_graph: AdjointBindingGraph


@dataclass(frozen=True)
class TrainingMotionSSALowering:
    """Direct ProcessGraph-to-repository-SSA lowering result."""

    module: Any
    function_name: str
    outputs: Mapping[str, int]
    shortfalls: tuple[Any, ...]


def abstract_tensor_program_to_process_graph(
    output: Any,
    *,
    bindings: Mapping[str, Any],
) -> ProcessGraph:
    """Recover semantic ProcessGraph operators from an SSA AbstractTensor run.

    ``SSATensorProgram`` is a source-producing AbstractTensor backend: it
    records calls to the repository's finite C/LLVM tensor basis without
    executing a Python numerical callback.  This adapter reverses only those
    authored call recipes (whose opcodes and output positions are explicit)
    into their canonical tensor identities.  It is not arbitrary binary/SSA
    decompilation and it never consults ``GradTape`` or a captured backward.

    ``bindings`` names every caller-owned input/parameter tensor.  Those names
    become the public ProcessGraph leaves used by the adjoint binding graph.
    """

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        C_TENSOR_OPCODE_ORDER,
    )

    tensor_value = getattr(output, "data", output)
    program = getattr(tensor_value, "program", None)
    if program is None or not hasattr(program, "function"):
        raise TypeError(
            "semantic ProcessGraph ingestion requires an SSATensorProgram output"
        )
    function = program.function
    block = function.blocks.get("entry")
    if block is None:
        raise ProcessGraphAutogradError(
            "SSA AbstractTensor program has no entry block"
        )

    graph = ProcessGraph(materialize_memory=False)
    constants: dict[int, Any] = {}
    named_values = {
        int(getattr(getattr(value, "data", value), "value").id): str(name)
        for name, value in bindings.items()
    }

    def add_node(
        value_id: int,
        op: str,
        parents: Iterable[int] = (),
        *,
        shape: Iterable[int] = (),
        dtype: str | None = "float64",
        attributes: Mapping[str, Any] | None = None,
        constant: Any = None,
    ) -> None:
        value_id = int(value_id)
        parent_ids = tuple(map(int, parents))
        parent_items = tuple(
            (parent, f"arg{index}")
            for index, parent in enumerate(parent_ids)
        )
        graph.G.add_node(
            value_id,
            op=str(op), type=str(op), label=str(op),
            parents=list(parent_items), children=[],
            attributes=dict(attributes or {}), extra_args={},
            tensor={
                "shape": tuple(map(int, shape or ())),
                "dtype": str(dtype or "float64"),
            },
            control={}, constant=copy.deepcopy(constant),
            expr_obj=None, store_id=None,
        )
        for parent, role in parent_items:
            if parent not in graph.G:
                raise ProcessGraphAutogradError(
                    f"SSA AbstractTensor value {value_id} references unknown {parent}"
                )
            graph.G.add_edge(parent, value_id, role=role)
            graph.G.nodes[parent].setdefault("children", []).append(
                (value_id, role)
            )

    for position, value in enumerate(function.args):
        value_id = int(value.id)
        name = named_values.get(value_id, f"argument_{position}")
        add_node(
            value_id, "input", shape=value.shape, dtype=value.dtype,
            attributes={
                "binding_name": name,
                "binding_kind": "parameter",
            },
        )
        graph.G.nodes[value_id]["label"] = name

    opcode_names = tuple(name.casefold() for name in C_TENSOR_OPCODE_ORDER)
    canonical_opcode = {
        "div": "truediv", "lt": "less", "le": "less_equal",
        "gt": "greater", "ge": "greater_equal", "eq": "equal",
        "ne": "not_equal",
    }
    reduction_names = {
        0: "sum", 1: "prod", 2: "min", 3: "max", 4: "any", 5: "all",
    }

    def const_value(value: Any) -> Any:
        try:
            return constants[int(value.id)]
        except KeyError as error:
            raise ProcessGraphAutogradError(
                f"SSA tensor call requires unresolved structural value {value.id}"
            ) from error

    for instruction in block.instrs:
        result = instruction.res
        if instruction.op == "Const" and result is not None:
            payload = instruction.attributes.get("values")
            if payload is None:
                payload = instruction.attributes.get("constant")
            if payload is None:
                payload = instruction.attributes.get("value")
            constants[int(result.id)] = copy.deepcopy(payload)
            add_node(
                int(result.id), "const", shape=result.shape,
                dtype=result.dtype, constant=payload,
                attributes={"value": copy.deepcopy(payload)},
            )
            continue
        if instruction.op in {"reshape", "view"} and result is not None:
            add_node(
                int(result.id),
                "reshape",
                (int(instruction.args[0].id),),
                shape=result.shape,
                dtype=result.dtype,
                attributes={
                    "shape": tuple(
                        instruction.attributes.get("shape") or result.shape
                    )
                },
            )
            continue
        if instruction.op not in {"Call", "call"} or result is None:
            continue
        callee = str(instruction.attributes.get("callee") or "")
        args = tuple(instruction.args)
        operation = None
        parents: tuple[int, ...] = ()
        attributes: dict[str, Any] = {}
        if callee == "matmul_double":
            operation, parents = "matmul", (int(args[0].id), int(args[1].id))
        elif callee == "fill_double":
            operation, parents = "const", ()
            attributes["value"] = const_value(args[1])
        elif callee in {"binary_double", "binary_scalar_double"}:
            opcode = int(const_value(args[4]))
            operation = canonical_opcode.get(
                opcode_names[opcode], opcode_names[opcode]
            )
            left, right = int(args[0].id), int(args[1].id)
            if callee == "binary_scalar_double" and int(const_value(args[5])):
                left, right = right, left
            parents = (left, right)
        elif callee == "unary_double":
            opcode = int(const_value(args[3]))
            operation = canonical_opcode.get(
                opcode_names[opcode], opcode_names[opcode]
            )
            parents = (int(args[0].id),)
        elif callee == "sum_double":
            operation, parents = "sum", (int(args[0].id),)
        elif callee == "broadcast_double":
            operation, parents = "broadcast_to", (int(args[0].id),)
            attributes["shape"] = tuple(map(int, result.shape))
        elif callee == "transpose_double":
            operation, parents = "transpose", (int(args[0].id),)
            attributes["dims"] = tuple(map(int, const_value(args[3])))
        elif callee == "reduce_dim_double":
            operation = reduction_names[int(const_value(args[5]))]
            parents = (int(args[0].id),)
            attributes["dim"] = int(const_value(args[4]))
            source_shape = tuple(graph.G.nodes[parents[0]]["tensor"]["shape"])
            attributes["keepdim"] = len(tuple(result.shape)) == len(source_shape)
        elif callee == "where_double":
            operation = "where"
            parents = tuple(int(value.id) for value in args[:3])
        elif callee == "index_select_double":
            source = int(args[0].id)
            dim = int(const_value(args[4]))
            indices = tuple(map(int, const_value(args[5])))
            selector: Any
            if not indices:
                selector = slice(0, 0, 1)
            elif len(indices) == 1:
                selector = slice(indices[0], indices[0] + 1, 1)
            else:
                step = indices[1] - indices[0]
                selector = (
                    slice(indices[0], indices[-1] + step, step)
                    if step and indices == tuple(range(
                        indices[0], indices[-1] + step, step,
                    ))
                    else indices
                )
            source_shape = tuple(
                (graph.G.nodes[source].get("tensor") or {}).get("shape") or ()
            )
            slices = [slice(None)] * len(source_shape)
            slices[dim] = selector
            operation, parents = "slice", (source,)
            attributes["slices"] = tuple(slices)
        elif callee == "index_set_double":
            operation = "index_set"
            parents = (int(args[0].id), int(args[6].id))
            attributes["idx"] = instruction.attributes.get("semantic_index")
        elif callee in {"unfold2d_double", "fold2d_double"}:
            dimensions = tuple(int(const_value(value)) for value in args[2:14])
            n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw = dimensions
            operation = "unfold2d" if callee == "unfold2d_double" else "fold2d"
            parents = (int(args[0].id),)
            attributes.update({
                "kernel_size": (kh, kw),
                "stride": (sh, sw),
                "padding": (ph, pw),
                "dilation": (dh, dw),
            })
            if operation == "fold2d":
                attributes["output_size"] = (n, c, h, w)
        if operation is None:
            raise ProcessGraphAutogradError(
                "SSA AbstractTensor semantic bridge has no exact authored "
                f"identity for {callee!r}"
            )
        if operation in {
            "add", "sub", "mul", "truediv", "pow", "mod",
            "maximum", "minimum",
        }:
            # The SSA backend materializes broadcast buffers for its finite
            # equal-shape kernel ABI.  ProcessGraph arithmetic owns the higher
            # implicit-broadcast semantics, so recover the original operands
            # and differentiate that semantic operation only once.
            parents = tuple(
                int(_parents(graph, parent)[0])
                if parent in graph.G
                and _operation(graph.G.nodes[parent]) == "broadcast_to"
                and _parents(graph, parent)
                else int(parent)
                for parent in parents
            )
        if operation == "truediv" and len(parents) == 2:
            numerator, denominator = parents
            numerator_data = graph.G.nodes.get(numerator)
            denominator_value = constants.get(denominator)
            if (
                numerator_data is not None
                and _operation(numerator_data) == "sum"
                and denominator_value is not None
                and _parents(graph, numerator)
            ):
                source = _parents(graph, numerator)[0]
                source_shape = tuple(
                    (graph.G.nodes[source].get("tensor") or {}).get("shape")
                    or ()
                )
                element_count = 1
                for extent in source_shape:
                    element_count *= int(extent)
                try:
                    denominator_scalar = float(denominator_value)
                except (TypeError, ValueError):
                    denominator_scalar = float("nan")
                if denominator_scalar == float(element_count):
                    operation, parents = "mean", (int(source),)
        add_node(
            int(result.id), operation, parents,
            shape=result.shape, dtype=result.dtype, attributes=attributes,
        )

    root_id = int(tensor_value.value.id)
    if root_id not in graph.G:
        raise ProcessGraphAutogradError(
            f"SSA AbstractTensor output {root_id} has no semantic producer"
        )
    reachable = nx.ancestors(graph.G, root_id) | {root_id}
    graph.G = graph.G.subgraph(reachable).copy()
    graph.roots = [root_id]
    graph.G.graph.update({
        "graph_kind": "abstract_tensor_semantic_forward",
        "semantic_authority": "SSATensorProgram authored call recipes",
        "python_backward_callbacks": False,
        "grad_tape_backward": False,
    })
    return graph


@dataclass(frozen=True)
class ControlProgramAdjoint:
    """Planner-owned reverse schedule for numerical adjoint regions."""

    forward: Any
    backward: Any
    forward_to_backward_regions: Mapping[int, int]


@dataclass(frozen=True)
class LoopAdjointContract:
    """Saved forward-loop cardinality required by a descending adjoint loop."""

    trip_count_value_id: int
    backward_recursion_region_id: int | None = None
    forward_carried_aliases: tuple[tuple[int, int], ...] = ()
    backward_carried_aliases: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ConditionalAdjointContract:
    """Explicit branch-local gradient identities for a carried merge."""

    forward_carried_aliases: tuple[tuple[int, int, int, int], ...]
    backward_carried_aliases: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class ProcessGraphProgramAdjoint:
    """Numeric and logical adjoints of one semantic ProcessGraph program."""

    numeric: ProcessGraphAdjoint
    control: ControlProgramAdjoint
    forward_region_nodes: Mapping[int, tuple[int, ...]]
    backward_region_nodes: Mapping[int, tuple[int, ...]]
    binding_graph: AdjointBindingGraph


@dataclass(frozen=True)
class ProcessGraphAdjointRegion:
    """One semantics-preserving numerical compartment of a program adjoint."""

    phase: str
    region_id: int
    graph: ProcessGraph
    node_ids: tuple[int, ...]
    input_value_ids: tuple[int, ...]
    output_value_ids: tuple[int, ...]


_SCHEDULED_REGION = re.compile(r"__scheduled_region_(\d+)__")


def _program_adjoint_binding_graph(
    forward: ProcessGraph,
    numeric: AdjointBindingGraph,
    control: Any,
    *,
    loop_adjoint_contracts: Mapping[int | str, LoopAdjointContract] | None,
) -> AdjointBindingGraph:
    """Add reverse-control facts to the graph-owned adjoint value ledger.

    Numerical rules retain products through ordinary backward inputs. Reverse
    control instead reads resident predicates, saved loop cardinalities, and
    carried state directly.  They belong in the same ledger even though they
    are not operands of a numerical ``BACKWARD_RULES`` call.
    """

    from .control_source import (
        CallBlock,
        ConditionalBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
    )

    graph = nx.DiGraph(copy.deepcopy(numeric.graph))
    contracts = dict(loop_adjoint_contracts or {})

    def retain(
        value_id: int,
        *,
        kind: str,
        path: str,
        role: str,
    ) -> None:
        value_id = int(value_id)
        if value_id not in forward.G:
            raise ProcessGraphAutogradError(
                f"{path} retains control value {value_id}, which is absent "
                "from the semantic ProcessGraph"
            )
        data = forward.G.nodes[value_id]
        attributes = dict(data.get("attributes") or {})
        tensor = dict(data.get("tensor") or {})
        if value_id not in graph:
            graph.add_node(
                value_id,
                forward_value_id=value_id,
                backward_input_id=None,
                kind=kind,
                storage="resident",
                shape=tuple(tensor.get("shape") or ()),
                dtype=str(tensor.get("dtype") or "float64"),
                binding_name=str(
                    attributes.get("binding_name")
                    or attributes.get("name")
                    or data.get("label")
                    or f"value_{value_id}"
                ),
                identity=None,
                version=None,
                alias_identity=attributes.get(
                    "alias_identity", data.get("store_id")
                ),
                mutable=kind == "mutable_state",
                predicate=kind == "predicate",
                parameter=False,
                backward_consumers=(),
            )
        node = graph.nodes[value_id]
        if kind in {"predicate", "loop_history"}:
            node["kind"] = kind
        node["storage"] = "resident"
        consumers = set(tuple(item) for item in node.get("control_consumers", ()))
        consumers.add((str(path), str(role)))
        node["control_consumers"] = tuple(sorted(consumers))

    def state_edge(source: int, target: int, *, path: str, role: str) -> None:
        retain(source, kind="mutable_state", path=path, role=f"{role}_source")
        retain(target, kind="mutable_state", path=path, role=f"{role}_target")
        graph.add_edge(
            int(source), int(target), relation="state_version", control_path=path,
        )

    def visit(block: Any, *, path: str) -> None:
        if isinstance(block, SequenceBlock):
            for index, child in enumerate(block.blocks):
                visit(child, path=f"{path}.sequence[{index}]")
            return
        if isinstance(block, ConditionalBlock):
            retain(
                block.predicate_value_id,
                kind="predicate",
                path=path,
                role="branch_selection",
            )
            for true_id, false_id, initial_id, merged_id in block.carried_aliases:
                for arm, value_id in (("true", true_id), ("false", false_id)):
                    state_edge(value_id, merged_id, path=path, role=f"{arm}_merge")
                if initial_id not in {true_id, false_id}:
                    state_edge(initial_id, merged_id, path=path, role="initial_merge")
                graph.add_edge(
                    int(block.predicate_value_id), int(merged_id),
                    relation="control_guard", control_path=path,
                )
            visit(block.body, path=f"{path}.body")
            if block.orelse is not None:
                visit(block.orelse, path=f"{path}.orelse")
            return
        if isinstance(block, LoopBlock):
            key: int | str = (
                int(block.recursion_region_id)
                if block.recursion_region_id is not None else path
            )
            contract = contracts.get(key)
            if not isinstance(contract, LoopAdjointContract):
                raise ProcessGraphAutogradError(
                    f"{path} requires a LoopAdjointContract keyed by {key!r}"
                )
            retain(
                contract.trip_count_value_id,
                kind="loop_history",
                path=path,
                role="reverse_trip_count",
            )
            for initial_id, updated_id in block.carried_aliases:
                state_edge(initial_id, updated_id, path=path, role="loop_carried")
            visit(block.body, path=f"{path}.body")
            return
        if isinstance(block, CallBlock):
            visit(block.callee, path=f"{path}.callee[{block.callsite_id}]")
            return
        if isinstance(block, StateMachineTick):
            for value, body in block.cases:
                visit(body, path=f"{path}.case[{value}]")
            if block.default is not None:
                visit(block.default, path=f"{path}.default")
            return
        if isinstance(block, ParallelDeployment):
            for index, lane in enumerate(block.lanes):
                visit(lane, path=f"{path}.lane[{index}]")

    visit(control.root, path="root")
    retained = set(map(int, graph))
    for left, right in forward.G.edges:
        if int(left) in retained and int(right) in retained and not graph.has_edge(left, right):
            graph.add_edge(int(left), int(right), relation="forward")
    graph.graph.update({
        "graph_kind": "program_adjoint_binding_cache",
        "schema_version": 1,
        "semantic_authority": "ProcessGraph+ControlProgram",
        "executable": False,
    })
    return AdjointBindingGraph(
        graph=nx.freeze(graph),
        forward_to_backward_input=dict(numeric.forward_to_backward_input),
    )


def differentiate_control_program(
    forward: Any,
    *,
    forward_to_backward_regions: Mapping[int, int],
    value_adjoint_ids: Mapping[int, int] | None = None,
    loop_adjoint_contracts: Mapping[int | str, LoopAdjointContract] | None = None,
    conditional_adjoint_contracts: (
        Mapping[int | str, ConditionalAdjointContract] | None
    ) = None,
) -> ControlProgramAdjoint:
    """Reverse a planner-owned acyclic region schedule without unrolling it.

    Conditions remain real conditions over the same resident predicate, and
    each selected arm runs its own reversed numerical-region schedule. Loops,
    branch-carried state, sequence effects, and publications are rejected
    until a saved-history/state-adjoint contract states how to reverse them.
    """

    from .control_source import (
        CallBlock,
        ConditionalBlock,
        ControlUniform,
        ControlProgram,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
        StatementBlock,
    )

    if not isinstance(forward, ControlProgram):
        raise TypeError("forward control must be a ControlProgram")
    region_map = {
        int(source): int(target)
        for source, target in forward_to_backward_regions.items()
    }
    adjoint_ids = {
        int(source): int(target)
        for source, target in (value_adjoint_ids or {}).items()
    }
    loop_contracts = dict(loop_adjoint_contracts or {})
    conditional_contracts = dict(conditional_adjoint_contracts or {})
    if forward.deployment_regions:
        raise ProcessGraphAutogradError(
            "ControlProgram deployment regions require adjoint lane remapping"
        )
    if (
        forward.value_aliases
        or forward.iterable_bindings
        or forward.static_iterable_bindings
        or forward.collection_bindings
        or forward.closure_iterable_bindings
        or forward.projected_iterable_bindings
    ):
        raise ProcessGraphAutogradError(
            "ControlProgram aliases/iterables require a carried-state adjoint contract"
        )

    def adjoint_binding(value_id: int) -> int:
        try:
            return adjoint_ids[int(value_id)]
        except KeyError as error:
            raise ProcessGraphAutogradError(
                f"call/control value {int(value_id)} has no adjoint identity"
            ) from error

    saved_loop_uniforms: dict[int, Any] = {}

    def reverse(block: Any, *, path: str) -> Any:
        if isinstance(block, StatementBlock):
            reversed_lines: list[str] = []
            for line in reversed(block.lines):
                match = _SCHEDULED_REGION.fullmatch(str(line))
                if match is None:
                    raise ProcessGraphAutogradError(
                        f"{path} contains untranslated control statement {line!r}; "
                        "its reverse effect is not defined"
                    )
                region = int(match.group(1))
                if region not in region_map:
                    raise ProcessGraphAutogradError(
                        f"{path} schedules region {region} without a backward region"
                    )
                reversed_lines.append(
                    f"__scheduled_region_{region_map[region]}__"
                )
            return StatementBlock(tuple(reversed_lines))
        if isinstance(block, SequenceBlock):
            return SequenceBlock(tuple(
                reverse(child, path=f"{path}.sequence[{index}]")
                for index, child in reversed(tuple(enumerate(block.blocks)))
            ))
        if isinstance(block, ConditionalBlock):
            backward_carried = ()
            if block.carried_aliases:
                key: int | str = (
                    int(block.source_node_id)
                    if block.source_node_id is not None else path
                )
                contract = conditional_contracts.get(key)
                if not isinstance(contract, ConditionalAdjointContract):
                    raise ProcessGraphAutogradError(
                        f"{path} has branch-carried values and requires a "
                        f"ConditionalAdjointContract keyed by {key!r}"
                    )
                if tuple(contract.forward_carried_aliases) != tuple(
                    block.carried_aliases
                ):
                    raise ProcessGraphAutogradError(
                        f"{path} conditional adjoint contract does not match "
                        "the authored forward carried aliases"
                    )
                backward_carried = tuple(contract.backward_carried_aliases)
                if len(backward_carried) != len(block.carried_aliases):
                    raise ProcessGraphAutogradError(
                        f"{path} conditional adjoint contract has the wrong "
                        "number of backward merges"
                    )
            return ConditionalBlock(
                predicate_value_id=int(block.predicate_value_id),
                body=reverse(block.body, path=f"{path}.body"),
                orelse=(
                    None if block.orelse is None
                    else reverse(block.orelse, path=f"{path}.orelse")
                ),
                expect_true=bool(block.expect_true),
                predicate_expression=block.predicate_expression,
                carried_aliases=backward_carried,
                source_node_id=block.source_node_id,
            )
        if isinstance(block, LoopBlock):
            if block.sequence_mutations:
                raise ProcessGraphAutogradError(
                    f"{path} mutates resident sequence state; no reverse-effect "
                    "contract is attached"
                )
            if block.parallel_iterations:
                raise ProcessGraphAutogradError(
                    f"{path} is a parallel loop; adjoint lane deployment must "
                    "be replanned from backward dependencies"
                )
            key: int | str = (
                int(block.recursion_region_id)
                if block.recursion_region_id is not None else path
            )
            contract = loop_contracts.get(key)
            if not isinstance(contract, LoopAdjointContract):
                raise ProcessGraphAutogradError(
                    f"{path} requires a LoopAdjointContract keyed by {key!r}"
                )
            if tuple(contract.forward_carried_aliases) != tuple(
                block.carried_aliases
            ):
                raise ProcessGraphAutogradError(
                    f"{path} loop adjoint contract does not match the authored "
                    "forward carried aliases"
                )
            if len(contract.backward_carried_aliases) != len(
                block.carried_aliases
            ):
                raise ProcessGraphAutogradError(
                    f"{path} loop adjoint contract has the wrong number of "
                    "backward carried aliases"
                )
            trip_count = int(contract.trip_count_value_id)
            saved_loop_uniforms.setdefault(
                trip_count,
                ControlUniform(
                    name=f"adjoint_trip_count_{trip_count}",
                    value_id=trip_count,
                    dtype="int",
                ),
            )
            reverse_start = (
                f"({block.start}) + ((value_{trip_count} - 1) * ({block.step}))"
            )
            reverse_stop = f"({block.start}) - 1"
            return LoopBlock(
                induction=block.induction,
                start=reverse_start,
                stop=reverse_stop,
                step=f"-({block.step})",
                body=reverse(block.body, path=f"{path}.body"),
                carried_aliases=tuple(contract.backward_carried_aliases),
                parallel_iterations=False,
                dispatch_shell=block.dispatch_shell,
                recursion_region_id=contract.backward_recursion_region_id,
                schedule_preference=block.schedule_preference,
                sequence_mutations=(),
                comparison="gt",
            )
        if isinstance(block, CallBlock):
            arguments = tuple(
                (adjoint_binding(caller_result), adjoint_binding(callee_result))
                for callee_result, caller_result in block.result_bindings
            )
            results = tuple(
                (adjoint_binding(callee_argument), adjoint_binding(caller_argument))
                for caller_argument, callee_argument in block.argument_bindings
            )
            return CallBlock(
                callsite_id=int(block.callsite_id),
                callee=reverse(block.callee, path=f"{path}.callee"),
                argument_bindings=arguments,
                result_bindings=results,
            )
        if isinstance(block, StateMachineTick):
            return StateMachineTick(
                state=block.state,
                cases=tuple(
                    (value, reverse(body, path=f"{path}.case[{value}]"))
                    for value, body in block.cases
                ),
                default=(
                    None if block.default is None
                    else reverse(block.default, path=f"{path}.default")
                ),
            )
        if isinstance(block, ParallelDeployment):
            return ParallelDeployment(
                lanes=tuple(
                    reverse(lane, path=f"{path}.lane[{index}]")
                    for index, lane in enumerate(block.lanes)
                ),
                schedule_preference=block.schedule_preference,
            )
        raise ProcessGraphAutogradError(
            f"{path} uses {type(block).__name__}; its saved-history/state "
            "adjoint contract is not implemented"
        )

    root = reverse(forward.root, path="root")
    backward = ControlProgram(
        root=root,
        region_indices=tuple(
            region_map[int(region)]
            for region in reversed(forward.region_indices)
        ),
        uniforms=tuple((*forward.uniforms, *saved_loop_uniforms.values())),
    )
    return ControlProgramAdjoint(
        forward=forward,
        backward=backward,
        forward_to_backward_regions=region_map,
    )


def differentiate_process_program(
    forward: ProcessGraph,
    control: Any,
    *,
    region_nodes: Mapping[int, Iterable[int]],
    outputs: Iterable[int] | None = None,
    wrt: Iterable[int] | None = None,
    loop_adjoint_contracts: Mapping[int | str, LoopAdjointContract] | None = None,
    conditional_adjoint_contracts: (
        Mapping[int | str, ConditionalAdjointContract] | None
    ) = None,
) -> ProcessGraphProgramAdjoint:
    """Differentiate numeric dataflow and its planner-owned region schedule."""

    normalized_regions = {
        int(region): tuple(map(int, nodes))
        for region, nodes in region_nodes.items()
    }
    owner: dict[int, int] = {}
    for region, nodes in normalized_regions.items():
        for node in nodes:
            if node not in forward.G:
                raise ProcessGraphAutogradError(
                    f"control region {region} names unknown forward node {node}"
                )
            if node in owner:
                raise ProcessGraphAutogradError(
                    f"forward node {node} belongs to regions {owner[node]} and {region}"
                )
            owner[node] = region
    missing = set(forward.G) - set(owner)
    if missing:
        raise ProcessGraphAutogradError(
            "forward nodes lack a control region: "
            + ", ".join(map(str, sorted(missing)))
        )

    numeric = differentiate_process_graph(forward, outputs=outputs, wrt=wrt)
    next_region = max(normalized_regions, default=-1) + 1
    region_map = {
        region: next_region + index
        for index, region in enumerate(sorted(normalized_regions))
    }
    backward_regions: dict[int, list[int]] = {
        target: [] for target in region_map.values()
    }
    for node_id, data in numeric.backward.G.nodes(data=True):
        source = (data.get("attributes") or {}).get("source_forward_id")
        if source is None or int(source) not in owner:
            raise ProcessGraphAutogradError(
                f"backward node {node_id} has no controlled forward identity"
            )
        backward_regions[region_map[owner[int(source)]]].append(int(node_id))

    control_adjoint = differentiate_control_program(
        control,
        forward_to_backward_regions=region_map,
        value_adjoint_ids={
            **{int(k): int(v) for k, v in numeric.gradient_value_ids.items()},
            **{int(k): int(v) for k, v in numeric.seed_value_ids.items()},
        },
        loop_adjoint_contracts=loop_adjoint_contracts,
        conditional_adjoint_contracts=conditional_adjoint_contracts,
    )
    binding_graph = _program_adjoint_binding_graph(
        forward,
        numeric.binding_graph,
        control,
        loop_adjoint_contracts=loop_adjoint_contracts,
    )
    return ProcessGraphProgramAdjoint(
        numeric=numeric,
        control=control_adjoint,
        forward_region_nodes=normalized_regions,
        backward_region_nodes={
            region: tuple(nodes) for region, nodes in backward_regions.items()
        },
        binding_graph=binding_graph,
    )


def _isolate_process_graph_region(
    graph: ProcessGraph,
    *,
    phase: str,
    region_id: int,
    node_ids: Iterable[int],
    required_outputs: Iterable[int] = (),
) -> ProcessGraphAdjointRegion:
    """Copy a numerical region with explicit, identity-preserving boundaries."""

    selected = {int(node_id) for node_id in node_ids}
    unknown = selected - set(graph.G)
    if unknown:
        raise ProcessGraphAutogradError(
            f"{phase} region {region_id} contains unknown nodes "
            + ", ".join(map(str, sorted(unknown)))
        )
    if not selected:
        isolated = copy.copy(graph)
        isolated.G = nx.DiGraph()
        isolated.G.graph.update({
            "graph_kind": "process_graph_adjoint_region",
            "semantic_authority": "ProcessGraph",
            "region_phase": str(phase),
            "region_id": int(region_id),
            "region_nodes": (),
            "region_inputs": (),
            "region_outputs": (),
            "source_graph_preserved": True,
            "fused_program_semantic_authority": False,
        })
        isolated.roots = []
        isolated.levels = {}
        isolated.scheduler = copy.copy(graph.scheduler)
        isolated.scheduler.G = isolated.G
        return ProcessGraphAdjointRegion(
            phase=str(phase), region_id=int(region_id), graph=isolated,
            node_ids=(), input_value_ids=(), output_value_ids=(),
        )

    def semantic_parents(node_id: int) -> set[int]:
        return {
            *map(int, graph.G.predecessors(node_id)),
            *map(int, _parents(graph, node_id)),
        }

    def semantic_children(node_id: int) -> set[int]:
        return {
            *map(int, graph.G.successors(node_id)),
            *(
                int(child)
                for child, _role in graph.G.nodes[node_id].get("children", ())
                if int(child) in graph.G
            ),
        }

    boundary = {
        parent
        for node_id in selected
        for parent in semantic_parents(node_id)
        if parent not in selected
    }
    included = selected | boundary
    isolated = copy.copy(graph)
    isolated.G = copy.deepcopy(graph.G.subgraph(included).copy())
    isolated.G.graph = copy.deepcopy(dict(graph.G.graph))
    isolated.scheduler = copy.copy(graph.scheduler)
    isolated.scheduler.G = isolated.G

    inputs: list[int] = sorted(
        node_id for node_id in selected
        if _operation(isolated.G.nodes[node_id]) == "input"
    )
    for node_id in sorted(boundary):
        data = isolated.G.nodes[node_id]
        if _operation(data) == "const":
            continue
        attributes = dict(data.get("attributes") or {})
        attributes.update({
            "binding_kind": "region_input",
            "source_value_id": int(node_id),
            "region_phase": str(phase),
            "region_id": int(region_id),
        })
        data.update({
            "type": "input",
            "op": "input",
            "label": str(data.get("label") or f"value_{node_id}"),
            "parents": [],
            "attributes": attributes,
            "extra_args": copy.deepcopy(attributes),
        })
        for parent in tuple(isolated.G.predecessors(node_id)):
            isolated.G.remove_edge(parent, node_id)
        inputs.append(int(node_id))
    inputs = sorted(set(inputs))

    required = set(map(int, required_outputs))
    outputs = tuple(sorted(
        node_id
        for node_id in selected
        if node_id in required
        or not semantic_children(node_id)
        or any(child not in selected for child in semantic_children(node_id))
    ))
    isolated.roots = list(outputs)
    isolated.levels = {
        int(node_id): int(level)
        for level, generation in enumerate(nx.topological_generations(isolated.G))
        for node_id in generation
    }
    isolated.G.graph.update({
        "graph_kind": "process_graph_adjoint_region",
        "semantic_authority": "ProcessGraph",
        "region_phase": str(phase),
        "region_id": int(region_id),
        "region_nodes": tuple(sorted(selected)),
        "region_inputs": tuple(inputs),
        "region_outputs": outputs,
        "source_graph_preserved": True,
        "fused_program_semantic_authority": False,
    })
    return ProcessGraphAdjointRegion(
        phase=str(phase),
        region_id=int(region_id),
        graph=isolated,
        node_ids=tuple(sorted(selected)),
        input_value_ids=tuple(inputs),
        output_value_ids=outputs,
    )


def isolate_process_program_adjoint_regions(
    program: ProcessGraphProgramAdjoint,
) -> tuple[ProcessGraphAdjointRegion, ...]:
    """Materialize both directions as explicit ProcessGraph compartments.

    Boundary nodes retain the source value id and become ordinary region
    inputs only in the copied compartment. The authoritative forward and
    backward graphs are never rewritten by isolation.
    """

    regions: list[ProcessGraphAdjointRegion] = []
    forward_outputs = set(map(int, program.numeric.output_value_ids))
    backward_outputs = set(map(int, program.numeric.backward.roots))
    for region_id, node_ids in sorted(program.forward_region_nodes.items()):
        regions.append(_isolate_process_graph_region(
            program.numeric.forward,
            phase="forward",
            region_id=int(region_id),
            node_ids=node_ids,
            required_outputs=forward_outputs,
        ))
    for region_id, node_ids in sorted(program.backward_region_nodes.items()):
        regions.append(_isolate_process_graph_region(
            program.numeric.backward,
            phase="backward",
            region_id=int(region_id),
            node_ids=node_ids,
            required_outputs=backward_outputs,
        ))
    return tuple(regions)


def _operation(data: Mapping[str, Any]) -> str:
    return str(data.get("op") or data.get("type") or "").casefold()


def _parents(graph: ProcessGraph, node_id: int) -> tuple[int, ...]:
    data = graph.G.nodes[node_id]
    declared = tuple(int(parent) for parent, _role in data.get("parents", ()))
    if declared:
        return declared
    return tuple(int(parent) for parent in graph.G.predecessors(node_id))


def _returned_value(graph: ProcessGraph, node_id: int) -> int:
    if _operation(graph.G.nodes[node_id]) != "return":
        return int(node_id)
    parents = _parents(graph, node_id)
    if len(parents) != 1:
        raise ProcessGraphAutogradError(
            f"return node {node_id} has {len(parents)} values; "
            "name each differentiated output explicitly"
        )
    return parents[0]


def _registry_function_source(opname: str, rule: Mapping[str, Any]) -> str:
    python_rule = dict(rule.get("python") or {})
    parameters = ", ".join(map(str, python_rule.get("parameters") or ()))
    statements = tuple(
        statement.strip()
        for statement in str(python_rule.get("body") or "").split(";")
        if statement.strip()
    )
    if not parameters or not statements:
        raise ProcessGraphAutogradError(
            f"BACKWARD_RULES[{opname!r}] has no complete Python definition"
        )
    return "\n".join((
        f"def bw_{opname}({parameters}):",
        *(f"    {statement}" for statement in statements),
        "",
    ))


@lru_cache(maxsize=1)
def _compiled_backward_rule_process_graph() -> ProcessGraph:
    """Compile the authored backward registry into one graph function table.

    This is source-to-ProcessGraph translation of the repository's existing
    mappings.  No rule is executed, traced, or replaced with a second
    derivative formula.
    """

    from ..common.tensors import backward_registry as registry
    from ..common.tensors.abstraction_methods.indexing import (
        NormalizedIndexAxis,
        flat_index_ids,
        normalize_index,
    )
    from ..common.tensors.topological_reducer import (
        UNTRANSLATED_NODE_TYPE,
        reduce_abstract_tensor_topology,
    )

    helpers = (
        registry.unbroadcast,
        registry.expand_to,
        registry.expand_reduction,
        registry.indicator,
        registry.reverse_cumsum,
        registry.index_adjoint,
        registry.I_like,
        registry.eps,
        registry.T,
        registry.matmul_vjp,
        NormalizedIndexAxis,
        normalize_index,
        flat_index_ids,
    )
    source = "\n\n".join((
        *(textwrap.dedent(inspect.getsource(helper)) for helper in helpers),
        *(
            _registry_function_source(opname, rule)
            for opname, rule in BACKWARD_RULES.items()
        ),
    ))
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {
        **vars(registry),
        "NormalizedIndexAxis": NormalizedIndexAxis,
        "normalize_index": normalize_index,
        "flat_index_ids": flat_index_ids,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(source, filename="<abstract-tensor-backward-rules>"),
            resolve_unresolved_parents=False,
        )
    reduce_abstract_tensor_topology(graph)
    missing = tuple(
        opname
        for opname in BACKWARD_RULES
        if graph.function_table.reference(f"bw_{opname}") is None
    )
    untranslated = tuple(
        (entry.qualified_name, int(node_id))
        for entry in graph.function_table
        if entry.graph is not None
        for node_id, data in entry.graph.G.nodes(data=True)
        if data.get("type") == UNTRANSLATED_NODE_TYPE
    )
    if missing or untranslated:
        raise ProcessGraphAutogradError(
            "backward registry did not compile completely to ProcessGraph: "
            f"missing={missing!r}, untranslated={untranslated!r}"
        )
    graph.G.graph.update({
        "semantic_authority": "BACKWARD_RULES->ProcessGraph",
        "backward_rule_count": len(BACKWARD_RULES),
        "python_backward_callbacks": False,
    })
    return graph


def _parameter_spec(specification: str) -> tuple[str, bool, Any]:
    text = str(specification).strip()
    variadic = text.startswith("*")
    text = text.lstrip("*")
    name, separator, default_source = text.partition("=")
    default = ...
    if separator:
        default = ast.literal_eval(default_source)
    return name.strip(), variadic, default


def _metadata_argument(
    name: str,
    attributes: Mapping[str, Any],
) -> tuple[bool, Any]:
    aliases = {
        "axis": ("axis", "dim"),
        "dim": ("dim", "axis"),
        "idx": ("idx", "index", "slices"),
        "index": ("index", "idx", "slices"),
        "slices": ("slices", "index", "idx"),
        "new_shape": ("new_shape", "shape"),
        "perm": ("perm", "order"),
    }
    for candidate in aliases.get(name, (name,)):
        if candidate in attributes:
            return True, copy.deepcopy(attributes[candidate])
    return False, None


def _retain_backward_rule_closure(
    graph: ProcessGraph,
    rule_names: Iterable[str],
) -> None:
    """Keep only graph functions reachable from the selected registry rules."""

    table = graph.function_table
    pending = [
        int(table.entry(f"bw_{name}").reference.address)
        for name in dict.fromkeys(map(str, rule_names))
    ]
    retained: set[int] = set()
    while pending:
        reference = int(pending.pop())
        if reference in retained:
            continue
        retained.add(reference)
        entry = table.entry(reference)
        if entry.graph is None:
            continue
        for _node_id, data in entry.graph.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            for key in ("callee_ref", "method_ref", "constructor_ref"):
                target = attributes.get(key)
                if target is not None and int(target) not in retained:
                    pending.append(int(target))
    table._entries = {
        reference: entry
        for reference, entry in table._entries.items()
        if int(reference.address) in retained
    }
    table._bindings = {
        name: reference
        for name, reference in table._bindings.items()
        if int(reference.address) in retained
    }
    table._qualified = {
        name: reference
        for name, reference in table._qualified.items()
        if int(reference.address) in retained
    }
    for entry in table:
        if entry.graph is not None:
            entry.graph.function_table = table
    graph.G.graph["backward_function_references"] = tuple(sorted(retained))


class _AdjointBindingGraphBuilder:
    """Build the deterministic forward-value cache consumed by an adjoint."""

    _PREDICATE_OPERATIONS = frozenset({
        "eq", "equal", "ne", "not_equal", "lt", "less", "le",
        "less_equal", "gt", "greater", "ge", "greater_equal",
        "logical_and", "logical_or", "logical_not", "boolop",
    })

    def __init__(self, forward: ProcessGraph) -> None:
        self.forward = forward
        self.graph = nx.DiGraph()
        self.backward_inputs: dict[int, int] = {}
        self.consumers: dict[int, set[tuple[int, str]]] = {}

    def _identity(self, forward_id: int) -> tuple[str | None, int | None]:
        identities = self.forward.G.graph.get("identity_table") or {}
        for name in sorted(map(str, identities)):
            history = tuple(map(int, identities[name]))
            if int(forward_id) in history:
                return name, history.index(int(forward_id))
        return None, None

    def retain(
        self,
        forward_id: int,
        backward_input_id: int,
        *,
        storage: str,
    ) -> None:
        forward_id = int(forward_id)
        backward_input_id = int(backward_input_id)
        data = self.forward.G.nodes[forward_id]
        attributes = dict(data.get("attributes") or {})
        operation = _operation(data)
        identity, version = self._identity(forward_id)
        binding_kind = str(attributes.get("binding_kind") or "")
        mutable = bool(
            attributes.get("mutable")
            or attributes.get("mutable_state")
            or binding_kind in {"mutable", "mutable_state", "state"}
        )
        predicate = operation in self._PREDICATE_OPERATIONS or bool(
            attributes.get("predicate")
            or attributes.get("control_predicate")
        )
        parameter = operation == "input" and binding_kind not in {
            "gradient_seed", "saved_forward",
        }
        kind = (
            "predicate" if predicate else
            "mutable_state" if mutable else
            "parameter" if parameter else
            "descriptor" if storage == "descriptor" else
            "product"
        )
        tensor = dict(data.get("tensor") or {})
        existing = self.graph.nodes.get(forward_id, {})
        effective_storage = (
            "resident"
            if storage == "resident" or existing.get("storage") == "resident"
            else "descriptor"
        )
        self.graph.add_node(
            forward_id,
            forward_value_id=forward_id,
            backward_input_id=backward_input_id,
            kind=kind,
            storage=effective_storage,
            shape=tuple(tensor.get("shape") or ()),
            dtype=str(tensor.get("dtype") or "float64"),
            binding_name=str(
                attributes.get("binding_name")
                or identity
                or data.get("label")
                or f"value_{forward_id}"
            ),
            identity=identity,
            version=version,
            alias_identity=attributes.get(
                "alias_identity", data.get("store_id")
            ),
            mutable=mutable,
            predicate=predicate,
            parameter=parameter,
        )
        prior = self.backward_inputs.setdefault(forward_id, backward_input_id)
        if prior != backward_input_id:
            raise ProcessGraphAutogradError(
                f"forward value {forward_id} acquired two backward inputs: "
                f"{prior} and {backward_input_id}"
            )

    def consume(
        self, forward_id: int, backward_node_id: int, role: str,
    ) -> None:
        forward_id = int(forward_id)
        if forward_id not in self.graph:
            raise ProcessGraphAutogradError(
                f"backward node {backward_node_id} consumes uncached forward "
                f"value {forward_id}"
            )
        self.consumers.setdefault(forward_id, set()).add((
            int(backward_node_id), str(role),
        ))

    def finish(self) -> AdjointBindingGraph:
        retained = set(map(int, self.graph))
        for left, right in self.forward.G.edges:
            if int(left) in retained and int(right) in retained:
                self.graph.add_edge(int(left), int(right), relation="forward")
        for forward_id in sorted(retained):
            self.graph.nodes[forward_id]["backward_consumers"] = tuple(
                sorted(self.consumers.get(forward_id, ()))
            )
        self.graph.graph.update({
            "graph_kind": "adjoint_binding_cache",
            "schema_version": 1,
            "semantic_authority": "ProcessGraph",
            "executable": False,
        })
        frozen = nx.freeze(copy.deepcopy(self.graph))
        return AdjointBindingGraph(
            graph=frozen,
            forward_to_backward_input=dict(sorted(self.backward_inputs.items())),
        )


class _AdjointBuilder:
    def __init__(self, forward: ProcessGraph) -> None:
        self.forward = forward
        registry_graph = _compiled_backward_rule_process_graph()
        self.backward = ProcessGraph(
            materialize_memory=False,
            function_table=copy.deepcopy(registry_graph.function_table),
            external_function_table=copy.deepcopy(
                registry_graph.external_function_table
            ),
        )
        self.next_id = 0
        self.saved: dict[int, int] = {}
        self.saved_storage: dict[int, str] = {}
        self.bindings = _AdjointBindingGraphBuilder(forward)

    def add(
        self,
        op: str,
        parents: Iterable[tuple[int, str]] = (),
        *,
        label: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        source_forward_id: int | None = None,
    ) -> int:
        node_id = self.next_id
        self.next_id += 1
        parent_items = tuple((int(parent), str(role)) for parent, role in parents)
        attrs = dict(attributes or {})
        if source_forward_id is not None:
            attrs.setdefault("source_forward_id", int(source_forward_id))
        self.backward.G.add_node(
            node_id,
            label=label or op,
            type=op,
            op=op,
            parents=list(parent_items),
            children=[],
            attributes=attrs,
            extra_args=copy.deepcopy(attrs),
            tensor={},
            control={},
            constant=None,
            expr_obj=None,
            store_id=None,
            schema_version=1,
        )
        for parent, role in parent_items:
            self.backward.G.add_edge(parent, node_id, role=role)
            self.backward.G.nodes[parent]["children"].append((node_id, role))
        return node_id

    def input(
        self, name: str, *, kind: str, source_forward_id: int,
    ) -> int:
        node_id = self.add(
            "input",
            label=name,
            attributes={
                "name": name,
                "binding_kind": kind,
                "source_forward_id": int(source_forward_id),
            },
            source_forward_id=source_forward_id,
        )
        self.backward.G.nodes[node_id]["tensor"] = copy.deepcopy(
            self.forward.G.nodes[source_forward_id].get("tensor") or {}
        )
        return node_id

    def saved_value(
        self, forward_id: int, *, storage: str = "resident",
    ) -> int:
        forward_id = int(forward_id)
        if forward_id not in self.saved:
            self.saved[forward_id] = self.input(
                f"saved_{forward_id}",
                kind="saved_forward",
                source_forward_id=forward_id,
            )
            self.saved_storage[forward_id] = str(storage)
        elif storage == "resident":
            self.saved_storage[forward_id] = "resident"
        self.bindings.retain(
            forward_id,
            self.saved[forward_id],
            storage=self.saved_storage[forward_id],
        )
        return self.saved[forward_id]

    def constant(self, value: Any, source: int) -> int:
        node_id = self.add(
            "const",
            label=repr(value),
            attributes={"values": copy.deepcopy(value)},
            source_forward_id=source,
        )
        self.backward.G.nodes[node_id]["constant"] = copy.deepcopy(value)
        return node_id

    def registry_rule(
        self,
        opname: str,
        gradient: int,
        forward_id: int,
        parents: tuple[int, ...],
        attributes: Mapping[str, Any],
    ) -> tuple[tuple[int, int], ...]:
        """Instantiate one existing backward mapping as a graph-backed call."""

        rule = BACKWARD_RULES[opname]
        python_rule = dict(rule.get("python") or {})
        parameter_specs = tuple(map(
            str, python_rule.get("parameters") or (),
        ))
        if not parameter_specs or _parameter_spec(parameter_specs[0])[0] != "g":
            raise ProcessGraphAutogradError(
                f"BACKWARD_RULES[{opname!r}] does not begin with gradient g"
            )
        body = str(python_rule.get("body") or "")
        arguments: list[int] = [int(gradient)]
        argument_names: list[str] = ["g"]
        argument_forward_sources: list[int | None] = [None]
        parent_names: list[str] = []
        parent_cursor = 0
        for specification in parameter_specs[1:]:
            name, variadic, default = _parameter_spec(specification)
            if variadic:
                while parent_cursor < len(parents):
                    parent = int(parents[parent_cursor])
                    arguments.append(self.saved_value(parent))
                    argument_names.append(name)
                    argument_forward_sources.append(parent)
                    parent_names.append(name)
                    parent_cursor += 1
                continue
            has_metadata, metadata = _metadata_argument(name, attributes)
            if has_metadata:
                arguments.append(self.constant(metadata, forward_id))
                argument_names.append(name)
                argument_forward_sources.append(None)
                continue
            if parent_cursor < len(parents):
                parent = int(parents[parent_cursor])
                descriptor_only = not bool(re.search(
                    rf"\b{re.escape(name)}\b(?!\.(?:shape|ndim|ndims|numel|dtype|device)\b)",
                    body,
                ))
                arguments.append(self.saved_value(
                    parent,
                    storage="descriptor" if descriptor_only else "resident",
                ))
                argument_names.append(name)
                argument_forward_sources.append(parent)
                parent_names.append(name)
                parent_cursor += 1
                continue
            if name == "y":
                arguments.append(self.saved_value(forward_id))
                argument_names.append(name)
                argument_forward_sources.append(int(forward_id))
                continue
            if default is not ...:
                arguments.append(self.constant(default, forward_id))
                argument_names.append(name)
                argument_forward_sources.append(None)
                continue
            raise ProcessGraphAutogradError(
                f"BACKWARD_RULES[{opname!r}] parameter {name!r} has no "
                f"ProcessGraph operand or operator metadata at node {forward_id}"
            )
        if parent_cursor != len(parents):
            raise ProcessGraphAutogradError(
                f"BACKWARD_RULES[{opname!r}] bound {parent_cursor} of "
                f"{len(parents)} forward operands at node {forward_id}"
            )
        entry = self.backward.function_table.entry(f"bw_{opname}")
        call = self.add(
            "Call",
            tuple(
                (argument, f"arg:{index}")
                for index, argument in enumerate(arguments)
            ),
            label=f"bw_{opname}",
            attributes={
                "callee_ref": int(entry.reference.address),
                "backward_rule": opname,
                "argument_names": tuple(argument_names),
            },
            source_forward_id=forward_id,
        )
        for index, source in enumerate(argument_forward_sources):
            if source is not None:
                self.bindings.consume(source, call, f"arg:{index}")
        differentiable = set(map(str, (rule.get("backward") or {}).keys()))
        variadic_differentiable = any(name.endswith("*") for name in differentiable)
        results: list[tuple[int, int]] = []
        multiple_results = len(parents) != 1 or variadic_differentiable
        for index, (parent, name) in enumerate(zip(parents, parent_names)):
            if name not in differentiable and not (
                variadic_differentiable
                and any(key.rstrip("*") == name for key in differentiable)
            ):
                continue
            value = call
            if multiple_results:
                index_value = self.constant(index, forward_id)
                value = self.add(
                    "Indexed",
                    ((call, "base"), (index_value, "index")),
                    attributes={"gradient_result_index": index},
                    source_forward_id=forward_id,
                )
            self.backward.G.nodes[value]["tensor"] = copy.deepcopy(
                self.forward.G.nodes[int(parent)].get("tensor") or {}
            )
            results.append((int(parent), int(value)))
        return tuple(results)

    def unary(self, op: str, value: int, source: int) -> int:
        return self.add(
            op, ((value, "operand"),), source_forward_id=source,
        )

    def binary(self, op: str, left: int, right: int, source: int) -> int:
        return self.add(
            op,
            ((left, "lhs"), (right, "rhs")),
            source_forward_id=source,
        )

    def reduce_to_shape(
        self,
        gradient: int,
        forward_id: int,
        source: int,
        *,
        gradient_shape: tuple[int, ...] | None = None,
    ) -> int:
        """Expand the registry's ``unbroadcast`` helper into existing ops."""

        source_shape = tuple(gradient_shape or (
            (self.forward.G.nodes[source].get("tensor") or {}).get("shape") or ()
        ))
        target_shape = tuple(
            (self.forward.G.nodes[forward_id].get("tensor") or {}).get("shape") or ()
        )
        if source_shape == target_shape:
            return gradient
        if not source_shape or not target_shape:
            raise ProcessGraphAutogradError(
                "canonical backward helper unbroadcast requires symbolic-shape "
                "ProcessGraph lowering before this region can dispatch natively; "
                f"source={source} shape={source_shape}, target={forward_id} "
                f"shape={target_shape}"
            )
        current = gradient
        current_shape = list(source_shape)
        if len(current_shape) > len(target_shape):
            for _ in range(len(current_shape) - len(target_shape)):
                current = self.add(
                    "sum",
                    ((current, "operand"),),
                    attributes={"dim": 0, "keepdim": False},
                    source_forward_id=source,
                )
                current_shape.pop(0)
        reduce_axes = tuple(
            axis
            for axis, (actual, target) in enumerate(
                zip(current_shape, target_shape)
            )
            if target == 1 and actual != 1
        )
        for axis in reduce_axes:
            current = self.add(
                "sum",
                ((current, "operand"),),
                attributes={"dim": int(axis), "keepdim": True},
                source_forward_id=source,
            )
        return self.add(
            "reshape",
            ((current, "operand"),),
            attributes={"shape": target_shape},
            source_forward_id=source,
        )

    def expand_reduction(
        self,
        gradient: int,
        forward_id: int,
        source: int,
        attributes: Mapping[str, Any],
    ) -> int:
        """Expand the registry's ``expand_reduction`` using canonical ops."""

        source_shape = tuple(
            (self.forward.G.nodes[source].get("tensor") or {}).get("shape") or ()
        )
        target_shape = tuple(
            (self.forward.G.nodes[forward_id].get("tensor") or {}).get("shape") or ()
        )
        if not target_shape:
            raise ProcessGraphAutogradError(
                "canonical backward helper expand_reduction requires a known "
                f"target shape for forward value {forward_id}"
            )
        axis = attributes.get("axis", attributes.get("dim"))
        axes = (
            tuple(range(len(target_shape))) if axis is None
            else tuple(int(item) for item in axis)
            if isinstance(axis, (tuple, list))
            else (int(axis),)
        )
        axes = tuple(sorted(item % len(target_shape) for item in axes))
        current = gradient
        if not bool(attributes.get("keepdim", False)):
            restored = list(source_shape)
            for item in axes:
                restored.insert(item, 1)
            current = self.add(
                "reshape",
                ((current, "operand"),),
                attributes={"shape": tuple(restored)},
                source_forward_id=source,
            )
        reference = self.saved_value(forward_id)
        zero = self.binary("sub", reference, reference, source)
        return self.binary(
            "add",
            zero,
            current,
            source,
        )


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    for dimensions in zip(*(
        (1,) * (max(map(len, shapes), default=0) - len(shape)) + tuple(shape)
        for shape in shapes
    )):
        concrete = {int(value) for value in dimensions if int(value) != 1}
        if len(concrete) > 1:
            raise ProcessGraphAutogradError(
                f"incompatible tensor broadcast shapes: {shapes!r}"
            )
        result.append(next(iter(concrete), 1))
    return tuple(result)


def _annotate_numeric_metadata(graph: ProcessGraph) -> None:
    """Propagate AbstractTensor shape/dtype facts through generated nodes."""

    for node_id in nx.topological_sort(graph.G):
        data = graph.G.nodes[node_id]
        tensor = dict(data.get("tensor") or {})
        parents = _parents(graph, int(node_id))
        parent_tensors = [graph.G.nodes[parent].get("tensor") or {} for parent in parents]
        parent_shapes = [tuple(item.get("shape") or ()) for item in parent_tensors]
        dtype = tensor.get("dtype") or next(
            (item.get("dtype") for item in parent_tensors if item.get("dtype")),
            "float64",
        )
        op = _operation(data)
        shape = tuple(tensor.get("shape") or ())
        attrs = data.get("attributes") or {}
        if op == "const":
            shape = tuple(tensor.get("shape") or ())
        elif op == "call":
            # A graph-backed backward rule may return one tensor or a
            # structural tuple of tensors. Projection nodes below carry the
            # concrete per-parent tensor descriptors.
            shape = tuple(tensor.get("shape") or ())
        elif op == "indexed" and shape:
            shape = tuple(shape)
        elif op == "slice" and shape:
            # Forward observation already records the exact indexed result;
            # broadcasting source and result shapes would erase the slice.
            shape = tuple(shape)
        elif op == "index_set" and parent_shapes:
            # Functional indexed assignment preserves the base tensor shape;
            # the assigned value has the selection shape and is not a
            # broadcast peer of the complete base.
            shape = parent_shapes[0]
        elif op == "unfold2d" and parent_shapes:
            n, c, h, w = parent_shapes[0]
            kh, kw = tuple(attrs["kernel_size"])
            sh, sw = tuple(attrs.get("stride", (1, 1)))
            ph, pw = tuple(attrs.get("padding", (0, 0)))
            dh, dw = tuple(attrs.get("dilation", (1, 1)))
            output_h = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
            output_w = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
            shape = (n, c * kh * kw, output_h * output_w)
        elif op == "fold2d":
            shape = tuple(attrs["output_size"])
        elif op == "reshape":
            shape = tuple(attrs.get("shape") or ())
        elif op in {"transpose", "swapaxes", "permute"} and parent_shapes:
            shape = parent_shapes[0]
            if len(shape) >= 2:
                shape = (*shape[:-2], shape[-1], shape[-2])
        elif op in {"matmul", "mm"} and len(parent_shapes) == 2:
            left, right = parent_shapes
            if len(left) < 2 or len(right) < 2:
                raise ProcessGraphAutogradError(
                    f"matmul node {node_id} requires rank-two tensor metadata"
                )
            batch = _broadcast_shape(left[:-2], right[:-2])
            shape = (*batch, left[-2], right[-1])
        elif op in {"sum", "mean"} and parent_shapes:
            source_shape = parent_shapes[0]
            axis = attrs.get("axis", attrs.get("dim"))
            axes = (
                tuple(range(len(source_shape))) if axis is None
                else tuple(int(item) % len(source_shape) for item in axis)
                if isinstance(axis, (tuple, list))
                else (int(axis) % len(source_shape),)
            )
            if bool(attrs.get("keepdim", False)):
                shape = tuple(
                    1 if index in axes else extent
                    for index, extent in enumerate(source_shape)
                )
            else:
                shape = tuple(
                    extent for index, extent in enumerate(source_shape)
                    if index not in axes
                )
        elif parent_shapes and op not in {"input", "return"}:
            shape = _broadcast_shape(*parent_shapes)
        data["tensor"] = {
            **tensor,
            "shape": tuple(shape),
            "dtype": dtype,
        }


def differentiate_process_graph(
    forward: ProcessGraph,
    *,
    outputs: Iterable[int] | None = None,
    wrt: Iterable[int] | None = None,
) -> ProcessGraphAdjoint:
    """Construct a first-class parametric backward ``ProcessGraph``.

    ``outputs`` names differentiated forward values (``return`` nodes are
    resolved to their value).  Each output receives an explicit upstream
    gradient input. ``wrt`` defaults to every forward ``input`` node.

    The pass is fail-closed.  It accepts no cycles and no reachable operation
    without a graph-native rule.  This prevents an observed branch, a Python
    function object, or a missing derivative from masquerading as compiled
    logical backward code.
    """

    if not nx.is_directed_acyclic_graph(forward.G):
        raise ProcessGraphAutogradError(
            "logical/cyclic ProcessGraph differentiation requires the "
            "planner-owned ControlProgram adjoint; refusing tape-like unrolling"
        )

    raw_outputs = tuple(outputs or getattr(forward, "roots", ()) or ())
    if not raw_outputs:
        raw_outputs = tuple(
            int(node_id) for node_id in forward.G
            if forward.G.out_degree(node_id) == 0
        )
    output_ids = tuple(_returned_value(forward, int(node)) for node in raw_outputs)
    if not output_ids:
        raise ProcessGraphAutogradError("forward graph has no differentiated output")

    wrt_ids = tuple(
        int(node_id) for node_id in (
            wrt if wrt is not None else (
                node for node, data in forward.G.nodes(data=True)
                if _operation(data) == "input"
            )
        )
    )
    unknown = (set(output_ids) | set(wrt_ids)) - set(forward.G)
    if unknown:
        raise ProcessGraphAutogradError(
            "unknown ProcessGraph value ids: " + ", ".join(map(str, sorted(unknown)))
        )

    reachable: set[int] = set()
    for output_id in output_ids:
        reachable |= nx.ancestors(forward.G, output_id) | {output_id}
    irrelevant_wrt = tuple(node for node in wrt_ids if node not in reachable)
    if irrelevant_wrt:
        raise ProcessGraphAutogradError(
            "requested gradients are disconnected from the outputs: "
            + ", ".join(map(str, irrelevant_wrt))
        )

    builder = _AdjointBuilder(forward)
    contributions: dict[int, list[int]] = {}
    seeds: dict[int, int] = {}
    gradients: dict[int, int] = {}
    registry_rules: dict[int, str] = {}

    for output_id in output_ids:
        seed = builder.input(
            f"grad_seed_{output_id}",
            kind="gradient_seed",
            source_forward_id=output_id,
        )
        seeds[output_id] = seed
        contributions.setdefault(output_id, []).append(seed)

    def total_gradient(forward_id: int) -> int | None:
        terms = contributions.get(forward_id, ())
        if not terms:
            return None
        result = terms[0]
        for term in terms[1:]:
            result = builder.binary("add", result, term, forward_id)
        gradients[forward_id] = result
        return result

    def contribute(parent_id: int, value_id: int) -> None:
        contributions.setdefault(int(parent_id), []).append(int(value_id))

    unsupported: list[tuple[int, str]] = []
    order = tuple(nx.topological_sort(forward.G.subgraph(reachable)))
    for node_id in reversed(order):
        gradient = total_gradient(int(node_id))
        if gradient is None:
            continue
        data = forward.G.nodes[node_id]
        op = _operation(data)
        parents = _parents(forward, int(node_id))
        if op in {"input", "const"}:
            continue
        attributes = data.get("attributes") or {}
        if (
            op in _AdjointBindingGraphBuilder._PREDICATE_OPERATIONS
            or bool(
                attributes.get("predicate")
                or attributes.get("control_predicate")
            )
        ):
            # Predicate values are resident backward-routing facts, not a
            # differentiable numerical surface. Gradient contributions can
            # reach a mask through ordinary arithmetic (for example the two
            # stable-sigmoid arms), but must stop at the comparison itself.
            # The exact predicate remains retained by the binding graph for
            # every backward rule that consumes it.
            continue
        registry_op = graph_adjoint_rule_name(op)
        if registry_op not in BACKWARD_RULES:
            unsupported.append((int(node_id), op))
            continue
        registry_rules[int(node_id)] = registry_op
        for parent, parent_gradient in builder.registry_rule(
            registry_op,
            gradient,
            int(node_id),
            parents,
            attributes,
        ):
            contribute(parent, parent_gradient)

    if unsupported:
        detail = ", ".join(f"{node}:{op or '?'}" for node, op in unsupported)
        raise ProcessGraphAutogradError(
            "ProcessGraph has no graph-native adjoint rule for " + detail
        )

    _retain_backward_rule_closure(builder.backward, registry_rules.values())

    for wrt_id in wrt_ids:
        if wrt_id not in gradients:
            total_gradient(wrt_id)
    missing = tuple(node for node in wrt_ids if node not in gradients)
    if missing:
        raise ProcessGraphAutogradError(
            "backward graph produced no gradient for "
            + ", ".join(map(str, missing))
        )

    builder.backward.roots = [gradients[node] for node in wrt_ids]
    # ``ProcessGraph.compute_levels`` is an execution scheduler and may
    # materialize Store nodes at roots. Differentiation is still constructing
    # semantic IR here, so record dependency levels without mutating it into a
    # deployment graph.
    builder.backward.levels = {
        int(node_id): int(level)
        for level, generation in enumerate(
            nx.topological_generations(builder.backward.G)
        )
        for node_id in generation
    }
    builder.backward.G.graph.update({
        "graph_kind": "parametric_backward",
        "schema_version": 1,
        "forward_output_ids": output_ids,
        "wrt_value_ids": wrt_ids,
        "gradient_outputs": {
            str(forward_id): gradients[forward_id] for forward_id in wrt_ids
        },
        "saved_forward_values": dict(builder.saved),
        "gradient_seeds": dict(seeds),
        "python_backward_callbacks": False,
        "backward_rule_registry": "src.common.tensors.backward_registry.BACKWARD_RULES",
        "backward_rule_nodes": dict(registry_rules),
    })
    execution_contract = forward.G.graph.get("execution_contract")
    if isinstance(execution_contract, Mapping):
        builder.backward.G.graph["execution_contract"] = copy.deepcopy(
            dict(execution_contract)
        )
        builder.backward.G.graph["deployment_role"] = (
            "opportunistic_numeric_dispatch"
            if execution_contract.get("native_lowering") == "opportunistic"
            else "required_numeric_program"
        )
    _annotate_numeric_metadata(builder.backward)
    binding_graph = builder.bindings.finish()
    saved_contracts = {
        int(forward_id): SavedValueContract(
            forward_value_id=int(forward_id),
            backward_input_id=int(backward_id),
            shape=tuple(
                (forward.G.nodes[forward_id].get("tensor") or {}).get("shape")
                or ()
            ),
            dtype=str(
                (forward.G.nodes[forward_id].get("tensor") or {}).get("dtype")
                or "float64"
            ),
            storage=builder.saved_storage.get(int(forward_id), "resident"),
        )
        for forward_id, backward_id in builder.saved.items()
    }
    gradient_contracts = {}
    for forward_id in wrt_ids:
        data = forward.G.nodes[forward_id]
        attrs = data.get("attributes") or {}
        binding_kind = str(attrs.get("binding_kind") or "value")
        binding_name = str(
            attrs.get("binding_name")
            or attrs.get("name")
            or data.get("label")
            or f"value_{forward_id}"
        )
        gradient_contracts[int(forward_id)] = GradientValueContract(
            forward_value_id=int(forward_id),
            backward_value_id=int(gradients[forward_id]),
            binding_kind=binding_kind,
            binding_name=binding_name,
            mutable=bool(
                attrs.get("mutable")
                or binding_kind in {"parameter", "state", "mutable_state"}
            ),
        )
    builder.backward.G.graph["saved_value_contracts"] = {
        str(key): vars(contract) for key, contract in saved_contracts.items()
    }
    builder.backward.G.graph["gradient_contracts"] = {
        str(key): vars(contract) for key, contract in gradient_contracts.items()
    }
    builder.backward.G.graph["adjoint_binding_contracts"] = {
        str(forward_id): copy.deepcopy(dict(data))
        for forward_id, data in binding_graph.graph.nodes(data=True)
    }
    builder.backward.G.graph["adjoint_binding_edges"] = tuple(
        (int(left), int(right), copy.deepcopy(dict(data)))
        for left, right, data in binding_graph.graph.edges(data=True)
    )
    return ProcessGraphAdjoint(
        forward=forward,
        backward=builder.backward,
        output_value_ids=output_ids,
        wrt_value_ids=wrt_ids,
        seed_value_ids=dict(seeds),
        saved_value_ids=dict(builder.saved),
        gradient_value_ids={node: gradients[node] for node in wrt_ids},
        saved_value_contracts=saved_contracts,
        gradient_contracts=gradient_contracts,
        binding_graph=binding_graph,
    )


def fuse_forward_loss_backward(
    adjoint: ProcessGraphAdjoint,
    *,
    unit_loss_seed: bool = True,
) -> ForwardLossBackwardMotion:
    """Compose a ProcessGraph-derived adjoint with its forward graph.

    Saved-value inputs disappear: each is wired directly to its authoritative
    forward producer. A scalar loss may use a graph constant seed of one; set
    ``unit_loss_seed=False`` to retain explicit upstream-gradient ABI inputs.
    The optimizer is intentionally absent from this graph.
    """

    forward = adjoint.forward
    backward = adjoint.backward
    motion = ProcessGraph(
        materialize_memory=False,
        function_table=copy.deepcopy(backward.function_table),
        external_function_table=copy.deepcopy(
            backward.external_function_table
        ),
    )
    forward_keep: set[int] = set()
    for loss_id in adjoint.output_value_ids:
        forward_keep |= nx.ancestors(forward.G, loss_id) | {loss_id}

    for node_id in nx.topological_sort(forward.G.subgraph(forward_keep)):
        data = copy.deepcopy(dict(forward.G.nodes[node_id]))
        data["parents"] = [
            (int(parent), str(role))
            for parent, role in data.get("parents", ())
            if int(parent) in forward_keep
        ]
        data["children"] = []
        attrs = dict(data.get("attributes") or {})
        attrs.setdefault("training_motion_phase", "forward")
        data["attributes"] = attrs
        data["extra_args"] = copy.deepcopy(attrs)
        motion.G.add_node(int(node_id), **data)
    for left, right, edge in forward.G.subgraph(forward_keep).edges(data=True):
        motion.G.add_edge(int(left), int(right), **copy.deepcopy(dict(edge)))
        role = str(edge.get("role") or "value")
        motion.G.nodes[int(left)]["children"].append((int(right), role))

    next_id = max((int(node) for node in motion.G), default=-1) + 1
    remap: dict[int, int] = {}
    seed_ids: dict[int, int] = {}
    saved_by_backward = {
        int(backward_id): int(forward_id)
        for forward_id, backward_id in adjoint.saved_value_ids.items()
    }
    output_by_seed = {
        int(seed_id): int(output_id)
        for output_id, seed_id in adjoint.seed_value_ids.items()
    }
    for node_id in nx.topological_sort(backward.G):
        node_id = int(node_id)
        if node_id in saved_by_backward:
            remap[node_id] = saved_by_backward[node_id]
            continue
        new_id = next_id
        next_id += 1
        remap[node_id] = new_id
        data = copy.deepcopy(dict(backward.G.nodes[node_id]))
        attrs = dict(data.get("attributes") or {})
        attrs["training_motion_phase"] = "backward"
        if node_id in output_by_seed and unit_loss_seed:
            data.update({
                "op": "const",
                "type": "const",
                "label": "1.0",
                "constant": 1.0,
            })
            attrs = {
                "values": 1.0,
                "gradient_seed_for": output_by_seed[node_id],
                "training_motion_phase": "backward",
            }
            data["parents"] = []
        else:
            data["parents"] = [
                (remap[int(parent)], str(role))
                for parent, role in data.get("parents", ())
            ]
        data["children"] = []
        data["attributes"] = attrs
        data["extra_args"] = copy.deepcopy(attrs)
        motion.G.add_node(new_id, **data)
        if node_id in output_by_seed:
            seed_ids[output_by_seed[node_id]] = new_id

    for node_id in nx.topological_sort(backward.G):
        node_id = int(node_id)
        if node_id in saved_by_backward:
            continue
        new_id = remap[node_id]
        for parent, role in backward.G.nodes[node_id].get("parents", ()):
            new_parent = remap[int(parent)]
            motion.G.add_edge(new_parent, new_id, role=str(role))
            motion.G.nodes[new_parent]["children"].append((new_id, str(role)))

    gradient_ids = {
        int(forward_id): remap[int(backward_id)]
        for forward_id, backward_id in adjoint.gradient_value_ids.items()
    }
    motion.roots = [*adjoint.output_value_ids, *gradient_ids.values()]
    motion.levels = {
        int(node_id): int(level)
        for level, generation in enumerate(nx.topological_generations(motion.G))
        for node_id in generation
    }
    execution_contract = forward.G.graph.get("execution_contract")
    motion.G.graph.update({
        "graph_kind": "forward_loss_backward_motion",
        "schema_version": 1,
        "loss_outputs": tuple(adjoint.output_value_ids),
        "gradient_outputs": dict(gradient_ids),
        "gradient_seeds": dict(seed_ids),
        "unit_loss_seed": bool(unit_loss_seed),
        "optimizer_included": False,
        "backward_rule_registry": backward.G.graph.get(
            "backward_rule_registry"
        ),
        "backward_rule_nodes": copy.deepcopy(
            backward.G.graph.get("backward_rule_nodes", {})
        ),
        "adjoint_binding_contracts": {
            str(forward_id): copy.deepcopy(dict(data))
            for forward_id, data in adjoint.binding_graph.graph.nodes(data=True)
        },
        "adjoint_binding_edges": tuple(
            (int(left), int(right), copy.deepcopy(dict(data)))
            for left, right, data in adjoint.binding_graph.graph.edges(data=True)
        ),
    })
    if isinstance(execution_contract, Mapping):
        motion.G.graph["execution_contract"] = copy.deepcopy(
            dict(execution_contract)
        )
    _annotate_numeric_metadata(motion)
    return ForwardLossBackwardMotion(
        graph=motion,
        loss_value_ids=tuple(adjoint.output_value_ids),
        gradient_value_ids=gradient_ids,
        seed_value_ids=seed_ids,
        binding_graph=adjoint.binding_graph,
    )


def compile_process_graph_backward(
    forward: ProcessGraph,
    *,
    outputs: Iterable[int] | None = None,
    wrt: Iterable[int] | None = None,
    packaging: str = "independent",
    unit_loss_seed: bool = True,
) -> ProcessGraphBackwardProduct:
    """Request backward code with its binding graph unconditionally attached.

    ``packaging='independent'`` selects the standalone parametric backward
    graph. ``packaging='combined'`` selects one forward/loss/backward motion.
    Both are generated from the same :func:`differentiate_process_graph`
    result and expose the identical :class:`AdjointBindingGraph`.  Combined
    packaging does not introduce or pass through ``FusedProgram``.
    """

    execution_contract = forward.G.graph.get("execution_contract")
    if isinstance(execution_contract, Mapping):
        backward_source = execution_contract.get("backward_source")
        if backward_source not in {None, "process_graph"}:
            raise ProcessGraphAutogradError(
                "execution contract selects backward_source="
                f"{backward_source!r}; refusing to replace it with graph "
                "inversion"
            )

    selected = str(packaging).strip().lower()
    if selected not in {"independent", "combined"}:
        raise ValueError(
            "backward packaging must be 'independent' or 'combined', got "
            f"{packaging!r}"
        )
    adjoint = differentiate_process_graph(forward, outputs=outputs, wrt=wrt)
    motion = (
        fuse_forward_loss_backward(adjoint, unit_loss_seed=unit_loss_seed)
        if selected == "combined"
        else None
    )
    return ProcessGraphBackwardProduct(
        graph=adjoint.backward if motion is None else motion.graph,
        packaging=selected,
        adjoint=adjoint,
        motion=motion,
        binding_graph=adjoint.binding_graph,
    )


def lower_training_motion_to_repository_ssa(
    motion: ForwardLossBackwardMotion,
    *,
    function_name: str = "forward_loss_backward",
    tensor_ssa_reference: Any | None = None,
    observed_outputs: Mapping[str, int] | None = None,
) -> TrainingMotionSSALowering:
    """Lower the complete semantic training motion through repository SSA.

    The motion remains a ``ProcessGraph`` program.  Its graph-backed calls to
    the authored ``BACKWARD_RULES`` closure therefore pass through the same
    control, call-linking, operator-region, and repository-SSA machinery as
    any other ingested program; this boundary neither projects a
    ``FusedProgram`` nor manufactures a second instruction language.
    """

    from types import SimpleNamespace

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from .fortran_c_shell import _class_surface_ssa_program
    from .glsl_deployment_strategy import strategize_shell_deployment
    from .shell_reference_tables import build_class_navigation_table

    graph = motion.graph
    _annotate_numeric_metadata(graph)

    observations = {
        str(name): int(value_id)
        for name, value_id in dict(observed_outputs or {}).items()
    }
    missing_observations = {
        name: value_id
        for name, value_id in observations.items()
        if value_id not in graph.G
    }
    if missing_observations:
        raise ProcessGraphAutogradError(
            "training-motion observations are not graph values: "
            f"{missing_observations!r}"
        )
    reserved_output_names = {
        *(f"loss_{index}" for index in range(len(motion.loss_value_ids))),
        *(f"grad_{forward_id}" for forward_id in motion.gradient_value_ids),
    }
    collisions = reserved_output_names.intersection(observations)
    if collisions:
        raise ProcessGraphAutogradError(
            "training-motion observation names collide with generated "
            f"outputs: {tuple(sorted(collisions))!r}"
        )
    outputs = {
        **observations,
        **{f"loss_{index}": int(value_id)
           for index, value_id in enumerate(motion.loss_value_ids)},
        **{f"grad_{forward_id}": int(value_id)
           for forward_id, value_id in motion.gradient_value_ids.items()},
    }

    used_names: set[str] = set()
    parameter_names: list[str] = []
    identity_table: dict[str, tuple[int, ...]] = {}
    for position, node_id in enumerate(nx.topological_sort(graph.G)):
        node_id = int(node_id)
        data = graph.G.nodes[node_id]
        if _operation(data) != "input":
            continue
        attributes = data.get("attributes") or {}
        base = str(
            attributes.get("binding_name")
            or attributes.get("name")
            or data.get("label")
            or f"argument_{position}"
        )
        name = base
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base}_{suffix}"
        used_names.add(name)
        parameter_names.append(name)
        identity_table[name] = (node_id,)
    identity_table.update({name: (node_id,) for name, node_id in outputs.items()})

    function_reference = graph.function_table.declare(
        function_name,
        qualified_name=function_name,
    )
    graph.function_table.resolve_graph(function_reference, graph)
    graph.G.graph.update({
        "function_ref": int(function_reference.address),
        "function_name": str(function_name),
        "function_parameters": tuple(parameter_names),
        "positional_parameters": tuple(parameter_names),
        "function_outputs": tuple(outputs),
        "identity_table": identity_table,
        "compile_targets": (str(function_name),),
        "semantic_authority": "ProcessGraph",
    })

    original_roots = list(graph.roots)
    original_function_outputs = tuple(
        graph.G.graph.get("function_outputs") or ()
    )
    original_identity_table = copy.deepcopy(
        graph.G.graph.get("identity_table") or {}
    )
    retained_forward_products = tuple(
        int(forward_id)
        for forward_id, data in motion.binding_graph.graph.nodes(data=True)
        if int(forward_id) in graph.G
        and str(data.get("storage") or "resident") == "resident"
        and _operation(graph.G.nodes[int(forward_id)]) not in {"input", "const"}
    )
    graph.roots = list(dict.fromkeys((
        *original_roots, *retained_forward_products,
    )))
    graph.G.graph["planning_retained_value_ids"] = retained_forward_products
    hidden_retained_names = tuple(
        f"__adjoint_saved_{value_id}"
        for value_id in retained_forward_products
    )
    graph.G.graph["function_outputs"] = tuple(dict.fromkeys((
        *original_function_outputs, *hidden_retained_names,
    )))
    graph.G.graph["identity_table"] = {
        **original_identity_table,
        **{
            name: (value_id,)
            for name, value_id in zip(
                hidden_retained_names, retained_forward_products
            )
        },
    }
    deployment = None
    try:
        deployment_type = strategize_shell_deployment(
            graph,
            backend="fortran",
            runtime_closure_only=True,
        )
        deployment = deployment_type(profiling=False, shell_language="glsl")
        deployment.compile_process_graph(prepare_ephemerals=False)
        deployment.prepare_graph_precompile(structural_ssa_only=True)
        compilation = SimpleNamespace(
            deployment=deployment,
            class_navigation=build_class_navigation_table(graph),
        )
        module, _section_outputs, exports = _class_surface_ssa_program(
            compilation,
            "training_motion",
            tensor_ssa_reference=(
                tensor_ssa_reference or c_backend_repository_ssa_reference()
            ),
        )
    finally:
        graph.roots = original_roots
        graph.G.graph["function_outputs"] = original_function_outputs
        graph.G.graph["identity_table"] = original_identity_table
        if deployment is not None:
            deployment.release()

    if module is None:
        raise ProcessGraphAutogradError(
            "repository SSA planning produced no training-motion functions"
        )
    from .tensor_ssa_lowering import (
        lower_tensor_calls_to_repository_ssa,
        propagate_repository_ssa_call_metadata,
        settle_shape_only_repository_returns,
        wire_repository_ssa_region_products,
    )
    reference = tensor_ssa_reference or c_backend_repository_ssa_reference()
    wire_repository_ssa_region_products(module)
    settle_shape_only_repository_returns(module)
    propagate_repository_ssa_call_metadata(module)
    tensor_shortfalls = lower_tensor_calls_to_repository_ssa(module, reference)
    propagate_repository_ssa_call_metadata(
        module, authoritative_returns=True,
    )
    if tensor_shortfalls:
        raise ProcessGraphAutogradError(
            "training-motion tensor SSA remained incomplete after call "
            "metadata settlement: "
            + "; ".join(
                f"{item.function}:{item.block}:{item.operation} ({item.reason})"
                for item in tensor_shortfalls
            )
        )
    expected_symbol = f"training_motion__{function_name}"
    root_symbol = next(
        (
            symbol for symbol in exports
            if symbol == expected_symbol or symbol.endswith(f"__{function_name}")
        ),
        None,
    )
    if root_symbol is None:
        raise ProcessGraphAutogradError(
            "repository SSA did not export the training-motion root; "
            f"exports={tuple(exports)!r}"
        )
    root_function = module.functions[str(root_symbol)]
    public_output_ids = set(map(int, outputs.values()))
    for block in root_function.blocks.values():
        for instruction in block.instrs:
            if instruction.op in {"Ret", "ret", "Return", "return"}:
                instruction.args = [
                    value for value in instruction.args
                    if int(value.id) in public_output_ids
                ]
    root_function.metadata["source_output_value_ids"] = tuple(
        int(value_id) for value_id in outputs.values()
    )
    root_function.metadata["named_outputs"] = tuple(
        (str(name), int(value_id)) for name, value_id in outputs.items()
    )
    return TrainingMotionSSALowering(
        module=module,
        function_name=str(root_symbol),
        outputs=outputs,
        shortfalls=(),
    )


__all__ = [
    "AdjointBindingGraph",
    "ConditionalAdjointContract",
    "ControlProgramAdjoint",
    "ForwardLossBackwardMotion",
    "LoopAdjointContract",
    "GradientValueContract",
    "ProcessGraphProgramAdjoint",
    "ProcessGraphAdjointRegion",
    "ProcessGraphBackwardProduct",
    "TrainingMotionSSALowering",
    "ProcessGraphAdjoint",
    "ProcessGraphAutogradError",
    "SavedValueContract",
    "abstract_tensor_program_to_process_graph",
    "differentiate_process_graph",
    "compile_process_graph_backward",
    "differentiate_control_program",
    "differentiate_process_program",
    "fuse_forward_loss_backward",
    "graph_adjoint_rule_name",
    "isolate_process_program_adjoint_regions",
    "lower_training_motion_to_repository_ssa",
]
