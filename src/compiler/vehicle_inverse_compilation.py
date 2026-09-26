"""Tape-free vehicle inverse/Adam source packaging.

The vehicle model is its equation.  JSON supplies values for the equation's
named parameters; this module never substitutes those values into the graph.
It adds a separate target/weight objective, derives a static ProcessGraph
adjoint, materializes that equation as AbstractTensor Python, composes the
repository AbstractNN functional Adam equation, then places two-limb promotion
around the complete Python program.  C AOT is a later compiler action.  The
browser route deliberately remains the canonical forward-only WASM emission.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import keyword
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import sympy

from ..common.tensors.backward_registry import BACKWARD_RULES
from ..transmogrifier.graph.graph_express2 import ProcessGraph
from .process_graph_autograd import (
    ProcessGraphBackwardProduct,
    _registry_function_source,
    compile_process_graph_backward,
)
from .symbolic_equation_compiler import SymbolicEquationCompilation


@dataclass(frozen=True, slots=True)
class VehicleObjectiveMetric:
    output: str
    target: float
    weight: float


@dataclass(frozen=True, slots=True)
class VehicleInverseSpecification:
    parameters: tuple[str, ...]
    metrics: tuple[VehicleObjectiveMetric, ...]
    adam_learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float


@dataclass(frozen=True, slots=True)
class VehicleInversePythonProgram:
    source: str
    reverse_entrypoint: str
    adam_entrypoint: str
    two_limb_entrypoint: str
    model_parameter_names: tuple[str, ...]
    optimized_parameter_names: tuple[str, ...]
    objective_parameter_names: tuple[str, ...]
    optimizer_parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    inverse_graph: ProcessGraphBackwardProduct
    manifest: Mapping[str, Any]


def load_vehicle_inverse_specification(path: str | Path) -> VehicleInverseSpecification:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != {"schema", "parameters", "metrics", "adam"}:
        raise ValueError("vehicle inverse specification has an unknown or missing field")
    if raw["schema"] != "turing.vehicle-inverse-objective.v1":
        raise ValueError("unknown vehicle inverse objective schema")
    parameters = tuple(map(str, raw["parameters"]))
    if not parameters or len(parameters) != len(set(parameters)):
        raise ValueError("vehicle inverse parameters must be nonempty and unique")
    metrics = tuple(
        VehicleObjectiveMetric(
            output=str(row["output"]), target=float(row["target"]),
            weight=float(row["weight"]),
        )
        for row in raw["metrics"]
    )
    if not metrics or len({row.output for row in metrics}) != len(metrics):
        raise ValueError("vehicle inverse metrics must be nonempty and unique")
    if any(not math.isfinite(row.target) or not math.isfinite(row.weight)
           or row.weight <= 0 for row in metrics):
        raise ValueError("vehicle inverse metric targets must be finite and weights positive")
    adam = raw["adam"]
    if set(adam) != {"learning_rate", "beta1", "beta2", "epsilon"}:
        raise ValueError("vehicle inverse Adam parameters are incomplete")
    values = tuple(float(adam[name]) for name in ("learning_rate", "beta1", "beta2", "epsilon"))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("vehicle inverse Adam parameters must be finite and positive")
    if values[1] >= 1 or values[2] >= 1:
        raise ValueError("vehicle inverse Adam beta parameters must be below one")
    return VehicleInverseSpecification(parameters, metrics, *values)


def _add_node(graph: ProcessGraph, operation: str, parents: Iterable[int] = (), *,
              label: str | None = None, attributes: Mapping[str, Any] | None = None,
              constant: Any = None) -> int:
    node_id = max(map(int, graph.G), default=-1) + 1
    parent_rows = [(int(parent), f"arg:{index}") for index, parent in enumerate(parents)]
    attrs = dict(attributes or {})
    graph.G.add_node(
        node_id, op=operation, type=operation, label=label or operation,
        parents=parent_rows, children=[], attributes=attrs,
        extra_args=copy.deepcopy(attrs), tensor={"shape": (), "dtype": "float64"},
        control={}, constant=constant, expr_obj=None, store_id=None,
    )
    for parent, role in parent_rows:
        graph.G.add_edge(parent, node_id, role=role)
        graph.G.nodes[parent].setdefault("children", []).append((node_id, role))
    return node_id


def build_vehicle_objective_graph(
    compilation: SymbolicEquationCompilation,
    specification: VehicleInverseSpecification,
) -> tuple[ProcessGraph, tuple[int, ...]]:
    """Add runtime target/weight inputs and one scalar loss to the model graph."""

    unknown_parameters = set(specification.parameters) - set(compilation.input_ids)
    unknown_metrics = {row.output for row in specification.metrics} - set(compilation.output_ids)
    if unknown_parameters or unknown_metrics:
        raise ValueError(
            f"vehicle inverse names are not in the model ABI: parameters={sorted(unknown_parameters)} "
            f"metrics={sorted(unknown_metrics)}")
    graph = copy.deepcopy(compilation.process_graph)
    terms: list[int] = []
    identities = dict(graph.G.graph.get("identity_table") or {})
    for metric in specification.metrics:
        target_name = f"objective_target__{metric.output}"
        weight_name = f"objective_weight__{metric.output}"
        target = _add_node(
            graph, "input", label=target_name,
            attributes={"binding_name": target_name, "binding_kind": "objective_parameter"})
        weight = _add_node(
            graph, "input", label=weight_name,
            attributes={"binding_name": weight_name, "binding_kind": "objective_parameter"})
        error = _add_node(graph, "Sub", (int(compilation.output_ids[metric.output]), target))
        squared = _add_node(graph, "Mul", (error, error))
        terms.append(_add_node(graph, "Mul", (weight, squared)))
        identities[target_name] = (target,)
        identities[weight_name] = (weight,)
    loss = terms[0]
    for term in terms[1:]:
        loss = _add_node(graph, "Add", (loss, term))
    graph.roots = [loss]
    identities["vehicle_stability_objective"] = (loss,)
    graph.G.graph.update({
        "identity_table": identities,
        "graph_kind": "vehicle-model-plus-runtime-objective",
        "model_equation_unchanged": True,
        "objective_values_are_runtime_parameters": True,
    })
    return graph, tuple(int(compilation.input_ids[name]) for name in specification.parameters)


def _identifier(value: str, used: set[str]) -> str:
    base = re.sub(r"\W", "_", str(value)) or "value"
    if base[0].isdigit() or keyword.iskeyword(base):
        base = "v_" + base
    candidate = base
    suffix = 1
    while candidate in used:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used.add(candidate)
    return candidate


def _literal(value: Any) -> str:
    if isinstance(value, sympy.Basic):
        return repr(float(sympy.N(value, 40)))
    return repr(value)


def _motion_source(product: ProcessGraphBackwardProduct, *, name: str) -> tuple[
    str, tuple[str, ...], tuple[str, ...]
]:
    if product.motion is None:
        raise ValueError("vehicle inverse Python materialization requires combined packaging")
    graph = product.motion.graph
    used: set[str] = set()
    names: dict[int, str] = {}
    arguments: list[str] = []
    for node_id in nx.topological_sort(graph.G):
        data = graph.G.nodes[node_id]
        operation = str(data.get("op") or data.get("type") or "").casefold()
        if operation != "input":
            continue
        attrs = data.get("attributes") or {}
        argument = _identifier(
            str(attrs.get("binding_name") or attrs.get("name") or data.get("label")
                or f"argument_{node_id}"), used)
        names[int(node_id)] = argument
        arguments.append(argument)

    lines: list[str] = [f"def {name}({', '.join(arguments)}):"]
    for node_id in nx.topological_sort(graph.G):
        node_id = int(node_id)
        if node_id in names:
            continue
        data = graph.G.nodes[node_id]
        operation = str(data.get("op") or data.get("type") or "").casefold()
        parents = [int(parent) for parent, _role in data.get("parents", ())]
        operands = [names[parent] for parent in parents]
        target = f"v_{node_id}"
        if operation in {"return", "store"}:
            continue
        if operation in {"const", "constant"}:
            value = data.get("constant")
            if value is None:
                attrs = data.get("attributes") or {}
                value = attrs.get("values", attrs.get("value", attrs.get("constant")))
            expression = _literal(value)
        elif operation == "pi":
            expression = "math.pi"
        elif operation in {"add", "sub", "mul", "div", "truediv", "pow"}:
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/",
                      "truediv": "/", "pow": "**"}[operation]
            expression = f"({operands[0]} {symbol} {operands[1]})"
        elif operation == "neg":
            expression = f"(-{operands[0]})"
        elif operation == "abs":
            expression = f"abs({operands[0]})"
        elif operation in {"sqrt", "tanh", "exp", "log", "sin", "cos"}:
            expression = f"{operands[0]}.{operation}()"
        elif operation in {"lt", "le", "gt", "ge", "eq", "ne"}:
            symbol = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}[operation]
            expression = f"({operands[0]} {symbol} {operands[1]})"
        elif operation in {"select", "piecewise"} and len(operands) == 3:
            expression = f"AbstractTensor.where({operands[0]}, {operands[1]}, {operands[2]})"
        elif operation == "call":
            rule = str((data.get("attributes") or {}).get("backward_rule") or "")
            if not rule:
                raise ValueError(f"inverse graph call {node_id} has no static backward rule")
            expression = f"bw_{rule}({', '.join(operands)})"
        elif operation in {"indexed", "getitem"}:
            expression = f"{operands[0]}[int({operands[1]})]"
        else:
            raise ValueError(f"inverse graph Python materializer has no rule for {node_id}:{operation}")
        names[node_id] = target
        lines.append(f"    {target} = {expression}")
    result_ids = [*product.motion.loss_value_ids, *product.motion.gradient_value_ids.values()]
    result_names = tuple(["loss", *(f"gradient__{value}" for value in product.motion.gradient_value_ids)])
    rendered = ", ".join(names[int(value_id)] for value_id in result_ids)
    lines.append(f"    return ({rendered},)")
    return "\n".join(lines) + "\n", tuple(arguments), result_names


def prepare_vehicle_inverse_adam_python(
    compilation: SymbolicEquationCompilation,
    specification: VehicleInverseSpecification,
    *, name: str = "vehicle_inverse",
) -> VehicleInversePythonProgram:
    objective, wrt_ids = build_vehicle_objective_graph(compilation, specification)
    product = compile_process_graph_backward(
        objective, outputs=objective.roots, wrt=wrt_ids,
        packaging="combined", unit_loss_seed=True)
    reverse_entry = f"{name}__forward_inverse"
    reverse_source, materialized_arguments, reverse_outputs = _motion_source(
        product, name=reverse_entry)
    rules = tuple(dict.fromkeys(
        str((data.get("attributes") or {}).get("backward_rule"))
        for _node, data in product.graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("backward_rule")
    ))
    rule_source = "\n".join(_registry_function_source(rule, BACKWARD_RULES[rule])
                              for rule in rules)

    model_parameters = tuple(compilation.input_ids)
    objective_parameters = tuple(
        name for metric in specification.metrics
        for name in (f"objective_target__{metric.output}",
                     f"objective_weight__{metric.output}")
    )
    reverse_arguments = (*model_parameters, *objective_parameters)
    if set(materialized_arguments) != set(reverse_arguments):
        raise RuntimeError(
            "materialized inverse ABI differs from model/objective parameter ABI: "
            f"missing={sorted(set(reverse_arguments) - set(materialized_arguments))} "
            f"extra={sorted(set(materialized_arguments) - set(reverse_arguments))}")
    optimizer_parameters = ("adam_step_index", "adam_learning_rate", "adam_beta1",
                            "adam_beta2", "adam_epsilon")
    moment_arguments = tuple(
        name for parameter in specification.parameters
        for name in (f"adam_m__{parameter}", f"adam_v__{parameter}")
    )
    adam_entry = f"{name}__adam"
    all_adam_arguments = (*reverse_arguments, *moment_arguments, *optimizer_parameters)
    lines = [f"def {adam_entry}({', '.join(all_adam_arguments)}):",
             f"    inverse = {reverse_entry}({', '.join(f'{item}={item}' for item in reverse_arguments)})",
             "    loss = inverse[0]"]
    output_names: list[str] = ["loss"]
    for index, parameter in enumerate(specification.parameters, start=1):
        m_name, v_name = f"adam_m__{parameter}", f"adam_v__{parameter}"
        prefix = f"updated__{parameter}"
        lines.append(
            f"    {prefix}, {prefix}__m, {prefix}__v, {prefix}__t = adam_step("
            f"{parameter}, inverse[{index}], {m_name}, {v_name}, adam_step_index, "
            "lr=adam_learning_rate, beta1=adam_beta1, beta2=adam_beta2, eps=adam_epsilon)")
        output_names.extend((prefix, f"{prefix}__m", f"{prefix}__v"))
    first = specification.parameters[0]
    lines.append(f"    return (loss, {', '.join(output_names[1:])}, updated__{first}__t)")
    adam_source = "\n".join(lines) + "\n"

    two_limb_entry = f"{name}__two_limb_c_entry"
    promoted = tuple(f"wide__{argument}" for argument in all_adam_arguments)
    wide_lines = [f"def {two_limb_entry}({', '.join(all_adam_arguments)}):"]
    for argument, wide in zip(all_adam_arguments, promoted):
        wide_lines.append(f"    {wide} = Precision.of({argument}, 2)")
    wide_lines.append(f"    result = {adam_entry}({', '.join(promoted)})")
    collapsed = []
    for index in range(len(output_names) + 1):
        local = f"result__{index}"
        wide_lines.append(f"    {local} = result[{index}].collapse()")
        collapsed.append(local)
    wide_lines.append(f"    return ({', '.join(collapsed)},)")
    precision_source = "\n".join(wide_lines) + "\n"
    source = "\n".join((
        "import math",
        "from src.common.tensors.abstraction import AbstractTensor",
        "from src.common.tensors.backward_registry import unbroadcast, eps",
        "from src.common.tensors.abstract_nn.optimizer import adam_step",
        "from src.common.tensors.extended_precision import Precision",
        rule_source, reverse_source, adam_source, precision_source,
    ))
    manifest = {
        "schema": "turing.vehicle-inverse-adam-python.v1",
        "model": "equation",
        "json_value_meaning": "runtime-parameter",
        "json_values_constant_folded": False,
        "tape_or_capture_used_for_reverse": False,
        "reverse_transform": "static-process-graph-adjoint",
        "python_materialization": "AbstractTensor",
        "adam": "src.common.tensors.abstract_nn.optimizer.adam_step",
        "precision_order": "compose-forward-inverse-adam-then-promote-all-inputs-to-two-limb",
        "adam_bias_correction": "two-limb-real-power-operator",
        "c_aot_entrypoint": two_limb_entry,
        "wasm": {"scope": "canonical-forward-only", "inverse": False, "adam": False},
    }
    return VehicleInversePythonProgram(
        source=source, reverse_entrypoint=reverse_entry, adam_entrypoint=adam_entry,
        two_limb_entrypoint=two_limb_entry, model_parameter_names=model_parameters,
        optimized_parameter_names=specification.parameters,
        objective_parameter_names=objective_parameters,
        optimizer_parameter_names=optimizer_parameters,
        output_names=tuple(["loss", *output_names[1:], "adam_step_index_next"]),
        inverse_graph=product, manifest=manifest,
    )


def vehicle_rig_outfit_stages() -> tuple[dict[str, Any], ...]:
    """Ordered validation gates over the unchanged vehicle equation/part ABI."""

    return (
        {"identity": "bare-structural-mockup", "parts": ["frame", "cage", "wrench-attachments"],
         "engine_enabled": False, "body": "bare-frame", "gate": "settle-and-contact-pass"},
        {"identity": "rolling-mockup", "parts": ["tires", "wheels", "suspension", "ballast"],
         "engine_enabled": False, "body": "bare-frame", "gate": "rough-terrain-pass"},
        {"identity": "powered-chassis", "parts": ["engine", "clutch", "transmission", "transfer-case",
                                                     "fuel", "electrical"],
         "engine_enabled": True, "body": "bare-frame", "gate": "powered-settle-and-rolling-start-pass"},
        {"identity": "outfitted-body", "parts": ["selected-body", "armor", "weapons", "accessories"],
         "engine_enabled": True, "body": "runtime-selection",
         "gate": "configuration-specific-rough-terrain-pass"},
        {"identity": "deployment", "parts": ["all-selected-json-parts", "deployment-terrain-profile"],
         "engine_enabled": True, "body": "runtime-selection", "gate": "deployment-metrics-pass"},
    )


def vehicle_rig_outfit_contract() -> dict[str, Any]:
    """The one runtime ABI used from an empty fixture through deployment."""

    return {
        "schema": "turing.vehicle-rig-outfit.v1",
        "model_authority": "same-vehicle-equation-and-parameter-abi-at-every-stage",
        "starts_with_equipment": False,
        "installable_part_classes": [
            "body", "engine", "clutch", "transmission", "transfer-case",
            "fuel-system", "electrical-system", "armor", "weapon", "accessory",
            "ballast", "bumper",
        ],
        "recompute_after_every_change": [
            "total-mass", "center-of-mass", "inertia-tensor", "axle-loads",
            "hardpoint-wrenches", "ground-clearance", "collision-geometry",
            "drag-vector-coefficients-and-reference-areas",
        ],
        "solve_targets": [
            "bare-quiescence", "rough-terrain-stability", "powered-response",
            "configuration-specific-deployment",
        ],
        "rule": (
            "a bare pass establishes the structural principle only; every outfitted "
            "configuration must pass its own solve with installed equipment represented"
        ),
        "stages": list(vehicle_rig_outfit_stages()),
    }


__all__ = [
    "VehicleInversePythonProgram", "VehicleInverseSpecification", "VehicleObjectiveMetric",
    "build_vehicle_objective_graph", "load_vehicle_inverse_specification",
    "prepare_vehicle_inverse_adam_python", "vehicle_rig_outfit_stages",
    "vehicle_rig_outfit_contract",
]
