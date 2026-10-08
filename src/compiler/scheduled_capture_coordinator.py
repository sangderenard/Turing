"""Scheduled capture coordinator: runs a planned shell once to capture its numeric regions; owns the capture-backend and planned-capture context variables, the node observer and the compiled structural object model."""

from __future__ import annotations

import ast
import builtins
import copy
import operator
import sys
import networkx as nx
import numpy as np
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, Mapping
from types import ModuleType

from .deployment_fifo import DeploymentFIFO
from .loop_composer import LoopStrategy, _destructure_loop_target
from ..common.tensors.abstraction import AbstractTensor, tensor_identity
from ..common.tensors.accelerator_backends.glsl_backend import execute_captured_fused_program
from ..common.tensors.accelerator_backends.glsl_tensor_backend import GLSLTensorOperations
from ..common.tensors.autograd import GradTape, autograd
from ..common.tensors.fused_ir import ordered_feed_ids
from ..common.tensors.operator_catalog import CREATION_OPERATORS
from ..transmogrifier.graph.edge_roles import ordered_arguments
from .source_node_facts import _constant_value, _static_python_value
from .call_argument_binding import _method_parameter_layout, _call_arguments
from .region_scheduling import _dependency_order
from .deployment_profiler import _shell_profile_name, _profile_event
from .callsite_literal_specialization import _source_static_value
from .captured_region_programs import (
    _compiler_input_name,
    _capture_storage_identity,
    _unique_runtime_feed_aliases,
)
from .validation_control import _validation_control_blocks
from .dispatch_metadata_classifier import _inert_routing_nodes


_scheduled_capture_backend: ContextVar[str] = ContextVar(
    "scheduled_capture_backend",
    default="glsl",
)


@contextmanager
def _use_scheduled_capture_backend(name: str):
    """Select the numerical observer used by one discovery traversal."""

    token = _scheduled_capture_backend.set(str(name))
    try:
        yield
    finally:
        _scheduled_capture_backend.reset(token)


_planned_capture_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "planned_process_graph_capture",
    default=None,
)


def _observe_process_graph_node(
    node_id: int,
    parent_ids: tuple[int, ...],
    parent_values: tuple[Any, ...],
    result: Any,
) -> Any:
    """Correlate one planned node with its immediate primitive occurrence.

    The ProcessGraph node ID is supplied by generated region code at the exact
    operation invocation.  The observer records only integer IDs and never
    compares payloads, wrappers, storage, shapes, or values.  It therefore
    cannot infer an alias or dependency; it only connects the already-planned
    operation to the primitive implementation occurrence revealed by the one
    forward capture.
    """

    context = _planned_capture_context.get()
    if context is None:
        return result
    if not isinstance(result, AbstractTensor):
        # This must be checked before the identity-aliasing logic below:
        # ``result is parent_value`` is only a meaningful alias signal for a
        # tensor, where object identity means shared storage.  For a plain
        # Python value it can be a pure CPython implementation detail --
        # small integers are interned, so ``0 + 1`` and the literal ``1``
        # operand satisfy ``is`` by cache coincidence, not because they are
        # the same computation.  Treating that as an alias would silently
        # drop the operation instead of recording it.
        #
        # Generated numeric-kernel code runs real Python operators, and this
        # observer is its only per-operation hook -- but everything below is
        # tape-primitive correlation, which exists to resolve which of
        # several possible dynamically-dispatched tensor primitives this
        # call produced.  A plain value has no such ambiguity: its own
        # graph node id already is its unambiguous identity, and it will
        # never enter ``tape._nodes`` no matter how long this waits.  Record
        # it directly, in exact execution order, the same way a reference
        # operator (SetAttr/GetAttr) already is.
        shell = context.get("shell")
        if shell is not None:
            shell.reference_operator_sequence.append(int(node_id))
        return result
    tape = context["tape"]
    capture_id = tensor_identity(result)
    aliased_parents = tuple(
        int(parent_id)
        for parent_id, parent_value in zip(parent_ids, parent_values)
        if result is parent_value
    )
    if len(aliased_parents) == 1 and int(node_id) != aliased_parents[0]:
        # The result is literally an existing explicit operand.  Its tape
        # entry therefore belongs to that pre-existing value; this operation
        # did not publish a new primitive result.  Record the ProcessGraph
        # endpoint as an SSA alias before primitive ownership is considered,
        # otherwise the same tape identity is incorrectly assigned both to
        # the call result and to its input socket.
        context["value_aliases"][int(node_id)] = aliased_parents[0]
        return result
    if capture_id in tape._nodes:
        primitive = tape._nodes[capture_id]
        self_parent_slots = tuple(
            int(slot)
            for primitive_id, slot in primitive.parents
            if int(primitive_id) == int(capture_id)
        )
        graph = context.get("graph")
        graph_node = (
            graph.G.nodes[int(node_id)]
            if graph is not None and int(node_id) in graph.G
            else {}
        )
        if (
            self_parent_slots
            and len(parent_ids) == 1
            and graph_node.get("type") not in {"Store", "IndexedStore"}
        ):
            # An identity-preserving tensor adapter can be recorded by the
            # tape under the same object ID as its operand.  Such a node is a
            # self-edge in object-keyed tape space, not a numerical kernel.
            # The ProcessGraph supplies the unambiguous SSA relationship.
            context["value_aliases"][int(node_id)] = int(parent_ids[0])
            return result
        existing_owner = primitive.ctx.get("process_graph_node_id")
        if existing_owner is not None:
            # A structural Store, call return, or other pass-through node may
            # observe the exact same tensor produced by an earlier operation.
            # That does not create a new primitive or a new value identity.
            # Producer ownership is single-assignment; the ProcessGraph's
            # explicit call/store binding carries the alias relationship.
            return result
        context["node_capture_ids"].setdefault(
            int(node_id), []
        ).append(int(capture_id))
        primitive.ctx["process_graph_node_id"] = int(node_id)
        tensor_op = (graph_node.get("attributes") or {}).get("tensor")
        if tensor_op in CREATION_OPERATORS and graph is not None:
            # The first positional argument is the size. Its role varies by
            # which ingestion path resolved this call -- "args" (generic
            # schema descent, taken here since a static class reference like
            # ``AbstractTensor.zeros`` doesn't go through the ordinary
            # receiver-argument rewiring), "arg0", or "arg:0" (other call
            # shapes) -- so match any arg-prefixed role instead of guessing
            # one spelling, and take the first by parent id for determinism.
            # Where the role states an index, use it: that is the actual
            # first argument rather than the smallest parent id, which was
            # only ever a stand-in for an order the caller could not
            # recover. Bare "arg" carries no index, so that spelling still
            # falls back to the deterministic-by-id choice.
            _declared_arguments = ordered_arguments(
                graph_node.get("parents") or ()
            )
            size_parent_id = (
                int(_declared_arguments[0]) if _declared_arguments
                else min(
                    (
                        int(parent)
                        for parent, role in (graph_node.get("parents") or ())
                        if str(role).lower().startswith("arg")
                    ),
                    default=None,
                )
            )
            if (
                size_parent_id is not None
                and size_parent_id in graph.G
                and not _source_static_value(graph, size_parent_id)
            ):
                # The size argument is itself computed (not a literal), so
                # this result's shape is only what this one discovery trace
                # happened to observe, not a compile-time fact.  Record the
                # real origin (see Meta.shape_source_ids) instead of letting
                # it silently freeze into a fixed number downstream.
                dimensions = len(tuple(getattr(result, "shape", ()) or ()))
                primitive.ctx["shape_source_ids"] = tuple(
                    size_parent_id for _ in range(dimensions)
                )
        collection_owners = context.get("collection_owner_ids", frozenset())

        def collection_source(value_id: int) -> int | None:
            current = int(value_id)
            seen = set()
            while current not in seen:
                seen.add(current)
                if current in collection_owners:
                    return current
                if graph is None or current not in graph.G:
                    return None
                attributes = graph.G.nodes[current].get("attributes") or {}
                if attributes.get("producer_kind") not in {
                    "aggregate_materialization",
                    "loop_materialization",
                }:
                    return None
                sources = tuple(map(
                    int,
                    attributes.get("materialized_source_value_ids", ()),
                ))
                if len(sources) != 1:
                    return None
                current = sources[0]
            return None

        collection_parents = tuple(dict.fromkeys(
            owner
            for parent_id in parent_ids
            if (owner := collection_source(int(parent_id))) is not None
        ))
        params = primitive.ctx.get("params", {})
        raw_materialization_dim = params.get("dim", 0)
        materialization_dim = int(
            0
            if raw_materialization_dim is None
            else raw_materialization_dim
        )
        if (
            len(collection_parents) == 1
            and str(primitive.op) in {"stack", "cat", "concat"}
            and materialization_dim == 0
        ):
            # Only a planner-owned resident collection may cross this
            # producer boundary as a zero-copy view.  A Python list/tuple
            # observed during discovery proves nothing about storage
            # ownership.  Leading-axis stack/cat preserves the resident
            # collection's contiguous publication order; other axes require
            # a real numerical materialization producer.
            context["collection_materializations"][int(capture_id)] = (
                collection_parents[0]
            )
            return result
        # The generated ProcessGraph callsite supplies each parent ID beside
        # the exact Python value used for that operand.  Use those pairs
        # directly.  GradTape's parent list/slot convention is backend
        # bookkeeping and is not the ProcessGraph ABI.
        primitive_parents = tuple(sorted(
            primitive.parents,
            key=lambda parent: int(parent[1]),
        ))
        if primitive_parents:
            primitive_ids = {
                int(primitive_id)
                for primitive_id, _slot in primitive_parents
            }
            exact = []
            matched_graph_ids = set()
            matched_primitive_ids = set()
            for graph_id, parent_value in zip(
                parent_ids, parent_values
            ):
                primitive_id = tensor_identity(parent_value)
                if primitive_id not in primitive_ids:
                    continue
                exact.append((int(primitive_id), int(graph_id)))
                matched_graph_ids.add(int(graph_id))
                matched_primitive_ids.add(int(primitive_id))
            remaining_primitives = tuple(
                int(primitive_id)
                for primitive_id, _slot in primitive_parents
                if int(primitive_id) not in matched_primitive_ids
            )
            remaining_graph_ids = tuple(
                int(graph_id)
                for graph_id in parent_ids
                if int(graph_id) not in matched_graph_ids
            )
            # A scalar literal can be tensorized inside the AbstractTensor
            # operator, so its source Python value has no tape ID.  Once all
            # exact tensor operands are removed, the remaining one-to-one
            # socket is unambiguous.
            if len(remaining_primitives) == len(remaining_graph_ids):
                exact.extend(zip(
                    remaining_primitives,
                    remaining_graph_ids,
                ))
            if exact:
                context["step_input_ids"][int(capture_id)] = tuple(
                    exact
                )
    return result


def _diagnostic_value_summary(value: Any) -> str:
    """Describe one routed value without creating AbstractTensor operations."""

    if isinstance(value, type):
        return f"type:{value.__module__}.{value.__qualname__}"
    shape = getattr(value, "shape", None)
    if callable(shape):
        # Namespace objects may export an operation called ``shape``.  Mirror
        # the runtime tensorization rule so failure reporting cannot mask the
        # original exception while trying to iterate that function.
        shape = None
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    prefix = type(value).__name__
    if shape is not None:
        prefix += f"(shape={tuple(shape)},dtype={dtype},device={device})"
    data = getattr(value, "data", value)
    materialize = getattr(data, "numpy", None)
    if not callable(materialize):
        return prefix + f":{value!r}" if shape is None else prefix
    try:
        import numpy as np

        array = np.asarray(materialize())
        if not array.size:
            return prefix + "[empty]"
        finite = np.isfinite(array)
        finite_count = int(finite.sum())
        if finite_count:
            selected = array[finite]
            stats = (
                f"min={selected.min():.7g},max={selected.max():.7g},"
                f"mean={selected.mean():.7g}"
            )
        else:
            stats = "min=nan,max=nan,mean=nan"
        return (
            prefix
            + f"[{stats},finite={finite_count}/{array.size}]"
        )
    except Exception as error:
        return prefix + f"[diagnostic={type(error).__name__}:{error}]"


def _graph_source_binding_name(graph: Any, node_id: int) -> str | None:
    """Recover a dotted public-input path for a numerical boundary node."""

    data = graph.G.nodes[int(node_id)]
    if data.get("type") == "Input":
        return str(data.get("label"))
    if data.get("type") != "Attribute":
        return None
    parent = next(
        (
            int(candidate)
            for candidate, role in (data.get("parents") or ())
            if str(role) == "value"
        ),
        None,
    )
    attribute = getattr(data.get("expr_obj"), "attr", None)
    if parent is None or attribute is None:
        return None
    root = _graph_source_binding_name(graph, parent)
    return None if root is None else f"{root}.{attribute}"


def _bind_capture_tape(
    value: Any,
    tape: Any,
    cache: dict[int, Any] | None = None,
) -> Any:
    """Bind tensor storage to the one discovery tape without copying values."""

    if isinstance(value, tuple):
        return tuple(_bind_capture_tape(item, tape, cache) for item in value)
    if isinstance(value, list):
        return [_bind_capture_tape(item, tape, cache) for item in value]
    if isinstance(value, dict):
        return {
            key: _bind_capture_tape(item, tape, cache)
            for key, item in value.items()
        }
    if not isinstance(value, AbstractTensor):
        return value
    if getattr(value, "_tape", None) is tape:
        return value
    identity = id(value)
    if cache is not None and identity in cache:
        return cache[identity]
    rebound = type(value)(
        track_time=value.track_time,
        requires_grad=value.requires_grad,
        tape=tape,
    )
    rebound.data = value.data
    if cache is not None:
        cache[identity] = rebound
    return rebound


class _SourceReturnSignal(Exception):
    """Internal non-error transfer for a compiled Python ``return``."""

    def __init__(self, value: Any):
        super().__init__()
        self.value = value


def _tensorize_graph_input(value: Any, *, device: Any) -> Any:
    """Move array-shaped public inputs onto the selected AbstractTensor backend."""

    def has_array_protocol(candidate: Any) -> bool:
        return (
            isinstance(candidate, np.ndarray)
            or callable(getattr(candidate, "__array__", None))
            or hasattr(candidate, "__cuda_array_interface__")
            or callable(getattr(candidate, "__dlpack__", None))
            or (
                getattr(candidate, "dtype", None) is not None
                and getattr(candidate, "shape", None) is not None
                and not callable(getattr(candidate, "shape", None))
            )
        )

    if isinstance(value, AbstractTensor):
        return value
    if isinstance(value, np.dtype):
        # NumPy dtype descriptors misleadingly expose ``shape == ()`` even
        # though they are type metadata, not zero-dimensional array payloads.
        # Dtype expressions belong to the AbstractTensor program and must
        # arrive at calls such as ``zeros(..., dtype=octets.dtype)`` unchanged.
        # Uploading the descriptor would manufacture an object-dtype tensor
        # and make the compiler fight the tensor operation's own type rules.
        return value
    if isinstance(value, _CompiledStructuralObject):
        # Dataclass/closure state can contain tensor fields and therefore
        # expose a derived ``shape`` through structural resolution.  The
        # aggregate itself is not an ndarray and must stay decomposable into
        # its named fields; coercing it produces an object-dtype array and
        # destroys the compiler-visible packet/closure schema.
        return value
    if isinstance(value, type):
        # A retained Python class is a structural reference the coordinator
        # calls through.  Its ``shape`` attribute is the unbound descriptor of
        # its instances, not the extent of an array-shaped input.
        return value
    if isinstance(value, ModuleType):
        return value
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict) and not callable(value):
        replacements = {}
        for name, field_value in fields.items():
            if isinstance(field_value, (type, ModuleType)) or callable(
                field_value
            ):
                continue
            if isinstance(field_value, AbstractTensor):
                continue
            if isinstance(field_value, np.dtype):
                continue
            field_shape = getattr(field_value, "shape", None)
            if (
                field_shape is None
                or callable(field_shape)
                or not has_array_protocol(field_value)
            ):
                continue
            replacements[str(name)] = AbstractTensor.tensor(
                field_value,
                device=device,
            )
        if replacements:
            # Discovery must not rewrite the caller's live Python object. A
            # shallow structural copy retains its scalar configuration and
            # the preallocated storage beneath each tensor wrapper, while
            # method assignments describe arena state on the capture copy.
            resident = copy.copy(value)
            for name, field_value in replacements.items():
                setattr(resident, name, field_value)
            return resident
    shape = getattr(value, "shape", None)
    if shape is None or callable(shape):
        # Imported scientific modules can export a public function named
        # ``shape`` (SymPy is one example).  Attribute presence alone is not
        # an array protocol: a callable shape describes an operation supplied
        # by the namespace, not the resident extent of the namespace object.
        # Keep such coordinator values intact so nested closures can continue
        # to call module constructors such as ``sympy.Symbol``.
        return value
    if not has_array_protocol(value):
        # Domain objects may expose a structural ``shape`` through dynamic
        # attribute routing.  Shape alone is not an upload protocol.
        return value
    return AbstractTensor.tensor(value, device=device)


def _coordinate_scheduled_capture_impl(
    shell: Any,
    initial_values: dict[str | int, Any],
    *,
    device: Any = None,
    capture: bool = True,
    discovery_session: dict[str, Any] | None = None,
) -> Any:
    """Execute structural boundaries and capture each planned numeric region.

    This is the first-invocation coordinator for a function deployment shell.
    Python objects remain structural values; array-shaped public inputs become
    resident tensors.  External calls and control/container nodes run between
    topologically closed numerical regions, while those regions alone enter
    forward capture for backend compilation.
    """

    if capture:
        invocation_slot = shell._capture_invocations
        shell._capture_invocations += 1
    else:
        invocation_slot = (
            shell._execute_invocations
            % max(1, shell.planned_invocation_slots)
        )
        shell._execute_invocations += 1
    graph = shell.process_graph
    supplied = dict(initial_values)
    live_binding_unavailable = object()

    def live_function_module_binding(name: str) -> Any:
        """Read a callee global from its owning live Python module.

        Static bindings describe the namespace at planning time.  A program
        may deliberately populate module globals later (lazy imports are the
        usual case), so a compiled callee must consult its own module at the
        source-ordered point where the name is first read.  The function
        table's qualified identity keeps this lookup lexical and prevents
        globals from unrelated discovered modules from being merged.
        """

        function_table = getattr(graph, "function_table", None)
        function_reference = graph.G.graph.get("function_ref")
        if function_table is None or function_reference is None:
            return live_binding_unavailable
        try:
            qualified_name = str(
                function_table.entry(int(function_reference)).qualified_name
            )
        except (KeyError, TypeError, ValueError):
            return live_binding_unavailable
        qualified_candidates = [qualified_name]
        method_owner = graph.G.graph.get("method_owner")
        if method_owner is not None:
            map_ir = graph.G.graph.get("map_ir") or {}
            for object_schema in map_ir.get("objects", ()):
                if str(object_schema.get("class_name")) == str(method_owner):
                    qualified_candidates.append(
                        str(object_schema.get("class_identity", ""))
                    )
        for qualified_candidate in qualified_candidates:
            components = qualified_candidate.split(".")
            for stop in range(len(components), 0, -1):
                module = sys.modules.get(".".join(components[:stop]))
                if module is None:
                    continue
                namespace = vars(module)
                return namespace.get(name, live_binding_unavailable)
        return live_binding_unavailable

    values: dict[int, Any] = {
        int(key): value
        for key, value in supplied.items()
        if isinstance(key, int)
    }
    function_identities = graph.G.graph.get("identity_table", {}) or {}

    def record_static_input_fields(
        binding_name: str,
        value: Any,
        path: tuple[str, ...] = (),
        visiting: frozenset[int] = frozenset(),
    ) -> None:
        if value is None or isinstance(
            value, (bool, int, float, str, bytes)
        ):
            shell._capture_input_static_fields[
                (str(binding_name), tuple(path))
            ] = value
            return
        identity = id(value)
        if identity in visiting or len(path) >= 8:
            return
        fields = getattr(value, "__dict__", None)
        if not isinstance(fields, dict) or callable(value):
            return
        next_visiting = visiting | {identity}
        for field_name, field_value in fields.items():
            record_static_input_fields(
                binding_name,
                field_value,
                (*path, str(field_name)),
                next_visiting,
            )

    def bound_target_names(target: ast.AST | None) -> tuple[str, ...]:
        if target is None:
            return ()
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, ast.Starred):
            return bound_target_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                name
                for item in target.elts
                for name in bound_target_names(item)
            )
        return ()

    source_with_bindings = frozenset(
        name
        for statement in graph.G.graph.get("function_body", ())
        for member in ast.walk(statement)
        if isinstance(member, ast.With)
        for item in member.items
        for name in bound_target_names(item.optional_vars)
    )

    def has_deferred_local_definition(name: str, input_id: int) -> bool:
        """Whether source execution, rather than invocation, defines name."""

        return name in source_with_bindings or any(
            int(identity) != int(input_id)
            and int(identity) in graph.G
            and str(graph.G.nodes[int(identity)].get("type"))
            not in {"Input", "input"}
            for identity in function_identities.get(name, ())
        )

    for node_id, data in graph.G.nodes(data=True):
        if str(data.get("type")) not in {"Input", "input"}:
            continue
        name = str(
            (data.get("attributes") or {}).get(
                "binding_name",
                data.get("label", ""),
            )
        )
        if node_id not in values:
            binding_kind = (
                (data.get("attributes") or {}).get("binding_kind")
            )
            if binding_kind in {"loop", "exception"}:
                continue
            live_binding = live_function_module_binding(name)
            if (
                binding_kind == "external"
                and live_binding is not live_binding_unavailable
            ):
                values[node_id] = live_binding
                shell.static_python_bindings[name] = live_binding
                continue
            class_descriptor = graph.G.graph.get(
                "class_table",
                {},
            ).get(name)
            if binding_kind == "external" and class_descriptor is not None:
                values[node_id] = _CompiledStructuralClass(
                    name,
                    class_descriptor,
                )
                continue
            if (
                binding_kind == "external"
                and name in shell.static_python_bindings
            ):
                values[node_id] = shell.static_python_bindings[name]
                continue
            if (
                binding_kind == "external"
                and name not in supplied
                and has_deferred_local_definition(name, node_id)
            ):
                # Control-flow normalization can retain an Input-shaped
                # occurrence before its assignment producer (notably a
                # local assigned inside ``try``).  It is a Python local, not
                # an invocation requirement; its use will resolve from the
                # source frame after the defining statement executes.
                continue
            if name not in supplied:
                raise KeyError(
                    f"missing ProcessGraph input {name!r} in "
                    f"{graph.G.graph.get('function_name', '?')} at node "
                    f"{node_id}; binding_kind={binding_kind!r}"
                )
            supplied_value = supplied[name]
            supplied_shape = getattr(supplied_value, "shape", None)
            if (
                isinstance(supplied_value, AbstractTensor)
                or isinstance(supplied_value, np.ndarray)
                or callable(getattr(supplied_value, "__array__", None))
                or (
                    supplied_shape is not None
                    and not callable(supplied_shape)
                    and getattr(supplied_value, "dtype", None) is not None
                )
            ):
                # Public array parameters enter the selected capture device
                # at the function boundary.  Deferring this until the first
                # numerical region lets an enclosing structural call observe
                # NumPy while its callee observes AbstractTensor, changing
                # source branches such as dt_system._restore_type and baking
                # an incidental host conversion into the graph.
                values[node_id] = _tensorize_graph_input(
                    supplied_value, device=device
                )
            else:
                values[node_id] = supplied_value
            shell._capture_input_type_names.setdefault(
                str(name), set()
            ).add(type(values[node_id]).__qualname__)
            if isinstance(values[node_id], AbstractTensor):
                shell._capture_tensor_input_names.add(str(name))
            record_static_input_fields(str(name), values[node_id])
            # Keep the name against both the object and the storage under
            # it. The object is wrapped into an AbstractTensor before it is
            # ever used, so object identity alone does not survive to the
            # feed; the resident range does, which is the same correlation
            # _capture_feed_aliases uses.
            try:
                shell._capture_input_names[id(values[node_id])] = str(name)
                storage = _capture_storage_identity(values[node_id])
                if storage is not None:
                    shell._capture_input_storage[storage] = str(name)
            except AttributeError:
                pass

    # Source execution can prove a graph-inert expression live at runtime
    # (for example the selected arm of an IfExp), so this coordinator view is
    # intentionally mutable even though the planner returns a frozenset.
    inert_nodes = set(_inert_routing_nodes(graph))
    regions = tuple(
        zip(
            shell.deep_compilers,
            shell.dispatch_subgraphs,
            shell.ephemeral_callables,
        )
    )
    region_for_node: dict[int, int] = {}
    coordinator_override_nodes: set[int] = set()
    for region_index, (_compiler, subgraph, _ephemeral) in enumerate(regions):
        if region_index in shell.coordinator_region_indices:
            continue
        for node_id in subgraph.G.graph.get("deployment_nodes", ()):
            if node_id in region_for_node:
                raise RuntimeError(
                    f"ProcessGraph node {node_id} belongs to two dispatches"
                )
            region_for_node[node_id] = region_index
        deployment_nodes = tuple(
            subgraph.G.graph.get("deployment_nodes", ())
        )
        if any(
            graph.G.nodes[node_id].get("type") == "Indexed"
            and any(
                parent in graph.G
                and graph.G.nodes[parent].get("type")
                in {"Const", "const", "Constant"}
                and isinstance(
                    _constant_value(graph.G.nodes[parent]),
                    (str, bytes),
                )
                for parent, role in (
                    graph.G.nodes[node_id].get("parents") or ()
                )
                if str(role) == "index"
            )
            for node_id in deployment_nodes
            if node_id in graph.G
        ):
            coordinator_override_nodes.update(deployment_nodes)
        if any(
            graph.G.nodes[node_id].get("type") == "IndexedStore"
            and any(
                isinstance(values.get(int(parent)), (dict, list, set))
                for parent, role in (
                    graph.G.nodes[node_id].get("parents") or ()
                )
                if str(role) == "base"
            )
            for node_id in deployment_nodes
            if node_id in graph.G
        ):
            # A fused region cannot clone or shader-store a Python container.
            # Runtime input type is authoritative during discovery; route the
            # whole coupled region through coordinator mutation so no sibling
            # dispatch claims the same structural store first.
            coordinator_override_nodes.update(deployment_nodes)

    controlled_nodes = {
        parent
        for _node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.If)
        for parent, role in data.get("parents", ())
        if str(role) in {"body", "orelse"}
    }
    expression_nodes = {
        id(data.get("expr_obj")): node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    function_body = tuple(graph.G.graph.get("function_body", ()))
    source_returns: list[ast.Return] = []
    nonlocal_names: set[str] = set()

    class _CurrentFunctionReturnVisitor(ast.NodeVisitor):
        def visit_Return(self, statement):
            if statement.value is not None:
                source_returns.append(statement)

        def visit_Nonlocal(self, statement):
            nonlocal_names.update(map(str, statement.names))

        def visit_FunctionDef(self, statement):
            return None

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, statement):
            return None

        def visit_ClassDef(self, statement):
            return None

    return_visitor = _CurrentFunctionReturnVisitor()
    for body_statement in function_body:
        return_visitor.visit(body_statement)
    ordered_return_values = tuple(
        (graph.G.graph.get("return_value_nodes", {}) or {}).values()
    )
    checkpoint_return_value_nodes = {
        id(statement): int(value_node)
        for statement, value_node in zip(
            source_returns, ordered_return_values
        )
        if int(value_node) in graph.G
    }
    for _control_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.If):
            controlled_statements = (
                *expression.body,
                *expression.orelse,
            )
        elif isinstance(expression, ast.IfExp):
            controlled_statements = (
                expression.body,
                expression.orelse,
            )
        elif isinstance(expression, ast.Try):
            controlled_statements = (
                *expression.body,
                *expression.handlers,
                *expression.orelse,
                *expression.finalbody,
            )
        else:
            continue
        for statement in controlled_statements:
            for member in ast.walk(statement):
                controlled = expression_nodes.get(id(member))
                if controlled is not None:
                    controlled_nodes.add(controlled)
    controlled_nodes.update(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if data.get("type") == "Input"
        and (
            data.get("attributes") or {}
        ).get("binding_kind") in {"loop", "exception"}
    )
    for loop_binding in tuple(controlled_nodes):
        if (
            loop_binding in graph.G
            and graph.G.nodes[loop_binding].get("type") == "Input"
            and (
                graph.G.nodes[loop_binding].get("attributes") or {}
            ).get("binding_kind") == "loop"
        ):
            controlled_nodes.update(
                nx.descendants(graph.G, loop_binding)
            )
    for boolean_node, boolean_data in graph.G.nodes(data=True):
        if isinstance(boolean_data.get("expr_obj"), ast.BoolOp):
            controlled_nodes.add(boolean_node)
            controlled_nodes.update(
                nx.ancestors(graph.G, boolean_node)
            )
    loop_plans_by_node = {
        plan.loop.node_id: plan for plan in shell.loop_plans
    }
    nested_body_nodes_by_loop = {
        int(owner.loop.node_id): frozenset(
            int(body_node)
            for nested in shell.loop_plans
            if (
                int(nested.loop.node_id) != int(owner.loop.node_id)
                and int(nested.loop.node_id) in owner.loop.body_nodes
            )
            for body_node in nested.loop.body_nodes
        )
        for owner in shell.loop_plans
    }
    comprehension_owner_by_binding: dict[int, int] = {}
    comprehension_owner_by_generator: dict[int, int] = {}
    loop_owner_by_binding = {
        int(binding): int(plan.loop.node_id)
        for plan in shell.loop_plans
        for _name, binding in plan.loop.target_bindings
    }
    for plan in shell.loop_plans:
        loop_expression = graph.G.nodes[
            plan.loop.node_id
        ].get("expr_obj")
        if not isinstance(loop_expression, ast.comprehension):
            continue
        owner = next(
            (
                successor
                for successor in graph.G.successors(plan.loop.node_id)
                if isinstance(
                    graph.G.nodes[successor].get("expr_obj"),
                    (
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                        ast.GeneratorExp,
                    ),
                )
            ),
            None,
        )
        if owner is not None:
            comprehension_owner_by_generator[plan.loop.node_id] = owner
            for _name, binding in plan.loop.target_bindings:
                comprehension_owner_by_binding[binding] = owner
    loop_invalidated_nodes: dict[int, set[int]] = {}
    for plan in shell.loop_plans:
        controlled_nodes.add(plan.loop.node_id)
        # Lexical normalization can fold or remove a body node after the loop
        # plan records its source identity.  The surviving graph is the
        # executable authority; stale source ids must not be handed to
        # NetworkX as traversal roots.
        live_body_nodes = {
            int(body_node)
            for body_node in plan.loop.body_nodes
            if int(body_node) in graph.G
        }
        controlled_nodes.update(live_body_nodes)
        invalidated = set(live_body_nodes)
        for body_node in live_body_nodes:
            invalidated.update(nx.descendants(graph.G, body_node))
        # Retained-loop backedges make the graph cyclic.  A raw descendants
        # walk can therefore wrap through the loop and reach invariant
        # producers that dominate the control node.  Clearing those values
        # replays effectful work performed before the loop (for example the
        # dt system's completed superstep while iterating its attempt log).
        # Body identities remain iteration-owned; other ancestors of the
        # loop control are already-computed invariants.
        invariant_ancestors = (
            set(nx.ancestors(graph.G, plan.loop.node_id))
            - live_body_nodes
        )
        invalidated.difference_update(invariant_ancestors)
        loop_invalidated_nodes[plan.loop.node_id] = invalidated
        # The planner loop owns the complete work cone fed by its body,
        # including numerical regions that consume the induction binding.
        # Leaving descendants in the ordinary topological sweep evaluates
        # them before the loop has supplied that binding.
        controlled_nodes.update(invalidated)
        controlled_nodes.update(
            binding for _name, binding in plan.loop.target_bindings
        )
        for successor in graph.G.successors(plan.loop.node_id):
            if isinstance(
                graph.G.nodes[successor].get("expr_obj"),
                (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
            ):
                controlled_nodes.update(nx.ancestors(graph.G, successor))
    loop_runtime_region_indices = {
        loop_node: tuple(
            region_index
            for region_index, (_compiler, subgraph, _ephemeral)
            in enumerate(regions)
            if set(
                subgraph.G.graph.get("deployment_nodes", ())
            ).intersection(invalidated)
            or set(
                subgraph.G.graph.get("deployment_outputs", ())
            ).intersection(invalidated)
        )
        for loop_node, invalidated in loop_invalidated_nodes.items()
    }
    downstream_loop_nodes = {
        candidate
        for candidate in loop_plans_by_node
        if any(
            owner != candidate
            and owner in nx.ancestors(graph.G, candidate)
            for owner in loop_plans_by_node
        )
    }
    has_comprehension_owner = any(
        isinstance(
            graph.G.nodes[loop_node].get("expr_obj"),
            ast.comprehension,
        )
        for loop_node in loop_plans_by_node
    )
    completed_regions: set[int] = set()
    active_nodes: set[int] = set()
    active_exceptions: dict[int, BaseException] = {}
    # One source function owns one discovery tape, and only the explicit
    # compilation/discovery invocation may create it.  Planned regions retain
    # separate output/feed cut points into that tape, but loops, frames and
    # repeated calls never create new tapes or capture observed values as new
    # constants.  Compilation consumes these cut points and then destroys the
    # tape before any installed shell can execute.
    #
    # In particular, do not "helpfully" create a tape for capture=False.
    # That mode is an ordinary coordinator execution and cannot rediscover,
    # extend, or rebuild the program.  Runtime must enter the installed shell;
    # retaining this non-capture path for diagnostics does not grant it any
    # compilation authority.
    discovery_tape = None
    if capture:
        if discovery_session is not None:
            # Nested source shells are compartments in the one source program,
            # not independently observed programs.  They record into the
            # owner's already-existing tape and may only contribute planner
            # cut points.  Creating or lowering another tape here would turn
            # function boundaries into executions and make the captured
            # example, rather than the planner, determine program count.
            discovery_tape = discovery_session["tape"]
            shell._discovery_tape = discovery_tape
            shell._discovery_tape_owner = False
        elif shell._discovery_tape is not None:
            discovery_tape = shell._discovery_tape
            discovery_session = shell._discovery_session
        else:
            if shell._discovery_tape_complete:
                raise RuntimeError(
                    "compiled source program attempted to create another "
                    "discovery tape"
                )
            discovery_tape = GradTape()
            shell._discovery_tape = discovery_tape
            shell._discovery_tape_owner = True
            shell._discovery_tape_creations += 1
            if shell._discovery_tape_creations != 1:
                raise RuntimeError(
                    "one source program created more than one discovery tape"
                )
            discovery_session = {
                "owner": shell,
                "tape": discovery_tape,
                "bindings": {},
                "lowered_program": None,
            }
            shell._discovery_session = discovery_session
    tape_bindings: dict[int, Any] = (
        discovery_session["bindings"]
        if capture and discovery_session is not None
        else {}
    )
    shell._discovery_tape_bindings = tape_bindings
    feed_maps = list(shell.forward_feed_ids)
    aggregate_feed_maps = list(shell.forward_aggregate_feed_paths)
    captured_subgraphs = list(shell.forward_subgraphs)
    captured_compilers = list(shell.forward_compilers)
    captured_region_indices = list(shell.forward_region_indices)
    captured_output_values = list(shell.forward_output_values)
    captured_region_node_ids = list(
        shell.forward_region_capture_node_ids
    )
    captured_region_planned_ids = list(
        shell.forward_region_planned_capture_ids
    )
    captured_region_planned_input_ids = list(
        shell.forward_region_planned_input_ids
    )
    captured_region_slots: dict[int, int] = {
        int(
            region_index
            if not isinstance(region_index, tuple)
            else region_index[-2]
        ): slot
        for slot, region_index in enumerate(captured_region_indices)
    }
    region_occurrences: dict[int, int] = {}

    def is_shader_internal_node(node_id: int) -> bool:
        region_index = region_for_node.get(node_id)
        if region_index is None:
            return False
        subgraph = regions[region_index][1]
        return (
            node_id in subgraph.G.graph.get("deployment_nodes", ())
            and node_id
            not in subgraph.G.graph.get("deployment_outputs", ())
        )

    def evaluate_region(region_index: int) -> None:
        if region_index in completed_regions:
            return
        occurrence = region_occurrences.get(region_index, 0)
        region_occurrences[region_index] = occurrence + 1
        region_key = (invocation_slot, region_index, occurrence)
        compiler, subgraph, ephemeral = regions[region_index]
        inputs: dict[str, Any] = {}
        for input_id in subgraph.G.graph["deployment_inputs"]:
            input_value = evaluate_node(input_id)
            tensorized_input = _tensorize_graph_input(
                input_value,
                device=device,
            )
            if tensorized_input is not input_value:
                # Object-field projections become numerical region feeds only
                # after structural attribute evaluation.  Public inputs were
                # tensorized earlier, but these late projections must join the
                # same resident/tape path or ndarray methods execute invisibly
                # in Python during discovery.
                values[input_id] = tensorized_input
                input_value = tensorized_input
            source_binding = _graph_source_binding_name(graph, input_id)
            if source_binding is not None:
                shell._capture_input_names[id(input_value)] = source_binding
                storage = _capture_storage_identity(input_value)
                if storage is not None:
                    shell._capture_input_storage[storage] = source_binding
            name = _compiler_input_name(
                subgraph.G.nodes[input_id]["label"]
            )
            inputs[name] = input_value

        deployment_nodes = tuple(
            subgraph.G.graph.get("deployment_nodes", ())
        )
        has_structural_store = any(
            str(
                graph.G.nodes[candidate].get("op")
                or graph.G.nodes[candidate].get("type")
            ) == "IndexedStore"
            and any(
                str(role) == "base"
                and isinstance(values.get(int(parent)), (dict, list, set))
                for parent, role in (
                    graph.G.nodes[candidate].get("parents") or ()
                )
            )
            for candidate in deployment_nodes
            if candidate in graph.G
        )
        has_structural_container_call = any(
            isinstance(expression := graph.G.nodes[candidate].get("expr_obj"), ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and (
                any(
                    str(role) in {"operand", "receiver"}
                    and isinstance(values.get(int(parent)), (dict, list, set))
                    for parent, role in (
                        graph.G.nodes[candidate].get("parents") or ()
                    )
                )
                or (
                    isinstance(expression.func.value, ast.Name)
                    and isinstance(
                        source_binding_values.get(expression.func.value.id),
                        (dict, list, set),
                    )
                )
            )
            for candidate in deployment_nodes
            if candidate in graph.G
        )
        has_structural_indexed_read = any(
            str(
                graph.G.nodes[candidate].get("op")
                or graph.G.nodes[candidate].get("type")
            ) == "Indexed"
            and any(
                str(role) == "base"
                and isinstance(
                    values.get(int(parent)), (dict, list, tuple, set)
                )
                for parent, role in (
                    graph.G.nodes[candidate].get("parents") or ()
                )
            )
            for candidate in deployment_nodes
            if candidate in graph.G
        )
        if (
            has_structural_store
            or has_structural_container_call
            or has_structural_indexed_read
        ):
            shell.coordinator_region_indices.add(region_index)
            coordinator_override_nodes.update(deployment_nodes)
            active_nodes.difference_update(map(int, deployment_nodes))
            for deployment_node in deployment_nodes:
                evaluate_node(int(deployment_node))
            completed_regions.add(region_index)
            return

        operations = [
            str(subgraph.G.nodes[node_id].get("op") or
                subgraph.G.nodes[node_id].get("type"))
            for node_id in subgraph.G.graph.get("deployment_nodes", ())
        ]
        label = f"region {region_index}: " + " -> ".join(operations)
        if shell._profiler.verbose:
            shell._profiler.trace(
                path=shell.profile_path,
                section="region-input",
                label=label,
                fields={
                    name: _diagnostic_value_summary(value)
                    for name, value in inputs.items()
                },
            )

        def run_region() -> None:
            captured_program = (
                shell.captured_region_programs.get(region_index)
                or shell.captured_region_programs.get(region_key)
            )
            if captured_program is not None:
                runtime_region_feeds = {
                    input_id: evaluate_node(input_id)
                    for input_id in subgraph.G.graph["deployment_inputs"]
                }
                missing_feed_ids = [
                    feed_id
                    for feed_id in ordered_feed_ids(captured_program.program)
                    if feed_id not in runtime_region_feeds
                ]
                unique_aliases = _unique_runtime_feed_aliases(
                    captured_program.program,
                    missing_feed_ids,
                    runtime_region_feeds,
                )
                runtime_region_feeds.update(unique_aliases)
                missing_feed_ids = [
                    feed_id
                    for feed_id in missing_feed_ids
                    if feed_id not in unique_aliases
                ]
                if shell._profiler.verbose:
                    shell._profiler.trace(
                        path=shell.profile_path,
                        section="feed-resolution",
                        label=label,
                        fields={
                            "program": tuple(
                                ordered_feed_ids(captured_program.program)
                            ),
                            "captured": tuple(
                                sorted(captured_program.feeds)
                            ),
                            "runtime": tuple(
                                sorted(runtime_region_feeds)
                            ),
                            "unresolved": tuple(missing_feed_ids),
                        },
                )
                if missing_feed_ids:
                    tensor_leaves = []
                    exactly_routed_tensors = {
                        id(value)
                        for input_id, value in runtime_region_feeds.items()
                        if (
                            input_id in captured_program.program.feeds
                            and isinstance(value, AbstractTensor)
                        )
                    }

                    def collect_tensor_leaves(value):
                        if isinstance(value, AbstractTensor):
                            if id(value) not in exactly_routed_tensors:
                                tensor_leaves.append(value)
                        elif isinstance(value, (tuple, list)):
                            for item in value:
                                collect_tensor_leaves(item)
                        elif isinstance(value, dict):
                            for item in value.values():
                                collect_tensor_leaves(item)

                    for input_id, value in runtime_region_feeds.items():
                        if input_id not in captured_program.program.feeds:
                            collect_tensor_leaves(value)
                    if len(tensor_leaves) != len(missing_feed_ids):
                        runtime_tensor_meta = []

                        def collect_runtime_meta(input_id, value):
                            if isinstance(value, AbstractTensor):
                                runtime_tensor_meta.append((
                                    input_id,
                                    tuple(value.shape),
                                    str(value.dtype),
                                ))
                            elif isinstance(value, (tuple, list)):
                                for item in value:
                                    collect_runtime_meta(input_id, item)
                            elif isinstance(value, dict):
                                for item in value.values():
                                    collect_runtime_meta(input_id, item)

                        for input_id, value in runtime_region_feeds.items():
                            collect_runtime_meta(input_id, value)
                        raise RuntimeError(
                            f"compiled region {region_index} ({tuple(operations)!r}) "
                            "cannot project structured tensor feeds exactly; "
                            f"missing_feeds={len(missing_feed_ids)} "
                            f"tensor_leaves={len(tensor_leaves)}; "
                            f"program_feed_ids={tuple(ordered_feed_ids(captured_program.program))!r}; "
                            f"runtime_feed_ids={tuple(sorted(runtime_region_feeds))!r}; "
                            f"unresolved_feed_ids={tuple(missing_feed_ids)!r}; "
                            "unresolved_meta="
                            f"{tuple((feed_id, captured_program.program.meta.get(feed_id)) for feed_id in missing_feed_ids)!r}; "
                            "unresolved_origins="
                            f"{tuple((feed_id, (captured_program.program.extras or {}).get('capture_feed_origins', {}).get(feed_id)) for feed_id in missing_feed_ids)!r}; "
                            f"runtime_tensor_meta={tuple(runtime_tensor_meta)!r}"
                        )
                    runtime_region_feeds.update(
                        zip(missing_feed_ids, tensor_leaves)
                    )
                try:
                    chunks = execute_captured_fused_program(
                        captured_program,
                        runtime_region_feeds,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"compiled shell {_shell_profile_name(shell)!r} "
                        f"id={id(shell)} region {region_index} "
                        f"({tuple(operations)!r}) "
                        "failed strict resident routing; "
                        f"cause={str(error)!r}; "
                        f"program_feeds={tuple(sorted(captured_program.program.feeds))!r}; "
                        f"runtime_feeds={tuple(sorted(runtime_region_feeds))!r}; "
                        f"step_inputs={tuple(tuple(step.input_ids) for step in captured_program.program.steps)!r}; "
                        f"stages={len(captured_program.stages)}"
                    ) from error
                for output_id in subgraph.G.graph["deployment_outputs"]:
                    output_name = f"value_{output_id}"
                    if output_name not in chunks:
                        def structural_value(structural_id):
                            if structural_id in values:
                                return values[structural_id]
                            data = subgraph.G.nodes[structural_id]
                            expression = data.get("expr_obj")
                            structural_parents = data.get("parents", ())
                            if isinstance(expression, ast.Constant):
                                result = expression.value
                            elif isinstance(expression, ast.Attribute):
                                parent = structural_value(
                                    structural_parents[0][0]
                                )
                                result = getattr(parent, expression.attr)
                            elif isinstance(expression, ast.Subscript):
                                base_id = next(
                                    parent
                                    for parent, role in structural_parents
                                    if str(role) == "base"
                                )
                                base = structural_value(base_id)
                                if isinstance(base, AbstractTensor):
                                    raise RuntimeError(
                                        "tensor indexing output was omitted "
                                        "from its compiled region"
                                    )
                                indices = [
                                    structural_value(parent)
                                    for parent, role in structural_parents
                                    if str(role) == "index"
                                ]
                                index = (
                                    indices[0]
                                    if len(indices) == 1
                                    else tuple(indices)
                                )
                                if isinstance(
                                    base, (dict, list, tuple, set)
                                ):
                                    index = coordinator_index(index)
                                result = base[index]
                            elif isinstance(expression, ast.BinOp):
                                operands = [
                                    structural_value(parent)
                                    for parent, _role in structural_parents
                                ]
                                if any(
                                    isinstance(value, AbstractTensor)
                                    for value in operands
                                ):
                                    raise RuntimeError(
                                        "tensor arithmetic output was omitted "
                                        "from its compiled region"
                                    )
                                binary = {
                                    ast.Add: operator.add,
                                    ast.Sub: operator.sub,
                                    ast.Mult: operator.mul,
                                    ast.Div: operator.truediv,
                                    ast.FloorDiv: operator.floordiv,
                                    ast.Mod: operator.mod,
                                }.get(type(expression.op))
                                if binary is None:
                                    raise KeyError(output_name)
                                result = binary(*operands)
                            else:
                                result = evaluate_node(structural_id)
                            values[structural_id] = result
                            return result

                        values[output_id] = structural_value(output_id)
                        continue
                    output = GLSLTensorOperations()
                    output.data = chunks[output_name]
                    values[output_id] = output
                completed_regions.add(region_index)
                return

            function = (
                shell.compiled_dispatch_functions[region_index]
                if shell.compiled_dispatch_functions
                else ephemeral
            )
            record_region = capture and region_index not in captured_region_slots
            if record_region:
                # The ProcessGraph planner already owns repetition,
                # cardinality, carried values and back-pressure.  Discovery
                # records exactly one primitive implementation exemplar for
                # this planned region.  Recording later loop occurrences
                # would unroll one observed execution into the program,
                # inflate the tape, and make runtime values decide program
                # structure.  Later occurrences may execute only to complete
                # this one discovery invocation; they cannot add tape nodes.
                capture_context = autograd.forward_capture(discovery_tape)
            else:
                capture_context = autograd.forward_observation()
            region_nodes_before = (
                set(discovery_tape._nodes)
                if capture and discovery_tape is not None
                else set()
            )
            with capture_context as tape:
                bound_inputs = (
                    {
                        name: _bind_capture_tape(value, tape, tape_bindings)
                        for name, value in inputs.items()
                    }
                    if capture
                    else inputs
                )
                planned_capture = {
                    "tape": discovery_tape,
                    "graph": graph,
                    "shell": shell,
                    "node_capture_ids": {},
                    "step_input_ids": {},
                    "collection_materializations": (
                        shell.forward_planned_collection_materializations
                    ),
                    "value_aliases": (
                        shell.compiled_process_graph_aliases
                    ),
                    "collection_owner_ids": frozenset(
                        int(collection_id)
                        for reduction in shell.loop_shader_reductions
                        if reduction.control_program is not None
                        for (
                            _source_id,
                            collection_id,
                            _induction,
                            _start,
                        ) in reduction.control_program.collection_bindings
                    ),
                }
                observer_token = (
                    _planned_capture_context.set(planned_capture)
                    if record_region
                    else None
                )
                try:
                    results = function(**bound_inputs)
                except Exception as error:
                    raise RuntimeError(
                        "ProcessGraph numerical region failed in "
                        f"{graph.G.graph.get('function_name', '?')} "
                        f"region {region_index}; "
                        f"nodes={subgraph.G.graph.get('deployment_nodes', ())!r}; "
                        f"inputs={subgraph.G.graph.get('deployment_inputs', ())!r}; "
                        f"outputs={subgraph.G.graph.get('deployment_outputs', ())!r}; "
                        f"bound={tuple((name, _diagnostic_value_summary(value)) for name, value in bound_inputs.items())!r}"
                    ) from error
                finally:
                    if observer_token is not None:
                        _planned_capture_context.reset(observer_token)
            if not isinstance(results, tuple):
                results = (results,)
            returned = dict(zip(compiler._outs, results))
            for output_id, store_id in zip(
                subgraph.G.graph["deployment_outputs"],
                subgraph.G.graph["deployment_store_nodes"],
            ):
                values[output_id] = returned[store_id]

            completed_regions.add(region_index)
            tensor_outputs_captured = any(
                isinstance(values.get(output_id), AbstractTensor)
                for output_id in subgraph.G.graph["deployment_outputs"]
            )
            if shell._profiler.verbose:
                shell._profiler.trace(
                    path=shell.profile_path,
                    section="capture-state",
                    label=label,
                    fields={
                        "record_region": record_region,
                        "tape_nodes": (
                            len(tape._nodes) if tape is not None else 0
                        ),
                        "tensor_outputs": tensor_outputs_captured,
                        "output_tape_state": tuple(
                            (
                                int(output_id),
                                id(values.get(output_id)),
                                (
                                    id(values.get(output_id))
                                    in discovery_tape._nodes
                                    if isinstance(
                                        values.get(output_id),
                                        AbstractTensor,
                                    )
                                    else None
                                ),
                                (
                                    getattr(
                                        values.get(output_id),
                                        "_tape",
                                        None,
                                    )
                                    is discovery_tape
                                    if isinstance(
                                        values.get(output_id),
                                        AbstractTensor,
                                    )
                                    else None
                                ),
                            )
                            for output_id in subgraph.G.graph[
                                "deployment_outputs"
                            ]
                        ),
                        "shell_id": id(shell),
                    },
                )
            occurrence_node_ids = tuple(
                int(node_id)
                for node_id in discovery_tape._nodes
                if int(node_id) not in region_nodes_before
            ) if record_region else ()
            # Repeated executions of one planned region are implementation
            # observations of the same ProcessGraph program.  Keep the first
            # occurrence as the primitive lowering exemplar; the planner's
            # LoopBlock owns repetition and cardinality.  Appending later tape
            # occurrences here would unroll one discovery execution into the
            # compiled program and create duplicate SSA writers.
            if (
                record_region
                and discovery_tape._nodes
                and occurrence_node_ids
            ):
                # Region ownership comes from the AST/ProcessGraph schedule,
                # not from the Python shape of its return value.  A region
                # may return a tuple or an ordinary object whose tensor fields
                # are consumed later.  Requiring the immediate graph output
                # itself to be an AbstractTensor silently drops every SSA
                # definition hidden behind that OOP boundary and forces a
                # later consumer to rediscover the producer by walking the
                # tape backward.  Record the region's owned forward nodes now;
                # lowering below computes only the live cross-region values.
                captured_region_slots[region_index] = len(
                    captured_region_indices
                )
                captured_region_node_ids.append(occurrence_node_ids)
                captured_region_planned_ids.append(tuple(
                    (
                        int(graph_node_id),
                        tuple(map(int, capture_ids)),
                    )
                    for graph_node_id, capture_ids
                    in planned_capture["node_capture_ids"].items()
                ))
                captured_region_planned_input_ids.append(tuple(
                    sorted(
                        (
                            int(result_capture_id),
                            tuple(
                                (
                                    int(primitive_id),
                                    int(graph_id),
                                )
                                for primitive_id, graph_id
                                in positional_inputs
                            ),
                        )
                        for result_capture_id, positional_inputs
                        in planned_capture["step_input_ids"].items()
                    )
                ))
                feed_maps.append(
                    {
                        id(bound_inputs[
                            _compiler_input_name(
                                subgraph.G.nodes[input_id]["label"]
                            )
                        ]): input_id
                        for input_id in subgraph.G.graph[
                            "deployment_inputs"
                        ]
                        if isinstance(
                            bound_inputs[
                                _compiler_input_name(
                                    subgraph.G.nodes[input_id]["label"]
                                )
                            ],
                            AbstractTensor,
                        )
                    }
                )
                aggregate_paths = {}
                rebound_originals = {}
                for original_id, rebound in tape_bindings.items():
                    rebound_originals.setdefault(
                        id(rebound), []
                    ).append(int(original_id))

                def record_aggregate_paths(
                    graph_input_id: int,
                    value: Any,
                    path: tuple[object, ...] = (),
                ) -> None:
                    if isinstance(value, AbstractTensor):
                        if path:
                            leaf = (
                                int(graph_input_id),
                                tuple(path),
                                _capture_storage_identity(value),
                            )
                            aggregate_paths[id(value)] = leaf
                            rebound_aliases = {id(value)}
                            pending_aliases = [id(value)]
                            while pending_aliases:
                                rebound_id = pending_aliases.pop()
                                for original_id in (
                                    rebound_originals.get(
                                        rebound_id, ()
                                    )
                                ):
                                    if original_id in rebound_aliases:
                                        continue
                                    rebound_aliases.add(original_id)
                                    pending_aliases.append(original_id)
                            aggregate_paths.update({
                                int(alias_id): leaf
                                for alias_id in rebound_aliases
                            })
                            # Some primitive hooks retain the tensor's resident
                            # storage wrapper as their operand.  It is a direct
                            # alias of this aggregate leaf, not a new value.
                            # Record that identity now so lowering can erase
                            # the Python wrapper without fabricating a shell
                            # input later.
                            storage_value = getattr(value, "data", None)
                            if storage_value is not None:
                                aggregate_paths[id(storage_value)] = leaf
                        return
                    if isinstance(value, (tuple, list)):
                        for index, item in enumerate(value):
                            record_aggregate_paths(
                                graph_input_id,
                                item,
                                (*path, int(index)),
                            )
                    elif isinstance(value, Mapping):
                        for name, item in value.items():
                            record_aggregate_paths(
                                graph_input_id,
                                item,
                                (*path, name),
                            )
                    elif isinstance(getattr(value, "__dict__", None), dict):
                        for name, item in vars(value).items():
                            record_aggregate_paths(
                                graph_input_id,
                                item,
                                (*path, str(name)),
                            )

                for input_id in subgraph.G.graph["deployment_inputs"]:
                    name = _compiler_input_name(
                        subgraph.G.nodes[input_id]["label"]
                    )
                    record_aggregate_paths(
                        int(input_id), bound_inputs[name]
                    )
                aggregate_by_storage = {
                    storage: (graph_input_id, path, storage)
                    for graph_input_id, path, storage
                    in aggregate_paths.values()
                    if storage is not None
                }
                for value in values.values():
                    if not isinstance(value, AbstractTensor):
                        continue
                    storage = _capture_storage_identity(value)
                    leaf = aggregate_by_storage.get(storage)
                    if leaf is None:
                        continue
                    aggregate_paths[id(value)] = leaf
                    storage_value = getattr(value, "data", None)
                    if storage_value is not None:
                        aggregate_paths[id(storage_value)] = leaf
                aggregate_feed_maps.append(aggregate_paths)
                captured_subgraphs.append(subgraph)
                captured_compilers.append(compiler)
                captured_region_indices.append(region_index)
                captured_output_values.append({
                    f"result_{index}": values[output_id]
                    for index, output_id in enumerate(
                        subgraph.G.graph["deployment_outputs"]
                    )
                    if isinstance(values.get(output_id), AbstractTensor)
                })

        with _profile_event(
            shell,
            "capture" if capture else "dispatch",
            label,
            gpu=True,
        ):
            run_region()
        if shell._profiler.verbose:
            shell._profiler.trace(
                path=shell.profile_path,
                section="region-output",
                label=label,
                fields={
                    f"value_{output_id}": _diagnostic_value_summary(
                        values.get(output_id)
                    )
                    for output_id in subgraph.G.graph[
                        "deployment_outputs"
                    ]
                },
            )

    class _LoopBreakSignal(Exception):
        pass

    class _LoopContinueSignal(Exception):
        pass

    def evaluate_static_expression(expression: ast.AST) -> Any:
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            if expression.id in shell.static_python_bindings:
                return shell.static_python_bindings[expression.id]
            defaults = graph.G.graph.get("parameter_defaults") or {}
            if expression.id in defaults:
                return defaults[expression.id]
            if expression.id in supplied:
                value = supplied[expression.id]
                if value is None or isinstance(
                    value,
                    (list, tuple, dict, set),
                ):
                    return value
        if isinstance(expression, ast.Attribute):
            return getattr(
                evaluate_static_expression(expression.value),
                expression.attr,
            )
        if (
            isinstance(expression, ast.Call)
            and not expression.args
            and not expression.keywords
        ):
            function = evaluate_static_expression(expression.func)
            if callable(function):
                return function()
        if (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and len(expression.comparators) == 1
            and isinstance(expression.ops[0], (ast.Is, ast.IsNot))
        ):
            left = evaluate_static_expression(expression.left)
            right = evaluate_static_expression(expression.comparators[0])
            result = left is right
            return not result if isinstance(expression.ops[0], ast.IsNot) else result
        raise KeyError(ast.unparse(expression))

    def evaluate_source_container_item(
        source: ast.AST,
        fallback_node: int | None = None,
    ) -> Any:
        """Evaluate a container element at its exact lexical source point."""

        if isinstance(source, ast.Name):
            if source.id in source_binding_values:
                return source_binding_values[source.id]
            if source.id in supplied:
                return supplied[source.id]
        if isinstance(source, ast.Dict):
            result = {}
            for key, value in zip(source.keys, source.values):
                item = evaluate_source_container_item(
                    value, expression_nodes.get(id(value))
                )
                if key is None:
                    result.update(dict(item))
                else:
                    result[evaluate_source_container_item(
                        key, expression_nodes.get(id(key))
                    )] = item
            return result
        if isinstance(source, (ast.Tuple, ast.List, ast.Set)):
            items = [
                evaluate_source_container_item(
                    item, expression_nodes.get(id(item))
                )
                for item in source.elts
            ]
            if isinstance(source, ast.Tuple):
                return tuple(items)
            if isinstance(source, ast.Set):
                return set(items)
            return items
        if isinstance(source, (ast.IfExp, ast.BoolOp)) or (
            isinstance(source, ast.Compare)
            and len(source.ops) == 1
            and isinstance(source.ops[0], (ast.Is, ast.IsNot))
        ):
            return evaluate_reduced_control_expression(source)
        node = (
            fallback_node
            if fallback_node is not None
            else expression_nodes.get(id(source))
        )
        if node is not None:
            return evaluate_node(int(node))
        return evaluate_reduced_control_expression(source)

    def coordinator_index(value: Any) -> Any:
        """Project tensor scalars into Python container index semantics."""

        if isinstance(value, AbstractTensor):
            if tuple(value.shape) == ():
                return value.item()
            return value
        if isinstance(value, tuple):
            return tuple(coordinator_index(item) for item in value)
        if isinstance(value, slice):
            return slice(
                coordinator_index(value.start),
                coordinator_index(value.stop),
                coordinator_index(value.step),
            )
        return value

    def _resolve_reference_node(expression: ast.expr) -> int | None:
        """Find the live graph node a reference-typed expression resolves to.

        A bound-method call's receiver (``machine``, ``machine.runner``, ...)
        is a reference value, not a tensor -- it has no numeric identity to
        fall back on. The existing, narrower check this replaces
        (``id(expression) in graph.G``) asks only "does this exact AST node
        still happen to be a node in the graph", which silently fails
        whenever an earlier reduction pass legitimately rebuilt or pruned
        that specific node while the value it represents is still perfectly
        live under its name. ``identity_table`` already tracks every SSA
        version of every name for exactly this reason (it is the same
        mechanism ``live_function_module_binding`` above uses); grounding
        receiver resolution in it instead of a raw ``id()`` presence check
        means a receiver resolves by what it *is*, not by whether one
        specific AST node object happened to survive unchanged.
        """

        if isinstance(expression, ast.Name):
            candidates = graph.G.graph.get("identity_table", {}).get(
                expression.id, ()
            )
            for candidate in reversed(candidates):
                if int(candidate) in graph.G:
                    return int(candidate)
            return None
        if isinstance(expression, ast.Attribute):
            if id(expression) in graph.G:
                return id(expression)
            base_node = _resolve_reference_node(expression.value)
            if base_node is None:
                return None
            for successor in graph.G.successors(base_node):
                successor_expr = graph.G.nodes[successor].get("expr_obj")
                if (
                    isinstance(successor_expr, ast.Attribute)
                    and successor_expr.attr == expression.attr
                ):
                    return successor
            return None
        return None

    def evaluate_node(node_id: int) -> Any:
        if node_id in values:
            cached_data = graph.G.nodes[node_id]
            if str(cached_data.get("type")) in {"Input", "input"}:
                if node_id in generator_binding_values:
                    values[node_id] = generator_binding_values[node_id]
                    return values[node_id]
                cached_attributes = cached_data.get("attributes") or {}
                cached_name = str(
                    cached_attributes.get(
                        "binding_name",
                        cached_data.get("label", node_id),
                    )
                )
                if cached_name in source_binding_values:
                    values[node_id] = source_binding_values[cached_name]
                    return values[node_id]
            return values[node_id]
        if node_id in inert_nodes:
            # A name or attribute lookup nobody reads.  Producing it would
            # demand a materialized receiver and force an otherwise
            # shader-local intermediate to become a published region output.
            return None
        if node_id in active_nodes:
            raise RuntimeError(
                f"recursive runtime dependency at ProcessGraph node {node_id}; "
                f"shell={_shell_profile_name(shell)!r}; "
                f"active_nodes={tuple(sorted(active_nodes))!r}; "
                f"region={region_for_node.get(node_id)!r}"
            )
        active_nodes.add(node_id)
        try:
            data = graph.G.nodes[node_id]
            node_type = str(data.get("type"))
            parents = tuple(data.get("parents") or ())
            if node_type == "Indexed":
                base_parent = next(
                    (
                        parent
                        for parent, role in parents
                        if str(role) == "base"
                    ),
                    None,
                )
                available_base = values.get(base_parent)
                available_indices = tuple(
                    values[parent]
                    for parent, role in parents
                    if str(role) == "index" and parent in values
                )
                if isinstance(
                    available_base,
                    (dict, list, tuple, set),
                ) or any(
                    isinstance(index, (str, bytes))
                    for index in available_indices
                ):
                    # Runtime loop bindings can carry structural keys that
                    # were not literals during planning. Container routing is
                    # a type fact, not value specialization, and must be
                    # decided before a numerical region claims the node.
                    coordinator_override_nodes.add(node_id)
            region_index = (
                None
                if node_id in coordinator_override_nodes
                else region_for_node.get(node_id)
            )
            if region_index is not None:
                evaluate_region(region_index)
                if node_id not in values:
                    subgraph = regions[region_index][1]
                    if is_shader_internal_node(node_id):
                        raise RuntimeError(
                            "shader-local ProcessGraph node was requested as "
                            "a compartment boundary value; "
                            f"node={node_id}; region={region_index}; "
                            f"shell={_shell_profile_name(shell)!r}. "
                            "The region extractor must export every value "
                            "observed by structural execution."
                        )
                    completed_regions.discard(region_index)
                    evaluate_region(region_index)
                if node_id not in values:
                    raise RuntimeError(
                        f"ProcessGraph region {region_index} did not publish "
                        f"requested boundary output node {node_id}; "
                        f"shell={_shell_profile_name(shell)!r}; "
                        f"deployment_outputs="
                        f"{subgraph.G.graph.get('deployment_outputs', ())!r}"
                    )
                return values[node_id]

            expression = data.get("expr_obj")
            attributes = data.get("attributes") or {}

            if node_type in {"Input", "input"}:
                if node_id in generator_binding_values:
                    result = generator_binding_values[node_id]
                    values[node_id] = result
                    return result
                # Retained stores and attribute writes may still point at an
                # Input-shaped identity for a local whose value is selected
                # by source control.  Once that statement has executed, the
                # active source frame is the authority for the spelling.
                name = str(
                    attributes.get(
                        "binding_name",
                        data.get("label", node_id),
                    )
                )
                if name in source_binding_values:
                    result = source_binding_values[name]
                    values[node_id] = result
                    return result
                comprehension_owner = (
                    comprehension_owner_by_binding.get(node_id)
                )
                if comprehension_owner is not None:
                    evaluate_node(comprehension_owner)
                    if node_id in generator_binding_values:
                        result = generator_binding_values[node_id]
                        values[node_id] = result
                        return result
                loop_owner = loop_owner_by_binding.get(node_id)
                if (
                    loop_owner is not None
                    and loop_owner not in active_nodes
                ):
                    # Normalized LoopExit/value dependencies can request an
                    # inner loop's iterable before the flat evaluator has
                    # visited the outer ``for`` statement.  A target is a
                    # definition produced by its owner loop, never a public
                    # function input.  Execute that owner first, then resolve
                    # the binding from the exact iteration/source frame.
                    evaluate_node(loop_owner)
                    if node_id in generator_binding_values:
                        result = generator_binding_values[node_id]
                        values[node_id] = result
                        return result
                    if node_id in values:
                        return values[node_id]
                    if name in source_binding_values:
                        result = source_binding_values[name]
                        values[node_id] = result
                        return result
                raise KeyError(
                    f"missing ProcessGraph input {name!r} in "
                    f"{graph.G.graph.get('function_name', '?')} at node "
                    f"{node_id}; loop targets="
                    f"{tuple(plan.loop.target_bindings for plan in shell.loop_plans)!r}; "
                    f"active generator bindings={tuple(generator_binding_values)!r}"
                )
            if node_type in {"Const", "const", "Constant"}:
                result = _constant_value(data)
            elif node_type in {"LoopExit", "LoopResult"}:
                by_role = {str(role): parent for parent, role in parents}
                control_id = int(by_role["control"])
                evaluate_node(control_id)
                if attributes.get("result_kind") == "carried":
                    binding_name = str(attributes.get("binding_name", ""))
                    carried_initial = next(
                        (
                            int(initial)
                            for name, initial, _updated in (
                                loop_plans_by_node[control_id]
                                .loop.carried_bindings
                            )
                            if str(name) == binding_name
                        ),
                        None,
                    )
                    if carried_initial is not None and carried_initial in values:
                        result = values[carried_initial]
                    elif (
                        carried_initial is not None
                        and carried_initial in graph.G
                    ):
                        result = evaluate_node(carried_initial)
                    elif binding_name in source_binding_values:
                        result = source_binding_values[binding_name]
                    else:
                        result = evaluate_node(by_role["value"])
                else:
                    result = evaluate_node(by_role["value"])
            elif node_type in {"LoopStateTransition", "LoopStatePort"}:
                by_role = {str(role): parent for parent, role in parents}
                # Retained-loop effects execute as members of the loop body.
                # A state port is only a publication boundary for the state
                # after that control has completed; replaying the effect here
                # both applies it twice and attempts to read loop-local target
                # bindings after their owning loop has gone out of scope.
                loop_id = (attributes or {}).get("loop_id")
                if loop_id is not None:
                    evaluate_node(int(loop_id))
                elif node_type == "LoopStateTransition":
                    evaluate_node(by_role["effect"])
                result = evaluate_node(by_role["state"])
            elif node_type == "LoopAggregateResult":
                result = tuple(
                    evaluate_node(parent)
                    for parent in ordered_arguments(parents)
                )
            elif isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
                element_parents = [parent for parent, _role in parents]
                items = [
                    evaluate_source_container_item(
                        item,
                        element_parents[index]
                        if index < len(element_parents)
                        else None,
                    )
                    for index, item in enumerate(expression.elts)
                ]
                if isinstance(expression, ast.Tuple):
                    result = tuple(items)
                elif isinstance(expression, ast.Set):
                    result = set(items)
                else:
                    result = items
            elif isinstance(expression, ast.Dict):
                result = evaluate_source_container_item(expression, node_id)
            elif (
                isinstance(expression, ast.Attribute)
                and node_type != "SetAttr"
            ):
                parent = next(
                    (
                        parent
                        for parent, role in parents
                        if str(role) == "value"
                    ),
                    None,
                )
                if parent is None:
                    raise RuntimeError(
                        "attribute node has no receiver in "
                        f"{graph.G.graph.get('function_name', '?')} at "
                        f"node {node_id}: "
                        f"{ast.dump(expression, include_attributes=False)}; "
                        f"parents={parents!r}; attributes="
                        f"{(data.get('attributes') or {})!r}"
                    )
                receiver = evaluate_node(parent)
                if isinstance(receiver, _CompiledStructuralObject):
                    # A field read on a compiler-owned structural object is
                    # not a static Python fact to freeze into ``values``:
                    # the field is genuine per-call state (see
                    # ``_CompiledStructuralObject.state``), and a plain
                    # ``getattr`` here would capture whatever this one
                    # discovery trace happened to observe as if it were a
                    # constant -- the exact one-value trap that makes
                    # ``counter.value`` invisible to region capture. Keep
                    # the field's own identity live instead of collapsing
                    # it to a bare Python value.
                    result = receiver.state.get(expression.attr)
                else:
                    try:
                        result = getattr(receiver, expression.attr)
                    except AttributeError as exc:
                        raise AttributeError(
                            f"{graph.G.graph.get('function_name', '?')} "
                            f"attribute {expression.attr!r} at node "
                            f"{node_id} has receiver "
                            f"{type(receiver).__name__}; parent={parent}, "
                            f"parents={parents!r}"
                        ) from exc
                if node_type == "GetAttr":
                    shell.reference_operator_sequence.append(int(node_id))
            elif isinstance(expression, ast.Slice):
                parts = {
                    str(role): evaluate_node(parent)
                    for parent, role in parents
                }
                result = slice(
                    parts.get("lower"),
                    parts.get("upper"),
                    parts.get("step"),
                )
            elif isinstance(expression, ast.Starred):
                result = evaluate_node(parents[0][0])
            elif isinstance(expression, ast.FormattedValue):
                value = evaluate_node(parents[0][0])
                conversion = expression.conversion
                if conversion == ord("r"):
                    value = repr(value)
                elif conversion == ord("a"):
                    value = ascii(value)
                elif conversion == ord("s"):
                    value = str(value)
                result = format(value)
            elif isinstance(expression, ast.JoinedStr):
                result = "".join(
                    str(evaluate_node(parent))
                    for parent, _role in parents
                )
            elif node_type == "IndexedStore":
                by_role: dict[str, list[int]] = {}
                for parent, role in parents:
                    by_role.setdefault(str(role), []).append(parent)
                base = evaluate_node(by_role["base"][0])
                if isinstance(expression, ast.Subscript):
                    index = evaluate_reduced_control_expression(
                        expression.slice
                    )
                else:
                    indices = [
                        evaluate_node(parent)
                        for parent in by_role.get("index", ())
                    ]
                    index = (
                        indices[0]
                        if len(indices) == 1
                        else tuple(indices)
                    )
                value = evaluate_node(by_role["value"][0])
                if isinstance(base, (dict, list, tuple, set)):
                    index = coordinator_index(index)
                base[index] = value
                result = base
            elif node_type == "Indexed":
                base = evaluate_node(
                    next(
                        parent
                        for parent, role in parents
                        if str(role) == "base"
                    )
                )
                indices = [
                    evaluate_node(parent)
                    for parent, role in parents
                    if str(role) == "index"
                ]
                index = indices[0] if len(indices) == 1 else tuple(indices)
                if isinstance(base, (dict, list, tuple, set)):
                    index = coordinator_index(index)
                result = base[index]
            elif isinstance(expression, ast.BoolOp):
                if isinstance(expression.op, ast.And):
                    result = True
                    for parent, role in parents:
                        result = evaluate_node(parent)
                        if shell._profiler.verbose:
                            shell._profiler.trace(
                                path=shell.profile_path,
                                section="boolop",
                                label=f"node {node_id} {role}",
                                fields={
                                    "value": _diagnostic_value_summary(
                                        result
                                    )
                                },
                            )
                        if not result:
                            break
                else:
                    result = False
                    for parent, _role in parents:
                        result = evaluate_node(parent)
                        if result:
                            break
            elif isinstance(expression, ast.IfExp):
                by_role = {
                    str(role): parent for parent, role in parents
                }
                branch = "body" if bool(
                    evaluate_node(by_role["test"])
                ) else "orelse"
                result = evaluate_node(by_role[branch])
            elif (
                attributes.get("materialization_kind")
                in {"unrolled_loop", "retained_loop_aggregate"}
                and isinstance(
                    expression,
                    (
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                        ast.GeneratorExp,
                    ),
                )
            ):
                try:
                    materialized = [
                        evaluate_node(parent)
                        for parent in ordered_arguments(parents)
                    ]
                except KeyError:
                    # An evaporated structural comprehension can retain a
                    # cloned target-shaped Input after its loop binding was
                    # removed. Reconstruct from source semantics, whose
                    # comprehension frame publishes every destructured name.
                    result = evaluate_reduced_control_expression(expression)
                else:
                    if not materialized:
                        result = evaluate_reduced_control_expression(
                            expression
                        )
                    elif isinstance(expression, ast.SetComp):
                        result = set(materialized)
                    elif isinstance(expression, ast.DictComp):
                        result = dict(materialized)
                    elif isinstance(expression, ast.GeneratorExp):
                        # Keep a replayable finite sequence during discovery.
                        # Consumers such as any/all/sum and starred calls preserve
                        # their Python argument semantics without re-entering the
                        # evaporated generator control node.
                        result = tuple(materialized)
                    else:
                        result = materialized
            elif isinstance(
                expression,
                (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
            ):
                generator_nodes = [
                    parent
                    for parent, role in parents
                    if str(role) == "generators"
                ]
                element_nodes = {
                    str(role): parent
                    for parent, role in parents
                    if str(role) in {"elt", "key", "value"}
                }
                collected = []

                def evaluate_generator(index: int) -> None:
                    if index == len(generator_nodes):
                        if isinstance(expression, ast.DictComp):
                            collected.append((
                                evaluate_node(element_nodes["key"]),
                                evaluate_node(element_nodes["value"]),
                            ))
                        else:
                            collected.append(
                                evaluate_node(element_nodes["elt"])
                            )
                        return
                    generator_node = generator_nodes[index]
                    try:
                        plan = loop_plans_by_node[generator_node]
                    except KeyError as error:
                        raise RuntimeError(
                            "aggregate retained an evaporated generator "
                            f"in {graph.G.graph.get('function_name', '?')}: "
                            f"aggregate={node_id}, generator={generator_node}, "
                            f"planned={tuple(loop_plans_by_node)!r}, "
                            f"specializations="
                            f"{graph.G.graph.get('planner_specializations')!r}, "
                            f"attributes={attributes!r}"
                        ) from error
                    loop = plan.loop
                    iterable = evaluate_node(loop.iterable_node)
                    if shell._profiler.verbose:
                        shell._profiler.trace(
                            path=shell.profile_path,
                            section="comprehension",
                            label=f"generator {generator_node}",
                            fields={
                                "iterable": _diagnostic_value_summary(iterable),
                                "targets": loop.target_bindings,
                            },
                        )
                    iterator = iter(iterable)
                    for item in iterator:
                        targets = loop.target_bindings
                        items_by_name = dict(_destructure_loop_target(
                            expression.generators[index].target,
                            item,
                        ))
                        # A later generator in the same comprehension may
                        # consume a destructured name that never enters a
                        # numerical target binding.  Publish the complete
                        # Python target frame, just as retained ``for`` source
                        # execution does.
                        source_binding_values.update(items_by_name)
                        invalidated = set()
                        assignments = tuple(
                            (
                                (name, binding),
                                items_by_name[str(name)],
                            )
                            for name, binding in targets
                        )
                        for (_name, binding), _value in assignments:
                            invalidated.update(
                                nx.descendants(graph.G, binding)
                            )
                        invalidated.discard(node_id)
                        for dependent in invalidated:
                            values.pop(dependent, None)
                        for (_name, binding), value in assignments:
                            generator_binding_values[binding] = value
                        values.update(generator_binding_values)
                        for region_index in loop_runtime_region_indices.get(
                            generator_node,
                            (),
                        ):
                            completed_regions.discard(region_index)
                        if all(
                            bool(evaluate_node(condition))
                            for condition in loop.condition_nodes
                        ):
                            evaluate_generator(index + 1)

                evaluate_generator(0)
                if isinstance(expression, ast.DictComp):
                    result = dict(collected)
                elif isinstance(expression, ast.SetComp):
                    result = set(collected)
                else:
                    result = collected
                if capture and isinstance(
                    expression, (ast.ListComp, ast.SetComp)
                ):
                    # Retain the actual wrappers until the one lowering.
                    # Integer-only identity tracking is invalid here because
                    # temporary wrappers can die during a long discovery run
                    # and CPython may reuse their addresses for unrelated
                    # values before the collection is materialized.
                    shell.collection_observations[int(node_id)] = tuple(
                        item
                        for item in result
                        if isinstance(item, AbstractTensor)
                    )
            elif isinstance(expression, ast.comprehension):
                owner = comprehension_owner_by_generator.get(node_id)
                if owner is None:
                    raise RuntimeError(
                        "comprehension control node has no planner-owned "
                        f"expression in {graph.G.graph.get('function_name', '?')} "
                        f"at node {node_id}"
                    )
                result = evaluate_node(owner)
            elif isinstance(expression, ast.Compare):
                by_role = {
                    str(role): int(parent) for parent, role in parents
                }
                source_operands = (
                    expression.left,
                    *expression.comparators,
                )
                operands = []
                for index, source_operand in enumerate(source_operands):
                    role = "lhs" if index == 0 else "rhs"
                    parent = by_role.get(role)
                    if parent is not None and (
                        index == 0 or len(expression.ops) == 1
                    ):
                        operands.append(evaluate_node(parent))
                    else:
                        operands.append(
                            evaluate_static_expression(source_operand)
                        )
                result = True
                for index, comparison in enumerate(expression.ops):
                    comparator = {
                        ast.Is: operator.is_,
                        ast.IsNot: operator.is_not,
                        ast.Eq: operator.eq,
                        ast.NotEq: operator.ne,
                        ast.Lt: operator.lt,
                        ast.LtE: operator.le,
                        ast.Gt: operator.gt,
                        ast.GtE: operator.ge,
                        ast.In: lambda left, right: left in right,
                        ast.NotIn: lambda left, right: left not in right,
                    }.get(type(comparison))
                    if comparator is None:
                        raise NotImplementedError(
                            "unsupported structural comparison "
                            f"{type(comparison).__name__}"
                        )
                    if not comparator(
                        operands[index],
                        operands[index + 1],
                    ):
                        result = False
                        break
            elif isinstance(expression, ast.UnaryOp):
                operand = evaluate_node(parents[0][0])
                unary = {
                    ast.USub: operator.neg,
                    ast.UAdd: operator.pos,
                    ast.Not: operator.not_,
                    ast.Invert: operator.invert,
                }.get(type(expression.op))
                if unary is None:
                    raise NotImplementedError(
                        "unsupported structural unary operator "
                        f"{type(expression.op).__name__}"
                    )
                result = unary(operand)
            elif isinstance(expression, (ast.BinOp, ast.AugAssign)):
                operands = [
                    evaluate_node(parent)
                    for parent, _role in parents
                ]
                binary = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv,
                    ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                    ast.LShift: operator.lshift,
                    ast.RShift: operator.rshift,
                    ast.BitOr: operator.or_,
                    ast.BitXor: operator.xor,
                    ast.BitAnd: operator.and_,
                    ast.MatMult: operator.matmul,
                }.get(type(expression.op))
                if binary is None:
                    raise NotImplementedError(
                        "unsupported structural binary operator "
                        f"{type(expression.op).__name__}"
                    )
                if len(operands) != 2:
                    raise RuntimeError(
                        "structural binary expression requires two operands "
                        f"in {graph.G.graph.get('function_name', '?')} at "
                        f"node {node_id}: "
                        f"{ast.dump(expression, include_attributes=False)}; "
                        f"parents={parents!r}"
                    )
                result = binary(operands[0], operands[1])
                if not isinstance(result, AbstractTensor):
                    # A tensor-bearing structural op still enters the tape
                    # (see _observe_process_graph_node); this branch runs for
                    # plain values too, and a plain value never touches the
                    # tape at all.  Record it the same way a reference
                    # operator is recorded, in exact order, so an arithmetic
                    # step feeding a field write is a real internal step
                    # instead of looking like a phantom external input.
                    shell.reference_operator_sequence.append(int(node_id))
            elif isinstance(expression, ast.Call):
                if (
                    isinstance(expression.func, ast.Attribute)
                    and isinstance(expression.func.value, ast.Name)
                    and expression.func.value.id in source_binding_values
                ):
                    receiver_parent = next(
                        (
                            int(parent)
                            for parent, role in parents
                            if str(role) in {"operand", "receiver"}
                        ),
                        None,
                    )
                    if receiver_parent is not None:
                        # The receiver visible at this lexical callsite is
                        # authoritative.  A normalized receiver edge can run
                        # through a later LoopExit carrying the same spelling,
                        # creating a false dependency cycle such as an outer
                        # ``mapping.items()`` requiring its inner loop before
                        # the outer target has been assigned.  Seed the exact
                        # receiver identity before traversing call parents;
                        # the ordinary callable/method routing below remains
                        # responsible for compiling or invoking the method.
                        values[receiver_parent] = source_binding_values[
                            expression.func.value.id
                        ]
                for parent, _role in parents:
                    evaluate_node(parent)
                attributes = data.get("attributes") or {}
                static_arguments = {
                    role: _static_python_value(
                        shell.static_python_bindings,
                        reference,
                    )
                    for role, reference in (
                        attributes.get("static_call_arguments") or {}
                    ).items()
                }
                args, kwargs = _call_arguments(
                    parents,
                    values,
                    static_arguments,
                    graph,
                )
                # Exact source-name arguments observe the active lexical
                # frame. A reduced edge may otherwise alias a saved value to
                # a later write bearing the same spelling.
                if not any(
                    isinstance(argument, ast.Starred)
                    for argument in expression.args
                ):
                    # Normalization can retain multiple SSA parents for one
                    # source spelling after loop-carried/container rebinding.
                    # Those are candidate identities, not additional Python
                    # arguments.  Source syntax owns call arity; keep exactly
                    # one planned slot per written positional argument before
                    # applying lexical/live-value corrections below.
                    source_args = list(args[: len(expression.args)])
                    for index, argument in enumerate(expression.args):
                        if (
                            isinstance(argument, ast.Name)
                            and argument.id in source_binding_values
                            and index < len(source_args)
                        ):
                            source_args[index] = source_binding_values[
                                argument.id
                            ]
                        elif (
                            isinstance(argument, ast.Name)
                            and argument.id not in supplied
                            and index < len(source_args)
                        ):
                            live_argument = live_function_module_binding(
                                argument.id
                            )
                            if live_argument is not live_binding_unavailable:
                                source_args[index] = live_argument
                        elif (
                            index < len(source_args)
                            and (
                                isinstance(argument, (ast.IfExp, ast.BoolOp))
                                or (
                                    isinstance(argument, ast.Compare)
                                    and len(argument.ops) == 1
                                    and isinstance(
                                        argument.ops[0],
                                        (ast.Is, ast.IsNot),
                                    )
                                )
                            )
                        ):
                            source_args[index] = (
                                evaluate_reduced_control_expression(argument)
                            )
                    args = tuple(source_args)
                for keyword in expression.keywords:
                    if (
                        keyword.arg is not None
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id in source_binding_values
                    ):
                        kwargs[keyword.arg] = source_binding_values[
                            keyword.value.id
                        ]
                    elif (
                        keyword.arg is not None
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id not in supplied
                    ):
                        live_argument = live_function_module_binding(
                            keyword.value.id
                        )
                        if live_argument is not live_binding_unavailable:
                            kwargs[keyword.arg] = live_argument
                external_ref = attributes.get("external_callee_ref")
                callee_ref = attributes.get(
                    "callee_ref", attributes.get("method_ref")
                )
                direct_self_recursion = (
                    isinstance(expression.func, ast.Name)
                    and expression.func.id
                    == graph.G.graph.get("function_name")
                )
                if direct_self_recursion:
                    callee_ref = graph.G.graph.get("function_ref", callee_ref)
                class_ref = attributes.get("class_ref")
                static_callable = None
                runtime_bound_callable = None
                static_reference = attributes.get(
                    "static_python_reference"
                )
                if isinstance(expression.func, ast.Name):
                    source_name = expression.func.id
                    exact_callable = None
                    if source_name in source_binding_values:
                        candidate = source_binding_values[source_name]
                        if callable(candidate):
                            exact_callable = candidate
                    elif source_name in supplied:
                        candidate = supplied[source_name]
                        if callable(candidate):
                            exact_callable = candidate
                    else:
                        candidate = shell.static_python_bindings.get(
                            source_name,
                            getattr(builtins, source_name, None),
                        )
                        if callable(candidate):
                            exact_callable = candidate
                    if exact_callable is not None:
                        # A bare source spelling has Python lexical authority.
                        # Normalization may merge call edges whose constructors
                        # have equal-looking scalar results (notably int(1)
                        # and bool(1)); it may not change which callable the
                        # source named.
                        static_callable = exact_callable
                        static_reference = source_name
                        external_ref = None
                        callable_name = getattr(
                            exact_callable, "__name__", None
                        )
                        internal_ref = callee_ref
                        if internal_ref is None:
                            for reference_name in (
                                source_name,
                                callable_name,
                            ):
                                if reference_name is None:
                                    continue
                                reference = graph.function_table.reference(
                                    reference_name
                                )
                                if reference is not None:
                                    internal_ref = reference.address
                                    if internal_ref in shell.function_shells:
                                        break
                        internal_shell = None
                        if internal_ref is not None:
                            internal_shell = shell.function_shells.get(
                                int(internal_ref)
                            )
                        internal_name = (
                            internal_shell.process_graph.G.graph.get(
                                "function_name"
                            )
                            if internal_shell is not None
                            else None
                        )
                        # Keep a source-ingested function linked to its
                        # planner shell.  Only discard a spelling-derived
                        # reference when it actually names a different
                        # callable (the int(1)/bool(1) collision described
                        # above).  Calling host Python with a compiled
                        # callback argument would escape the compiler and is
                        # invalid.
                        callee_ref = (
                            internal_ref
                            if internal_shell is not None
                            and str(internal_name) == str(callable_name)
                            else None
                        )
                        class_ref = None
                if static_reference is not None:
                    try:
                        candidate = _static_python_value(
                            shell.static_python_bindings,
                            static_reference,
                        )
                    except (AttributeError, KeyError):
                        candidate = None
                    if callable(candidate):
                        static_callable = candidate
                        if getattr(candidate, "__self__", None) is not None:
                            # Exact bound-call authority outranks a bare-name
                            # function-table collision and retains its receiver.
                            callee_ref = None
                if isinstance(expression.func, ast.Attribute):
                    receiver_parent = next(
                        (
                            int(parent)
                            for parent, role in parents
                            if str(role) in {"operand", "receiver"}
                        ),
                        None,
                    )
                    if (
                        receiver_parent is None
                        and id(expression.func.value) in graph.G
                    ):
                        # Older compiled-plan checkpoints may have already
                        # consumed the explicit operand edge while retaining
                        # the original Attribute AST.  Its value node remains
                        # the exact receiver identity and is safe to recover.
                        receiver_parent = id(expression.func.value)
                    source_receiver_name = (
                        expression.func.value.id
                        if isinstance(expression.func.value, ast.Name)
                        else None
                    )
                    receiver_value = (
                        source_binding_values[source_receiver_name]
                        if source_receiver_name in source_binding_values
                        else evaluate_node(receiver_parent)
                        if receiver_parent is not None
                        else None
                    )
                    if isinstance(
                        receiver_value,
                        (_CompiledStructuralObject, _CompiledStructuralClass),
                    ):
                        bound_method = getattr(
                            receiver_value, expression.func.attr, None
                        )
                        if isinstance(bound_method, _CompiledStructuralMethod):
                            # Runtime structural type identity is stronger than
                            # a spelling-only method-table guess.
                            callee_ref = bound_method.method_ref
                    elif receiver_value is not None:
                        candidate = getattr(
                            receiver_value, expression.func.attr, None
                        )
                        if callable(candidate):
                            candidate_name = getattr(
                                getattr(candidate, "__func__", candidate),
                                "__name__",
                                None,
                            )
                            internal_ref = callee_ref
                            if internal_ref is None and candidate_name:
                                reference = graph.function_table.reference(
                                    candidate_name
                                )
                                if reference is not None:
                                    internal_ref = reference.address
                            internal_shell = (
                                shell.function_shells.get(int(internal_ref))
                                if internal_ref is not None
                                else None
                            )
                            internal_graph = (
                                internal_shell.process_graph.G.graph
                                if internal_shell is not None
                                else {}
                            )
                            method_owner = internal_graph.get("method_owner")
                            receiver_type = type(receiver_value)
                            receiver_owner_names = {
                                receiver_type.__name__,
                                receiver_type.__qualname__,
                                f"{receiver_type.__module__}."
                                f"{receiver_type.__qualname__}",
                            }
                            if (
                                internal_shell is not None
                                and internal_graph.get("method_binding")
                                == "instance"
                                and str(internal_graph.get("function_name"))
                                == str(candidate_name)
                                and str(method_owner) in receiver_owner_names
                            ):
                                # The exact runtime bound method validates the
                                # planner's class-method reference.  Keep the
                                # call inside the source hierarchy so its
                                # numerical regions enter the one discovery
                                # tape.  Spelling alone remains insufficient:
                                # unrelated same-named methods still take the
                                # host-bound path below.
                                callee_ref = internal_ref
                                runtime_bound_callable = None
                            else:
                                runtime_bound_callable = candidate
                                callee_ref = None
                                external_ref = None
                if class_ref is not None:
                    descriptor = (
                        graph.G.graph.get("class_table", {}).get(class_ref)
                    )
                    if descriptor is None:
                        raise RuntimeError(
                            f"unknown compiled class reference {class_ref!r}"
                        )
                    host_factory = (
                        static_callable
                        if isinstance(static_callable, type)
                        and getattr(static_callable, "__new__", object.__new__)
                        is not object.__new__
                        else None
                    )
                    if host_factory is not None:
                        # A custom ``__new__`` is an allocation/factory
                        # protocol, not an initializer.  Until allocation IR
                        # exists, replacing it with a field-only structural
                        # object is observably wrong (Path -> WindowsPath is
                        # the canonical example). Preserve the exact available
                        # factory and let subsequent bound calls retain their
                        # real receiver.
                        result = host_factory(*args, **kwargs)
                        values[node_id] = result
                        return result
                    result = _CompiledStructuralObject(
                        class_ref,
                        descriptor,
                        args,
                        kwargs,
                    )
                    initializer_ref = (
                        descriptor.get("methods") or {}
                    ).get("__init__")
                    initializer = (
                        shell.function_shells.get(int(initializer_ref))
                        if initializer_ref is not None
                        else None
                    )
                    if initializer is not None:
                        initializer.static_python_bindings = {
                            **shell.static_python_bindings,
                            **initializer.static_python_bindings,
                        }
                        receiver_parameter, positional_parameters, _ = (
                            _method_parameter_layout(
                                initializer.process_graph.G
                            )
                        )
                        initializer_inputs = dict(
                            zip(positional_parameters, args)
                        ) | kwargs
                        if receiver_parameter is not None:
                            initializer_inputs[receiver_parameter] = result
                        for name, default in (
                            initializer.process_graph.G.graph.get(
                                "parameter_defaults",
                                {},
                            ).items()
                        ):
                            initializer_inputs.setdefault(name, default)
                        if capture:
                            _coordinate_scheduled_capture(
                                initializer,
                                initializer_inputs,
                                device=device,
                                capture=True,
                                discovery_session=discovery_session,
                            )
                        elif initializer.whole_program_compiled:
                            initializer.execute_process_graph(
                                initializer_inputs
                            )
                        else:
                            initializer.coordinate_first_invocation(
                                initializer_inputs,
                                device=device,
                            )
                elif runtime_bound_callable is not None:
                    try:
                        result = runtime_bound_callable(*args, **kwargs)
                    except Exception as error:
                        if hasattr(error, "add_note"):
                            error.add_note(
                                "runtime-bound Python call failed; "
                                f"shell={shell.profile_path!r}; "
                                f"node={node_id}; "
                                f"call={ast.dump(expression, include_attributes=False)!r}; "
                                f"callable={_diagnostic_value_summary(runtime_bound_callable)!r}; "
                                f"args={tuple(_diagnostic_value_summary(value) for value in args)!r}; "
                                f"kwargs={{{', '.join(f'{name!r}: {_diagnostic_value_summary(value)!r}' for name, value in kwargs.items())}}}; "
                                f"parents={parents!r}; "
                                f"static_arguments={static_arguments!r}"
                            )
                        raise
                elif external_ref is not None:
                    try:
                        external_label = (
                            shell.external_function_table
                            .entry(external_ref)
                            .qualified_name
                        )
                    except (AttributeError, KeyError):
                        external_label = str(external_ref)
                    with _profile_event(
                        shell,
                        "external",
                        external_label,
                        gpu=True,
                    ):
                        result = shell.call_external(
                            external_ref,
                            *args,
                            **kwargs,
                        )
                elif (
                    callee_ref is not None
                ):
                    nested = (
                        shell
                        if direct_self_recursion
                        else getattr(
                            shell, "callsite_function_shells", {}
                        ).get(node_id)
                        or shell.function_shells[int(callee_ref)]
                    )
                    nested.static_python_bindings = {
                        **shell.static_python_bindings,
                        **nested.static_python_bindings,
                    }
                    if shell._profiler.verbose:
                        shell._profiler.trace(
                            path=shell.profile_path,
                            section="function-call",
                            label=(
                                f"{graph.G.graph.get('function_name', '?')} "
                                f"node {node_id} -> "
                                f"{nested.process_graph.G.graph.get('function_name', callee_ref)}"
                            ),
                            fields={
                                "positional": tuple(
                                    _diagnostic_value_summary(value)
                                    for value in args
                                ),
                                "keywords": {
                                    name: _diagnostic_value_summary(value)
                                    for name, value in kwargs.items()
                                },
                            },
                        )
                    if nested.process_graph_boundary:
                        if nested.python_callable is None:
                            raise RuntimeError(
                                "host-boundary ProcessGraph function has no "
                                f"Python implementation: {callee_ref}"
                            )
                        result = nested.python_callable(*args, **kwargs)
                        values[node_id] = result
                        return result
                    nested_graph_metadata = nested.process_graph.G.graph
                    receiver_parameter, positional_parameters, _ = (
                        _method_parameter_layout(
                            nested.process_graph.G
                        )
                    )
                    receiver_value = None
                    call_args = args
                    if nested_graph_metadata.get("method_binding") == "class":
                        owner = nested_graph_metadata.get("method_owner")
                        descriptor = (
                            nested_graph_metadata.get("class_table", {})
                            .get(owner)
                        )
                        if descriptor is None:
                            raise RuntimeError(
                                "compiled classmethod has no class descriptor: "
                                f"{owner!r}"
                            )
                        receiver_value = _CompiledStructuralClass(
                            owner, descriptor
                        )
                    elif nested_graph_metadata.get("method_binding") == "instance":
                        receiver_parent = next(
                            (
                                int(parent)
                                for parent, role in parents
                                if str(role) in {"operand", "receiver"}
                            ),
                            None,
                        )
                        if (
                            receiver_parent is None
                            and isinstance(expression.func, ast.Attribute)
                        ):
                            receiver_parent = _resolve_reference_node(
                                expression.func.value
                            )
                        if receiver_parent is not None:
                            receiver_value = evaluate_node(receiver_parent)
                        elif args:
                            # A method discovered through a named Python
                            # binding is an unbound function call: its first
                            # explicit positional argument is the instance.
                            receiver_value = args[0]
                            call_args = args[1:]
                    nested_inputs = dict(
                        zip(positional_parameters, call_args)
                    ) | kwargs
                    if receiver_parameter is not None and receiver_value is not None:
                        nested_inputs[receiver_parameter] = receiver_value
                    caller_identities = graph.G.graph.get(
                        "identity_table",
                        {},
                    )
                    for _input_id, input_data in (
                        nested.process_graph.G.nodes(data=True)
                    ):
                        input_attributes = (
                            input_data.get("attributes") or {}
                        )
                        if (
                            input_data.get("type") != "Input"
                            or input_attributes.get("binding_kind")
                            != "external"
                        ):
                            continue
                        name = input_attributes.get("binding_name")
                        if name in nested_inputs:
                            continue
                        resolution_errors = []
                        if (
                            name not in nested_inputs
                            and discovery_session is not None
                        ):
                            for lexical_frame in reversed(
                                discovery_session.get(
                                    "lexical_frames", ()
                                )
                            ):
                                try:
                                    resolved, lexical_value = (
                                        lexical_frame["resolve"](name)
                                    )
                                except (KeyError, RuntimeError):
                                    continue
                                if resolved:
                                    nested_inputs[name] = lexical_value
                                    break
                        if name not in nested_inputs and name in supplied:
                            nested_inputs[name] = supplied[name]
                        if name not in nested_inputs:
                            for identity in reversed(
                                caller_identities.get(name, ())
                            ):
                                try:
                                    nested_inputs[name] = evaluate_node(
                                        identity
                                    )
                                except (KeyError, RuntimeError) as error:
                                    resolution_errors.append(
                                        (identity, str(error))
                                    )
                                    continue
                                break
                        if (
                            name not in nested_inputs
                            and caller_identities.get(name)
                        ):
                            raise RuntimeError(
                                f"failed to route closure {name!r} into "
                                f"{nested.process_graph.G.graph.get('function_name', '?')}; "
                                f"caller identities={caller_identities[name]!r}; "
                                f"errors={tuple(resolution_errors)!r}"
                            )
                    for name, value in (
                        nested.process_graph.G.graph.get(
                            "parameter_defaults",
                            {},
                        ).items()
                    ):
                        nested_inputs.setdefault(name, value)
                    # Literal callsite specializations were propagated across
                    # the function table before deployment planning.  Values
                    # observed here are runtime inputs, not authority to
                    # rebuild loop IR or invalidate the region schedule.
                    if capture:
                        _coordinate_scheduled_capture(
                            nested,
                            nested_inputs,
                            device=device,
                            capture=True,
                            discovery_session=discovery_session,
                        )
                    elif nested.whole_program_compiled:
                        nested.execute_process_graph(nested_inputs)
                    else:
                        nested.coordinate_first_invocation(
                            nested_inputs,
                            device=device,
                        )
                    result = nested.last_result
                    if discovery_session is not None:
                        history = discovery_session.setdefault(
                            "call_return_history", []
                        )
                        history.append((
                            nested.profile_path,
                            _diagnostic_value_summary(result),
                        ))
                        del history[:-64]
                elif static_reference:
                    # Exact lexical/builtin resolution can establish a static
                    # source reference even when the normalized call node did
                    # not originally carry metadata for one.
                    reference = static_reference
                    structural_constructors = {
                        "tuple": tuple,
                        "list": list,
                        "set": set,
                        "dict": dict,
                    }
                    constructor = structural_constructors.get(reference)
                    if constructor is not None:
                        if (
                            attributes.get("materialization_kind")
                            in {
                                "unrolled_loop",
                                "retained_loop_aggregate",
                            }
                        ):
                            if kwargs:
                                raise TypeError(
                                    "loop materialization cannot have "
                                    "keyword arguments"
                                )
                            result = constructor(args)
                        else:
                            try:
                                result = constructor(*args, **kwargs)
                            except Exception as error:
                                if hasattr(error, "add_note"):
                                    error.add_note(
                                        "static structural constructor failed; "
                                        f"shell={shell.profile_path!r}; "
                                        f"node={node_id}; "
                                        f"call={ast.dump(expression, include_attributes=False)!r}; "
                                        f"args={tuple(_diagnostic_value_summary(value) for value in args)!r}; "
                                        f"kwargs={{{', '.join(f'{name!r}: {_diagnostic_value_summary(value)!r}' for name, value in kwargs.items())}}}"
                                    )
                                raise
                    else:
                        callable_value = static_callable or _static_python_value(
                            shell.static_python_bindings, reference
                        )
                        if any(
                            isinstance(value, _CompiledStructuralFunction)
                            for value in (*args, *kwargs.values())
                        ):
                            available_shells = tuple(
                                (
                                    int(address),
                                    child.process_graph.G.graph.get(
                                        "function_name"
                                    ),
                                )
                                for address, child in sorted(
                                    shell.function_shells.items()
                                )
                            )
                            raise RuntimeError(
                                "source callable with compiled callback was "
                                "not linked to a planner shell; "
                                f"reference={reference!r}; "
                                f"callable={getattr(callable_value, '__name__', callable_value)!r}; "
                                f"callee_ref={attributes.get('callee_ref')!r}; "
                                f"available_shells={available_shells!r}"
                            )
                        if (
                            reference in {"max", "min"}
                            and not kwargs
                            and len(args) >= 2
                            and any(
                                isinstance(value, AbstractTensor)
                                for value in args
                            )
                        ):
                            # Python's scalar max/min returns one of its
                            # operands by identity.  During tensor discovery
                            # that would collapse the Call result onto the
                            # selected input from this one sample, assigning
                            # one primitive occurrence to two ProcessGraph
                            # endpoints and (worse) specializing away the
                            # runtime comparison.  Preserve the source scalar
                            # semantics as an elementwise tensor operation;
                            # scalar and singleton arenas are its degenerate
                            # case and therefore retain a distinct SSA result.
                            result = args[0]
                            method_name = (
                                "maximum" if reference == "max" else "minimum"
                            )
                            for operand in args[1:]:
                                if not isinstance(result, AbstractTensor):
                                    result = AbstractTensor.tensor(result)
                                result = getattr(result, method_name)(operand)
                        else:
                            call_args = args
                            if (
                                reference == "next"
                                and call_args
                                and isinstance(call_args[0], (list, tuple))
                            ):
                                # A GeneratorExp/comprehension argument is
                                # traced as an already-materialized list here
                                # (this simulator has no lazy generator
                                # value), but the real builtin `next()` --
                                # reached as a static host call because it
                                # has no compiled lowering -- requires an
                                # actual iterator, not a list, and raises
                                # TypeError otherwise. `next(genexpr, default)`
                                # is an ordinary, common idiom (used by
                                # MachineExecutionOrchestrator._relative_target,
                                # among others); wrap the materialized
                                # sequence in iter() so it behaves exactly
                                # like the real generator would have.
                                call_args = (iter(call_args[0]), *call_args[1:])
                            try:
                                result = callable_value(*call_args, **kwargs)
                            except Exception as error:
                                if hasattr(error, "add_note"):
                                    error.add_note(
                                        "static Python host call failed; "
                                        f"shell={shell.profile_path!r}; "
                                        f"node={node_id}; "
                                        f"reference={reference!r}; "
                                        f"call={ast.dump(expression, include_attributes=False)!r}; "
                                        f"callable={_diagnostic_value_summary(callable_value)!r}; "
                                        f"args={tuple(_diagnostic_value_summary(value) for value in call_args)!r}; "
                                        f"kwargs={{{', '.join(f'{name!r}: {_diagnostic_value_summary(value)!r}' for name, value in kwargs.items())}}}"
                                    )
                                raise
                elif any(
                    str(role) == "callee" for _parent, role in parents
                ):
                    callee_parent = next(
                        parent
                        for parent, role in parents
                        if str(role) == "callee"
                    )
                    callee_value = evaluate_node(callee_parent)
                    if isinstance(callee_value, _CompiledStructuralFunction):
                        nested = (
                            getattr(shell, "callsite_function_shells", {}).get(
                                node_id
                            )
                            or shell.function_shells[
                                int(callee_value.function_ref)
                            ]
                        )
                        _receiver, positional_parameters, _all_parameters = (
                            _method_parameter_layout(nested.process_graph.G)
                        )
                        nested_inputs = dict(
                            zip(positional_parameters, args)
                        ) | kwargs
                        caller_identities = graph.G.graph.get(
                            "identity_table", {}
                        )
                        for _input_id, input_data in (
                            nested.process_graph.G.nodes(data=True)
                        ):
                            input_attributes = input_data.get(
                                "attributes"
                            ) or {}
                            if (
                                input_data.get("type") != "Input"
                                or input_attributes.get("binding_kind")
                                != "external"
                            ):
                                continue
                            name = input_attributes.get("binding_name")
                            if name in nested_inputs:
                                continue
                            for identity in reversed(
                                caller_identities.get(name, ())
                            ):
                                try:
                                    nested_inputs[name] = evaluate_node(identity)
                                except (KeyError, RuntimeError):
                                    continue
                                break
                        for name, value in (
                            nested.process_graph.G.graph.get(
                                "parameter_defaults", {}
                            ).items()
                        ):
                            nested_inputs.setdefault(name, value)
                        if capture:
                            _coordinate_scheduled_capture(
                                nested,
                                nested_inputs,
                                device=device,
                                capture=True,
                                discovery_session=discovery_session,
                            )
                        elif nested.whole_program_compiled:
                            nested.execute_process_graph(nested_inputs)
                        else:
                            nested.coordinate_first_invocation(
                                nested_inputs,
                                device=device,
                            )
                        result = nested.last_result
                        values[node_id] = result
                        return result
                    if callable(callee_value) and not isinstance(
                        callee_value,
                        (
                            _CompiledStructuralClass,
                            _CompiledStructuralMethod,
                        ),
                    ):
                        if not capture:
                            raise RuntimeError(
                                "an uncaptured runtime callable reached "
                                f"compiled coordinator execution at node {node_id}"
                            )
                        # A callback parameter is an invocation boundary, not
                        # a name-based function-table reference.  During the
                        # one authorized discovery pass, invoke the supplied
                        # callable under the active AbstractTensor backend so
                        # its canonical tensor work is appended to the same
                        # program tape.  Installed execution may never fall
                        # back to the Python callable: its work must already
                        # have been absorbed by this capture.
                        result = callee_value(*args, **kwargs)
                        values[node_id] = result
                        return result
                    if not isinstance(
                        callee_value,
                        (
                            _CompiledStructuralClass,
                            _CompiledStructuralMethod,
                        ),
                    ):
                        raise RuntimeError(
                            "runtime callee edge is not a compiled "
                            f"structural class at node {node_id}: "
                            f"{type(callee_value).__name__}"
                        )
                    if isinstance(
                        callee_value,
                        _CompiledStructuralClass,
                    ):
                        result = _CompiledStructuralObject(
                            callee_value.class_ref,
                            callee_value.descriptor,
                            args,
                            kwargs,
                        )
                        initializer_ref = (
                            callee_value.descriptor.get("methods") or {}
                        ).get("__init__")
                        if initializer_ref is not None:
                            nested = shell.function_shells[int(initializer_ref)]
                            (
                                receiver_parameter,
                                positional_parameters,
                                _,
                            ) = _method_parameter_layout(
                                nested.process_graph.G
                            )
                            nested_inputs = dict(
                                zip(positional_parameters, args)
                            ) | kwargs
                            if receiver_parameter is not None:
                                nested_inputs[receiver_parameter] = result
                            for name, value in (
                                nested.process_graph.G.graph.get(
                                    "parameter_defaults", {}
                                ).items()
                            ):
                                nested_inputs.setdefault(name, value)
                            if capture:
                                _coordinate_scheduled_capture(
                                    nested,
                                    nested_inputs,
                                    device=device,
                                    capture=True,
                                    discovery_session=discovery_session,
                                )
                            elif nested.whole_program_compiled:
                                nested.execute_process_graph(nested_inputs)
                            else:
                                nested.coordinate_first_invocation(
                                    nested_inputs,
                                    device=device,
                                )
                    else:
                        nested = (
                            getattr(shell, "callsite_function_shells", {}).get(
                                node_id
                            )
                            or shell.function_shells[
                                int(callee_value.method_ref)
                            ]
                        )
                        receiver_parameter, positional_parameters, _ = (
                            _method_parameter_layout(
                                nested.process_graph.G
                            )
                        )
                        nested_inputs = dict(
                            zip(positional_parameters, args)
                        ) | kwargs
                        if receiver_parameter is not None:
                            nested_inputs[receiver_parameter] = (
                                callee_value.receiver
                            )
                        for name, value in (
                            nested.process_graph.G.graph.get(
                                "parameter_defaults",
                                {},
                            ).items()
                        ):
                            nested_inputs.setdefault(name, value)
                        if capture:
                            _coordinate_scheduled_capture(
                                nested,
                                nested_inputs,
                                device=device,
                                capture=True,
                                discovery_session=discovery_session,
                            )
                        elif nested.whole_program_compiled:
                            nested.execute_process_graph(nested_inputs)
                        else:
                            nested.coordinate_first_invocation(
                                nested_inputs,
                                device=device,
                            )
                        result = nested.last_result
                elif isinstance(expression.func, ast.Attribute):
                    callable_parent = next(
                        (
                            parent
                            for parent, role in parents
                            if str(role) in {"func", "callee"}
                        ),
                        None,
                    )
                    if callable_parent is not None:
                        result = evaluate_node(callable_parent)(
                            *args,
                            **kwargs,
                        )
                    else:
                        receiver_parent = next(
                            (
                                parent
                                for parent, role in parents
                                if str(role) == "operand"
                            ),
                            None,
                        )
                        if receiver_parent is None:
                            live_callable = (
                                evaluate_reduced_control_expression(
                                    expression.func
                                )
                            )
                            if callable(live_callable):
                                result = live_callable(*args, **kwargs)
                                values[node_id] = result
                                return result
                            raise RuntimeError(
                                "attribute call has neither a callable nor "
                                f"receiver edge at ProcessGraph node {node_id}: "
                                f"{ast.dump(expression, include_attributes=False)}; "
                                f"parents={parents!r}"
                            )
                        receiver = evaluate_node(receiver_parent)
                        if isinstance(
                            receiver,
                            (_CompiledStructuralObject, _CompiledStructuralClass),
                        ):
                            methods = (
                                receiver.methods
                                if isinstance(receiver, _CompiledStructuralObject)
                                else receiver.descriptor.get("methods") or {}
                            )
                            method_ref = methods.get(expression.func.attr)
                            if method_ref is None:
                                raise RuntimeError(
                                    f"compiled class {receiver.class_ref!r} "
                                    "has no retained method "
                                    f"{expression.func.attr!r}"
                                )
                            nested = (
                                getattr(
                                    shell,
                                    "callsite_function_shells",
                                    {},
                                ).get(node_id)
                                or shell.function_shells[int(method_ref)]
                            )
                            (
                                receiver_parameter,
                                positional_parameters,
                                _,
                            ) = _method_parameter_layout(
                                nested.process_graph.G
                            )
                            nested_inputs = dict(
                                zip(positional_parameters, args)
                            ) | kwargs
                            if receiver_parameter is not None:
                                nested_inputs[receiver_parameter] = receiver
                            for name, value in (
                                nested.process_graph.G.graph.get(
                                    "parameter_defaults",
                                    {},
                                ).items()
                            ):
                                nested_inputs.setdefault(name, value)
                            if capture:
                                _coordinate_scheduled_capture(
                                    nested,
                                    nested_inputs,
                                    device=device,
                                    capture=True,
                                    discovery_session=discovery_session,
                                )
                            elif nested.whole_program_compiled:
                                nested.execute_process_graph(nested_inputs)
                            else:
                                nested.coordinate_first_invocation(
                                    nested_inputs,
                                    device=device,
                                )
                            result = nested.last_result
                        else:
                            result = getattr(
                                receiver,
                                expression.func.attr,
                            )(*args, **kwargs)
                else:
                    raise RuntimeError(
                        f"unresolved ProcessGraph Call node {node_id} in "
                        f"{graph.G.graph.get('function_name', '?')}: "
                        f"{ast.dump(expression, include_attributes=False)}; "
                        f"parents={parents!r}; attributes={attributes!r}"
                    )
            elif isinstance(expression, ast.If):
                test_id = next(
                    parent
                    for parent, role in parents
                    if str(role) == "test"
                )
                test = bool(evaluate_node(test_id))
                branch = "body" if test else "orelse"
                branch_nodes = [
                    parent
                    for parent, role in parents
                    if str(role) == branch
                ]
                result = None
                for branch_node in branch_nodes:
                    result = evaluate_node(branch_node)
            elif node_type == "Phi":
                by_role = {
                    str(role): parent for parent, role in parents
                }
                selected = (
                    by_role["body"]
                    if bool(evaluate_node(by_role["test"]))
                    else by_role["orelse"]
                )
                result = evaluate_node(selected)
            elif isinstance(expression, ast.Try):
                by_role: dict[str, list[int]] = {}
                for parent, role in parents:
                    by_role.setdefault(str(role), []).append(parent)
                result = None
                try:
                    for body_node in by_role.get("body", ()):
                        result = evaluate_node(body_node)
                except Exception as error:
                    selected_handler = None
                    for handler_id in by_role.get("handlers", ()):
                        handler_expression = graph.G.nodes[
                            handler_id
                        ].get("expr_obj")
                        if not isinstance(
                            handler_expression,
                            ast.ExceptHandler,
                        ):
                            continue
                        type_expression = handler_expression.type
                        if type_expression is None:
                            selected_handler = handler_id
                            break
                        names = (
                            tuple(type_expression.elts)
                            if isinstance(type_expression, ast.Tuple)
                            else (type_expression,)
                        )
                        exception_types = []
                        for name in names:
                            if isinstance(name, ast.Name):
                                candidate = shell.static_python_bindings.get(
                                    name.id,
                                    getattr(builtins, name.id, None),
                                )
                                if isinstance(candidate, type) and issubclass(
                                    candidate,
                                    BaseException,
                                ):
                                    exception_types.append(candidate)
                        if exception_types and isinstance(
                            error,
                            tuple(exception_types),
                        ):
                            selected_handler = handler_id
                            break
                    shell._profiler.record_exception(
                        error,
                        path=shell.profile_path,
                        phase="graph_try",
                        node_id=node_id,
                        handled=selected_handler is not None,
                    )
                    if selected_handler is None:
                        raise
                    active_exceptions[selected_handler] = error
                    try:
                        result = evaluate_node(selected_handler)
                    finally:
                        active_exceptions.pop(selected_handler, None)
                else:
                    for else_node in by_role.get("orelse", ()):
                        result = evaluate_node(else_node)
                finally:
                    for final_node in by_role.get("finalbody", ()):
                        result = evaluate_node(final_node)
            elif isinstance(expression, ast.ExceptHandler):
                error = active_exceptions.get(node_id)
                if error is None:
                    raise RuntimeError(
                        "ProcessGraph exception handler executed without "
                        f"an active exception at node {node_id}"
                    )
                if expression.name:
                    for binding in graph.G.graph.get(
                        "identity_table",
                        {},
                    ).get(expression.name, ()):
                        values[binding] = error
                result = None
                for parent, role in parents:
                    if str(role) == "body":
                        result = evaluate_node(parent)
            elif isinstance(expression, (ast.For, ast.While)):
                plan = loop_plans_by_node[node_id]
                loop = plan.loop
                if isinstance(expression, ast.For):
                    if loop.iterator_kind == "arithmetic_sequence":
                        by_role = {
                            str(role): parent for parent, role in parents
                        }
                        start = (
                            loop.start
                            if loop.start is not None
                            else evaluate_node(by_role["start"])
                        )
                        stop = (
                            loop.stop
                            if loop.stop is not None
                            else evaluate_node(by_role["stop"])
                        )
                        step = (
                            loop.step
                            if loop.step is not None
                            else evaluate_node(by_role["step"])
                        )
                        iterator = iter(range(
                            int(start),
                            int(stop),
                            int(step),
                        ))
                    else:
                        iterator = iter(evaluate_node(loop.iterable_node))
                else:
                    iterator = None

                result = None
                iterations_completed = 0
                target_initials = (
                    (graph.G.nodes[node_id].get("attributes") or {})
                    .get("loop_target_initials") or {}
                )
                for name, binding in loop.target_bindings:
                    initial = target_initials.get(name)
                    if initial is not None:
                        values[binding] = evaluate_node(int(initial))
                if shell._profiler.verbose:
                    shell._profiler.trace(
                        path=shell.profile_path,
                        section="loop",
                        label=f"node {node_id} begin",
                        fields={
                            "strategy": plan.strategy.value,
                            "trip_count": loop.trip_count,
                            "carried": tuple(
                                name
                                for name, _initial, _updated
                                in loop.carried_bindings
                            ),
                        },
                    )
                while True:
                    if isinstance(expression, ast.For):
                        try:
                            item = next(iterator)
                        except StopIteration:
                            break
                        targets = loop.target_bindings
                        items_by_name = dict(_destructure_loop_target(
                            expression.target, item
                        ))
                        target_assignments = tuple(
                            (
                                (name, binding),
                                items_by_name[str(name)],
                            )
                            for name, binding in targets
                        )
                    else:
                        target_assignments = ()
                        test_parent = next(
                            parent
                            for parent, role in parents
                            if str(role) == "test"
                        )
                        if not bool(evaluate_node(test_parent)):
                            break

                    # Generated binding nodes (for example tuple-unpack
                    # ``Indexed`` nodes) can sit just beyond the lexical AST
                    # body while still depending on this iteration.  Clear
                    # the full downstream value cone so those bindings and
                    # any prematurely cached post-loop values cannot retain
                    # the preceding iteration.
                    for dependent in loop_invalidated_nodes[node_id]:
                        values.pop(dependent, None)
                    for (_name, binding), value in target_assignments:
                        values[binding] = value
                        generator_binding_values[binding] = value
                    # A flat graph dependency can demand a nested loop before
                    # the source-statement walker reaches that loop.  Its
                    # destructured targets are still ordinary Python lexical
                    # assignments, so publish every leaf into both runtime
                    # identity tables and the source frame.  Keeping only the
                    # planner-selected bindings makes a sibling target (for
                    # example ``value_ids`` in ``for name, value_ids in ...``)
                    # look like a missing function input when the inner loop
                    # consumes it.
                    source_binding_values.update(items_by_name)
                    for region_index in loop_runtime_region_indices.get(
                        node_id,
                        (),
                    ):
                        completed_regions.discard(region_index)
                    loop_signal = None
                    try:
                        for body_node in (
                            candidate
                            for candidate in loop.body_nodes
                            if (
                                candidate not in nested_body_nodes_by_loop[
                                    int(node_id)
                                ]
                            )
                        ):
                            body_region = region_for_node.get(body_node)
                            if is_shader_internal_node(body_node):
                                evaluate_region(body_region)
                            else:
                                result = evaluate_node(body_node)
                    except _LoopBreakSignal:
                        loop_signal = "break"
                    except _LoopContinueSignal:
                        loop_signal = "continue"
                    if loop_signal == "break":
                        iterations_completed += 1
                        break
                    for carried_name, initial, updated in loop.carried_bindings:
                        if (
                            loop_signal == "continue"
                            and carried_name in source_binding_values
                        ):
                            values[initial] = source_binding_values[
                                carried_name
                            ]
                        elif loop_signal != "continue":
                            values[initial] = evaluate_node(updated)
                    iterations_completed += 1
                    if loop_signal == "continue":
                        continue
                    if shell._profiler.verbose:
                        shell._profiler.trace(
                            path=shell.profile_path,
                            section="loop-iteration",
                            label=(
                                f"node {node_id} iteration "
                                f"{iterations_completed}"
                            ),
                            fields={
                                name: _diagnostic_value_summary(
                                    values.get(initial)
                                )
                                for name, initial, _updated
                                in loop.carried_bindings
                            },
                        )
                    if (
                        isinstance(expression, ast.While)
                        and iterations_completed > 1_000_000
                    ):
                        raise RuntimeError(
                            "ProcessGraph while loop exceeded safety limit"
                        )
                if shell._profiler.verbose:
                    shell._profiler.trace(
                        path=shell.profile_path,
                        section="loop",
                        label=f"node {node_id} end",
                        fields={"iterations": iterations_completed},
                    )
            elif isinstance(expression, ast.Raise):
                exception = (
                    evaluate_node(parents[0][0])
                    if parents
                    else RuntimeError("ProcessGraph bare raise")
                )
                if hasattr(exception, "add_note"):
                    input_state = tuple(
                        (
                            int(input_id),
                            (input_data.get("attributes") or {}).get(
                                "binding_name",
                                input_data.get("label"),
                            ),
                            _diagnostic_value_summary(values.get(input_id)),
                        )
                        for input_id, input_data in graph.G.nodes(data=True)
                        if input_data.get("type") == "Input"
                        and input_id in values
                    )
                    exception.add_note(
                        "raised by ProcessGraph "
                        f"{graph.G.graph.get('function_name', '?')} at node "
                        f"{node_id}; inputs={input_state!r}"
                    )
                raise exception
            elif isinstance(expression, ast.Assert):
                by_role = {
                    str(role): parent for parent, role in parents
                }
                if not bool(evaluate_node(by_role["test"])):
                    message_id = by_role.get("msg")
                    message = (
                        evaluate_node(message_id)
                        if message_id is not None
                        else None
                    )
                    raise AssertionError(message)
                result = None
            elif isinstance(expression, ast.Pass):
                result = None
            elif isinstance(expression, ast.Break):
                raise _LoopBreakSignal()
            elif isinstance(expression, ast.Continue):
                raise _LoopContinueSignal()
            elif node_type == "SetAttr":
                by_role = {
                    str(role): parent for parent, role in parents
                }
                receiver = evaluate_node(by_role["object"])
                value = evaluate_node(by_role["value"])
                setattr(
                    receiver,
                    (data.get("attributes") or {})["attribute"],
                    value,
                )
                result = value
                shell.reference_operator_sequence.append(int(node_id))
            elif node_type == "DelAttr":
                by_role = {
                    str(role): parent for parent, role in parents
                }
                receiver = evaluate_node(by_role["object"])
                delattr(
                    receiver,
                    (data.get("attributes") or {})["attribute"],
                )
                result = None
            elif node_type == "DelItem":
                base = evaluate_node(
                    next(
                        parent
                        for parent, role in parents
                        if str(role) == "base"
                    )
                )
                indices = [
                    evaluate_node(parent)
                    for parent, role in parents
                    if str(role) == "index"
                ]
                index = indices[0] if len(indices) == 1 else tuple(indices)
                del base[index]
                result = None
            elif node_type == "StaticReference":
                # This is a compiler-owned link to a function subgraph or
                # structural symbol.  It is deliberately not a Python object
                # and performs no runtime work.
                attributes = data.get("attributes") or {}
                first_class_ref = attributes.get(
                    "first_class_function_ref"
                )
                static_reference = attributes.get("static_python_reference")
                static_value = None
                if static_reference is not None:
                    try:
                        static_value = _static_python_value(
                            shell.static_python_bindings,
                            static_reference,
                        )
                    except (AttributeError, KeyError):
                        pass
                if isinstance(static_value, type):
                    # Classes used as first-class values (most importantly in
                    # isinstance/issubclass tuples) retain Python type
                    # identity. Constructor calls remain governed by the
                    # class_ref branch above.
                    result = static_value
                else:
                    result = (
                        _CompiledStructuralFunction(first_class_ref)
                        if first_class_ref is not None
                        else attributes.get(
                            "function_ref",
                            static_reference,
                        )
                    )
            elif isinstance(expression, ast.Lambda):
                lambda_expression = expression
                positional_parameters = tuple(
                    (*lambda_expression.args.posonlyargs,
                     *lambda_expression.args.args)
                )
                positional_defaults = {
                    parameter.arg: evaluate_reduced_control_expression(default)
                    for parameter, default in zip(
                        positional_parameters[
                            len(positional_parameters)
                            - len(lambda_expression.args.defaults):
                        ],
                        lambda_expression.args.defaults,
                    )
                }
                keyword_defaults = {
                    parameter.arg: evaluate_reduced_control_expression(default)
                    for parameter, default in zip(
                        lambda_expression.args.kwonlyargs,
                        lambda_expression.args.kw_defaults,
                    )
                    if default is not None
                }

                def structural_lambda(*call_args, **call_kwargs):
                    remaining_kwargs = dict(call_kwargs)
                    bound = {}
                    if (
                        len(call_args) > len(positional_parameters)
                        and lambda_expression.args.vararg is None
                    ):
                        raise TypeError("lambda received too many arguments")
                    for index, parameter in enumerate(positional_parameters):
                        if index < len(call_args):
                            if parameter.arg in remaining_kwargs:
                                raise TypeError(
                                    f"lambda got multiple values for "
                                    f"{parameter.arg!r}"
                                )
                            bound[parameter.arg] = call_args[index]
                        elif parameter.arg in remaining_kwargs:
                            bound[parameter.arg] = remaining_kwargs.pop(
                                parameter.arg
                            )
                        elif parameter.arg in positional_defaults:
                            bound[parameter.arg] = positional_defaults[
                                parameter.arg
                            ]
                        else:
                            raise TypeError(
                                f"lambda missing argument {parameter.arg!r}"
                            )
                    if lambda_expression.args.vararg is not None:
                        bound[lambda_expression.args.vararg.arg] = tuple(
                            call_args[len(positional_parameters):]
                        )
                    for parameter in lambda_expression.args.kwonlyargs:
                        if parameter.arg in remaining_kwargs:
                            bound[parameter.arg] = remaining_kwargs.pop(
                                parameter.arg
                            )
                        elif parameter.arg in keyword_defaults:
                            bound[parameter.arg] = keyword_defaults[
                                parameter.arg
                            ]
                        else:
                            raise TypeError(
                                "lambda missing keyword-only argument "
                                f"{parameter.arg!r}"
                            )
                    if lambda_expression.args.kwarg is not None:
                        bound[lambda_expression.args.kwarg.arg] = (
                            remaining_kwargs
                        )
                        remaining_kwargs = {}
                    if remaining_kwargs:
                        unexpected = next(iter(remaining_kwargs))
                        raise TypeError(
                            f"lambda got unexpected keyword {unexpected!r}"
                        )
                    missing = object()
                    previous = {
                        name: source_binding_values.get(name, missing)
                        for name in bound
                    }
                    source_binding_values.update(bound)
                    try:
                        return evaluate_reduced_control_expression(
                            lambda_expression.body
                        )
                    finally:
                        for name, prior in previous.items():
                            if prior is missing:
                                source_binding_values.pop(name, None)
                            else:
                                source_binding_values[name] = prior

                result = structural_lambda
            elif isinstance(expression, ast.Return) or node_type in {
                "Return",
                "return",
            }:
                result = (
                    evaluate_node(parents[0][0])
                    if parents
                    else None
                )
                values[node_id] = result
                if capture:
                    shell.forward_feed_ids = tuple(feed_maps)
                    shell.forward_aggregate_feed_paths = tuple(
                        aggregate_feed_maps
                    )
                    shell.forward_subgraphs = tuple(captured_subgraphs)
                    shell.forward_compilers = tuple(captured_compilers)
                    shell.forward_region_indices = tuple(
                        captured_region_indices
                    )
                    shell.forward_output_values = tuple(
                        captured_output_values
                    )
                    shell.forward_region_capture_node_ids = tuple(
                        captured_region_node_ids
                    )
                    shell.forward_region_planned_capture_ids = tuple(
                        captured_region_planned_ids
                    )
                    shell.forward_region_planned_input_ids = tuple(
                        captured_region_planned_input_ids
                    )
                shell.captured_values = values
                shell.last_result = result
                raise _SourceReturnSignal(result)
            elif node_type in {"Store", "store", "Output", "output"}:
                result = (
                    evaluate_node(parents[0][0])
                    if parents
                    else None
                )
            else:
                raise NotImplementedError(
                    "scheduled coordinator cannot resolve structural "
                    f"node {node_id} ({node_type})"
                )
            values[node_id] = result
            return result
        finally:
            # Coordinator region arbitration can deliberately clear a
            # region's active markers before replaying it structurally.
            active_nodes.discard(node_id)

    generator_binding_values: dict[int, Any] = {}
    source_binding_values: dict[str, Any] = {
        str((data.get("attributes") or {}).get(
            "binding_name", data.get("label", "")
        )): values[int(node_id)]
        for node_id, data in graph.G.nodes(data=True)
        if str(data.get("type")) in {"Input", "input"}
        and (data.get("attributes") or {}).get("binding_kind")
        == "parameter"
        and int(node_id) in values
    }
    source_binding_node_ids: dict[str, int] = {
        str((data.get("attributes") or {}).get(
            "binding_name", data.get("label", "")
        )): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if str(data.get("type")) in {"Input", "input"}
        and (data.get("attributes") or {}).get("binding_kind")
        == "parameter"
        and int(node_id) in values
    }

    def has_live_source_root(expression: ast.AST) -> bool:
        current = expression
        while isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        return (
            isinstance(current, ast.Name)
            and (
                current.id in source_binding_values
                or current.id in supplied
                or live_function_module_binding(current.id)
                is not live_binding_unavailable
            )
        )

    def evaluate_reduced_control_expression(expression: ast.AST) -> Any:
        if isinstance(
            expression,
            (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp),
        ):
            def target_names(target: ast.AST) -> tuple[str, ...]:
                if isinstance(target, ast.Name):
                    return (target.id,)
                if isinstance(target, (ast.Tuple, ast.List)):
                    return tuple(
                        name
                        for item in target.elts
                        for name in target_names(item)
                    )
                raise NotImplementedError(
                    "reduced comprehension target is not supported: "
                    f"{ast.unparse(target)}"
                )

            def bind_target(target: ast.AST, value: Any) -> None:
                if isinstance(target, ast.Name):
                    source_binding_values[target.id] = value
                    return
                if isinstance(target, (ast.Tuple, ast.List)):
                    unpacked = tuple(value)
                    if len(unpacked) != len(target.elts):
                        raise ValueError(
                            "comprehension target/value arity mismatch"
                        )
                    for item, item_value in zip(target.elts, unpacked):
                        bind_target(item, item_value)
                    return
                raise NotImplementedError(
                    "reduced comprehension target is not supported: "
                    f"{ast.unparse(target)}"
                )

            missing = object()

            def iterate(generators, index=0):
                if index >= len(generators):
                    if isinstance(expression, ast.DictComp):
                        yield (
                            evaluate_reduced_control_expression(
                                expression.key
                            ),
                            evaluate_reduced_control_expression(
                                expression.value
                            ),
                        )
                    else:
                        yield evaluate_reduced_control_expression(
                            expression.elt
                        )
                    return
                generator = generators[index]
                if generator.is_async:
                    raise NotImplementedError(
                        "async reduced comprehensions are not supported"
                    )
                iterable = evaluate_reduced_control_expression(
                    generator.iter
                )
                names = target_names(generator.target)
                previous = {
                    name: source_binding_values.get(name, missing)
                    for name in names
                }
                try:
                    for item in iterable:
                        bind_target(generator.target, item)
                        if all(
                            bool(evaluate_reduced_control_expression(test))
                            for test in generator.ifs
                        ):
                            yield from iterate(generators, index + 1)
                finally:
                    for name, value in previous.items():
                        if value is missing:
                            source_binding_values.pop(name, None)
                        else:
                            source_binding_values[name] = value

            items = iterate(expression.generators)
            if isinstance(expression, ast.GeneratorExp):
                return items
            if isinstance(expression, ast.ListComp):
                return list(items)
            if isinstance(expression, ast.SetComp):
                return set(items)
            return dict(items)
        if isinstance(expression, ast.Name):
            if expression.id in source_binding_values:
                return source_binding_values[expression.id]
            if expression.id in supplied:
                return supplied[expression.id]
            live_binding = live_function_module_binding(expression.id)
            if live_binding is not live_binding_unavailable:
                return live_binding
            identities = graph.G.graph.get("identity_table", {}) or {}
            for identity in reversed(identities.get(expression.id, ())):
                identity = int(identity)
                if identity in values or identity in graph.G:
                    return evaluate_node(identity)
            return evaluate_static_expression(expression)
        if isinstance(expression, ast.IfExp):
            branch = (
                expression.body
                if bool(evaluate_reduced_control_expression(expression.test))
                else expression.orelse
            )
            return evaluate_reduced_control_expression(branch)
        if isinstance(expression, ast.BoolOp):
            if isinstance(expression.op, ast.And):
                result = True
                for operand in expression.values:
                    result = evaluate_reduced_control_expression(operand)
                    if not result:
                        break
                return result
            result = False
            for operand in expression.values:
                result = evaluate_reduced_control_expression(operand)
                if result:
                    break
            return result
        if isinstance(expression, ast.UnaryOp):
            operand = evaluate_reduced_control_expression(
                expression.operand
            )
            unary = {
                ast.Not: operator.not_,
                ast.UAdd: operator.pos,
                ast.USub: operator.neg,
                ast.Invert: operator.invert,
            }.get(type(expression.op))
            if unary is None:
                raise NotImplementedError(
                    "reduced unary operator is not supported: "
                    f"{ast.unparse(expression)}"
                )
            return unary(operand)
        if isinstance(expression, ast.BinOp):
            left = evaluate_reduced_control_expression(expression.left)
            right = evaluate_reduced_control_expression(expression.right)
            binary = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.MatMult: operator.matmul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.LShift: operator.lshift,
                ast.RShift: operator.rshift,
                ast.BitOr: operator.or_,
                ast.BitXor: operator.xor,
                ast.BitAnd: operator.and_,
            }.get(type(expression.op))
            if binary is None:
                raise NotImplementedError(
                    "reduced binary operator is not supported: "
                    f"{ast.unparse(expression)}"
                )
            return binary(left, right)
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
        ):
            source_callable = shell.static_python_bindings.get(
                expression.func.id
            )
            if discovery_session is not None:
                history = discovery_session.setdefault(
                    "structural_predicate_history", []
                )
                history.append((
                    "source-call",
                    expression.func.id,
                    repr(source_callable),
                    bool(
                        isinstance(source_callable, type)
                        and getattr(
                            source_callable, "__new__", object.__new__
                        ) is not object.__new__
                    ),
                ))
                del history[:-32]
            if (
                isinstance(source_callable, type)
                and getattr(source_callable, "__new__", object.__new__)
                is not object.__new__
            ):
                positional = []
                for argument in expression.args:
                    if isinstance(argument, ast.Starred):
                        positional.extend(
                            evaluate_reduced_control_expression(
                                argument.value
                            )
                        )
                    else:
                        positional.append(
                            evaluate_reduced_control_expression(argument)
                        )
                keywords = {}
                for keyword in expression.keywords:
                    value = evaluate_reduced_control_expression(
                        keyword.value
                    )
                    if keyword.arg is None:
                        keywords.update(value)
                    else:
                        keywords[keyword.arg] = value
                return source_callable(*positional, **keywords)
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"isinstance", "issubclass"}
            and len(expression.args) == 2
            and not expression.keywords
        ):
            def evaluate_type_spec(type_expression: ast.AST) -> Any:
                if isinstance(type_expression, ast.Tuple):
                    return tuple(
                        evaluate_type_spec(item)
                        for item in type_expression.elts
                    )
                if isinstance(type_expression, ast.Name):
                    if type_expression.id in shell.static_python_bindings:
                        return shell.static_python_bindings[type_expression.id]
                    builtin_type = getattr(
                        builtins, type_expression.id, None
                    )
                    if isinstance(builtin_type, type):
                        return builtin_type
                if isinstance(type_expression, ast.Attribute):
                    return getattr(
                        evaluate_type_spec(type_expression.value),
                        type_expression.attr,
                    )
                return evaluate_reduced_control_expression(type_expression)

            predicate = (
                isinstance
                if expression.func.id == "isinstance"
                else issubclass
            )
            subject = evaluate_reduced_control_expression(expression.args[0])
            type_spec = evaluate_type_spec(expression.args[1])
            result = predicate(subject, type_spec)
            if discovery_session is not None:
                history = discovery_session.setdefault(
                    "structural_predicate_history", []
                )
                history.append((
                    expression.func.id,
                    _diagnostic_value_summary(subject),
                    repr(type_spec),
                    bool(result),
                ))
                del history[:-32]
            return result
        if (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and len(expression.comparators) == 1
            and isinstance(expression.ops[0], (ast.Is, ast.IsNot))
        ):
            def identity_operand(operand: ast.AST) -> Any:
                # Identity tests are coordinator/source semantics all the way
                # through an attribute path.  Do not let a retained Attribute
                # node substitute an equal normalized value: SymPy's ``One``
                # and ``true`` are especially sensitive to singleton identity.
                if isinstance(operand, ast.Attribute):
                    return getattr(
                        identity_operand(operand.value), operand.attr
                    )
                return evaluate_reduced_control_expression(operand)

            left = identity_operand(expression.left)
            right = identity_operand(expression.comparators[0])
            result = left is right
            return (
                not result
                if isinstance(expression.ops[0], ast.IsNot)
                else result
            )
        if isinstance(expression, ast.Compare):
            operands = [
                evaluate_reduced_control_expression(expression.left),
                *(
                    evaluate_reduced_control_expression(comparator)
                    for comparator in expression.comparators
                ),
            ]
            for index, comparison in enumerate(expression.ops):
                comparator = {
                    ast.Is: operator.is_,
                    ast.IsNot: operator.is_not,
                    ast.Eq: operator.eq,
                    ast.NotEq: operator.ne,
                    ast.Lt: operator.lt,
                    ast.LtE: operator.le,
                    ast.Gt: operator.gt,
                    ast.GtE: operator.ge,
                    ast.In: lambda left, right: left in right,
                    ast.NotIn: lambda left, right: left not in right,
                }.get(type(comparison))
                if comparator is None:
                    break
                if not bool(comparator(operands[index], operands[index + 1])):
                    return False
            else:
                return True
        if isinstance(expression, ast.Subscript):
            base = evaluate_reduced_control_expression(expression.value)
            slice_expression = expression.slice
            if isinstance(slice_expression, ast.Slice):
                index = slice(
                    evaluate_reduced_control_expression(
                        slice_expression.lower
                    ) if slice_expression.lower is not None else None,
                    evaluate_reduced_control_expression(
                        slice_expression.upper
                    ) if slice_expression.upper is not None else None,
                    evaluate_reduced_control_expression(
                        slice_expression.step
                    ) if slice_expression.step is not None else None,
                )
            elif isinstance(slice_expression, ast.Tuple):
                index = tuple(
                    evaluate_reduced_control_expression(item)
                    for item in slice_expression.elts
                )
            else:
                index = evaluate_reduced_control_expression(
                    slice_expression
                )
            if isinstance(base, (dict, list, tuple, set)):
                index = coordinator_index(index)
            return base[index]
        if (
            isinstance(expression, ast.Attribute)
            and has_live_source_root(expression)
        ):
            return getattr(
                evaluate_reduced_control_expression(expression.value),
                expression.attr,
            )
        if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
            items = [
                evaluate_reduced_control_expression(item)
                for item in expression.elts
            ]
            if isinstance(expression, ast.Tuple):
                return tuple(items)
            if isinstance(expression, ast.Set):
                return set(items)
            return items
        if isinstance(expression, ast.Dict):
            result = {}
            for key, value in zip(expression.keys, expression.values):
                item = evaluate_reduced_control_expression(value)
                if key is None:
                    result.update(dict(item))
                else:
                    result[
                        evaluate_reduced_control_expression(key)
                    ] = item
            return result
        retained_node = expression_nodes.get(id(expression))
        if retained_node is not None:
            # Reduction can classify a value as graph-inert when its only
            # consumer is a source assignment/control expression. Source
            # execution is itself a real consumer: once it demands the AST
            # expression, suppressing the node would replace the value with
            # None and violate Python branch semantics.
            inert_nodes.discard(retained_node)
            return evaluate_node(retained_node)
        if isinstance(expression, ast.Call):
            # A structural method call can disappear from the numerical
            # graph while its result is still consumed by source control.
            # Resolve it against the live lexical bindings (not only the
            # restricted static-binding table) and preserve Python's
            # starred argument/keyword semantics.  State snapshots such as
            # ``saved = state.copy_shallow()`` in the dt controller are the
            # canonical case.
            function = evaluate_reduced_control_expression(expression.func)
            positional = []
            for argument in expression.args:
                if isinstance(argument, ast.Starred):
                    positional.extend(
                        evaluate_reduced_control_expression(argument.value)
                    )
                else:
                    positional.append(
                        evaluate_reduced_control_expression(argument)
                    )
            keywords = {}
            for keyword in expression.keywords:
                value = evaluate_reduced_control_expression(keyword.value)
                if keyword.arg is None:
                    keywords.update(value)
                else:
                    keywords[keyword.arg] = value
            return function(*positional, **keywords)
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Attribute):
            return getattr(
                evaluate_reduced_control_expression(expression.value),
                expression.attr,
            )
        return evaluate_static_expression(expression)

    def execute_generator_statements(statements):
        source_value_unavailable = object()

        def publish_nonlocal(name: str, value: Any) -> None:
            if discovery_session is None:
                return
            for lexical_frame in reversed(
                discovery_session.get("lexical_frames", ())
            ):
                if lexical_frame.get("shell") is shell:
                    continue
                if lexical_frame["assign"](name, value):
                    return

        def assign_source_target(target: ast.AST, value: Any) -> None:
            if isinstance(target, ast.Name):
                source_binding_values[target.id] = value
                if discovery_session is not None:
                    history = discovery_session.setdefault(
                        "source_assignment_history", []
                    )
                    history.append((
                        shell.profile_path,
                        target.id,
                        _diagnostic_value_summary(value),
                    ))
                    del history[:-64]
                if target.id in nonlocal_names:
                    publish_nonlocal(target.id, value)
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                unpacked = tuple(value)
                starred = tuple(
                    index
                    for index, item in enumerate(target.elts)
                    if isinstance(item, ast.Starred)
                )
                if len(starred) > 1:
                    raise ValueError(
                        "multiple starred assignment targets are invalid"
                    )
                if not starred:
                    if len(unpacked) != len(target.elts):
                        raise ValueError(
                            "assignment target/value arity mismatch"
                        )
                    assigned = tuple(zip(target.elts, unpacked))
                else:
                    star_index = starred[0]
                    trailing = len(target.elts) - star_index - 1
                    minimum = len(target.elts) - 1
                    if len(unpacked) < minimum:
                        raise ValueError(
                            "not enough values to unpack assignment"
                        )
                    assigned = (
                        tuple(zip(target.elts[:star_index], unpacked[:star_index]))
                        + ((
                            target.elts[star_index].value,
                            list(
                                unpacked[
                                    star_index:
                                    len(unpacked) - trailing
                                    if trailing
                                    else len(unpacked)
                                ]
                            ),
                        ),)
                        + tuple(zip(
                            target.elts[star_index + 1 :],
                            unpacked[len(unpacked) - trailing :]
                            if trailing
                            else (),
                        ))
                    )
                for item_target, item_value in assigned:
                    assign_source_target(item_target, item_value)
                return

        for statement in statements:
            if isinstance(statement, ast.Break):
                # Source control remains authoritative even when reduction
                # removes the marker node because it has no numerical value.
                raise _LoopBreakSignal()
            if isinstance(statement, ast.Continue):
                raise _LoopContinueSignal()
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value,
                (ast.Yield, ast.YieldFrom),
            ):
                yielded = statement.value
                value_node = (
                    expression_nodes.get(id(yielded.value))
                    if yielded.value is not None
                    else None
                )
                if yielded.value is not None and value_node is None:
                    yield_node = expression_nodes.get(id(yielded))
                    if yield_node is not None:
                        value_node = next(
                            (
                                parent
                                for parent, role in (
                                    graph.G.nodes[yield_node].get(
                                        "parents",
                                        (),
                                    )
                                )
                                if str(role) in {"value", "operand"}
                            ),
                            None,
                        )
                if yielded.value is not None and value_node is None:
                    raise RuntimeError(
                        "generator yield value was not retained in "
                        f"{graph.G.graph.get('function_name', '?')}: "
                        f"{ast.dump(yielded, include_attributes=False)}"
                    )
                value = (
                    evaluate_node(value_node)
                    if value_node is not None
                    else None
                )
                if isinstance(yielded, ast.YieldFrom):
                    yield from value
                else:
                    yield value
                continue
            if isinstance(statement, ast.If):
                try:
                    test = evaluate_reduced_control_expression(
                        statement.test
                    )
                except (KeyError, TypeError) as error:
                    raise RuntimeError(
                        "generator branch test was not retained in "
                        f"{graph.G.graph.get('function_name', '?')}: "
                        f"{ast.dump(statement.test, include_attributes=False)}"
                    ) from error
                if discovery_session is not None:
                    history = discovery_session.setdefault(
                        "source_branch_history", []
                    )
                    history.append((
                        shell.profile_path,
                        ast.dump(statement.test, include_attributes=False),
                        _diagnostic_value_summary(test),
                    ))
                    del history[:-512]
                shell._captured_source_branches.append((
                    ast.dump(statement.test, include_attributes=False),
                    _diagnostic_value_summary(test),
                ))
                del shell._captured_source_branches[:-64]
                if shell._profiler.verbose:
                    shell._profiler.trace(
                        path=shell.profile_path,
                        section="generator-branch",
                        label=ast.dump(
                            statement.test,
                            include_attributes=False,
                        ),
                        fields={
                            "value": _diagnostic_value_summary(test),
                            "selected": (
                                "body" if bool(test) else "orelse"
                            ),
                        },
                    )
                branch = statement.body if bool(test) else statement.orelse
                yield from execute_generator_statements(branch)
                continue
            if isinstance(statement, ast.With):
                # ``with`` is source-ordered Python control, including for
                # context managers whose enter/exit effects do not produce a
                # numerical graph value (locks are a common example).  Let
                # ExitStack implement Python's ordered entry, reverse exit,
                # and exception-suppression protocol while the coordinator
                # remains responsible for evaluating expressions and binding
                # optional ``as`` targets in the active lexical frame.
                with ExitStack() as context_stack:
                    for item in statement.items:
                        context_manager = (
                            evaluate_reduced_control_expression(
                                item.context_expr
                            )
                        )
                        entered_value = context_stack.enter_context(
                            context_manager
                        )
                        if item.optional_vars is not None:
                            assign_source_target(
                                item.optional_vars, entered_value
                            )
                    yield from execute_generator_statements(statement.body)
                continue
            if isinstance(statement, ast.Try):
                try:
                    try:
                        yield from execute_generator_statements(statement.body)
                    except (
                        _SourceReturnSignal,
                        _LoopBreakSignal,
                        _LoopContinueSignal,
                    ):
                        raise
                    except BaseException as error:
                        selected_handler = None
                        for handler in statement.handlers:
                            if handler.type is None:
                                selected_handler = handler
                                break
                            exception_type = (
                                evaluate_reduced_control_expression(
                                    handler.type
                                )
                            )
                            if isinstance(error, exception_type):
                                selected_handler = handler
                                break
                        if selected_handler is None:
                            raise
                        exception_name = selected_handler.name
                        if exception_name is not None:
                            source_binding_values[exception_name] = error
                        try:
                            yield from execute_generator_statements(
                                selected_handler.body
                            )
                        finally:
                            # CPython clears an ``except ... as name`` target
                            # when the handler suite exits.
                            if exception_name is not None:
                                source_binding_values.pop(exception_name, None)
                    else:
                        yield from execute_generator_statements(
                            statement.orelse
                        )
                finally:
                    yield from execute_generator_statements(
                        statement.finalbody
                    )
                continue
            if isinstance(statement, ast.For):
                loop_node = expression_nodes.get(id(statement))
                if loop_node is None:
                    # A statically bounded loop may already have evaporated
                    # into cloned straight-line graph nodes.  Those nodes are
                    # evaluated by the ordinary dependency sweep; replaying
                    # the removed source loop would duplicate the body.
                    continue
                plan = loop_plans_by_node[loop_node]
                loop = plan.loop
                target_initials = (
                    (graph.G.nodes[loop_node].get("attributes") or {})
                    .get("loop_target_initials") or {}
                )
                for name, binding in loop.target_bindings:
                    initial = target_initials.get(name)
                    if initial is not None:
                        values[binding] = evaluate_node(int(initial))
                if loop.iterator_kind == "arithmetic_sequence":
                    by_role = {
                        str(role): parent
                        for parent, role in (
                            graph.G.nodes[loop_node].get("parents") or ()
                        )
                    }
                    start = (
                        loop.start
                        if loop.start is not None
                        else evaluate_node(by_role["start"])
                    )
                    stop = (
                        loop.stop
                        if loop.stop is not None
                        else evaluate_node(by_role["stop"])
                    )
                    step = (
                        loop.step
                        if loop.step is not None
                        else evaluate_node(by_role["step"])
                    )
                    iterator = iter(range(
                        int(start),
                        int(stop),
                        int(step),
                    ))
                else:
                    iterator = iter(evaluate_node(loop.iterable_node))
                loop_broken = False
                # Source execution is now the authoritative owner of this
                # retained loop.  Publish that fact before entering the body:
                # a LoopExit demanded by a body expression must not recursively
                # replay the generic flat-loop evaluator with an empty target
                # frame while the source loop is already active.
                values[loop_node] = None
                iteration_count = 0
                for item in iterator:
                    iteration_count += 1
                    items_by_name = dict(_destructure_loop_target(
                        statement.target, item
                    ))
                    # Numerical reduction is allowed to omit loop targets
                    # that never enter a tensor region, but Python source
                    # control may still read any destructured target.  Keep
                    # the complete source binding environment in step with
                    # the iterator rather than treating target_bindings as a
                    # complete account of Python assignment semantics.
                    source_binding_values.update(items_by_name)
                    assignments = tuple(
                        (
                            (name, binding),
                            items_by_name[str(name)],
                        )
                        for name, binding in loop.target_bindings
                    )
                    for dependent in loop_invalidated_nodes[loop_node]:
                        values.pop(dependent, None)
                    for (_name, binding), bound_value in assignments:
                        generator_binding_values[binding] = bound_value
                    values.update(generator_binding_values)
                    for region_index in loop_runtime_region_indices.get(
                        loop_node,
                        (),
                    ):
                        completed_regions.discard(region_index)
                    try:
                        yield from execute_generator_statements(statement.body)
                    except _LoopBreakSignal:
                        loop_broken = True
                        break
                    except _LoopContinueSignal:
                        loop_continued = True
                    else:
                        loop_continued = False
                    for carried_name, initial, updated in loop.carried_bindings:
                        if (
                            loop_continued
                            and carried_name in source_binding_values
                        ):
                            values[initial] = source_binding_values[
                                carried_name
                            ]
                        elif not loop_continued:
                            values[initial] = evaluate_node(updated)
                # The generator interpreter is the planner-owned execution of
                # this loop.  Publish completion so a downstream LoopExit
                # cannot replay the same body through the generic flat loop
                # evaluator (which would also lose nested branch semantics).
                values[loop_node] = None
                shell._captured_loop_iterations[int(loop_node)] = (
                    int(iteration_count)
                )
                if not loop_broken:
                    yield from execute_generator_statements(statement.orelse)
                continue
            if isinstance(statement, ast.While):
                loop_node = expression_nodes.get(id(statement))
                if loop_node is None:
                    continue
                source_loop_nodes = frozenset(
                    int(member_node)
                    for member in ast.walk(statement)
                    if (
                        member_node := expression_nodes.get(id(member))
                    ) is not None
                    and int(member_node) != int(loop_node)
                )
                iterations = 0
                loop_broken = False
                plan = loop_plans_by_node[loop_node]
                if plan.strategy is LoopStrategy.NATIVE_SOURCE:
                    # Discovery traces the body of a retained native loop; it
                    # must not execute the runtime trip count.  This is vital
                    # for event/render loops whose configured bound is
                    # intentionally infinite.  The semantic loop/control IR
                    # owns repetition after capture.
                    for dependent in (
                        *loop_invalidated_nodes[loop_node],
                        *source_loop_nodes,
                    ):
                        values.pop(dependent, None)
                    for region_index in loop_runtime_region_indices.get(
                        loop_node, ()
                    ):
                        completed_regions.discard(region_index)
                    test = evaluate_reduced_control_expression(statement.test)
                    if bool(test):
                        try:
                            yield from execute_generator_statements(
                                statement.body
                            )
                        except (_LoopBreakSignal, _LoopContinueSignal):
                            pass
                    values[loop_node] = None
                    # The runtime loop, not discovery, decides whether it
                    # exhausts normally and therefore enters ``else``.
                    continue
                while True:
                    for dependent in (
                        *loop_invalidated_nodes[loop_node],
                        *source_loop_nodes,
                    ):
                        values.pop(dependent, None)
                    for region_index in loop_runtime_region_indices.get(
                        loop_node,
                        (),
                    ):
                        completed_regions.discard(region_index)
                    test = evaluate_reduced_control_expression(
                        statement.test
                    )
                    if not bool(test):
                        break
                    try:
                        yield from execute_generator_statements(statement.body)
                    except _LoopBreakSignal:
                        loop_broken = True
                        break
                    except _LoopContinueSignal:
                        pass
                    iterations += 1
                    if iterations > 1_000_000:
                        raise RuntimeError(
                            "ProcessGraph generator while loop exceeded "
                            "safety limit"
                        )
                values[loop_node] = None
                if not loop_broken:
                    yield from execute_generator_statements(statement.orelse)
                continue
            if isinstance(statement, ast.Return):
                source_return = object()
                value = source_return
                if isinstance(statement.value, ast.Name):
                    if statement.value.id in source_binding_values:
                        value = source_binding_values[statement.value.id]
                    elif statement.value.id in supplied:
                        value = supplied[statement.value.id]
                elif (
                    statement.value is not None
                    and has_live_source_root(statement.value)
                ):
                    value = evaluate_reduced_control_expression(
                        statement.value
                    )
                # The executing Return statement and its value expression are
                # the exact source-order authority.  Normalized/checkpointed
                # ``return_value_nodes`` can collapse multiple returns onto a
                # function's syntactically last output identity; consulting
                # it first made an executed early tensor return appear to be
                # the later ``float(value.item())`` branch.  Prefer the live
                # expression node and retain both metadata tables solely as
                # compatibility fallbacks when AST identity was serialized.
                value_node = (
                    source_binding_node_ids.get(statement.value.id)
                    if isinstance(statement.value, ast.Name)
                    else expression_nodes.get(id(statement.value))
                    if statement.value is not None
                    else None
                )
                if value_node is None and statement.value is not None:
                    value_node = graph.G.graph.get(
                        "return_value_nodes", {}
                    ).get(id(statement))
                if value_node is None and statement.value is not None:
                    value_node = checkpoint_return_value_nodes.get(
                        id(statement)
                    )
                if value_node is not None and value_node not in graph.G:
                    output_names = tuple(
                        graph.G.graph.get("function_outputs", ())
                    )
                    identities = graph.G.graph.get("identity_table", {})
                    candidates = tuple(
                        int(identities[name][-1])
                        for name in output_names
                        if identities.get(name)
                        and int(identities[name][-1]) in graph.G
                    )
                    value_node = (
                        candidates[0]
                        if len(candidates) == 1
                        else graph.roots[0]
                        if len(graph.roots) == 1
                        else None
                    )
                if capture and value_node is not None:
                    shell._captured_return_value_ids.add(int(value_node))
                if value is source_return:
                    value = (
                        values[value_node]
                        if value_node in values
                        else evaluate_node(value_node)
                        if value_node is not None
                        else None
                    )
                if capture:
                    shell.forward_feed_ids = tuple(feed_maps)
                    shell.forward_aggregate_feed_paths = tuple(
                        aggregate_feed_maps
                    )
                    shell.forward_subgraphs = tuple(captured_subgraphs)
                    shell.forward_compilers = tuple(captured_compilers)
                    shell.forward_region_indices = tuple(
                        captured_region_indices
                    )
                    shell.forward_output_values = tuple(
                        captured_output_values
                    )
                    shell.forward_region_capture_node_ids = tuple(
                        captured_region_node_ids
                    )
                    shell.forward_region_planned_capture_ids = tuple(
                        captured_region_planned_ids
                    )
                    shell.forward_region_planned_input_ids = tuple(
                        captured_region_planned_input_ids
                    )
                shell.captured_values = values
                shell.last_result = value
                raise _SourceReturnSignal(value)
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value_expression = statement.value
                assigned_value = None
                if value_expression is not None:
                    value_node = expression_nodes.get(id(value_expression))
                    # A bare source name denotes the value visible at this
                    # statement, not necessarily the last normalized SSA
                    # identity bearing that spelling.  The distinction is
                    # observable whenever a value is saved before a later
                    # write, for example ``saved = counter; counter += 1``.
                    # Consult the active lexical/source frame first; retained
                    # compound expressions remain owned by the planned graph.
                    if isinstance(
                        value_expression,
                        (ast.Name, ast.IfExp, ast.BoolOp),
                    ):
                        assigned_value = evaluate_reduced_control_expression(
                            value_expression
                        )
                    elif value_node is not None:
                        if is_shader_internal_node(value_node):
                            evaluate_region(region_for_node[value_node])
                            assigned_value = values.get(
                                value_node,
                                source_value_unavailable,
                            )
                        else:
                            assigned_value = evaluate_node(value_node)
                    else:
                        assigned_value = evaluate_reduced_control_expression(
                            value_expression
                        )
                targets = (
                    tuple(statement.targets)
                    if isinstance(statement, ast.Assign)
                    else (statement.target,)
                )
                for target in targets:
                    if (
                        assigned_value is not source_value_unavailable
                        and isinstance(target, (ast.Name, ast.Tuple, ast.List))
                    ):
                        assign_source_target(target, assigned_value)
                    target_node = expression_nodes.get(id(target))
                    if (
                        target_node is not None
                        and target_node in graph.G
                        and graph.G.nodes[target_node].get("type")
                        in {"SetAttr", "IndexedStore"}
                    ):
                        if is_shader_internal_node(target_node):
                            evaluate_region(region_for_node[target_node])
                            continue
                        # The source statement has already evaluated its RHS
                        # at this exact program point.  A normalized store can
                        # still refer to the final SSA identity with the same
                        # spelling (for example ``result`` assigned in many
                        # arms of a long if/elif chain).  Publish the live RHS
                        # into the store's value port so structural mutation
                        # observes Python's source-order value, not a later
                        # predicate or branch identity.
                        for parent, role in graph.G.nodes[target_node].get(
                            "parents", ()
                        ):
                            if (
                                str(role) == "value"
                                and assigned_value
                                is not source_value_unavailable
                            ):
                                values[int(parent)] = assigned_value
                                inert_nodes.discard(int(parent))
                        evaluate_node(target_node)
                continue
            if isinstance(statement, ast.AugAssign):
                statement_node = expression_nodes.get(id(statement))
                if (
                    statement_node is not None
                    and is_shader_internal_node(statement_node)
                ):
                    evaluate_region(region_for_node[statement_node])
                    if isinstance(statement.target, ast.Name):
                        source_binding_values.pop(
                            statement.target.id,
                            None,
                        )
                    continue
                assigned_value = None
                scalar_source_update = False
                if (
                    isinstance(statement.target, ast.Name)
                    and statement.target.id in source_binding_values
                    and not isinstance(
                        source_binding_values[statement.target.id],
                        AbstractTensor,
                    )
                ):
                    try:
                        right_value = evaluate_reduced_control_expression(
                            statement.value
                        )
                    except (KeyError, TypeError):
                        right_value = None
                    if not isinstance(right_value, AbstractTensor):
                        binary = {
                            ast.Add: operator.add,
                            ast.Sub: operator.sub,
                            ast.Mult: operator.mul,
                            ast.Div: operator.truediv,
                            ast.FloorDiv: operator.floordiv,
                            ast.Mod: operator.mod,
                            ast.Pow: operator.pow,
                            ast.LShift: operator.lshift,
                            ast.RShift: operator.rshift,
                            ast.BitOr: operator.or_,
                            ast.BitXor: operator.xor,
                            ast.BitAnd: operator.and_,
                            ast.MatMult: operator.matmul,
                        }.get(type(statement.op))
                        if binary is not None:
                            assigned_value = binary(
                                source_binding_values[statement.target.id],
                                right_value,
                            )
                            scalar_source_update = True
                if statement_node is not None:
                    values.pop(statement_node, None)
                    statement_region = region_for_node.get(statement_node)
                    if statement_region is not None:
                        completed_regions.discard(statement_region)
                if not scalar_source_update:
                    assigned_value = (
                        evaluate_node(statement_node)
                        if statement_node is not None
                        else evaluate_reduced_control_expression(statement)
                    )
                if isinstance(statement.target, ast.Name):
                    source_binding_values[
                        statement.target.id
                    ] = assigned_value
                    if statement.target.id in nonlocal_names:
                        publish_nonlocal(
                            statement.target.id, assigned_value
                        )
                continue
            statement_node = expression_nodes.get(id(statement))
            if statement_node is not None and statement_node in graph.G:
                if is_shader_internal_node(statement_node):
                    evaluate_region(region_for_node[statement_node])
                else:
                    evaluate_node(statement_node)
                continue
            for member in ast.iter_child_nodes(statement):
                member_node = expression_nodes.get(id(member))
                if member_node is not None and member_node in graph.G:
                    if is_shader_internal_node(member_node):
                        evaluate_region(region_for_node[member_node])
                    else:
                        evaluate_node(member_node)

    if graph.G.graph.get("generator_stream"):
        for input_id in tuple(values):
            previous = values[input_id]
            values[input_id] = _tensorize_graph_input(
                previous,
                device=device,
            )
            # Tensorising makes a new object; carry the name across or it is
            # lost exactly when it starts to matter.
            named = getattr(shell, "_capture_input_names", None)
            if named is not None and id(previous) in named:
                named[id(values[input_id])] = named[id(previous)]

        def generator_producer():
            try:
                with AbstractTensor.use_backend(
                    _scheduled_capture_backend.get(),
                    device,
                ):
                    yield from execute_generator_statements(
                        graph.G.graph.get("function_body", ())
                    )
            finally:
                if capture:
                    shell.forward_feed_ids = tuple(feed_maps)
                    shell.forward_aggregate_feed_paths = tuple(
                        aggregate_feed_maps
                    )
                    shell.forward_subgraphs = tuple(captured_subgraphs)
                    shell.forward_compilers = tuple(captured_compilers)
                    shell.forward_region_indices = tuple(
                        captured_region_indices
                    )
                    shell.forward_output_values = tuple(
                        captured_output_values
                    )
                    shell.forward_region_capture_node_ids = tuple(
                        captured_region_node_ids
                    )
                    shell.forward_region_planned_capture_ids = tuple(
                        captured_region_planned_ids
                    )
                    shell.forward_region_planned_input_ids = tuple(
                        captured_region_planned_input_ids
                    )
                shell.captured_values = values

        shell.last_result = _GreedyGeneratorStream(
            generator_producer(),
            fifo_slots=2,
        )
        return shell.last_result

    if discovery_session is not None:
        def resolve_lexical_name(name: str) -> tuple[bool, Any]:
            if name in source_binding_values:
                return True, source_binding_values[name]
            if name in supplied:
                return True, supplied[name]
            for identity in reversed(
                graph.G.graph.get("identity_table", {}).get(name, ())
            ):
                try:
                    return True, evaluate_node(identity)
                except (KeyError, RuntimeError):
                    continue
            return False, None

        def assign_lexical_name(name: str, value: Any) -> bool:
            if (
                name not in source_binding_values
                and name not in supplied
                and name not in graph.G.graph.get("identity_table", {})
            ):
                return False
            source_binding_values[name] = value
            return True

        discovery_session.setdefault("lexical_frames", []).append({
            "shell": shell,
            "resolve": resolve_lexical_name,
            "assign": assign_lexical_name,
        })

    with AbstractTensor.use_backend(
        _scheduled_capture_backend.get(),
        device,
    ):
        # Public array inputs are tensor values even when their first consumer
        # is structural control (for example a defensive ``isinstance``
        # guard).  Tensorise under the scheduled backend before control
        # ownership removes those inputs from the ordinary sweep.
        for input_id, previous in tuple(values.items()):
            resident = _tensorize_graph_input(previous, device=device)
            if resident is previous:
                continue
            values[input_id] = resident
            named = getattr(shell, "_capture_input_names", None)
            if named is not None and (carried := named.get(id(previous))):
                named[id(resident)] = carried
                storage = _capture_storage_identity(resident)
                if storage is not None:
                    shell._capture_input_storage[storage] = carried

        # Runtime control operands are outputs of the compiled program just as
        # surely as returned tensors are.  A predicate used only by
        # ``if tensor_scalar: raise`` may otherwise remain hidden beneath the
        # planner-owned statement branch: ordinary topological demand never
        # reaches it, then control-source composition discovers the predicate
        # ID after numerical capture has already ended.
        #
        # Demand every retained validation predicate here, while the source
        # function's one discovery tape is active.  This does not execute the
        # validation on the host, replay a region, or create a second tape. It
        # merely makes the predicate-producing region a terminal of the same
        # program observation so its resident slot can be consumed by the
        # compiled shell's ValidationBlock.
        if capture:
            for validation in _validation_control_blocks(shell):
                evaluate_node(int(validation.predicate_value_id))

        # Structural Python control is ordered by its retained source body,
        # not by numerical dependency levels.  In particular, a later branch
        # predicate must not run after an earlier branch returned.  Statement
        # execution still demands each planned numerical region through
        # evaluate_node; it does not replace or reinterpret numeric lowering.
        if function_body:
            for _yielded in execute_generator_statements(function_body):
                raise RuntimeError(
                    "non-generator ProcessGraph produced a yielded value"
                )

        for node_id in _dependency_order(graph):
            if node_id in inert_nodes:
                continue
            if node_id in controlled_nodes:
                # The planner owns statement loops and comprehension loops.
                # A statement loop with no controlling loop ancestor may
                # execute here; downstream loops defer until their owner,
                # LoopExit, returned value, or graph root demands them.
                if (
                    isinstance(
                        graph.G.nodes[node_id].get("expr_obj"),
                        (ast.For, ast.While),
                    )
                    and node_id not in downstream_loop_nodes
                    and not has_comprehension_owner
                ):
                    evaluate_node(node_id)
                continue
            if node_id in values:
                try:
                    previous = values[node_id]
                    values[node_id] = _tensorize_graph_input(
                        previous,
                        device=device,
                    )
                    # This is where a supplied array becomes the tensor the
                    # tape sees, so it is the identity a feed will carry.
                    # Recording the name only at binding time meant a
                    # parameter used directly by its own function -- rather
                    # than passed on to a callee, whose shell records it
                    # again -- lost its name here and could only fall back to
                    # a positional one.
                    named = getattr(shell, "_capture_input_names", None)
                    if named is not None:
                        carried = named.get(id(previous))
                        if carried:
                            named[id(values[node_id])] = carried
                            storage = _capture_storage_identity(values[node_id])
                            if storage is not None:
                                shell._capture_input_storage[storage] = carried
                except Exception as error:
                    value = values[node_id]
                    node = graph.G.nodes[node_id]
                    value_dtype = getattr(value, "dtype", None)
                    value_shape = getattr(value, "shape", None)
                    raise TypeError(
                        "failed to tensorize a scheduled graph input: "
                        f"shell={shell.profile_path!r}, node={node_id!r}, "
                        f"label={node.get('label')!r}, "
                        f"op={node.get('op')!r}, "
                        f"value_type={type(value).__module__}."
                        f"{type(value).__qualname__}, "
                        f"shape={value_shape!r}, dtype={value_dtype!r}, "
                        f"value={value!r}"
                    ) from error
            elif is_shader_internal_node(node_id):
                # Topological traversal schedules its containing region, but
                # does not ask the region to materialize this shader-local
                # intermediate as a coordinator-visible value.
                evaluate_region(region_for_node[node_id])
            else:
                evaluate_node(node_id)

    roots = tuple(evaluate_node(node_id) for node_id in graph.roots)
    if capture:
        shell.forward_feed_ids = tuple(feed_maps)
        shell.forward_aggregate_feed_paths = tuple(aggregate_feed_maps)
        shell.forward_subgraphs = tuple(captured_subgraphs)
        shell.forward_compilers = tuple(captured_compilers)
        shell.forward_region_indices = tuple(captured_region_indices)
        shell.forward_output_values = tuple(captured_output_values)
        shell.forward_region_capture_node_ids = tuple(
            captured_region_node_ids
        )
        shell.forward_region_planned_capture_ids = tuple(
            captured_region_planned_ids
        )
        shell.forward_region_planned_input_ids = tuple(
            captured_region_planned_input_ids
        )
    shell.captured_values = values
    shell.last_result = roots[0] if len(roots) == 1 else roots
    return shell.last_result


def _coordinate_scheduled_capture(
    shell: Any,
    initial_values: dict[str | int, Any],
    *,
    device: Any = None,
    capture: bool = True,
    discovery_session: dict[str, Any] | None = None,
) -> Any:
    """Profile-aware boundary around one scheduled shell invocation."""

    if capture:
        # Discovery/compilation diagnostics remain available through verbose
        # trace and the error buffer, but they are not runtime frames.  Mixing
        # their eager region dispatches into steady-state shell statistics made
        # one execution report as two invocations and obscured CPU churn.
        shell._profiler._runtime_suppression += 1
    token = shell._profiler.begin_shell(shell.profile_path)
    try:
        return _coordinate_scheduled_capture_impl(
            shell,
            initial_values,
            device=device,
            capture=capture,
            discovery_session=discovery_session,
        )
    except _SourceReturnSignal as signal:
        shell.last_result = signal.value
        return signal.value
    except Exception as error:
        if hasattr(error, "add_note"):
            structural_state = tuple(
                (
                    str(name),
                    {
                        str(field): _diagnostic_value_summary(field_value)
                        for field, field_value in list(value.state.items())[:24]
                    },
                )
                for name, value in initial_values.items()
                if isinstance(value, _CompiledStructuralObject)
            )
            active_session = discovery_session or getattr(
                shell, "_discovery_session", None
            )
            call_returns = tuple(
                (active_session or {}).get("call_return_history", ())
            )
            structural_predicates = tuple(
                (active_session or {}).get(
                    "structural_predicate_history", ()
                )
            )
            source_assignments = tuple(
                (active_session or {}).get(
                    "source_assignment_history", ()
                )
            )
            source_branches = tuple(
                (active_session or {}).get("source_branch_history", ())
            )
            error.add_note(
                "while coordinating ProcessGraph shell "
                f"{shell.profile_path!r} with inputs="
                f"{tuple((str(name), _diagnostic_value_summary(value)) for name, value in initial_values.items())!r}; "
                f"structural_state={structural_state!r}; "
                f"recent_call_returns={call_returns!r}; "
                f"recent_structural_predicates={structural_predicates!r}; "
                f"recent_source_assignments={source_assignments!r}; "
                f"recent_source_branches={source_branches!r}"
            )
        shell._profiler.record_exception(
            error,
            path=shell.profile_path,
            phase="execution",
        )
        raise
    finally:
        active_session = discovery_session or getattr(
            shell, "_discovery_session", None
        )
        if active_session is not None:
            frames = active_session.get("lexical_frames", [])
            for index in range(len(frames) - 1, -1, -1):
                if frames[index].get("shell") is shell:
                    del frames[index]
                    break
        shell._profiler.end_shell(shell.profile_path, token)
        if capture:
            shell._profiler._runtime_suppression -= 1


class _CompiledStructuralObject:
    """Shell-owned state whose methods are compiled function subgraphs."""

    __slots__ = ("class_ref", "methods", "state")

    def __init__(self, class_ref, descriptor, args, kwargs):
        self.class_ref = str(class_ref)
        self.methods = dict(descriptor.get("methods") or {})
        fields = tuple(descriptor.get("fields") or ())
        defaults = descriptor.get("field_defaults") or {}
        self.state = {
            field: copy.deepcopy(defaults[field])
            for field in fields
            if field in defaults
        }
        self.state.update(zip(fields, args))
        self.state.update(kwargs)
        for field in fields:
            self.state.setdefault(field, None)

    def __getattr__(self, name):
        method_ref = self.methods.get(name)
        if method_ref is not None:
            return _CompiledStructuralMethod(self, method_ref)
        try:
            return self.state[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name in {"class_ref", "methods", "state"}:
            object.__setattr__(self, name, value)
        else:
            self.state[name] = value


class _CompiledStructuralMethod:
    """A method edge into a compiled function shell."""

    __slots__ = ("receiver", "method_ref")

    def __init__(self, receiver, method_ref):
        self.receiver = receiver
        self.method_ref = int(method_ref)


class _CompiledStructuralFunction:
    """First-class edge to a source function already in the function table."""

    __slots__ = ("function_ref",)

    def __init__(self, function_ref):
        self.function_ref = int(function_ref)


class _CompiledStructuralClass:
    """Shell primitive for a compiled class, never a retained Python class."""

    __slots__ = ("class_ref", "descriptor")

    def __init__(self, class_ref, descriptor):
        self.class_ref = str(class_ref)
        self.descriptor = descriptor

    def __call__(self, *args, **kwargs):
        return _CompiledStructuralObject(
            self.class_ref,
            self.descriptor,
            args,
            kwargs,
        )

    def __getattr__(self, name):
        method_ref = (self.descriptor.get("methods") or {}).get(name)
        if method_ref is None:
            raise AttributeError(name)
        return _CompiledStructuralMethod(self, method_ref)


class _GreedyGeneratorStream:
    """Pull-driven shell stream that fills available FIFO capacity per pull."""

    def __init__(self, producer, *, fifo_slots=2):
        self._producer = iter(producer)
        self._fifo = DeploymentFIFO(slots=fifo_slots)
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        while not self._exhausted and self._fifo.unread < self._fifo.slots:
            try:
                item = next(self._producer)
            except StopIteration:
                self._exhausted = True
                break
            if not self._fifo.publish(item):
                break
        available, value = self._fifo.consume()
        if not available:
            raise StopIteration
        return value
