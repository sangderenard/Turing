"""Expand canonical tensor instructions into ordinary repository SSA.

The code reference owns implementations; this module only supplies the ABI
recipe which turns a canonical tensor call into calls to those implementations.
No tensor executor, backend object, or late runtime dispatch survives this pass.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from math import prod
from typing import Any, Iterable

from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import c_tensor_opcode
from ..transmogrifier.ssa import (
    IRModule,
    Instr,
    SSAValue,
    SSATensorDescriptor,
    SSATensorTable,
)
from ..transmogrifier.ssa_registry import Handler
from ..transmogrifier.tensor_ssa_reference import SSATensorCodeReference
from .ssa_aggregate_abi import (
    legalize_aggregate_adapters,
    legalize_aggregate_output_views,
)


@dataclass(frozen=True, order=True)
class TensorSSALoweringShortfall:
    function: str
    block: str
    operation: str
    reason: str


def wire_repository_ssa_region_products(module: IRModule) -> bool:
    """Connect region outputs to later region feeds inside each function.

    Region planning records semantic source identities in ``output_ids`` and
    ``feed_ids``.  Source-call linking can temporarily materialize a
    caller-owned frame argument when a producer region has not yet been
    inserted.  Once the complete repository module exists, the declared
    identities provide an exact, deterministic replacement: use the
    producer's projected SSA value and schedule its call/projection cluster
    before the consumer.
    """

    changed = False
    for function in module.functions.values():
        for block in function.blocks.values():
            instructions = block.instrs
            producer_by_source: dict[int, tuple[Instr, SSAValue, list[Instr]]] = {}
            for index, instruction in enumerate(tuple(instructions)):
                if (
                    instruction.op not in {"Call", "call"}
                    or instruction.res is None
                    or not tuple(instruction.attributes.get("output_ids", ()))
                ):
                    continue
                cluster = [instruction]
                cursor = index + 1
                while cursor < len(instructions) and instructions[cursor].op in {
                    "Const", "GetElementPtr", "getelementptr", "Load", "load",
                }:
                    cluster.append(instructions[cursor])
                    cursor += 1
                for projected in cluster:
                    if (
                        projected.op in {"Load", "load"}
                        and projected.res is not None
                        and projected.attributes.get("source_output_id") is not None
                    ):
                        producer_by_source[int(
                            projected.attributes["source_output_id"]
                        )] = (instruction, projected.res, cluster)

            for consumer in tuple(instructions):
                if consumer.op not in {"Call", "call"}:
                    continue
                feeds = tuple(map(
                    int, consumer.attributes.get("feed_ids", ())
                ))
                if not feeds:
                    continue
                for position, feed_id in enumerate(feeds):
                    producer = producer_by_source.get(feed_id)
                    if producer is None or producer[0] is consumer:
                        continue
                    producer_call, produced_value, cluster = producer
                    if position >= len(consumer.args):
                        continue
                    previous = consumer.args[position]
                    # A repeated seed-producing region can occur inside a
                    # loop body even though the live value for this semantic
                    # feed is the header Phi.  Control lowering marks that
                    # exact case.  Do not let whole-module product wiring
                    # replace the Phi with the repeated seed projection and
                    # reset the carried value on every iteration.
                    if (
                        (previous.accounting or {}).get(
                            "ssa_loop_carried_feed"
                        ) == int(feed_id)
                    ):
                        continue
                    if int(consumer.args[position].id) != int(produced_value.id):
                        declared_shapes = tuple(
                            consumer.attributes.get("feed_shapes", ())
                        )
                        declared_dtypes = tuple(
                            consumer.attributes.get("feed_dtypes", ())
                        )
                        view_shape = (
                            tuple(declared_shapes[position])
                            if position < len(declared_shapes)
                            else tuple(previous.shape or produced_value.shape)
                        )
                        view_dtype = (
                            str(declared_dtypes[position])
                            if position < len(declared_dtypes)
                            and str(declared_dtypes[position])
                            else previous.dtype or produced_value.dtype
                        )
                        consumer.args[position] = SSAValue(
                            int(produced_value.id),
                            dtype=view_dtype,
                            shape=view_shape,
                            device=previous.device or produced_value.device,
                            accounting={
                                **dict(produced_value.accounting or {}),
                                **dict(previous.accounting or {}),
                                "ssa_storage_alias": int(produced_value.id),
                                "ssa_region_feed": (int(feed_id), int(position)),
                            },
                        )
                        changed = True
                    producer_index = instructions.index(producer_call)
                    consumer_index = instructions.index(consumer)
                    if producer_index > consumer_index:
                        for member in cluster:
                            instructions.remove(member)
                        consumer_index = instructions.index(consumer)
                        instructions[consumer_index:consumer_index] = cluster
                        changed = True
    return changed


def settle_shape_only_repository_returns(module: IRModule) -> bool:
    """Restore identity returns erased from pure shape-only regions.

    The numeric isolator can place a terminal ``reshape`` whose source and
    target descriptors are already identical on the boundary of a planned
    region.  Structural SSA then quite correctly removes the data-moving
    operation, but older region assembly also removes the return itself.  The
    caller still records the exact one-input/one-output region contract.  Use
    that contract to spell the remaining operation as an identity return so
    the caller-owned output buffer is populated instead of left uninitialized.

    This is deliberately limited to pure regions and an unambiguous unary
    call edge with an identical tensor descriptor.  Numeric regions, scalar
    transforms, broadcasts, and shape-changing views are not inferred here.
    """

    calls_by_callee: dict[str, list[tuple[Any, Instr]]] = {}
    for caller in module.functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "call"}:
                    continue
                callee = str(instruction.attributes.get("callee") or "")
                if callee in module.functions:
                    calls_by_callee.setdefault(callee, []).append((
                        caller, instruction,
                    ))

    changed = False
    harmless = {"Const", "StaticRef", "extent", "getattr", "GetAttr"}
    return_ops = {Handler.Ret.value, "ret", "Return", "return"}
    for name, function in module.functions.items():
        instructions = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
        ]
        if any(instruction.op in return_ops for instruction in instructions):
            continue
        if any(instruction.op not in harmless for instruction in instructions):
            continue
        if len(function.args) != 1:
            continue
        callsites = calls_by_callee.get(name, ())
        if not callsites:
            continue
        source = function.args[0]
        source_shape = tuple(source.shape or ())
        source_dtype = source.dtype
        exact = True
        for caller, call in callsites:
            output_ids = tuple(map(
                int, call.attributes.get("output_ids", ())
            ))
            if len(call.args) != 1 or len(output_ids) != 1:
                exact = False
                break
            projected = next((
                instruction.res
                for block in caller.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Load", "load"}
                and instruction.res is not None
                and int(instruction.attributes.get(
                    "source_output_id", -1
                )) == output_ids[0]
            ), None)
            if projected is None:
                exact = False
                break
            if tuple(projected.shape or ()) != source_shape:
                exact = False
                break
            if (
                source_dtype is not None
                and projected.dtype is not None
                and source_dtype != projected.dtype
            ):
                exact = False
                break
        if not exact:
            continue
        last_block = next(reversed(tuple(function.blocks.values())), None)
        if last_block is None:
            continue
        last_block.instrs.append(Instr(Handler.Ret.value, [source], None))
        function.metadata["source_output_value_ids"] = (int(source.id),)
        changed = True
    return changed


_VIEW_OPERATIONS = frozenset({
    "reshape", "view", "flatten", "unsqueeze", "squeeze", "contiguous",
    "detach",
})
_CAST_OPERATIONS = {
    # Value-precision casts under the double-typed working representation.
    # The reference semantics is the numpy backend's ``_cast_`` map:
    # float -> float32 values, double -> float64 values (a copying identity
    # here), int/long -> truncated integer values, bool -> nonzero.  ``double``
    # previously shared the narrowing kernel and silently truncated every
    # mantissa to single precision.
    "float": "cast_double_to_float_values",
    "double": "cast_double_to_double_values",
    "bool": "cast_double_to_bool_values",
    "to_dtype": None,
    "astype": None,
    "to": None,
    "long": "cast_double_to_int_values",
    "int": "cast_double_to_int_values",
    "long_cast": "cast_double_to_int_values",
}
_REDUCTION_CODES = {"sum": 0, "prod": 1, "min": 2, "max": 3, "any": 4, "all": 5}
_SHAPED_SSA_OPERATIONS = {
    "Indexed": "basic_index", "indexed": "basic_index",
    "IndexedStore": "basic_index_store",
    "index_set": "basic_index_store",
    "MatMul": "matmul", "matmul": "matmul",
    "Add": "add", "add": "add",
    "Sub": "sub", "sub": "sub",
    "Mul": "mul", "Mult": "mul", "mul": "mul",
    "Div": "truediv", "truediv": "truediv",
    "Pow": "pow", "pow": "pow",
    "sum": "sum", "mean": "mean", "prod": "prod",
    "min": "min", "max": "max", "any": "any", "all": "all",
    # SymPy's equation graph uses class-style opcodes for elementwise
    # maximum/minimum.  Tensor methods named ``max``/``min`` remain reductions;
    # these spellings are the binary symbolic operators.
    "Max": "maximum", "Min": "minimum",
    "matmul": "matmul", "transpose": "transpose",
    "broadcast_to": "broadcast_to", "expand": "expand",
    "unfold2d": "unfold2d", "fold2d": "fold2d",
}


def _used_value_ids(module: IRModule) -> set[int]:
    return {
        int(value.id)
        for function in module.functions.values()
        for value in function.args
    } | {
        int(value.id)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        for value in (*instruction.args, instruction.res)
        if value is not None
    }


def _constant_payload(instruction: Instr) -> Any:
    if instruction.op != Handler.Const.value:
        return None
    attributes = instruction.attributes
    if attributes.get("values") is not None:
        # The whole-object compiler spells a scalar constant with the same
        # "values" key a vector constant uses; a scalar payload stays scalar.
        payload = attributes["values"]
        sequence = _as_sequence(payload)
        return sequence if sequence is not None else payload
    for key in ("constant", "value", "data"):
        if key in attributes and attributes[key] is not None:
            return attributes[key]
    return None


def _as_sequence(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return None


def _known_count(value: SSAValue) -> int | None:
    shape = tuple(value.shape)
    if shape:
        return prod(int(extent) for extent in shape)
    # A typed rank-zero value is a real scalar. An untyped rank-zero value is
    # the whole-object compiler's current spelling for an extent not yet known.
    return 1 if value.dtype is not None else None


def _attribute(attributes: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in attributes and attributes[name] is not None:
            return attributes[name]
    return None


def settle_static_repository_view_shapes(module: IRModule) -> bool:
    """Resolve literal reshape/view extents before region ABI propagation.

    View lowering aliases storage and therefore normally computes its result
    descriptor while rewriting the instruction.  A later numerical region may
    consume that view, however, so the descriptor has to cross the call edge
    before kernel selection starts.  This pass evaluates only the already-SSA
    literal shape operand; it does not execute Python or infer data values.
    """

    changed = False
    for function in module.functions.values():
        constants = {
            int(instruction.res.id): payload
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
            and (payload := _constant_payload(instruction)) is not None
        }
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is None or len(instruction.args) < 2:
                    continue
                operation = str(
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                    or instruction.attributes.get("tensor_candidate")
                    or instruction.op
                ).casefold()
                if operation not in {"reshape", "view"}:
                    continue
                requested = _as_sequence(constants.get(
                    int(instruction.args[1].id)
                ))
                if requested is None:
                    continue
                resolved = [int(extent) for extent in requested]
                inferred = [
                    index for index, extent in enumerate(resolved)
                    if extent == -1
                ]
                if len(inferred) > 1 or any(
                    extent <= 0 and extent != -1 for extent in resolved
                ):
                    continue
                source_count = _known_count(instruction.args[0])
                known_count = prod(
                    extent for extent in resolved if extent != -1
                )
                if inferred:
                    if (
                        source_count is None or not known_count
                        or source_count % known_count
                    ):
                        continue
                    resolved[inferred[0]] = source_count // known_count
                result_shape = tuple(resolved)
                if tuple(instruction.res.shape) != result_shape:
                    instruction.res.shape = result_shape
                    changed = True
    return changed


def settle_canonical_value_metadata(module: IRModule) -> bool:
    """Fill empty tensor metadata when one canonical value id is unanimous."""

    values: dict[tuple[str, int], list[SSAValue]] = {}
    for function in module.functions.values():
        region_receipt = function.metadata.get("source_region_integral") or {}
        owner = str(region_receipt.get("owner") or function.name)
        for value in function.args:
            values.setdefault((owner, int(value.id)), []).append(value)
        for block in function.blocks.values():
            for instruction in block.instrs:
                for value in (*instruction.args, instruction.res):
                    if value is not None:
                        values.setdefault(
                            (owner, int(value.id)), []
                        ).append(value)
    changed = False
    excluded = {"ssa.aggregate", "ptr", "pointer", "ptrptr_float64"}
    for occurrences in values.values():
        numerical = [
            value for value in occurrences
            if str(value.dtype or "") not in excluded
        ]
        shapes = {
            tuple(value.shape) for value in numerical if tuple(value.shape)
        }
        if len(shapes) != 1:
            continue
        shape = next(iter(shapes))
        dtype = next((
            value.dtype for value in numerical
            if tuple(value.shape) == shape and value.dtype is not None
        ), None)
        for value in numerical:
            if not tuple(value.shape):
                value.shape = shape
                if value.dtype is None and dtype is not None:
                    value.dtype = dtype
                changed = True
    return changed


def settle_repository_ssa_static_extent_operands(module: IRModule) -> bool:
    """Restamp explicit kernel extents after whole-module ABI settlement.

    Tensor lowering materializes shape/rank operands as ordinary constants.
    A producer's exact allocation contract can settle later, when all region
    descriptors coexist.  Keep those already-lowered operands synchronized so
    backends never observe an obsolete view shape.
    """

    changed = False
    for function_name, function in module.functions.items():
        table = getattr(module, "tensor_tables", {}).get(function_name)
        if table is None:
            continue
        definitions: dict[int, list[Instr]] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op == "Const" and instruction.res is not None:
                    definitions.setdefault(
                        int(instruction.res.id), []
                    ).append(instruction)

        def stamp(value: SSAValue, payload: Any, *, vector: bool) -> None:
            nonlocal changed
            for definition in definitions.get(int(value.id), ()):
                attributes = dict(definition.attributes)
                if vector:
                    wanted = tuple(map(int, payload))
                    if tuple(attributes.get("values") or ()) == wanted:
                        continue
                    attributes["values"] = wanted
                    attributes["constant"] = None
                else:
                    wanted = int(payload)
                    if attributes.get("constant") == wanted:
                        continue
                    attributes["constant"] = wanted
                definition.attributes = attributes
                changed = True

        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op != "Call"
                    or instruction.attributes.get("callee")
                    != "broadcast_double"
                    or len(instruction.args) < 6
                ):
                    continue
                source_shape = tuple(instruction.args[0].shape or ())
                output_shape = tuple(instruction.args[1].shape or ())
                # Exact SSA occurrences, not canonical integer ids, own view
                # shape.  Consulting the table first would replace legitimate
                # reshape/broadcast views with their allocation owner's shape.
                if source_shape:
                    stamp(instruction.args[2], source_shape, vector=True)
                    stamp(instruction.args[3], len(source_shape), vector=False)
                if output_shape:
                    stamp(instruction.args[4], output_shape, vector=True)
                    stamp(instruction.args[5], len(output_shape), vector=False)
    return changed


def propagate_repository_ssa_call_metadata(
    module: IRModule, *, authoritative_returns: bool = False,
) -> bool:
    """Settle tensor dtype/shape facts across repository-SSA call edges.

    Planned regions return through explicit aggregate projections while source
    functions use ordinary ``Ret`` values.  This pass correlates both forms,
    including an aggregate forwarded into a projection-only adapter.  It only
    copies already-proven SSA metadata; it executes no source code and invents
    no extents.
    """

    changed_any = False

    # The SSA instruction topology is immutable throughout this metadata
    # fixed point.  Only ``SSAValue.shape``, ``dtype``, and ``accounting`` are
    # enriched.  The old implementation nevertheless rebuilt whole-function
    # value, constant, return, and aggregate-projection indices for every
    # call edge on every iteration.  A training motion has many calls into
    # the same planned regions, so that made propagation a repeated
    # all-module scan rather than a metadata fixed point.
    #
    # Cache references to the live SSAValue objects, not copies of their
    # metadata.  Enrichment therefore remains immediately visible through
    # every cached index and the convergence semantics are unchanged.
    value_cache: dict[int, tuple[SSAValue, ...]] = {}
    values_by_id_cache: dict[int, dict[int, tuple[SSAValue, ...]]] = {}
    constant_result_ids_cache: dict[int, frozenset[int]] = {}
    constant_indices_cache: dict[int, dict[int, int]] = {}
    returned_cache: dict[int, tuple[SSAValue, ...]] = {}
    projection_cache: dict[tuple[int, int], dict[int, SSAValue]] = {}

    def values(function) -> tuple[SSAValue, ...]:
        key = id(function)
        cached = value_cache.get(key)
        if cached is not None:
            return cached
        discovered = [
            *function.args,
            *(
                value
                for block in function.blocks.values()
                for instruction in block.instrs
                for value in (*instruction.args, instruction.res)
                if value is not None
            ),
        ]
        # One SSAValue is normally referenced by several instructions.  Keep
        # each live object once while retaining distinct objects that share a
        # canonical integer value id and all need the same metadata.
        cached = tuple({id(value): value for value in discovered}.values())
        value_cache[key] = cached
        return cached

    def values_by_id(function) -> dict[int, tuple[SSAValue, ...]]:
        key = id(function)
        cached = values_by_id_cache.get(key)
        if cached is not None:
            return cached
        grouped: dict[int, list[SSAValue]] = {}
        for value in values(function):
            grouped.setdefault(int(value.id), []).append(value)
        cached = {
            value_id: tuple(group)
            for value_id, group in grouped.items()
        }
        values_by_id_cache[key] = cached
        return cached

    def constant_result_ids(function) -> frozenset[int]:
        key = id(function)
        cached = constant_result_ids_cache.get(key)
        if cached is not None:
            return cached
        cached = frozenset(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Const" and instruction.res is not None
        )
        constant_result_ids_cache[key] = cached
        return cached

    def enrich(
        function, value_id: int, source: SSAValue, *, authoritative: bool = False,
    ) -> bool:
        changed = False
        source_shape = tuple(source.shape or ())
        source_dtype = source.dtype
        source_aggregate = tuple(
            (source.accounting or {}).get("ssa_aggregate_outputs", ())
        )
        source_physical_dtype = (source.accounting or {}).get(
            "physical_dtype"
        )
        fixed_constant = int(value_id) in constant_result_ids(function)
        for value in values_by_id(function).get(int(value_id), ()):
            if (
                not fixed_constant
                and authoritative
                and tuple(value.shape or ()) != source_shape
            ):
                value.shape = source_shape
                changed = True
            elif (
                not fixed_constant
                and source_shape
                and not tuple(value.shape or ())
            ):
                value.shape = source_shape
                changed = True
            if (
                source_dtype is not None
                and (value.dtype is None or authoritative)
                and value.dtype != source_dtype
            ):
                value.dtype = source_dtype
                changed = True
            if (
                source_physical_dtype is not None
                and (value.accounting or {}).get("physical_dtype") is None
            ):
                value.accounting = {
                    **dict(value.accounting or {}),
                    "physical_dtype": str(source_physical_dtype),
                }
                changed = True
            if (
                source_aggregate
                and not tuple(
                    (value.accounting or {}).get(
                        "ssa_aggregate_outputs", ()
                    )
                )
            ):
                value.accounting = {
                    **dict(value.accounting or {}),
                    "ssa_aggregate_outputs": source_aggregate,
                }
                changed = True
        return changed

    def settle_specialized_formal_descriptor(
        callee_name: str, formal: SSAValue, source: SSAValue,
    ) -> bool:
        """Keep a region input descriptor aligned with its exact call ABI."""

        shape = tuple(map(int, source.shape or ()))
        if not shape:
            return False
        table = getattr(module, "tensor_tables", {}).get(callee_name)
        descriptor = (
            table.by_id(int(formal.id)) if table is not None else None
        )
        if (
            descriptor is None
            or descriptor.storage != "input"
            or tuple(descriptor.shape) == shape
        ):
            return False
        stride = 1
        reversed_strides = []
        for extent in reversed(shape):
            reversed_strides.append(stride)
            stride *= int(extent)
        dtype_bytes = {
            "bool": 1, "i1": 1, "int8": 1, "uint8": 1,
            "int16": 2, "uint16": 2,
            "float32": 4, "float": 4, "int32": 4, "i32": 4,
            "float64": 8, "double": 8, "int64": 8, "i64": 8,
        }.get(str(source.dtype or descriptor.dtype).lower(), 8)
        table.tensors[int(formal.id)] = dataclasses.replace(
            descriptor,
            dtype=str(source.dtype or descriptor.dtype),
            shape=shape,
            strides=tuple(reversed(reversed_strides)),
            byte_size=prod(shape) * dtype_bytes,
            metadata_state="static",
        )
        return True

    def returned(function) -> tuple[SSAValue, ...]:
        key = id(function)
        cached = returned_cache.get(key)
        if cached is not None:
            return cached
        cached = next((
            tuple(instruction.args)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        ), ())
        returned_cache[key] = cached
        return cached

    def constant_indices(function) -> dict[int, int]:
        key = id(function)
        cached = constant_indices_cache.get(key)
        if cached is not None:
            return cached
        result = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op != "Const" or instruction.res is None:
                    continue
                payload = _constant_payload(instruction)
                if isinstance(payload, (int, float)):
                    result[int(instruction.res.id)] = int(payload)
        constant_indices_cache[key] = result
        return result

    def projections(function, aggregate_id: int) -> dict[int, SSAValue]:
        key = (id(function), int(aggregate_id))
        cached = projection_cache.get(key)
        if cached is not None:
            return cached
        constants = constant_indices(function)
        addresses: dict[int, int] = {}
        result: dict[int, SSAValue] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op in {"GetElementPtr", "getelementptr"}
                    and instruction.res is not None
                    and instruction.args
                    and int(instruction.args[0].id) == int(aggregate_id)
                ):
                    position = instruction.attributes.get("aggregate_index")
                    if position is None and len(instruction.args) > 1:
                        position = constants.get(int(instruction.args[1].id))
                    if position is not None:
                        addresses[int(instruction.res.id)] = int(position)
                elif (
                    instruction.op in {"Load", "load"}
                    and instruction.res is not None
                    and instruction.args
                    and int(instruction.args[0].id) in addresses
                ):
                    result[addresses[int(instruction.args[0].id)]] = instruction.res
        projection_cache[key] = result
        return result

    changed = True
    settle_exact_formals = True
    # Authoritative return metadata is a settling phase, not a permanent
    # overwrite mode.  Leaving it enabled for every fixed-point round lets
    # distinct SSA aliases with the same canonical id overwrite one another
    # on every scan, so ``changed`` can remain true forever even though no new
    # fact enters the module.  Apply the exact return contract once, then let
    # the ordinary fill-only propagation carry those settled facts outward.
    settle_exact_returns = bool(authoritative_returns)
    while changed:
        changed = False
        for function_name, function in module.functions.items():
            # Elementwise/reduction results inherit a shaped numerical input;
            # exact result extents already present elsewhere always win.
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.res is None:
                        continue
                    operation = str(
                        instruction.attributes.get("tensor_candidate")
                        or instruction.attributes.get("tensor_operation")
                        or instruction.op
                    )
                    opcode = c_tensor_opcode(operation)
                    if operation not in {
                        "Div", "Sub", "Add", "Mul",
                        "div", "truediv", "sub", "add", "mul",
                        "broadcast_to", "expand",
                    } and not (opcode is not None and opcode[0] == "unary"):
                        continue
                    shaped = next(
                        (value for value in instruction.args if tuple(value.shape or ())),
                        None,
                    )
                    if shaped is not None:
                        changed |= enrich(function, int(instruction.res.id), shaped)

            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {"Call", "call"}:
                        continue
                    callee_name = str(instruction.attributes.get("callee") or "")
                    callee = module.functions.get(callee_name)
                    if callee is None:
                        continue
                    if instruction.attributes.get("ssa_output_argument") is not None:
                        # Authored repository kernels are generic definitions
                        # shared by every tensor shape. Their concrete call
                        # operands are already complete; propagating one
                        # invocation's shapes through the shared formals and
                        # back into another invocation corrupts metadata.
                        continue
                    specialized_call = (
                        "__planned_region_" in callee_name
                    )
                    feed_ids = tuple(map(
                        int, instruction.attributes.get("feed_ids", ())
                    ))
                    for position, (actual, formal) in enumerate(zip(
                        instruction.args, callee.args
                    )):
                        semantic_actual = actual
                        if (
                            position < len(feed_ids)
                            and not tuple(actual.shape or ())
                        ):
                            candidates = values_by_id(function).get(
                                feed_ids[position], ()
                            )
                            numerical_candidates = tuple(
                                candidate for candidate in candidates
                                if str(candidate.dtype or "") not in {
                                    "ssa.aggregate", "ptr", "pointer",
                                    "ptrptr_float64",
                                }
                            )
                            shaped_candidates = tuple(
                                candidate for candidate in numerical_candidates
                                if tuple(candidate.shape or ())
                            )
                            shaped_contracts = {
                                (
                                    tuple(candidate.shape),
                                    str(candidate.dtype or ""),
                                )
                                for candidate in shaped_candidates
                            }
                            # Reshapes are storage aliases and deliberately
                            # permit one semantic feed id to have several
                            # shaped views.  An empty materialized operand may
                            # inherit only a unanimous contract; choosing the
                            # first view would make source order decide ABI.
                            if len(shaped_contracts) == 1:
                                semantic_actual = shaped_candidates[0]
                            elif not shaped_contracts:
                                semantic_actual = next(
                                    iter(numerical_candidates), actual,
                                )
                            # The call operand may be a caller-owned storage
                            # formal while feed_ids names the canonical source
                            # value.  They are one dependency edge: carry the
                            # Program-ABI/source descriptor to both endpoints.
                            if (
                                tuple(semantic_actual.shape or ())
                                and not tuple(actual.shape or ())
                            ):
                                actual.shape = tuple(semantic_actual.shape)
                                changed = True
                            if (
                                semantic_actual.dtype is not None
                                and actual.dtype is None
                            ):
                                actual.dtype = semantic_actual.dtype
                                changed = True
                        # The caller's actual value is the concrete contract
                        # for this specialized formal, including the
                        # distinction between a scalar ``()`` and a tensor.
                        # Reverse propagation remains fill-only so a stale
                        # placeholder cannot reshape the actual argument.
                        changed |= enrich(
                            callee, int(formal.id), semantic_actual,
                            authoritative=(
                                specialized_call and settle_exact_formals
                            ),
                        )
                        if specialized_call:
                            changed |= settle_specialized_formal_descriptor(
                                callee_name, formal, semantic_actual,
                            )
                        changed |= enrich(function, int(actual.id), formal)

                    callee_returns = returned(callee)
                    declared = tuple(map(
                        int, instruction.attributes.get("output_ids", ())
                    ))
                    if declared and instruction.res is not None:
                        caller_outputs = projections(function, int(instruction.res.id))
                        callee_values = {
                            value_id: candidates[0]
                            for value_id, candidates in values_by_id(callee).items()
                        }
                        for position, output_id in enumerate(declared):
                            caller_value = caller_outputs.get(position)
                            tensor_table = getattr(
                                module, "tensor_tables", {}
                            ).get(callee_name)
                            descriptor = (
                                tensor_table.by_id(output_id)
                                if tensor_table is not None else None
                            )
                            material_descriptor = (
                                descriptor
                                if descriptor is not None
                                and bool(tuple(descriptor.shape))
                                and bool(descriptor.owns_allocation)
                                and int(descriptor.data_value_id)
                                == int(output_id)
                                else None
                            )
                            if material_descriptor is not None:
                                # The tensor table is the allocation contract
                                # for a lowered region output.  One canonical
                                # source id can also name earlier singleton
                                # views; choosing the first SSA occurrence
                                # stamped those view extents onto a fully
                                # materialized output at the next call edge.
                                callee_value = SSAValue(
                                    int(output_id),
                                    dtype=str(material_descriptor.dtype),
                                    shape=tuple(material_descriptor.shape),
                                )
                            else:
                                callee_value = callee_values.get(output_id)
                                if callee_value is None and descriptor is not None:
                                    callee_value = callee_values.get(
                                        int(descriptor.data_value_id)
                                    )
                            if caller_value is None or callee_value is None:
                                continue
                            changed |= enrich(
                                function, int(caller_value.id), callee_value,
                                authoritative=(
                                    settle_exact_returns
                                    or material_descriptor is not None
                                ),
                            )
                            if not authoritative_returns:
                                changed |= enrich(
                                    callee, int(callee_value.id), caller_value
                                )
                    elif len(callee_returns) == 1 and instruction.res is not None:
                        changed |= enrich(
                            function, int(instruction.res.id), callee_returns[0],
                            authoritative=settle_exact_returns,
                        )
                        if not authoritative_returns:
                            changed |= enrich(
                                callee, int(callee_returns[0].id),
                                instruction.res,
                            )
                    elif len(callee_returns) > 1 and instruction.res is not None:
                        # An aggregate may be forwarded into a small adapter
                        # whose sole job is to project its members.
                        for consumer_block in function.blocks.values():
                            for consumer in consumer_block.instrs:
                                if consumer.op not in {"Call", "call"}:
                                    continue
                                adapter = module.functions.get(str(
                                    consumer.attributes.get("callee") or ""
                                ))
                                if adapter is None:
                                    continue
                                for position, actual in enumerate(consumer.args):
                                    if int(actual.id) != int(instruction.res.id):
                                        continue
                                    formal = adapter.args[position]
                                    adapter_outputs = projections(adapter, int(formal.id))
                                    for output_index, source in enumerate(callee_returns):
                                        projected = adapter_outputs.get(output_index)
                                        if projected is not None:
                                            changed |= enrich(
                                                adapter, int(projected.id), source,
                                                authoritative=settle_exact_returns,
                                            )
        settle_exact_formals = False
        settle_exact_returns = False
        changed_any |= changed
    return changed_any


def lower_tensor_calls_to_repository_ssa(
    module: IRModule,
    reference: SSATensorCodeReference,
) -> tuple[TensorSSALoweringShortfall, ...]:
    """Replace tensor calls with fully explicit calls into ``reference``.

    Static shapes are converted to ordinary integer/vector constants. Metadata
    views become SSA aliases. If a runtime extent is genuinely absent, the call
    remains visible and a precise shortfall is returned instead of compiling it
    as a one-element tensor.
    """

    # Direct source compilation and autograd both consume the same repository
    # SSA.  Region calls must therefore settle their exact actual/formal and
    # return metadata here, at the common tensor boundary, rather than relying
    # on one caller to remember a private prelude.  All three operations are
    # idempotent; autograd's historical explicit calls remain harmless.
    settle_shape_only_repository_returns(module)
    # Resolve facts owned by instructions/identity before the first call-edge
    # pass.  Specialized-formal settlement is exact on its first round; if it
    # runs first, an empty downstream placeholder can overwrite a literal
    # reshape result before that result has stated its extents.
    settle_static_repository_view_shapes(module)
    settle_canonical_value_metadata(module)
    wire_repository_ssa_region_products(module)
    settle_canonical_value_metadata(module)
    propagate_repository_ssa_call_metadata(module)
    if settle_canonical_value_metadata(module):
        propagate_repository_ssa_call_metadata(module)

    next_id = max(_used_value_ids(module), default=-1) + 1
    linked_roots: set[str] = set()
    shortfalls: list[TensorSSALoweringShortfall] = []

    def fresh(*, shape=(), dtype: str | None = "float64") -> SSAValue:
        nonlocal next_id
        result = SSAValue(next_id, dtype=dtype, shape=tuple(shape))
        next_id += 1
        return result

    def constant(value: Any, dtype: str) -> tuple[SSAValue, Instr]:
        result = fresh(dtype=dtype)
        return result, Instr(Handler.Const.value, [], result, attributes={"constant": value})

    def int_vector(values: Iterable[int]) -> tuple[SSAValue, Instr]:
        sequence = tuple(int(value) for value in values)
        result = fresh(shape=(len(sequence),), dtype="int32")
        return result, Instr(
            Handler.Const.value, [], result,
            attributes={"values": sequence, "constant": None},
        )

    def call(
        callee: str,
        arguments: list[SSAValue],
        result: SSAValue,
        source: Instr,
        *,
        output_argument: int | None = None,
    ) -> Instr:
        # The imported C/LLVM tensor kernels use double-backed buffers even
        # when a value's semantic dtype is Boolean (comparisons and masks are
        # represented as 0.0/1.0). Preserve that physical ABI on every shaped
        # Boolean occurrence crossing one of these calls. Without it, native C
        # allocates one byte per element and ``broadcast_double``/``where``
        # read or write eight, corrupting the activation heap.
        for value in (*arguments, result):
            if (
                tuple(value.shape or ())
                and str(value.dtype or "").casefold() in {"bool", "i1"}
            ):
                value.accounting = {
                    **dict(value.accounting or {}),
                    "physical_dtype": "float64",
                }
        linked_roots.add(callee)
        attributes = {
            key: value for key, value in source.attributes.items()
            if key not in {
                "tensor_operation", "tensor", "tensor_candidate",
                "callee", "lowered_from"
            }
        }
        attributes["callee"] = callee
        if output_argument is not None:
            attributes["ssa_output_argument"] = output_argument
        return Instr(
            Handler.Call.value, arguments, result, attributes=attributes,
            source_span=source.source_span,
        )

    for function_name, function in tuple(module.functions.items()):
        if not hasattr(module, "tensor_tables"):
            module.tensor_tables = {}
        tensor_table = module.tensor_tables.setdefault(
            function_name, SSATensorTable()
        )
        function_argument_ids = {int(value.id) for value in function.args}
        unresolved_argument_ids = {
            int(value.id)
            for value in function.args
            if value.dtype is None and not tuple(value.shape)
        }

        def shape_unknown(value: SSAValue) -> bool:
            """True only when a value's shape is genuinely not known.

            An EMPTY shape with a known dtype is a rank-0 scalar, a fully
            static shape (``alpha = (microstep + 1.0) / microstep_count``).
            Treating every empty shape as "unknown" routed such scalars
            through runtime broadcasting with minted extent queries, and the
            dynamic state then cascaded into every elementwise result after
            them, none of which had a public extent origin at emission.
            """

            if tuple(value.shape or ()):
                return False
            if value.dtype is None or int(value.id) in unresolved_argument_ids:
                return True
            existing = tensor_table.by_id(int(value.id))
            return existing is not None and existing.metadata_state != "static"

        def strides(shape: tuple[int, ...]) -> tuple[int, ...]:
            stride = 1
            result = []
            for extent in reversed(shape):
                result.append(stride)
                stride *= int(extent)
            return tuple(reversed(result))

        def register_tensor(
            value: SSAValue,
            *,
            storage: str,
            metadata_state: str | None = None,
            alias_of: int | None = None,
            data_value_id: int | None = None,
            extent_ids_from: SSATensorDescriptor | None = None,
        ) -> SSATensorDescriptor:
            tensor_id = int(value.id)
            existing = tensor_table.by_id(tensor_id)
            if existing is not None:
                return existing
            shape = tuple(map(int, value.shape))
            state = metadata_state or (
                "unresolved"
                if tensor_id in unresolved_argument_ids
                else "static"
            )
            owner = tensor_table.by_id(alias_of) if alias_of is not None else None
            physical_dtype = str(
                (value.accounting or {}).get(
                    "physical_dtype", value.dtype or "float64"
                )
            ).lower()
            dtype_bytes = {
                "bool": 1, "i1": 1, "int8": 1, "uint8": 1,
                "int16": 2, "uint16": 2,
                "float32": 4, "float": 4, "int32": 4, "i32": 4,
                "float64": 8, "double": 8, "int64": 8, "i64": 8,
            }.get(physical_dtype, 8)
            element_count = prod(shape) if shape else 1
            # A dynamic descriptor requires all three extent values; a result
            # inheriting a dynamic source's state inherits its extents too
            # (elementwise results share the source's runtime metadata).
            extent_ids: dict[str, int | None] = {
                "shape_value_id": None,
                "rank_value_id": None,
                "element_count_value_id": None,
            }
            if state == "dynamic":
                if extent_ids_from is None or extent_ids_from.shape_value_id is None:
                    state = "unresolved"
                else:
                    extent_ids = {
                        "shape_value_id": extent_ids_from.shape_value_id,
                        "rank_value_id": extent_ids_from.rank_value_id,
                        "element_count_value_id": extent_ids_from.element_count_value_id,
                    }
            return tensor_table.register(SSATensorDescriptor(
                tensor_id=tensor_id,
                data_value_id=int(
                    tensor_id if data_value_id is None else data_value_id
                ),
                dtype=str(value.dtype or "float64"),
                shape=shape,
                strides=strides(shape),
                storage=storage,
                metadata_state=state,
                **extent_ids,
                arena_id=owner.arena_id if owner is not None else tensor_id,
                allocation_owner=(
                    owner.allocation_owner if owner is not None else tensor_id
                ),
                owns_allocation=owner is None,
                element_offset=(owner.element_offset if owner is not None else 0),
                byte_offset=(owner.byte_offset if owner is not None else 0),
                byte_size=(
                    element_count * dtype_bytes if state == "static" else None
                ),
                alias_of=alias_of,
                writable=storage != "input",
            ))

        constants: dict[int, Any] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    payload = _constant_payload(instruction)
                    if payload is not None:
                        constants[int(instruction.res.id)] = payload

        # Runtime extents: when a static shape is absent (the whole-program
        # capture keeps parameters symbolic), extent metadata becomes ordinary
        # SSA via the ``extent`` tensor operation -- the same contract the
        # descriptor was designed for ("shape/stride information is either
        # static (the tuples) or itself ordinary SSA (the optional value
        # ids)") and the Fortran emitter already recognises. Each backend
        # lowers ``extent`` its own way; nothing here is float- or
        # backend-specific.
        dynamic_extents: dict[int, dict[str, SSAValue]] = {}

        def ensure_dynamic(
            prefix: list[Instr], tensor: SSAValue
        ) -> dict[str, SSAValue]:
            """Cross ``tensor`` to dynamic metadata, whole.

            The descriptor's own validation states the contract: a dynamic
            tensor carries shape, rank, AND element-count as ordinary SSA,
            never a subset. All three ``extent`` instructions are minted on
            the first request and reused after.
            """

            tensor_id = int(tensor.id)
            cached = dynamic_extents.get(tensor_id)
            if cached is not None:
                return cached
            minted: dict[str, SSAValue] = {}
            for kind in ("shape", "rank", "element_count"):
                # Shape metadata is an integer vector with one slot per
                # tensor axis.  Treating it as a scalar under-allocates the
                # native frame and lets extent emission overwrite the
                # adjacent rank/count values.  Rank and element count remain
                # ordinary scalar metadata.
                value = fresh(
                    shape=(len(tuple(tensor.shape or ())),)
                    if kind == "shape" else (),
                    dtype="int32",
                )
                prefix.append(Instr(
                    "extent", [tensor], value,
                    attributes={
                        "tensor_operation": "extent", "extent_kind": kind,
                    },
                ))
                minted[kind] = value
            dynamic_extents[tensor_id] = minted
            existing = tensor_table.by_id(tensor_id)
            if existing is not None:
                tensor_table.tensors[tensor_id] = dataclasses.replace(
                    existing,
                    metadata_state="dynamic",
                    byte_size=None,
                    shape_value_id=int(minted["shape"].id),
                    rank_value_id=int(minted["rank"].id),
                    element_count_value_id=int(minted["element_count"].id),
                )
            return minted

        per_dim_extents: dict[tuple[int, int], SSAValue] = {}

        def dim_extent(
            prefix: list[Instr], tensor: SSAValue, axis: int
        ) -> SSAValue:
            key = (int(tensor.id), int(axis))
            cached = per_dim_extents.get(key)
            if cached is not None:
                return cached
            value = fresh(dtype="int32")
            prefix.append(Instr(
                "extent", [tensor], value,
                attributes={
                    "tensor_operation": "extent", "extent_kind": "dim",
                    "axis": int(axis),
                },
            ))
            per_dim_extents[key] = value
            return value

        aliases: dict[int, SSAValue] = {}

        def resolve(value: SSAValue) -> SSAValue:
            seen: set[int] = set()
            while int(value.id) in aliases and int(value.id) not in seen:
                seen.add(int(value.id))
                value = aliases[int(value.id)]
            return value

        for block_name, block in function.blocks.items():
            rewritten: list[Instr] = []
            for original in block.instrs:
                instruction = dataclasses.replace(
                    original, args=[resolve(argument) for argument in original.args]
                )
                if (
                    instruction.op in {"Call", "call"}
                    and instruction.attributes.get("call_role_set") == "blas"
                    and instruction.attributes.get("call_role") == "gemm"
                    and instruction.attributes.get("callee") == "blas.gemm"
                ):
                    # AbstractTensor submitted a finite BLAS role call.  Its
                    # semantic identity remains attached while the universal
                    # tensor lowering supplies the ordinary executable
                    # fallback. A backend identity may subsequently replace
                    # that fallback by qualified location.
                    attributes = dict(instruction.attributes)
                    attributes.pop("callee", None)
                    instruction = dataclasses.replace(
                        instruction,
                        op="matmul",
                        attributes=attributes,
                    )
                if (
                    instruction.op in {"Call", "call"}
                    and instruction.attributes.get("callee") is not None
                ):
                    raw_axes = instruction.attributes.get(
                        "basic_index_axes"
                    )
                    source_shape = tuple(map(
                        int,
                        instruction.attributes.get(
                            "basic_index_source_shape", ()
                        ),
                    ))
                    axes = (
                        None if raw_axes is None else tuple(
                            (tuple(map(int, indices)), bool(drop_axis))
                            for indices, drop_axis in raw_axes
                        )
                    )
                    if (
                        instruction.attributes.get("callee")
                        == "index_select_double"
                        and instruction.res is not None
                        and instruction.args
                        and axes is not None
                        and len(axes) == len(source_shape)
                        and all(
                            drop_axis and len(indices) == 1
                            for indices, drop_axis in axes
                        )
                    ):
                        # A region may already have selected the repository
                        # index kernel before whole-module tensor lowering.
                        # Legalize the same dropped-axis scalar contract here
                        # rather than preserving a one-element tensor call.
                        linear_index = 0
                        for axis, (indices, _drop_axis) in enumerate(axes):
                            linear_index = (
                                linear_index * source_shape[axis]
                                + int(indices[0])
                            )
                        index_value, index_definition = constant(
                            linear_index, "int64"
                        )
                        address = fresh(dtype="ptr")
                        instruction.res.shape = ()
                        rewritten.extend((
                            index_definition,
                            Instr(
                                "GetElementPtr",
                                [instruction.args[0], index_value],
                                address,
                                attributes={
                                    "basic_index_linearized": True,
                                },
                                source_span=instruction.source_span,
                            ),
                            Instr(
                                "Load", [address], instruction.res,
                                attributes={
                                    "basic_index_linearized": True,
                                },
                                source_span=instruction.source_span,
                            ),
                        ))
                        continue
                    # AOT numerical-region capture may already have selected
                    # the universal repository fallback before this pass sees
                    # the call.  Keep the AbstractTensor semantic identity on
                    # that fallback so a destination can still replace it by
                    # its qualified GEMM intrinsic.  Without this, authored
                    # ``left @ right``/``left.matmul(right)`` reaches here as
                    # an otherwise-correct ``matmul_double`` call, but the
                    # WebGPU identity has nothing left to recognize.
                    tensor_spelling = (
                        instruction.attributes.get("tensor_operation")
                        or instruction.attributes.get("tensor_candidate")
                    )
                    if (
                        str(tensor_spelling) == "matmul"
                        and instruction.attributes.get("callee")
                        == "matmul_double"
                    ):
                        attributes = dict(instruction.attributes)
                        attributes.setdefault(
                            "backend_intrinsic_candidate",
                            {
                                "semantic_identity": (
                                    "src.common.tensors.abstraction."
                                    "AbstractTensor.matmul"
                                ),
                                "lowering_namespace": "abstract_tensor",
                                "ingested_fallback": False,
                            },
                        )
                        attributes.setdefault(
                            "backend_intrinsic_family", "blas.gemm"
                        )
                        instruction = dataclasses.replace(
                            instruction, attributes=attributes
                        )
                    rewritten.append(instruction)
                    continue
                candidate_only = bool(
                    instruction.attributes.get("tensor_candidate") is not None
                    and instruction.attributes.get("tensor_operation") is None
                    and instruction.attributes.get("tensor") is None
                )
                operation_value = (
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                    or instruction.attributes.get("tensor_candidate")
                )
                # ProcessGraph spells Python arithmetic as ordinary SSA
                # opcodes rather than attaching tensor metadata. Once shape
                # propagation proves the operands are tensors, route those
                # opcodes through the authored row-major tensor source.
                if (
                    operation_value is None
                    and (
                        instruction.op in _SHAPED_SSA_OPERATIONS
                        or c_tensor_opcode(str(instruction.op)) is not None
                    )
                    and instruction.res is not None
                    and (
                        tuple(instruction.res.shape)
                        or any(tuple(argument.shape) for argument in instruction.args)
                        # Symbolic shapes prove nothing either way; an operand
                        # already registered as a tensor descriptor (a kernel
                        # call's result) is proof enough of tensorhood.
                        or any(
                            tensor_table.by_id(int(argument.id)) is not None
                            for argument in instruction.args
                        )
                    )
                ):
                    operation_value = _SHAPED_SSA_OPERATIONS.get(
                        instruction.op, str(instruction.op)
                    )
                if operation_value is None or instruction.res is None:
                    rewritten.append(instruction)
                    continue
                operation = str(operation_value)
                result = instruction.res
                args = list(instruction.args)
                raw_basic_axes = instruction.attributes.get(
                    "basic_index_axes"
                )
                basic_source_shape = tuple(map(
                    int,
                    instruction.attributes.get(
                        "basic_index_source_shape", ()
                    ),
                ))
                basic_axes = (
                    None if raw_basic_axes is None else tuple(
                        (tuple(map(int, indices)), bool(drop_axis))
                        for indices, drop_axis in raw_basic_axes
                    )
                )
                if (
                    args
                    and basic_axes is not None
                    and len(basic_axes) == len(basic_source_shape)
                    and all(
                        drop_axis and len(indices) == 1
                        for indices, drop_axis in basic_axes
                    )
                ):
                    # ``slice`` and ``basic_index`` are two planner spellings
                    # of the same normalized index contract.  Decide scalar
                    # semantics from the authoritative dropped-axis metadata
                    # before either spelling selects a tensor kernel.
                    linear_index = 0
                    for axis, (indices, _drop_axis) in enumerate(basic_axes):
                        linear_index = (
                            linear_index * basic_source_shape[axis]
                            + int(indices[0])
                        )
                    index_value, index_definition = constant(
                        linear_index, "int64"
                    )
                    address = fresh(dtype="ptr")
                    result.shape = ()
                    rewritten.extend((
                        index_definition,
                        Instr(
                            "GetElementPtr", [args[0], index_value], address,
                            attributes={"basic_index_linearized": True},
                            source_span=instruction.source_span,
                        ),
                        Instr(
                            "Load", [address], result,
                            attributes={"basic_index_linearized": True},
                            source_span=instruction.source_span,
                        ),
                    ))
                    continue
                if (
                    len(args) == 2
                    and operation in {"min", "max"}
                    and all(_known_count(argument) == 1 for argument in args)
                    and _known_count(result) == 1
                ):
                    # The planned graph can retain a count-one tensor shape on
                    # a scalar reduction result. Python's two-argument
                    # built-ins still select between two scalar values; they
                    # do not require a dynamic tensor-extent kernel.
                    result.shape = ()
                    rewritten.append(dataclasses.replace(
                        instruction,
                        op={"min": "Min", "max": "Max"}[operation],
                        attributes={
                            key: value
                            for key, value in instruction.attributes.items()
                            if key not in {
                                "tensor_operation", "tensor",
                                "tensor_candidate",
                            }
                        },
                    ))
                    continue
                # Python's two-argument built-ins are the scalar spelling of
                # the same elementwise operations exposed by AbstractTensor.
                # ProcessGraph retains the legal source names ``min``/``max``;
                # normalize them here, after call arity is known, so they use
                # the existing broadcast/scalar repository kernels.  Do not
                # rewrite one-argument iterable reductions.
                if len(args) == 2 and operation in {"min", "max"}:
                    operation = {
                        "min": "minimum",
                        "max": "maximum",
                    }[operation]
                prefix: list[Instr] = []
                emitted: list[Instr] = []

                if operation in {"basic_index", "basic_index_store"}:
                    raw_axes = instruction.attributes.get(
                        "basic_index_axes"
                    )
                    if raw_axes is not None and args:
                        axes = tuple(
                            (tuple(map(int, indices)), bool(drop_axis))
                            for indices, drop_axis in raw_axes
                        )
                        source = args[0]
                        source_shape = tuple(map(
                            int,
                            instruction.attributes.get(
                                "basic_index_source_shape",
                                tuple(source.shape or ()),
                            ),
                        ))
                        if source_shape:
                            result.shape = tuple(
                                (
                                    len(axes[axis][0])
                                    if axis < len(axes) else source_shape[axis]
                                )
                                for axis in range(len(source_shape))
                                if axis >= len(axes) or not axes[axis][1]
                            )
                        source_descriptor = register_tensor(
                            source,
                            storage=(
                                "input"
                                if int(source.id) in function_argument_ids
                                else "temporary"
                            ),
                        )
                        if operation == "basic_index_store" and len(args) >= 2:
                            value = args[-1]
                            register_tensor(
                                value,
                                storage=(
                                    "input"
                                    if int(value.id) in function_argument_ids
                                    else "temporary"
                                ),
                            )
                            offsets = [0]
                            flattened_indices = []
                            for indices, _drop_axis in axes:
                                flattened_indices.extend(indices)
                                offsets.append(len(flattened_indices))
                            shape_value, shape_def = int_vector(source_shape)
                            offsets_value, offsets_def = int_vector(offsets)
                            indices_value, indices_def = int_vector(
                                flattened_indices
                            )
                            rank_value, rank_def = constant(
                                len(source_shape), "int32"
                            )
                            value_count, value_count_def = constant(
                                prod(len(indices) for indices, _drop in axes),
                                "int32",
                            )
                            result.shape = source_shape
                            inplace_store = str(instruction.op) == "IndexedStore"
                            if inplace_store:
                                register_tensor(
                                    result,
                                    storage="view",
                                    alias_of=int(source_descriptor.tensor_id),
                                    data_value_id=int(
                                        source_descriptor.data_value_id
                                    ),
                                )
                                aliases[int(result.id)] = source
                            else:
                                register_tensor(result, storage="temporary")
                            prefix.extend((
                                shape_def, rank_def, offsets_def, indices_def,
                                value_count_def,
                            ))
                            emitted.append(call(
                                (
                                    "index_assign_double"
                                    if inplace_store else "index_set_double"
                                ),
                                (
                                    [
                                        source, shape_value, rank_value,
                                        offsets_value, indices_value, value,
                                        value_count,
                                    ]
                                    if inplace_store else [
                                        source, result, shape_value, rank_value,
                                        offsets_value, indices_value, value,
                                        value_count,
                                    ]
                                ),
                                result,
                                instruction,
                                output_argument=(None if inplace_store else 1),
                            ))
                            rewritten.extend((*prefix, *emitted))
                            continue
                        if operation == "basic_index":
                            scalar_selection = (
                                len(axes) == len(source_shape)
                                and all(
                                    bool(drop_axis) and len(indices) == 1
                                    for indices, drop_axis in axes
                                )
                            )
                            if scalar_selection:
                                # Every integer-indexed axis is removed by
                                # Python indexing.  The result is one scalar,
                                # not a dynamic one-element tensor requiring
                                # index_select/broadcast extent metadata.
                                linear_index = 0
                                for axis, (indices, _drop_axis) in enumerate(
                                    axes
                                ):
                                    linear_index = (
                                        linear_index * source_shape[axis]
                                        + int(indices[0])
                                    )
                                index_value, index_definition = constant(
                                    linear_index, "int64"
                                )
                                address = fresh(dtype="ptr")
                                result.shape = ()
                                rewritten.extend((
                                    index_definition,
                                    Instr(
                                        "GetElementPtr",
                                        [source, index_value],
                                        address,
                                        attributes={
                                            "basic_index_linearized": True,
                                        },
                                        source_span=instruction.source_span,
                                    ),
                                    Instr(
                                        "Load", [address], result,
                                        attributes={
                                            "basic_index_linearized": True,
                                        },
                                        source_span=instruction.source_span,
                                    ),
                                ))
                                continue
                            changed_axes = tuple(
                                (axis, indices)
                                for axis, (indices, _drop_axis) in enumerate(axes)
                                if tuple(indices) != tuple(range(source_shape[axis]))
                            )
                            if not changed_axes:
                                register_tensor(
                                    result,
                                    storage="view",
                                    alias_of=source_descriptor.tensor_id,
                                    data_value_id=source_descriptor.data_value_id,
                                )
                                aliases[int(result.id)] = SSAValue(
                                    source.id,
                                    dtype=result.dtype or source.dtype,
                                    shape=tuple(result.shape),
                                    device=result.device or source.device,
                                    accounting=dict(source.accounting),
                                )
                                continue
                            current = source
                            current_shape = list(source_shape)
                            for selection_index, (axis, indices) in enumerate(
                                reversed(changed_axes)
                            ):
                                last = selection_index == len(changed_axes) - 1
                                destination_shape = list(current_shape)
                                destination_shape[axis] = len(indices)
                                destination = (
                                    result if last else fresh(
                                        shape=tuple(destination_shape),
                                        dtype=result.dtype or source.dtype,
                                    )
                                )
                                register_tensor(destination, storage="temporary")
                                shape_value, shape_def = int_vector(current_shape)
                                indices_value, indices_def = int_vector(indices)
                                rank_value, rank_def = constant(
                                    len(current_shape), "int32"
                                )
                                axis_value, axis_def = constant(axis, "int32")
                                count_value, count_def = constant(
                                    len(indices), "int32"
                                )
                                prefix.extend((
                                    shape_def, rank_def, axis_def,
                                    indices_def, count_def,
                                ))
                                emitted.append(call(
                                    "index_select_double",
                                    [
                                        current, destination, shape_value,
                                        rank_value, axis_value, indices_value,
                                        count_value,
                                    ],
                                    destination, instruction,
                                    output_argument=1,
                                ))
                                current = destination
                                current_shape = destination_shape
                            rewritten.extend((*prefix, *emitted))
                            continue

                # Shape-only operations do not exist at runtime. Preserve the
                # result's shape/dtype annotation while aliasing its storage.
                if operation in _VIEW_OPERATIONS and args:
                    source = args[0]
                    if operation in {"reshape", "view"} and len(args) > 1:
                        requested = _as_sequence(constants.get(int(args[1].id)))
                        if requested is not None:
                            resolved = [int(extent) for extent in requested]
                            inferred = [
                                index for index, extent in enumerate(resolved)
                                if extent == -1
                            ]
                            if len(inferred) <= 1 and all(
                                extent > 0 or extent == -1
                                for extent in resolved
                            ):
                                source_count = _known_count(source)
                                known_count = prod(
                                    extent for extent in resolved
                                    if extent != -1
                                )
                                if (
                                    inferred and source_count is not None
                                    and known_count
                                    and source_count % known_count == 0
                                ):
                                    resolved[inferred[0]] = (
                                        source_count // known_count
                                    )
                                if -1 not in resolved:
                                    result.shape = tuple(resolved)
                    source_descriptor = register_tensor(
                        source,
                        storage=(
                            "input"
                            if int(source.id) in function_argument_ids
                            else "temporary"
                        ),
                    )
                    register_tensor(
                        result,
                        storage="view",
                        metadata_state=source_descriptor.metadata_state,
                        alias_of=source_descriptor.tensor_id,
                        data_value_id=source_descriptor.data_value_id,
                    )
                    aliases[int(result.id)] = SSAValue(
                        source.id,
                        dtype=result.dtype or source.dtype,
                        shape=tuple(result.shape) or tuple(source.shape),
                        device=result.device or source.device,
                        accounting=dict(source.accounting),
                    )
                    continue

                # Constants after the first data operand are structural call
                # operands (axis, shape, dtype token, keepdim, ...).
                # Operand zero is the tensor value even when constant folding
                # proved its payload.  Only later constant operands are call
                # structure (shape, axis, dtype, keepdim, ...).  Dropping a
                # constant first operand made calls such as
                # ``broadcast_to(1.0, (m, n))`` appear to have no source.
                data_positions = (
                    frozenset({0, 1, 2})
                    if operation == "where" else frozenset({0})
                )
                data_args = [
                    argument for position, argument in enumerate(args)
                    if position in data_positions
                    or int(argument.id) not in constants
                ]
                metadata = [
                    constants[int(argument.id)]
                    for position, argument in enumerate(args)
                    if position not in data_positions
                    and int(argument.id) in constants
                ]
                source = data_args[0] if data_args else (args[0] if args else None)
                tensor_opcode = c_tensor_opcode(operation)
                if (
                    tensor_opcode is not None
                    and data_args
                    and all(
                        not tuple(argument.shape or ())
                        and int((argument.accounting or {}).get(
                            "program_abi_rank", 0
                        ) or 0) == 0
                        for argument in data_args
                    )
                ):
                    # Exact scalar operands override provisional count-one
                    # shapes propagated before dropped-axis indexing was
                    # legalized.  Otherwise ordinary scalar arithmetic is
                    # sent through dynamic broadcast kernels solely because
                    # stale planner metadata still says ``(1,)``.
                    result.shape = ()
                if (
                    source is not None
                    and tuple(source.shape or ())
                    and (operation in _REDUCTION_CODES or operation == "mean")
                ):
                    reduction_axis = _attribute(
                        instruction.attributes, "axis", "dim"
                    )
                    if reduction_axis is None:
                        reduction_axis = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                            or _as_sequence(item) is not None
                        ), None)
                    reduction_keepdim = bool(
                        _attribute(instruction.attributes, "keepdim")
                        or False
                    )
                    if reduction_axis is None and operation in {"sum", "mean"}:
                        result.shape = ()
                    elif reduction_axis is not None:
                        raw_axes = _as_sequence(reduction_axis)
                        axes = (
                            tuple(raw_axes)
                            if raw_axes is not None
                            else (reduction_axis,)
                        )
                        rank = len(source.shape)
                        normalized = {
                            int(axis) % rank for axis in axes
                        }
                        result.shape = tuple(
                            1 if reduction_keepdim and index in normalized else extent
                            for index, extent in enumerate(source.shape)
                            if reduction_keepdim or index not in normalized
                        )
                sequence_metadata = tuple(
                    tuple(map(int, sequence))
                    for item in metadata
                    if (sequence := _as_sequence(item)) is not None
                    and all(isinstance(member, (int, float)) for member in sequence)
                )
                if (
                    operation == "unfold2d"
                    and source is not None
                    and len(tuple(source.shape or ())) == 4
                ):
                    n, channels, height, width = map(int, source.shape)
                    kernel = _as_sequence(_attribute(
                        instruction.attributes, "kernel_size"
                    ))
                    stride = _as_sequence(_attribute(
                        instruction.attributes, "stride"
                    )) or (1, 1)
                    padding = _as_sequence(_attribute(
                        instruction.attributes, "padding"
                    )) or (0, 0)
                    dilation = _as_sequence(_attribute(
                        instruction.attributes, "dilation"
                    )) or (1, 1)
                    if kernel is not None and all(
                        len(pair) == 2
                        for pair in (kernel, stride, padding, dilation)
                    ):
                        kh, kw = map(int, kernel)
                        sh, sw = map(int, stride)
                        ph, pw = map(int, padding)
                        dh, dw = map(int, dilation)
                        output_h = (
                            height + 2 * ph - dh * (kh - 1) - 1
                        ) // sh + 1
                        output_w = (
                            width + 2 * pw - dw * (kw - 1) - 1
                        ) // sw + 1
                        result.shape = (
                            n, channels * kh * kw, output_h * output_w,
                        )
                elif operation == "fold2d":
                    output_size = _as_sequence(_attribute(
                        instruction.attributes, "output_size"
                    ))
                    if output_size is None and sequence_metadata:
                        output_size = sequence_metadata[0]
                    if output_size is not None and len(output_size) == 4:
                        result.shape = tuple(map(int, output_size))
                elif operation == "arange":
                    count_value = instruction.attributes.get("arange_count")
                    start_value = instruction.attributes.get(
                        "arange_start", 0.0,
                    )
                    step_value = instruction.attributes.get(
                        "arange_step", 1.0,
                    )
                    if count_value is not None and int(count_value) > 0:
                        start, start_def = constant(
                            float(start_value), "float64",
                        )
                        step, step_def = constant(
                            float(step_value), "float64",
                        )
                        count, count_def = constant(
                            int(count_value), "int32",
                        )
                        result.shape = (int(count_value),)
                        prefix.extend((start_def, step_def, count_def))
                        emitted.append(call(
                            "create_arange",
                            [start, step, count, result],
                            result,
                            instruction,
                            output_argument=3,
                        ))
                elif operation == "gather" and len(data_args) >= 2:
                    indices = data_args[1]
                    if not tuple(source.shape or ()):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name,
                            block_name,
                            operation,
                            (
                                "gather source has no declared rank; "
                                f"source=%{source.id}, indices=%{indices.id}, "
                                "argument_shapes="
                                f"{tuple(tuple(arg.shape or ()) for arg in data_args)!r}"
                            ),
                        ))
                        rewritten.append(instruction)
                        continue
                    raw_dim = _attribute(
                        instruction.attributes, "axis", "dim"
                    )
                    if raw_dim is None:
                        raw_dim = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                        ), 0)
                    axis = int(raw_dim) % len(source.shape)
                    source_shape, source_shape_def = int_vector(source.shape)
                    rank, rank_def = constant(len(source.shape), "int32")
                    axis_value, axis_def = constant(axis, "int32")
                    index_count, index_count_def = constant(
                        _known_count(indices) or 1, "int32"
                    )
                    prefix.extend((
                        source_shape_def, rank_def, axis_def,
                        index_count_def,
                    ))
                    emitted.append(call(
                        "gather_values_double",
                        [
                            source, result, source_shape, rank, axis_value,
                            indices, index_count,
                        ],
                        result, instruction, output_argument=1,
                    ))
                opcode_contract = (
                    None if emitted else c_tensor_opcode(operation)
                )
                if (
                    source is not None
                    and opcode_contract is not None
                ):
                    result.accounting = {
                        **dict(result.accounting or {}),
                        # Both authored C dispatch kernels take ``double *``
                        # outputs. Tensor comparison masks may remain
                        # semantically Boolean while retaining that physical
                        # storage ABI.
                        "physical_dtype": "float64",
                    }
                    if opcode_contract[0] == "binary":
                        shaped_operands = [
                            tuple(operand.shape or ())
                            for operand in data_args
                            if tuple(operand.shape or ())
                        ]
                        if shaped_operands:
                            import numpy as _np

                            try:
                                result.shape = tuple(_np.broadcast_shapes(
                                    *shaped_operands
                                ))
                            except ValueError:
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function_name,
                                    block_name,
                                    operation,
                                    (
                                        "tensor operands have incompatible "
                                        f"shapes {tuple(shaped_operands)!r}; "
                                        f"result=%{result.id}"
                                    ),
                                ))
                                rewritten.append(instruction)
                                continue
                        elif len(data_args) == 1:
                            result.shape = tuple(source.shape or ())
                if (
                    source is not None
                    and opcode_contract is not None
                    and opcode_contract[0] == "unary"
                ):
                    # ``unary_double`` is an elementwise authored kernel: its
                    # output descriptor is identical to its data source.  A
                    # stale cross-call hint must not turn that identity into
                    # a fabricated broadcast (or size an internal frame for
                    # a different model layer).
                    result.shape = tuple(source.shape or ())
                    if source.dtype is not None:
                        result.dtype = source.dtype
                if operation == "where" and len(data_args) == 3:
                    # Settle the broadcast contract before registering the
                    # result descriptor.  Registration owns allocation size;
                    # changing only SSAValue.shape later would leave a
                    # one-element arena behind even if the kernel count were
                    # subsequently corrected.
                    shaped_operands = tuple(
                        tuple(operand.shape or ())
                        for operand in data_args
                        if tuple(operand.shape or ())
                    )
                    if shaped_operands:
                        import numpy as _np

                        try:
                            result.shape = tuple(_np.broadcast_shapes(
                                *shaped_operands
                            ))
                        except ValueError:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name,
                                block_name,
                                operation,
                                "where operands have incompatible shapes "
                                f"{shaped_operands!r}; result=%{result.id}",
                            ))
                            rewritten.append(instruction)
                            continue
                # Registration fixes the physical allocation width, so the
                # repository's double-backed Boolean ABI must be stamped
                # before any descriptor is created.  Doing this only while
                # constructing the eventual Call is too late: byte_size has
                # already been frozen as one byte per semantic bool.
                for value in (*data_args, result):
                    if (
                        tuple(value.shape or ())
                        and str(value.dtype or "").casefold() in {"bool", "i1"}
                    ):
                        value.accounting = {
                            **dict(value.accounting or {}),
                            "physical_dtype": "float64",
                        }
                source_descriptor = None
                if source is not None:
                    source_descriptor = register_tensor(
                        source,
                        storage=(
                            "input"
                            if int(source.id) in function_argument_ids
                            else "temporary"
                        ),
                    )
                for operand in data_args[1:]:
                    register_tensor(
                        operand,
                        storage=(
                            "input"
                            if int(operand.id) in function_argument_ids
                            else "temporary"
                        ),
                    )
                result_descriptor = register_tensor(
                    result,
                    storage="temporary",
                    metadata_state=(
                        source_descriptor.metadata_state
                        if source_descriptor is not None
                        and not tuple(result.shape)
                        else None
                    ),
                    extent_ids_from=source_descriptor,
                )
                source_count = _known_count(source) if source is not None else None
                result_count = _known_count(result)

                def need_count(
                    of: SSAValue | None, count: int | None
                ) -> SSAValue | None:
                    if count is not None:
                        value, definition = constant(count, "int32")
                        prefix.append(definition)
                        return value
                    if of is None:
                        return None
                    # Extent unknown statically: mint it as ordinary SSA.
                    return ensure_dynamic(prefix, of)["element_count"]

                # Derived source definition: cbrt(x) = sign(x)*pow(abs(x),1/3).
                # It is deliberately expressed through the finite primitive
                # basis instead of becoming a new runtime intrinsic.
                if emitted:
                    pass
                elif operation in {
                    "fill", "full", "full_like", "zeros", "zeros_like",
                    "empty", "empty_like", "ones", "ones_like",
                }:
                    like_operation = operation.endswith("_like")
                    if like_operation and source is not None:
                        result.shape = tuple(source.shape or ())
                        if source.dtype is not None:
                            result.dtype = source.dtype
                    fill_value: float | None
                    if operation in {"ones", "ones_like"}:
                        fill_value = 1.0
                    elif operation in {
                        "zeros", "zeros_like", "empty", "empty_like",
                    }:
                        fill_value = 0.0
                    else:
                        explicit = _attribute(
                            instruction.attributes, "fill_value", "value"
                        )
                        if explicit is None:
                            explicit = next((
                                item for item in metadata
                                if isinstance(item, (int, float))
                                and not isinstance(item, bool)
                            ), None)
                        fill_value = (
                            None if explicit is None else float(explicit)
                        )
                    count = need_count(result, _known_count(result))
                    if fill_value is not None and count is not None:
                        scalar, scalar_def = constant(fill_value, "float64")
                        prefix.append(scalar_def)
                        emitted.append(call(
                            "fill_double", [result, scalar, count], result,
                            instruction, output_argument=0,
                        ))
                elif operation == "cbrt" and source is not None:
                    count = need_count(source, source_count)
                    abs_result = fresh(shape=source.shape, dtype=source.dtype or "float64")
                    sign_result = fresh(shape=source.shape, dtype=source.dtype or "float64")
                    power_result = fresh(shape=source.shape, dtype=source.dtype or "float64")
                    unary_abs, abs_def = constant(c_tensor_opcode("abs")[1], "int32")
                    binary_pow, pow_def = constant(c_tensor_opcode("pow")[1], "int32")
                    third, third_def = constant(1.0 / 3.0, "float64")
                    reverse, reverse_def = constant(0, "int32")
                    binary_mul, mul_def = constant(c_tensor_opcode("mul")[1], "int32")
                    prefix.extend((abs_def, pow_def, third_def, reverse_def, mul_def))
                    emitted.extend((
                        call("unary_double", [source, abs_result, count, unary_abs], abs_result, instruction, output_argument=1),
                        call("sign_double", [source, sign_result, count], sign_result, instruction, output_argument=1),
                        call("binary_scalar_double", [abs_result, third, power_result, count, binary_pow, reverse], power_result, instruction, output_argument=2),
                        call("binary_double", [sign_result, power_result, result, count, binary_mul], result, instruction, output_argument=2),
                    ))

                elif operation in _CAST_OPERATIONS and source is not None:
                    callee = _CAST_OPERATIONS[operation]
                    dtype_hint = str(
                        _attribute(instruction.attributes, "dtype", "target_dtype")
                        or (metadata[-1] if metadata else result.dtype or "")
                    ).lower()
                    if callee is None:
                        # Dispatch on the requested dtype exactly.  The old
                        # two-way split sent every non-integer request --
                        # including an explicit ``to_dtype("float64")`` --
                        # through the single-precision narrowing kernel, and
                        # sent ``bool`` through integer truncation (2.7 -> 2
                        # where the reference says nonzero -> 1).
                        if any(
                            token in dtype_hint
                            for token in ("bool", "logical")
                        ):
                            callee = "cast_double_to_bool_values"
                        elif any(
                            token in dtype_hint for token in ("int", "long")
                        ):
                            callee = "cast_double_to_int_values"
                        elif any(
                            token in dtype_hint
                            for token in ("float64", "double")
                        ):
                            callee = "cast_double_to_double_values"
                        else:
                            callee = "cast_double_to_float_values"
                    count = need_count(source, source_count)
                    if count is not None:
                        emitted.append(call(callee, [source, result, count], result, instruction, output_argument=1))

                elif (
                    operation in {"transpose", "swapaxes", "permute"}
                    and source is not None
                    and not source.shape
                ):
                    # Symbolic source: a swap of axes 0 and 1 states rank two
                    # (the same contract matmul_double already carries), so
                    # the axes vector is static while shape and rank ride as
                    # runtime extents. Any other permutation of a symbolic
                    # tensor stays a refusal until ranks are inferred.
                    dims = [int(item) for item in metadata if isinstance(item, (int, float))]
                    dim0 = _attribute(instruction.attributes, "dim0", "axis1")
                    dim1 = _attribute(instruction.attributes, "dim1", "axis2")
                    if dim0 is not None and dim1 is not None:
                        dims = [int(dim0), int(dim1)]
                    if set(dims) == {0, 1}:
                        extents = ensure_dynamic(prefix, source)
                        ensure_dynamic(prefix, result)
                        axes_value, axes_def = int_vector((1, 0))
                        ndim, ndim_def = constant(2, "int32")
                        prefix.extend((axes_def, ndim_def))
                        emitted.append(call(
                            "transpose_double",
                            [source, result, extents["shape"], axes_value, ndim],
                            result, instruction, output_argument=1,
                        ))

                elif operation in {"transpose", "swapaxes", "permute"} and source is not None and source.shape:
                    rank = len(source.shape)
                    axes = _attribute(instruction.attributes, "dims", "axes", "permutation")
                    if axes is None and operation == "permute":
                        axes = next((_as_sequence(item) for item in metadata if _as_sequence(item) is not None), None)
                    if axes is None and operation in {"transpose", "swapaxes"}:
                        dims = [int(item) for item in metadata if isinstance(item, (int, float))]
                        dim0 = _attribute(instruction.attributes, "dim0", "axis1")
                        dim1 = _attribute(instruction.attributes, "dim1", "axis2")
                        if dim0 is not None and dim1 is not None:
                            dims = [int(dim0), int(dim1)]
                        if len(dims) >= 2:
                            axes_list = list(range(rank))
                            a, b = dims[-2] % rank, dims[-1] % rank
                            axes_list[a], axes_list[b] = axes_list[b], axes_list[a]
                            axes = tuple(axes_list)
                    axes_seq = _as_sequence(axes)
                    if axes_seq is not None and len(axes_seq) == rank:
                        shape_value, shape_def = int_vector(source.shape)
                        axes_value, axes_def = int_vector(int(axis) % rank for axis in axes_seq)
                        ndim, ndim_def = constant(rank, "int32")
                        prefix.extend((shape_def, axes_def, ndim_def))
                        emitted.append(call(
                            "transpose_double", [source, result, shape_value, axes_value, ndim],
                            result, instruction, output_argument=1,
                        ))

                elif (
                    operation in _REDUCTION_CODES or operation == "mean"
                ) and source is not None:
                    axis = _attribute(instruction.attributes, "axis", "dim")
                    if axis is None:
                        axis = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                            or (
                                _as_sequence(item) is not None
                                and all(
                                    isinstance(member, (int, float))
                                    for member in _as_sequence(item) or ()
                                )
                            )
                        ), None)
                    axis_sequence = _as_sequence(axis)
                    if axis_sequence is not None and len(axis_sequence) == 1:
                        axis = axis_sequence[0]
                    if axis is None and operation in {"sum", "mean"}:
                        count = need_count(source, source_count)
                        if count is not None and operation == "sum":
                            emitted.append(call("sum_double", [source, count], result, instruction))
                        elif count is not None:
                            # mean = flat sum scaled by the element count; the
                            # division is ordinary scalar SSA, not a kernel.
                            total = fresh(dtype=result.dtype or "float64")
                            emitted.append(call("sum_double", [source, count], total, instruction))
                            emitted.append(Instr(
                                "Div", [total, count], result,
                                attributes={"lowered_from": "mean"},
                                source_span=instruction.source_span,
                            ))
                    elif axis is not None and shape_unknown(source) and operation in _REDUCTION_CODES:
                        # Symbolic source: shape and rank ride as runtime
                        # extents; the kernel already takes them as operands.
                        extents = ensure_dynamic(prefix, source)
                        ensure_dynamic(prefix, result)
                        dim, dim_def = constant(int(axis), "int32")
                        code, code_def = constant(_REDUCTION_CODES[operation], "int32")
                        prefix.extend((dim_def, code_def))
                        emitted.append(call(
                            "reduce_dim_double",
                            [source, result, extents["shape"], extents["rank"], dim, code],
                            result, instruction, output_argument=1,
                        ))
                    elif axis is not None and source.shape and operation in _REDUCTION_CODES:
                        rank = len(source.shape)
                        axis_value = int(axis) % rank
                        shape_value, shape_def = int_vector(source.shape)
                        ndim, ndim_def = constant(rank, "int32")
                        dim, dim_def = constant(axis_value, "int32")
                        code, code_def = constant(_REDUCTION_CODES[operation], "int32")
                        prefix.extend((shape_def, ndim_def, dim_def, code_def))
                        emitted.append(call(
                            "reduce_dim_double", [source, result, shape_value, ndim, dim, code],
                            result, instruction, output_argument=1,
                        ))

                elif operation == "cumsum" and source is not None and source.shape:
                    axis = _attribute(instruction.attributes, "axis", "dim")
                    if axis is None:
                        axis = next((item for item in metadata if isinstance(item, (int, float))), 0)
                    rank = len(source.shape)
                    shape_value, shape_def = int_vector(source.shape)
                    ndim, ndim_def = constant(rank, "int32")
                    dim, dim_def = constant(int(axis) % rank, "int32")
                    prefix.extend((shape_def, ndim_def, dim_def))
                    emitted.append(call(
                        "cumsum_dim_double", [source, result, shape_value, ndim, dim],
                        result, instruction, output_argument=1,
                    ))

                elif operation == "stack" and data_args:
                    operand_shape = tuple(source.shape or ())
                    statically_known = (
                        bool(operand_shape)
                        or all(
                            int(operand.id) not in unresolved_argument_ids
                            for operand in data_args
                        )
                    )
                    if statically_known and all(
                        tuple(operand.shape or ()) == operand_shape
                        for operand in data_args
                    ):
                        rank = len(operand_shape)
                        raw_dim = _attribute(instruction.attributes, "axis", "dim")
                        if raw_dim is None:
                            raw_dim = next((
                                item for item in metadata
                                if isinstance(item, (int, float))
                            ), 0)
                        dim_index = int(raw_dim) % (rank + 1)
                        result.shape = (
                            *operand_shape[:dim_index], len(data_args),
                            *operand_shape[dim_index:],
                        )
                        pointers = fresh(
                            shape=(len(data_args),), dtype="ptrptr_float64"
                        )
                        shape_value, shape_def = int_vector(operand_shape)
                        count_value, count_def = constant(
                            len(data_args), "int32"
                        )
                        rank_value, rank_def = constant(rank, "int32")
                        dim_value, dim_def = constant(dim_index, "int32")
                        prefix.extend((
                            Instr("PointerArray", data_args, pointers),
                            shape_def, count_def, rank_def, dim_def,
                        ))
                        emitted.append(call(
                            "stack_double",
                            [
                                pointers, count_value, shape_value,
                                rank_value, dim_value, result,
                            ],
                            result, instruction, output_argument=5,
                        ))

                elif operation in {"cat", "concat", "concatenate"} and data_args:
                    rank = len(tuple(source.shape or ()))
                    raw_dim = _attribute(instruction.attributes, "axis", "dim")
                    if raw_dim is None:
                        raw_dim = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                        ), 0)
                    dim_index = int(raw_dim) % rank if rank else 0
                    shapes = tuple(tuple(value.shape or ()) for value in data_args)
                    compatible = bool(rank) and all(
                        len(shape) == rank
                        and all(
                            axis == dim_index or shape[axis] == source.shape[axis]
                            for axis in range(rank)
                        )
                        for shape in shapes
                    )
                    if compatible:
                        result_shape = list(map(int, source.shape))
                        result_shape[dim_index] = sum(
                            int(shape[dim_index]) for shape in shapes
                        )
                        result.shape = tuple(result_shape)
                        pointers = fresh(
                            shape=(len(data_args),), dtype="ptrptr_float64"
                        )
                        sizes, sizes_def = int_vector(
                            shape[dim_index] for shape in shapes
                        )
                        shape_value, shape_def = int_vector(source.shape)
                        count_value, count_def = constant(
                            len(data_args), "int32"
                        )
                        rank_value, rank_def = constant(rank, "int32")
                        dim_value, dim_def = constant(dim_index, "int32")
                        prefix.extend((
                            Instr("PointerArray", data_args, pointers),
                            sizes_def, shape_def, count_def, rank_def, dim_def,
                        ))
                        emitted.append(call(
                            "cat_double",
                            [
                                pointers, sizes, count_value, shape_value,
                                rank_value, dim_value, result,
                            ],
                            result, instruction, output_argument=6,
                        ))

                elif operation == "where" and len(data_args) == 3:
                    count = need_count(result, result_count)
                    conformed = []
                    for operand in data_args:
                        if tuple(operand.shape or ()) == tuple(result.shape or ()):
                            conformed.append(operand)
                            continue
                        temporary = fresh(
                            dtype=result.dtype or "float64",
                            shape=tuple(result.shape or ()),
                        )
                        register_tensor(temporary, storage="temporary")
                        if not tuple(operand.shape or ()):
                            scalar = operand
                            payload = constants.get(int(operand.id))
                            if (
                                isinstance(payload, (int, float))
                                and not isinstance(payload, bool)
                                and str(operand.dtype or "").casefold()
                                not in {"double", "float64"}
                            ):
                                scalar, scalar_def = constant(
                                    float(payload), "float64"
                                )
                                prefix.append(scalar_def)
                            emitted.append(call(
                                "fill_double", [temporary, scalar, count],
                                temporary, instruction, output_argument=0,
                            ))
                        elif operand.shape and result.shape:
                            source_shape, source_shape_def = int_vector(operand.shape)
                            source_rank, source_rank_def = constant(
                                len(operand.shape), "int32"
                            )
                            output_shape, output_shape_def = int_vector(result.shape)
                            output_rank, output_rank_def = constant(
                                len(result.shape), "int32"
                            )
                            prefix.extend((
                                source_shape_def, source_rank_def,
                                output_shape_def, output_rank_def,
                            ))
                            emitted.append(call(
                                "broadcast_double",
                                [
                                    operand, temporary, source_shape, source_rank,
                                    output_shape, output_rank,
                                ],
                                temporary, instruction, output_argument=1,
                            ))
                        else:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name,
                                block_name,
                                operation,
                                "where operand cannot be conformed without "
                                f"a result shape; operand=%{operand.id} "
                                f"result=%{result.id}",
                            ))
                            emitted.clear()
                            break
                        conformed.append(temporary)
                    if len(conformed) != 3:
                        rewritten.append(instruction)
                        continue
                    emitted.append(call(
                        "where_double", [*conformed, result, count], result,
                        instruction, output_argument=3,
                    ))

                elif operation in {"broadcast_to", "expand"} and source is not None:
                    requested_shape = _attribute(
                        instruction.attributes, "shape", "size", "sizes"
                    )
                    if requested_shape is None:
                        requested_shape = next(
                            (
                                item for item in metadata
                                if _as_sequence(item) is not None
                            ),
                            None,
                        )
                    output_shape = _as_sequence(requested_shape)
                    if output_shape is None and result.shape:
                        output_shape = tuple(result.shape)
                    if output_shape is not None and (
                        source.shape or source.dtype is not None
                    ):
                        output_shape = tuple(map(int, output_shape))
                        result.shape = output_shape
                        input_shape, input_shape_def = int_vector(source.shape)
                        input_rank, input_rank_def = constant(
                            len(source.shape), "int32"
                        )
                        result_shape, result_shape_def = int_vector(output_shape)
                        result_rank, result_rank_def = constant(
                            len(output_shape), "int32"
                        )
                        prefix.extend((
                            input_shape_def,
                            input_rank_def,
                            result_shape_def,
                            result_rank_def,
                        ))
                        emitted.append(call(
                            "broadcast_double",
                            [
                                source, result, input_shape, input_rank,
                                result_shape, result_rank,
                            ],
                            result, instruction, output_argument=1,
                        ))
                    elif output_shape is not None and shape_unknown(source):
                        output_shape = tuple(map(int, output_shape))
                        result.shape = output_shape
                        source_extents = ensure_dynamic(prefix, source)
                        result_shape, result_shape_def = int_vector(output_shape)
                        result_rank, result_rank_def = constant(
                            len(output_shape), "int32"
                        )
                        prefix.extend((result_shape_def, result_rank_def))
                        emitted.append(call(
                            "broadcast_double",
                            [
                                source, result,
                                source_extents["shape"],
                                source_extents["rank"],
                                result_shape, result_rank,
                            ],
                            result, instruction, output_argument=1,
                        ))

                elif operation == "unfold2d" and source is not None:
                    kernel = _as_sequence(_attribute(
                        instruction.attributes, "kernel_size"
                    ))
                    stride = _as_sequence(_attribute(
                        instruction.attributes, "stride"
                    )) or (1, 1)
                    padding = _as_sequence(_attribute(
                        instruction.attributes, "padding"
                    )) or (0, 0)
                    dilation = _as_sequence(_attribute(
                        instruction.attributes, "dilation"
                    )) or (1, 1)
                    if (
                        len(tuple(source.shape or ())) == 4
                        and kernel is not None
                        and all(
                            len(pair) == 2
                            for pair in (kernel, stride, padding, dilation)
                        )
                    ):
                        dimensions = (
                            *map(int, source.shape), *map(int, kernel),
                            *map(int, stride), *map(int, padding),
                            *map(int, dilation),
                        )
                        dimension_values = []
                        for extent in dimensions:
                            value, definition = constant(extent, "int32")
                            dimension_values.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "unfold2d_double",
                            [source, result, *dimension_values],
                            result, instruction, output_argument=1,
                        ))

                elif operation == "fold2d" and source is not None:
                    sequences = list(sequence_metadata)
                    output_size = _as_sequence(_attribute(
                        instruction.attributes, "output_size"
                    ))
                    kernel = _as_sequence(_attribute(
                        instruction.attributes, "kernel_size"
                    ))
                    stride = _as_sequence(_attribute(
                        instruction.attributes, "stride"
                    ))
                    padding = _as_sequence(_attribute(
                        instruction.attributes, "padding"
                    ))
                    dilation = _as_sequence(_attribute(
                        instruction.attributes, "dilation"
                    ))
                    if output_size is None and sequences:
                        output_size = sequences.pop(0)
                    resolved = []
                    for explicit, default in (
                        (kernel, None), (stride, (1, 1)),
                        (padding, (0, 0)), (dilation, (1, 1)),
                    ):
                        if explicit is not None:
                            resolved.append(tuple(explicit))
                        elif sequences:
                            resolved.append(tuple(sequences.pop(0)))
                        else:
                            resolved.append(default)
                    kernel, stride, padding, dilation = resolved
                    if (
                        output_size is not None
                        and len(output_size) == 4
                        and kernel is not None
                        and all(
                            pair is not None and len(pair) == 2
                            for pair in (kernel, stride, padding, dilation)
                        )
                    ):
                        dimensions = (
                            *map(int, output_size), *map(int, kernel),
                            *map(int, stride), *map(int, padding),
                            *map(int, dilation),
                        )
                        dimension_values = []
                        for extent in dimensions:
                            value, definition = constant(extent, "int32")
                            dimension_values.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "fold2d_double",
                            [source, result, *dimension_values],
                            result, instruction, output_argument=1,
                        ))

                elif operation == "sign" and source is not None:
                    count = need_count(source, source_count)
                    emitted.append(call("sign_double", [source, result, count], result, instruction, output_argument=1))

                elif operation == "matmul" and len(data_args) == 2:
                    left, right = data_args
                    if len(left.shape) == len(right.shape) == 2 and left.shape[1] == right.shape[0]:
                        dimensions = []
                        for extent in (left.shape[0], left.shape[1], right.shape[1]):
                            value, definition = constant(int(extent), "int32")
                            dimensions.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "matmul_double", [left, right, result, *dimensions], result,
                            instruction, output_argument=2,
                        ))
                    elif (
                        len(left.shape) >= 2
                        and len(right.shape) >= 2
                        and left.shape[-1] == right.shape[-2]
                    ):
                        left_batch = tuple(map(int, left.shape[:-2]))
                        right_batch = tuple(map(int, right.shape[:-2]))
                        batch_rank = max(len(left_batch), len(right_batch))
                        left_aligned = (1,) * (batch_rank - len(left_batch)) + left_batch
                        right_aligned = (1,) * (batch_rank - len(right_batch)) + right_batch
                        batch_shape: list[int] = []
                        compatible = True
                        for left_extent, right_extent in zip(
                            left_aligned, right_aligned
                        ):
                            if left_extent == right_extent or left_extent == 1:
                                batch_shape.append(right_extent)
                            elif right_extent == 1:
                                batch_shape.append(left_extent)
                            else:
                                compatible = False
                                break
                        if compatible:
                            m = int(left.shape[-2])
                            n = int(left.shape[-1])
                            p = int(right.shape[-1])
                            result.shape = (*batch_shape, m, p)

                            # The authored C kernel accepts one element offset
                            # per broadcasted batch. Build that finite routing
                            # table here from the static tensor descriptors;
                            # matrix values remain ordinary repository tensors.
                            left_matrix_size = m * n
                            right_matrix_size = n * p
                            left_offsets: list[int] = []
                            right_offsets: list[int] = []
                            batch_count = prod(batch_shape)
                            for flat_index in range(batch_count):
                                remaining = flat_index
                                coordinates = [0] * batch_rank
                                for axis in range(batch_rank - 1, -1, -1):
                                    extent = batch_shape[axis]
                                    coordinates[axis] = remaining % extent
                                    remaining //= extent

                                left_batch_index = 0
                                right_batch_index = 0
                                for coordinate, left_extent, right_extent in zip(
                                    coordinates, left_aligned, right_aligned
                                ):
                                    left_batch_index = (
                                        left_batch_index * left_extent
                                        + (0 if left_extent == 1 else coordinate)
                                    )
                                    right_batch_index = (
                                        right_batch_index * right_extent
                                        + (0 if right_extent == 1 else coordinate)
                                    )
                                left_offsets.append(
                                    left_batch_index * left_matrix_size
                                )
                                right_offsets.append(
                                    right_batch_index * right_matrix_size
                                )

                            left_offsets_value, left_offsets_def = int_vector(
                                left_offsets
                            )
                            right_offsets_value, right_offsets_def = int_vector(
                                right_offsets
                            )
                            prefix.extend((left_offsets_def, right_offsets_def))
                            dimensions = []
                            for extent in (batch_count, m, n, p):
                                value, definition = constant(extent, "int32")
                                dimensions.append(value)
                                prefix.append(definition)
                            emitted.append(call(
                                "batched_matmul_indexed_double",
                                [
                                    left, right, result,
                                    left_offsets_value, right_offsets_value,
                                    *dimensions,
                                ],
                                result, instruction, output_argument=2,
                            ))
                    elif shape_unknown(left) and shape_unknown(right):
                        # Symbolic operands: matmul_double's contract is rank
                        # two, so the three dims become runtime extents.
                        ensure_dynamic(prefix, left)
                        ensure_dynamic(prefix, right)
                        ensure_dynamic(prefix, result)
                        dimensions = [
                            dim_extent(prefix, left, 0),
                            dim_extent(prefix, left, 1),
                            dim_extent(prefix, right, 1),
                        ]
                        emitted.append(call(
                            "matmul_double", [left, right, result, *dimensions], result,
                            instruction, output_argument=2,
                        ))

                else:
                    opcode = c_tensor_opcode(operation)
                    if opcode is not None and source is not None:
                        kind, opcode_value = opcode
                        count_value = source_count if kind == "unary" else result_count
                        count = need_count(source if kind == "unary" else result, count_value)
                        opcode_ssa, opcode_instruction = constant(opcode_value, "int32")
                        if count is not None:
                            prefix.append(opcode_instruction)
                            if kind == "unary":
                                emitted.append(call(
                                    "unary_double", [source, result, count, opcode_ssa], result,
                                    instruction, output_argument=1,
                                ))
                            elif kind == "binary" and len(data_args) == 2:
                                left_constant = constants.get(
                                    int(data_args[0].id)
                                )
                                if (
                                    int(data_args[0].id) in constants
                                    and not tuple(data_args[0].shape)
                                    and tuple(data_args[1].shape)
                                    and isinstance(left_constant, (int, float))
                                    and not isinstance(left_constant, bool)
                                ):
                                    # AST arithmetic retains a literal in
                                    # operand zero (``1 + tensor``).  The
                                    # repository tensor ABI is double-based;
                                    # broadcasting the original i32 storage
                                    # through ``broadcast_double`` reads four
                                    # uninitialised bytes and corrupts the
                                    # result.  Materialize the numerical
                                    # scalar at the ABI dtype and use the
                                    # existing scalar kernel, exactly as the
                                    # right-literal path below already does.
                                    scalar, scalar_def = constant(
                                        float(left_constant), "float64"
                                    )
                                    reverse, reverse_def = constant(1, "int32")
                                    prefix.extend((scalar_def, reverse_def))
                                    emitted.append(call(
                                        "binary_scalar_double",
                                        [
                                            data_args[1], scalar, result,
                                            count, opcode_ssa, reverse,
                                        ],
                                        result,
                                        instruction,
                                        output_argument=2,
                                    ))
                                else:
                                    conformed_args = []
                                    for operand in data_args:
                                        if tuple(operand.shape) == tuple(result.shape):
                                            conformed_args.append(operand)
                                            continue
                                        if shape_unknown(operand) or shape_unknown(result):
                                            # Shapes unknown statically: conforming
                                            # is a runtime decision, so route the
                                            # operand through broadcast_double with
                                            # extent metadata -- never a silent
                                            # mismatched elementwise call.
                                            operand_extents = ensure_dynamic(prefix, operand)
                                            result_extents = ensure_dynamic(prefix, result)
                                            broadcasted = fresh(
                                                shape=result.shape,
                                                dtype=operand.dtype or "float64",
                                            )
                                            emitted.append(call(
                                                "broadcast_double",
                                                [
                                                    operand,
                                                    broadcasted,
                                                    operand_extents["shape"],
                                                    operand_extents["rank"],
                                                    result_extents["shape"],
                                                    result_extents["rank"],
                                                ],
                                                broadcasted,
                                                instruction,
                                                output_argument=1,
                                            ))
                                            register_tensor(
                                                broadcasted, storage="temporary",
                                                metadata_state="dynamic",
                                                extent_ids_from=tensor_table.by_id(int(result.id)),
                                            )
                                            conformed_args.append(broadcasted)
                                            continue
                                        broadcasted = fresh(
                                            shape=result.shape,
                                            dtype=operand.dtype or "float64",
                                        )
                                        input_shape, input_shape_def = int_vector(
                                            operand.shape
                                        )
                                        input_rank, input_rank_def = constant(
                                            len(operand.shape), "int32"
                                        )
                                        output_shape, output_shape_def = int_vector(
                                            result.shape
                                        )
                                        output_rank, output_rank_def = constant(
                                            len(result.shape), "int32"
                                        )
                                        prefix.extend((
                                            input_shape_def,
                                            input_rank_def,
                                            output_shape_def,
                                            output_rank_def,
                                        ))
                                        emitted.append(call(
                                            "broadcast_double",
                                            [
                                                operand,
                                                broadcasted,
                                                input_shape,
                                                input_rank,
                                                output_shape,
                                                output_rank,
                                            ],
                                            broadcasted,
                                            instruction,
                                            output_argument=1,
                                        ))
                                        register_tensor(
                                            broadcasted, storage="temporary"
                                        )
                                        conformed_args.append(broadcasted)
                                    emitted.append(call(
                                        "binary_double", [conformed_args[0], conformed_args[1], result, count, opcode_ssa],
                                        result, instruction, output_argument=2,
                                    ))
                            elif kind == "binary" and len(data_args) == 1:
                                scalar_key = next((key for key in ("right_scalar", "left_scalar") if key in instruction.attributes), None)
                                scalar_payload = instruction.attributes.get(scalar_key) if scalar_key else (metadata[-1] if metadata else None)
                                if scalar_payload is not None and not isinstance(scalar_payload, (tuple, list)):
                                    scalar, scalar_def = constant(float(scalar_payload), "float64")
                                    reverse, reverse_def = constant(int(scalar_key == "left_scalar"), "int32")
                                    prefix.extend((scalar_def, reverse_def))
                                    emitted.append(call(
                                        "binary_scalar_double", [source, scalar, result, count, opcode_ssa, reverse],
                                        result, instruction, output_argument=2,
                                    ))

                if emitted:
                    rewritten.extend(prefix)
                    rewritten.extend(emitted)
                    continue

                recognized = (
                    operation in _VIEW_OPERATIONS
                    or operation in _CAST_OPERATIONS
                    or operation == "cbrt"
                    or reference.operation(operation) is not None
                )
                if recognized and not candidate_only:
                    reason = (
                        "tensor extent/shape or structural operands are not available in SSA"
                        if source is not None and _known_count(source) is None
                        else "referenced SSA exists but its concrete call operands cannot be derived from this instruction"
                    )
                    reason += (
                        "; argument_shapes="
                        f"{tuple(tuple(argument.shape or ()) for argument in args)!r}, "
                        f"result=%{result.id} shape={tuple(result.shape or ())!r}"
                    )
                    shortfalls.append(TensorSSALoweringShortfall(
                        function_name, block_name, operation, reason,
                    ))
                rewritten.append(instruction)
            block.instrs[:] = rewritten

        if aliases:
            for block in function.blocks.values():
                for index, instruction in enumerate(block.instrs):
                    replacement_args = [resolve(argument) for argument in instruction.args]
                    if replacement_args != instruction.args:
                        block.instrs[index] = dataclasses.replace(instruction, args=replacement_args)

        returned_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {Handler.Ret.value, "ret", "Return", "return"}
            for argument in instruction.args
        }
        for tensor_id in returned_ids:
            descriptor = tensor_table.by_id(tensor_id)
            if descriptor is not None:
                tensor_table.tensors[tensor_id] = dataclasses.replace(
                    descriptor, storage="output", writable=True
                )

    for name, function in reference.dependency_closure(*sorted(linked_roots)).items():
        existing = module.functions.get(name)
        if existing is not None and existing is not function:
            raise ValueError(f"repository SSA function collision for {name!r}")
        module.functions[name] = function
        reference_table = getattr(reference.module, "tensor_tables", {}).get(name)
        if reference_table is not None:
            existing_table = module.tensor_tables.get(name)
            if existing_table is not None and existing_table != reference_table:
                raise ValueError(f"repository SSA tensor-table collision for {name!r}")
            module.tensor_tables[name] = reference_table
    if legalize_aggregate_adapters(module):
        settle_canonical_value_metadata(module)
    # Tensor descriptors are created while rewriting functions above.  A
    # producer's final allocation shape can therefore become authoritative
    # only after the whole module has been visited, regardless of whether an
    # aggregate adapter happened to be removed on this invocation.
    propagate_repository_ssa_call_metadata(module)
    settle_repository_ssa_static_extent_operands(module)
    legalize_aggregate_output_views(module)
    return tuple(shortfalls)


__all__ = [
    "TensorSSALoweringShortfall",
    "lower_tensor_calls_to_repository_ssa",
    "legalize_aggregate_adapters",
    "legalize_aggregate_output_views",
    "propagate_repository_ssa_call_metadata",
    "settle_shape_only_repository_returns",
    "wire_repository_ssa_region_products",
]
