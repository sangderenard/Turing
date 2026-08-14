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


@dataclass(frozen=True, order=True)
class TensorSSALoweringShortfall:
    function: str
    block: str
    operation: str
    reason: str


_VIEW_OPERATIONS = frozenset({
    "reshape", "view", "flatten", "unsqueeze", "squeeze", "contiguous",
})
_CAST_OPERATIONS = {
    "float": "cast_double_to_float_values",
    "double": "cast_double_to_float_values",
    "to_dtype": None,
    "astype": None,
    "to": None,
    "long": "cast_double_to_int_values",
    "int": "cast_double_to_int_values",
    "long_cast": "cast_double_to_int_values",
}
_REDUCTION_CODES = {"sum": 0, "prod": 1, "min": 2, "max": 3, "any": 4, "all": 5}
_SHAPED_SSA_OPERATIONS = {
    "MatMul": "matmul", "matmul": "matmul",
    "Add": "add", "add": "add",
    "Sub": "sub", "sub": "sub",
    "Mul": "mul", "Mult": "mul", "mul": "mul",
    "Div": "truediv", "truediv": "truediv",
    "Pow": "pow", "pow": "pow",
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
        linked_roots.add(callee)
        attributes = {
            key: value for key, value in source.attributes.items()
            if key not in {
                "tensor_operation", "tensor", "callee", "lowered_from"
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
            dtype_bytes = {
                "bool": 1, "i1": 1, "int8": 1, "uint8": 1,
                "int16": 2, "uint16": 2,
                "float32": 4, "float": 4, "int32": 4, "i32": 4,
                "float64": 8, "double": 8, "int64": 8, "i64": 8,
            }.get(str(value.dtype or "float64").lower(), 8)
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
                value = fresh(dtype="int32")
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
                operation_value = (
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                )
                # ProcessGraph spells Python arithmetic as ordinary SSA
                # opcodes rather than attaching tensor metadata. Once shape
                # propagation proves the operands are tensors, route those
                # opcodes through the authored row-major tensor source.
                if (
                    operation_value is None
                    and instruction.op in _SHAPED_SSA_OPERATIONS
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
                    operation_value = _SHAPED_SSA_OPERATIONS[instruction.op]
                if operation_value is None or instruction.res is None:
                    rewritten.append(instruction)
                    continue
                operation = str(operation_value)
                result = instruction.res
                args = list(instruction.args)
                prefix: list[Instr] = []
                emitted: list[Instr] = []

                # Shape-only operations do not exist at runtime. Preserve the
                # result's shape/dtype annotation while aliasing its storage.
                if operation in _VIEW_OPERATIONS and args:
                    source = args[0]
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
                data_args = [argument for argument in args if int(argument.id) not in constants]
                metadata = [constants[int(argument.id)] for argument in args if int(argument.id) in constants]
                source = data_args[0] if data_args else (args[0] if args else None)
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
                if operation == "cbrt" and source is not None:
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
                        callee = (
                            "cast_double_to_int_values"
                            if any(token in dtype_hint for token in ("int", "long", "bool"))
                            else "cast_double_to_float_values"
                        )
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
                        axis = next((item for item in metadata if isinstance(item, (int, float))), None)
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
                    elif axis is not None and not source.shape and operation in _REDUCTION_CODES:
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

                elif operation == "where" and len(data_args) == 3:
                    count = need_count(result, result_count)
                    emitted.append(call(
                        "where_double", [*data_args, result, count], result,
                        instruction, output_argument=3,
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
                    elif not left.shape and not right.shape:
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
                            if kind == "unary" and len(data_args) == 1:
                                emitted.append(call(
                                    "unary_double", [source, result, count, opcode_ssa], result,
                                    instruction, output_argument=1,
                                ))
                            elif kind == "binary" and len(data_args) == 2:
                                conformed_args = []
                                for operand in data_args:
                                    if tuple(operand.shape) == tuple(result.shape):
                                        conformed_args.append(operand)
                                        continue
                                    if not operand.shape or not result.shape:
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
                if recognized:
                    reason = (
                        "tensor extent/shape or structural operands are not available in SSA"
                        if source is not None and _known_count(source) is None
                        else "referenced SSA exists but its concrete call operands cannot be derived from this instruction"
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
    return tuple(shortfalls)


__all__ = ["TensorSSALoweringShortfall", "lower_tensor_calls_to_repository_ssa"]
