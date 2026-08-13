"""An AbstractTensor backend whose storage is compilable repository SSA.

This is a source-producing backend, not an executor.  AbstractTensor's public
methods run against :class:`SSATensorOperations` in the ordinary way.  The
finite backend hooks create calls to implementations in one
:class:`SSATensorCodeReference` and immediately copy each implementation's
complete ordinary-SSA dependency closure into the program module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any

from ..abstraction import AbstractTensor
from ....transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSAValue,
    SSATensorDescriptor,
    SSATensorTable,
)
from ....transmogrifier.ssa_registry import Handler
from ....transmogrifier.tensor_ssa_reference import SSATensorCodeReference
from .c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
    c_tensor_opcode,
)


# These two C entrypoints take pointer tables (double**), while repository SSA
# currently has only a one-level address value.  Their complete source remains
# available for ingestion; advanced stack/cat can already compose structurally.
# The other 22 primitive functions form the directly compilable Fortran ABI.
SSA_TENSOR_FORTRAN_SOURCE_ONLY = frozenset({"stack_double", "cat_double"})


@dataclass(frozen=True, slots=True)
class SSATensorValue:
    """One tensor value owned by an :class:`SSATensorProgram`."""

    program: "SSATensorProgram"
    value: SSAValue
    tensor_id: int

    @property
    def descriptor(self) -> SSATensorDescriptor:
        descriptor = self.program.tensor_table.by_id(self.tensor_id)
        if descriptor is None:
            raise KeyError(f"SSA tensor {self.tensor_id} is not registered")
        return descriptor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.value.shape)

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def dtype(self) -> str | None:
        return self.value.dtype


class SSATensorProgram:
    """Accumulate a tensor expression and all source needed to compile it."""

    def __init__(
        self,
        name: str = "ssa_tensor_program",
        *,
        reference: SSATensorCodeReference | None = None,
    ) -> None:
        self.reference = reference or c_backend_repository_ssa_reference()
        self.function = Function(
            str(name), [], {"entry": BasicBlock("entry")},
        )
        self.tensor_table = SSATensorTable()
        self.module = IRModule(
            {self.function.name: self.function},
            tensor_tables={self.function.name: self.tensor_table},
        )
        self._next_value_id = 0
        self._next_tensor_id = 0
        self._finished = False

    @property
    def block(self) -> BasicBlock:
        return self.function.blocks["entry"]

    def _fresh(self, *, shape=(), dtype="float64") -> SSAValue:
        value = SSAValue(
            self._next_value_id,
            dtype=str(dtype),
            shape=tuple(int(size) for size in shape),
        )
        self._next_value_id += 1
        return value

    def _constant(self, value: Any, dtype: str) -> SSAValue:
        result = self._fresh(dtype=dtype)
        self.block.instrs.append(Instr(
            Handler.Const.value, [], result, attributes={"constant": value}
        ))
        return result

    @staticmethod
    def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        stride = 1
        result = []
        for extent in reversed(shape):
            result.append(stride)
            stride *= int(extent)
        return tuple(reversed(result))

    @staticmethod
    def _dtype_bytes(dtype: str | None) -> int:
        return {
            "bool": 1, "i1": 1,
            "int8": 1, "uint8": 1,
            "int16": 2, "uint16": 2,
            "float32": 4, "float": 4, "int32": 4, "i32": 4,
            "float64": 8, "double": 8, "int64": 8, "i64": 8,
        }.get(str(dtype or "float64").lower(), 8)

    def _tensor(
        self,
        value: SSAValue,
        *,
        storage: str = "temporary",
        alias_of: int | None = None,
        data_value_id: int | None = None,
        writable: bool = True,
    ) -> SSATensorValue:
        tensor_id = self._next_tensor_id
        self._next_tensor_id += 1
        shape = tuple(map(int, value.shape))
        owner_descriptor = (
            self.tensor_table.by_id(alias_of)
            if alias_of is not None else None
        )
        element_count = prod(shape) if shape else 1
        element_offset = (
            int(owner_descriptor.element_offset)
            if owner_descriptor is not None else 0
        )
        byte_offset = (
            int(owner_descriptor.byte_offset)
            if owner_descriptor is not None else 0
        )
        self.tensor_table.register(SSATensorDescriptor(
            tensor_id=tensor_id,
            data_value_id=int(value.id if data_value_id is None else data_value_id),
            dtype=str(value.dtype or "float64"),
            shape=shape,
            strides=self._row_major_strides(shape),
            storage=storage,
            arena_id=(
                owner_descriptor.arena_id
                if owner_descriptor is not None else tensor_id
            ),
            allocation_owner=(
                owner_descriptor.allocation_owner
                if owner_descriptor is not None else tensor_id
            ),
            owns_allocation=owner_descriptor is None,
            element_offset=element_offset,
            byte_offset=byte_offset,
            byte_size=element_count * self._dtype_bytes(value.dtype),
            alias_of=alias_of,
            writable=bool(writable),
        ))
        return SSATensorValue(self, value, tensor_id)

    def _link(self, *roots: str) -> None:
        for name, function in self.reference.dependency_closure(*roots).items():
            existing = self.module.functions.get(name)
            if existing is not None and existing is not function:
                raise ValueError(f"repository SSA function collision for {name!r}")
            self.module.functions[name] = function
            reference_table = getattr(
                self.reference.module, "tensor_tables", {}
            ).get(name)
            if reference_table is not None:
                existing_table = self.module.tensor_tables.get(name)
                if existing_table is not None and existing_table != reference_table:
                    raise ValueError(f"repository SSA tensor-table collision for {name!r}")
                self.module.tensor_tables[name] = reference_table

    def _call(
        self,
        callee: str,
        arguments: list[SSAValue],
        result: SSAValue,
        *,
        output_argument: int | None = None,
    ) -> SSATensorValue:
        attributes: dict[str, Any] = {"callee": str(callee)}
        if output_argument is not None:
            attributes["ssa_output_argument"] = int(output_argument)
        self.block.instrs.append(Instr(
            Handler.Call.value, arguments, result, attributes=attributes
        ))
        self._link(callee)
        return self._tensor(result)

    def input(self, shape: tuple[int, ...], *, dtype: str = "float64") -> SSATensorValue:
        if self._finished:
            raise RuntimeError("cannot add an input after finishing an SSA tensor program")
        value = self._fresh(shape=shape, dtype=dtype)
        self.function.args.append(value)
        return self._tensor(value, storage="input", writable=False)

    def input_tensor(
        self, shape: tuple[int, ...], *, dtype: str = "float64"
    ) -> "SSATensorOperations":
        return SSATensorOperations.input(self, shape, dtype=dtype)

    def full_tensor(
        self, shape: tuple[int, ...], value: float = 0.0
    ) -> "SSATensorOperations":
        tensor = SSATensorOperations()
        tensor._program = self
        tensor.data = self.full(tuple(shape), value)
        return tensor

    def arange_tensor(
        self, start: float, end: float, step: float = 1.0
    ) -> "SSATensorOperations":
        tensor = SSATensorOperations()
        tensor._program = self
        tensor.data = self.arange(start, end, step)
        return tensor

    @staticmethod
    def _broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        result: list[int] = []
        for offset in range(1, max(len(left), len(right)) + 1):
            a = left[-offset] if offset <= len(left) else 1
            b = right[-offset] if offset <= len(right) else 1
            if a == b or b == 1:
                result.append(a)
            elif a == 1:
                result.append(b)
            else:
                raise ValueError(f"SSA tensor operands are not broadcastable: {left}, {right}")
        return tuple(reversed(result))

    def operation(
        self,
        op: str,
        left: Any,
        right: Any,
        *,
        result_shape: tuple[int, ...] | None = None,
    ) -> SSATensorValue:
        """Materialize one fundamental backend operation and its source closure."""

        if self._finished:
            raise RuntimeError("cannot append operations after finishing an SSA tensor program")
        canonical = str(op)
        if canonical.startswith(("i", "r")) and canonical[1:] in self.reference.operations:
            canonical = canonical[1:]

        tensor_operands = [
            operand for operand in (left, right)
            if isinstance(operand, SSATensorValue)
        ]
        if not tensor_operands:
            raise TypeError("an SSA tensor operation requires at least one tensor operand")
        if any(operand.program is not self for operand in tensor_operands):
            raise ValueError("SSA tensor operands must belong to the same source program")

        tensor_scalar_broadcast = None
        if canonical == "matmul" and len(tensor_operands) == 2:
            left_shape, right_shape = tensor_operands[0].shape, tensor_operands[1].shape
            if len(left_shape) != 2 or len(right_shape) != 2 or left_shape[1] != right_shape[0]:
                raise NotImplementedError("the fundamental SSA matmul kernel is rank-2")
            shape = (left_shape[0], right_shape[1])
        elif right is None:
            shape = tensor_operands[0].shape
        elif len(tensor_operands) == 2:
            shape = self._broadcast_shape(tensor_operands[0].shape, tensor_operands[1].shape)
            if tensor_operands[0].shape != shape or tensor_operands[1].shape != shape:
                left_size, right_size = (
                    tensor_operands[0].size, tensor_operands[1].size
                )
                if left_size == 1 and right_size != 1:
                    tensor_scalar_broadcast = (
                        tensor_operands[1], tensor_operands[0], 1
                    )
                elif right_size == 1 and left_size != 1:
                    tensor_scalar_broadcast = (
                        tensor_operands[0], tensor_operands[1], 0
                    )
                else:
                    if isinstance(left, SSATensorValue) and left.shape != shape:
                        left = self.broadcast(left, shape)
                    if isinstance(right, SSATensorValue) and right.shape != shape:
                        right = self.broadcast(right, shape)
                    tensor_operands = [
                        operand for operand in (left, right)
                        if isinstance(operand, SSATensorValue)
                    ]
        else:
            shape = tensor_operands[0].shape

        shape = tuple(result_shape) if result_shape is not None else shape
        result = self._fresh(shape=shape)
        if canonical == "matmul":
            m = self._constant(shape[0], "int32")
            n = self._constant(tensor_operands[0].shape[1], "int32")
            p = self._constant(shape[1], "int32")
            return self._call(
                "matmul_double",
                [tensor_operands[0].value, tensor_operands[1].value, result, m, n, p],
                result,
                output_argument=2,
            )
        if canonical == "sign" and right is None:
            count = self._constant(tensor_operands[0].size, "int32")
            return self._call(
                "sign_double", [tensor_operands[0].value, result, count], result,
                output_argument=1,
            )
        if canonical == "sum" and right is None:
            count = self._constant(tensor_operands[0].size, "int32")
            return self._call(
                "sum_double", [tensor_operands[0].value, count], result
            )
        opcode = c_tensor_opcode(canonical)
        if opcode is None:
            raise NotImplementedError(
                f"SSA tensor code reference has no fundamental call recipe for {canonical!r}"
            )
        kind, opcode_value = opcode
        count = self._constant(prod(shape) if shape else 1, "int32")
        opcode_ssa = self._constant(opcode_value, "int32")
        if kind == "unary" and right is None:
            return self._call(
                "unary_double",
                [tensor_operands[0].value, result, count, opcode_ssa],
                result,
                output_argument=1,
            )
        if kind == "binary" and len(tensor_operands) == 2:
            if tensor_scalar_broadcast is not None:
                array, scalar, reverse_value = tensor_scalar_broadcast
                reverse = self._constant(reverse_value, "int32")
                return self._call(
                    "binary_scalar_double",
                    [array.value, scalar.value, result, count, opcode_ssa, reverse],
                    result,
                    output_argument=2,
                )
            return self._call(
                "binary_double",
                [tensor_operands[0].value, tensor_operands[1].value, result, count, opcode_ssa],
                result,
                output_argument=2,
            )
        if kind == "binary" and len(tensor_operands) == 1:
            scalar = right if isinstance(left, SSATensorValue) else left
            scalar_ssa = self._constant(float(scalar), "float64")
            reverse = self._constant(int(not isinstance(left, SSATensorValue)), "int32")
            return self._call(
                "binary_scalar_double",
                [tensor_operands[0].value, scalar_ssa, result, count, opcode_ssa, reverse],
                result,
                output_argument=2,
            )
        raise NotImplementedError(f"invalid SSA tensor operand layout for {canonical!r}")

    def full(self, shape: tuple[int, ...], value: float = 0.0) -> SSATensorValue:
        result = self._fresh(shape=shape)
        scalar = self._constant(float(value), "float64")
        count = self._constant(prod(shape) if shape else 1, "int32")
        return self._call(
            "fill_double", [result, scalar, count], result, output_argument=0
        )

    def constant(
        self, values: Any, shape: tuple[int, ...], *, dtype: str = "float64"
    ) -> SSATensorValue:
        result = self._fresh(shape=shape, dtype=dtype)

        def flatten(value):
            if isinstance(value, (list, tuple)):
                return [item for part in value for item in flatten(part)]
            if str(dtype) in {"int", "int32", "int64", "i32", "i64"}:
                return [int(value)]
            if str(dtype) in {"bool", "logical", "i1"}:
                return [bool(value)]
            return [float(value)]

        flat = flatten(values)
        if len(flat) != (prod(shape) if shape else 1):
            raise ValueError("SSA tensor literal does not match its inferred shape")
        self.block.instrs.append(Instr(
            Handler.Const.value,
            [],
            result,
            attributes={"values": tuple(flat), "constant": None},
        ))
        return self._tensor(result, storage="constant", writable=False)

    def _int_vector(self, values) -> SSAValue:
        sequence = tuple(int(value) for value in values)
        return self.constant(sequence, (len(sequence),), dtype="int32").value

    def broadcast(
        self, source: SSATensorValue, shape: tuple[int, ...]
    ) -> SSATensorValue:
        target_shape = tuple(map(int, shape))
        if source.shape == target_shape:
            return source
        if self._broadcast_shape(source.shape, target_shape) != target_shape:
            raise ValueError(
                f"cannot broadcast SSA tensor {source.shape} to {target_shape}"
            )
        result = self._fresh(shape=target_shape, dtype=source.dtype or "float64")
        input_shape = self._int_vector(source.shape)
        input_rank = self._constant(len(source.shape), "int32")
        output_shape = self._int_vector(target_shape)
        output_rank = self._constant(len(target_shape), "int32")
        return self._call(
            "broadcast_double",
            [
                source.value,
                result,
                input_shape,
                input_rank,
                output_shape,
                output_rank,
            ],
            result,
            output_argument=1,
        )

    def transpose(self, source: SSATensorValue, axes) -> SSATensorValue:
        axes = tuple(int(axis) % len(source.shape) for axis in axes)
        if sorted(axes) != list(range(len(source.shape))):
            raise ValueError("SSA transpose axes must be a permutation")
        result = self._fresh(shape=tuple(source.shape[axis] for axis in axes))
        shape = self._int_vector(source.shape)
        axes_value = self._int_vector(axes)
        ndim = self._constant(len(source.shape), "int32")
        return self._call(
            "transpose_double", [source.value, result, shape, axes_value, ndim],
            result, output_argument=1,
        )

    def reduce_dim(
        self, source: SSATensorValue, dim: int, operation: int,
        *, keepdim: bool = False,
    ) -> SSATensorValue:
        dim %= len(source.shape)
        shape = (
            source.shape[:dim] + (1,) + source.shape[dim + 1:]
            if keepdim else source.shape[:dim] + source.shape[dim + 1:]
        )
        result = self._fresh(shape=shape)
        shape_value = self._int_vector(source.shape)
        ndim = self._constant(len(source.shape), "int32")
        dim_value = self._constant(dim, "int32")
        op_value = self._constant(operation, "int32")
        return self._call(
            "reduce_dim_double",
            [source.value, result, shape_value, ndim, dim_value, op_value],
            result, output_argument=1,
        )

    def cumsum(self, source: SSATensorValue, dim: int) -> SSATensorValue:
        dim %= len(source.shape)
        result = self._fresh(shape=source.shape)
        shape = self._int_vector(source.shape)
        ndim = self._constant(len(source.shape), "int32")
        dim_value = self._constant(dim, "int32")
        return self._call(
            "cumsum_dim_double",
            [source.value, result, shape, ndim, dim_value],
            result, output_argument=1,
        )

    def index_select(
        self, source: SSATensorValue, dim: int, indices,
    ) -> SSATensorValue:
        dim %= len(source.shape)
        indices = tuple(int(index) for index in indices)
        if any(index < 0 or index >= source.shape[dim] for index in indices):
            raise IndexError("SSA index_select index out of range")
        shape = list(source.shape)
        shape[dim] = len(indices)
        result = self._fresh(shape=tuple(shape))
        source_shape = self._int_vector(source.shape)
        ndim = self._constant(len(source.shape), "int32")
        dim_value = self._constant(dim, "int32")
        index_value = self._int_vector(indices)
        count = self._constant(len(indices), "int32")
        return self._call(
            "index_select_double",
            [source.value, result, source_shape, ndim, dim_value, index_value, count],
            result, output_argument=1,
        )

    def pad(self, source: SSATensorValue, padding, value: float) -> SSATensorValue:
        padding = tuple(int(item) for item in padding)
        if len(padding) % 2 or len(padding) // 2 > len(source.shape):
            raise ValueError("invalid SSA tensor padding specification")
        left = [0] * len(source.shape)
        right = [0] * len(source.shape)
        count = len(padding) // 2
        for index in range(count):
            left[len(source.shape) - count + index] = padding[-2 * (index + 1)]
            right[len(source.shape) - count + index] = padding[-2 * (index + 1) + 1]
        new_shape = tuple(
            source.shape[index] + left[index] + right[index]
            for index in range(len(source.shape))
        )
        result = self._fresh(shape=new_shape)
        source_shape = self._int_vector(source.shape)
        output_shape = self._int_vector(new_shape)
        left_value = self._int_vector(left)
        ndim = self._constant(len(source.shape), "int32")
        fill = self._constant(float(value), "float64")
        return self._call(
            "pad_double_nd",
            [source.value, result, source_shape, output_shape, left_value, ndim, fill],
            result, output_argument=1,
        )

    def where(
        self, condition: SSATensorValue, when_true: SSATensorValue,
        when_false: SSATensorValue,
    ) -> SSATensorValue:
        operands = (condition, when_true, when_false)
        if any(operand.program is not self for operand in operands):
            raise ValueError("SSA where operands must belong to the same source program")
        shape = self._broadcast_shape(when_true.shape, when_false.shape)
        if any(operand.shape != shape for operand in operands):
            raise NotImplementedError("SSA where broadcasting requires the broadcast primitive")
        result = self._fresh(shape=shape)
        count = self._constant(prod(shape) if shape else 1, "int32")
        return self._call(
            "where_double",
            [condition.value, when_true.value, when_false.value, result, count],
            result,
            output_argument=3,
        )

    def arange(self, start: float, end: float, step: float) -> SSATensorValue:
        count_value = max(0, int((end - start) / step))
        result = self._fresh(shape=(count_value,))
        start_ssa = self._constant(float(start), "float64")
        step_ssa = self._constant(float(step), "float64")
        count = self._constant(count_value, "int32")
        return self._call(
            "create_arange", [start_ssa, step_ssa, count, result], result,
            output_argument=3,
        )

    def finish(self, output: SSATensorValue) -> IRModule:
        """Return the complete compilable SSA record for ``output``."""

        if output.program is not self:
            raise ValueError("SSA tensor output belongs to a different source program")
        if not self._finished:
            descriptor = self.tensor_table.by_id(output.tensor_id)
            if descriptor is None:
                raise KeyError(f"SSA tensor {output.tensor_id} is not registered")
            self.tensor_table.tensors[output.tensor_id] = replace(
                descriptor, storage="output"
            )
            self.block.instrs.append(Instr(Handler.Ret.value, [output.value], None))
            self.function.metadata["named_outputs"] = (("result", output.value.id),)
            self._finished = True
        return self.module


class SSATensorOperations(AbstractTensor):
    """The minimal AbstractTensor backend that dispenses compilable SSA source."""

    tensor_type_ = SSATensorValue
    supports_native_batched_matmul = True
    long_dtype_ = "int64"
    bool_dtype_ = "bool"
    float_dtype_ = "float64"

    @classmethod
    def _tensor_from_list(
        cls,
        data,
        dtype=None,
        device=None,
        tape=None,
        *,
        like=None,
        requires_grad=False,
    ) -> "SSATensorOperations":
        program = getattr(like, "_program", None)
        if program is None and isinstance(
            getattr(like, "data", None), SSATensorValue
        ):
            program = like.data.program
        if program is None:
            program = SSATensorProgram("ssa_tensor_literal")
        result = cls(track_time=False, tape=tape)
        result._program = program
        result.data = result.tensor_from_list_(data, dtype, device)
        if requires_grad:
            result._requires_grad = True
        return result

    @classmethod
    def input(
        cls,
        program: SSATensorProgram,
        shape: tuple[int, ...],
        *,
        dtype: str = "float64",
    ) -> "SSATensorOperations":
        tensor = cls()
        tensor._program = program
        tensor.data = program.input(shape, dtype=dtype)
        return tensor

    @staticmethod
    def _normalized_dtype(dtype: Any) -> str:
        text = str(dtype or "float64").lower()
        for canonical in (
            "float64", "float32", "int64", "int32", "int16", "int8",
            "uint64", "uint32", "uint16", "uint8", "bool",
        ):
            if canonical in text:
                return canonical
        if text in {"float", "double"}:
            return "float64" if text == "double" else "float32"
        if text in {"long", "int"}:
            return "int64" if text == "long" else "int32"
        return "float64"

    @classmethod
    def replace_abstract_tensor(
        cls,
        program: SSATensorProgram,
        source: AbstractTensor,
        *,
        snapshot_content: bool = False,
        input_name: str | None = None,
    ) -> "SSATensorOperations":
        """Replace an incoming tensor with an SSA-owned tensor record.

        The returned value retains no reference to ``source`` or its backend.
        By default its payload becomes a compiled-program input: incoming
        discovery content is deliberately erased while shape/dtype become SSA
        facts. ``snapshot_content=True`` instead copies a detached literal into
        SSA Const source, useful when the payload is authored program data.
        """

        if not isinstance(source, AbstractTensor):
            raise TypeError("SSA replacement requires an AbstractTensor")
        shape = tuple(int(extent) for extent in source.shape)
        dtype = cls._normalized_dtype(getattr(source, "dtype", None))
        result = cls()
        result._program = program
        if snapshot_content:
            data = getattr(source, "data", source)
            if hasattr(data, "tolist"):
                data = data.tolist()

            def detached(value):
                if isinstance(value, (list, tuple)):
                    return [detached(item) for item in value]
                if hasattr(value, "item"):
                    try:
                        return value.item()
                    except (TypeError, ValueError):
                        pass
                if isinstance(value, (bool, int, float)):
                    return value
                return float(value)

            result.data = program.constant(detached(data), shape, dtype=dtype)
        else:
            result.data = program.input(shape, dtype=dtype)
            if input_name is not None:
                names = dict(program.function.metadata.get("tensor_input_names", ()))
                names[int(result.data.tensor_id)] = str(input_name)
                program.function.metadata["tensor_input_names"] = tuple(names.items())
        return result

    def _apply_operator__(self, op: str, left: Any, right: Any) -> SSATensorValue:
        owner = left if isinstance(left, SSATensorValue) else right
        if not isinstance(owner, SSATensorValue):
            raise TypeError("SSA tensor arithmetic requires an SSA tensor operand")
        return owner.program.operation(op, left, right)

    def _apply_operator(self, op: str, left: Any, right: Any):
        """Use AbstractTensor's operator surface without materializing runtime tensors."""

        left_data = self._data(left)
        right_data = self._data(right)
        result = type(self)()
        result.data = self._apply_operator__(op, left_data, right_data)
        return result

    def sign(self):
        return self._apply_operator("sign", self, None)

    def maximum(self, other):
        return self._apply_operator("maximum", self, other)

    def minimum(self, other):
        return self._apply_operator("minimum", self, other)

    @staticmethod
    def where(condition, when_true, when_false, *, allow_scalar=True):
        if not isinstance(condition, SSATensorOperations):
            raise TypeError("SSA tensor where requires an SSA tensor condition")
        program = condition.data.program

        def materialize(value):
            if isinstance(value, SSATensorOperations):
                return value.data
            if isinstance(value, SSATensorValue):
                return value
            if allow_scalar and isinstance(value, (int, float, bool)):
                return program.full(condition.data.shape, float(value))
            raise TypeError("SSA tensor where branches must be tensors or scalars")

        result = type(condition)()
        result.data = program.where(
            condition.data, materialize(when_true), materialize(when_false)
        )
        return result

    def _comparison(self, operation: str, other):
        return self._apply_operator(operation, self, other)

    def __eq__(self, other):
        return self._comparison("equal", other)

    def __ne__(self, other):
        return self._comparison("not_equal", other)

    def __lt__(self, other):
        return self._comparison("less", other)

    def __le__(self, other):
        return self._comparison("less_equal", other)

    def __gt__(self, other):
        return self._comparison("greater", other)

    def __ge__(self, other):
        return self._comparison("greater_equal", other)

    @staticmethod
    def _data(value: Any) -> Any:
        return value.data if isinstance(value, AbstractTensor) else value

    def tensor_from_list_(self, data, dtype=None, device=None) -> SSATensorValue:
        def shape_of(value):
            if not isinstance(value, (list, tuple)):
                return ()
            if not value:
                return (0,)
            child = shape_of(value[0])
            if any(shape_of(item) != child for item in value):
                raise ValueError("SSA tensor literals must be rectangular")
            return (len(value),) + child

        program = getattr(self, "_program", None) or SSATensorProgram("ssa_tensor_literal")
        self._program = program
        return program.constant(data, shape_of(data))

    def full_(self, size, fill_value, dtype=None, device=None) -> SSATensorValue:
        program = getattr(self, "_program", None)
        if program is None:
            raise ValueError("construct SSA tensors through an SSATensorProgram")
        return program.full(tuple(size), float(fill_value))

    def zeros_(self, size, dtype=None, device=None) -> SSATensorValue:
        return self.full_(size, 0.0, dtype, device)

    def ones_(self, size, dtype=None, device=None) -> SSATensorValue:
        return self.full_(size, 1.0, dtype, device)

    def clone_(self, tensor: SSATensorValue | None = None) -> SSATensorValue:
        source = tensor or self.data
        return source.program.operation("add", source, 0.0)

    def _view_value(self, shape: tuple[int, ...]) -> SSATensorValue:
        value = SSAValue(
            self.data.value.id,
            dtype=self.data.dtype,
            shape=tuple(shape),
        )
        return self.data.program._tensor(
            value,
            storage="view",
            alias_of=self.data.tensor_id,
            data_value_id=self.data.descriptor.data_value_id,
            writable=self.data.descriptor.writable,
        )

    def reshape_(self, shape) -> SSATensorValue:
        requested = list(shape)
        unknown = [index for index, size in enumerate(requested) if int(size) == -1]
        if len(unknown) > 1:
            raise ValueError("only one inferred SSA tensor dimension is permitted")
        known = prod(int(size) for size in requested if int(size) != -1)
        if unknown:
            if not known or self.data.size % known:
                raise ValueError("SSA tensor reshape has incompatible size")
            requested[unknown[0]] = self.data.size // known
        target = tuple(int(size) for size in requested)
        if (prod(target) if target else 1) != self.data.size:
            raise ValueError("SSA tensor reshape has incompatible size")
        return self._view_value(target)

    def flatten_(self, start_dim=0, end_dim=-1) -> SSATensorValue:
        shape = self.data.shape
        start_dim %= len(shape)
        end_dim %= len(shape)
        target = shape[:start_dim] + (prod(shape[start_dim:end_dim + 1]),) + shape[end_dim + 1:]
        return self._view_value(target)

    def unsqueeze_(self, dim) -> SSATensorValue:
        dim %= len(self.data.shape) + 1
        target = self.data.shape[:dim] + (1,) + self.data.shape[dim:]
        return self._view_value(target)

    def squeeze_(self, dim=None) -> SSATensorValue:
        shape = self.data.shape
        if dim is None:
            target = tuple(size for size in shape if size != 1)
        else:
            dim %= len(shape)
            target = shape[:dim] + shape[dim + 1:] if shape[dim] == 1 else shape
        return self._view_value(target)

    def sum_(self, dim=None, keepdim=False) -> SSATensorValue:
        if dim is not None:
            return self.data.program.reduce_dim(self.data, int(dim), 0, keepdim=keepdim)
        shape = (1,) * len(self.data.shape) if keepdim else ()
        return self.data.program.operation("sum", self.data, None, result_shape=shape)

    def mean_(self, dim=None, keepdim=False) -> SSATensorValue:
        if dim is None:
            total = self.data.program.operation(
                "sum", self.data, None, result_shape=(() if not keepdim else (1,) * len(self.data.shape))
            )
            return self.data.program.operation(
                "truediv", total, float(self.data.size)
            )
        dim = int(dim) % len(self.data.shape)
        total = self.data.program.reduce_dim(
            self.data, dim, 0, keepdim=keepdim
        )
        return self.data.program.operation(
            "truediv", total, float(self.data.shape[dim])
        )

    def prod_(self, dim=None, keepdim=False) -> SSATensorValue:
        if dim is None:
            tensor = self.reshape_((self.data.size,))
            return tensor.program.reduce_dim(tensor, 0, 1, keepdim=keepdim)
        return self.data.program.reduce_dim(self.data, int(dim), 1, keepdim=keepdim)

    def min_(self, dim=None, keepdim=False) -> SSATensorValue:
        source = self.data if dim is not None else self.reshape_((self.data.size,))
        return source.program.reduce_dim(source, 0 if dim is None else int(dim), 2, keepdim=keepdim)

    def max_(self, dim=None, keepdim=False) -> SSATensorValue:
        source = self.data if dim is not None else self.reshape_((self.data.size,))
        return source.program.reduce_dim(source, 0 if dim is None else int(dim), 3, keepdim=keepdim)

    def any_(self, dim=None) -> SSATensorValue:
        source = self.data if dim is not None else self.reshape_((self.data.size,))
        return source.program.reduce_dim(source, 0 if dim is None else int(dim), 4)

    def all_(self, dim=None) -> SSATensorValue:
        source = self.data if dim is not None else self.reshape_((self.data.size,))
        return source.program.reduce_dim(source, 0 if dim is None else int(dim), 5)

    def cumsum_(self, dim=0) -> SSATensorValue:
        return self.data.program.cumsum(self.data, int(dim))

    def softmax_(self, dim=-1) -> SSATensorValue:
        dim = int(dim) % len(self.data.shape)
        maximum = self.data.program.reduce_dim(
            self.data, dim, 3, keepdim=True
        )
        shifted = self.data.program.operation("sub", self.data, maximum)
        numerator = self.data.program.operation("exp", shifted, None)
        denominator = self.data.program.reduce_dim(
            numerator, dim, 0, keepdim=True
        )
        return self.data.program.operation("truediv", numerator, denominator)

    def log_softmax_(self, dim=-1) -> SSATensorValue:
        dim = int(dim) % len(self.data.shape)
        maximum = self.data.program.reduce_dim(
            self.data, dim, 3, keepdim=True
        )
        shifted = self.data.program.operation("sub", self.data, maximum)
        exponentials = self.data.program.operation("exp", shifted, None)
        denominator = self.data.program.reduce_dim(
            exponentials, dim, 0, keepdim=True
        )
        log_denominator = self.data.program.operation("log", denominator, None)
        return self.data.program.operation("sub", shifted, log_denominator)

    def permute_(self, dims) -> SSATensorValue:
        return self.data.program.transpose(self.data, dims)

    def transpose_(self, dim0, dim1) -> SSATensorValue:
        axes = list(range(len(self.data.shape)))
        dim0 %= len(axes)
        dim1 %= len(axes)
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.data.program.transpose(self.data, axes)

    def swapaxes_(self, axis1, axis2) -> SSATensorValue:
        return self.transpose_(axis1, axis2)

    def index_select_(self, dim, indices) -> SSATensorValue:
        if isinstance(indices, AbstractTensor):
            raise NotImplementedError("dynamic SSA index vectors require an SSA input vector")
        return self.data.program.index_select(self.data, int(dim), indices)

    def pad_(self, pad, value=0, mode="constant") -> SSATensorValue:
        if mode != "constant":
            raise NotImplementedError("the fundamental SSA pad kernel is constant mode")
        return self.data.program.pad(self.data, pad, float(value))

    def matmul_(self, tensor, other) -> SSATensorValue:
        left, right = self._data(tensor), self._data(other)
        return left.program.operation("matmul", left, right)

    def where_(self, x, y) -> SSATensorValue:
        return self.data.program.where(self.data, self._data(x), self._data(y))

    def arange_(self, start, end=None, step=1, *, dtype=None, device=None) -> SSATensorValue:
        if end is None:
            start, end = 0, start
        program = getattr(self, "_program", None)
        if program is None:
            raise ValueError("construct SSA tensors through an SSATensorProgram")
        return program.arange(float(start), float(end), float(step))

    def get_shape(self) -> tuple[int, ...]:
        return self.data.shape

    def get_ndims(self) -> int:
        return len(self.data.shape)

    def get_dtype_(self, tensor: SSATensorValue | None = None) -> str | None:
        return (tensor or self.data).dtype

    def get_device_(self, tensor: SSATensorValue | None = None) -> str:
        return "ssa"

    def numel_(self, tensor: SSATensorValue | None = None) -> int:
        return (tensor or self.data).size

    def compilable_ssa(self) -> IRModule:
        """Dispense the caller plus the full transitive primitive source record."""

        return self.data.program.finish(self.data)


def _operator_hook(operation: str):
    def hook(self, other=None, *unused, **unused_keywords):
        right = self._data(other) if other is not None else None
        return self.data.program.operation(operation, self.data, right)
    hook.__name__ = f"{operation}_"
    return hook


for _operation in (
    "sqrt", "exp", "log", "neg", "abs", "trunc", "floor", "ceil",
    "isfinite", "isnan", "isinf", "logical_not", "tanh", "sin", "cos", "tan",
    "asin", "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh", "sign",
    "less", "less_equal", "greater", "greater_equal", "equal", "not_equal",
    "maximum", "minimum",
):
    setattr(SSATensorOperations, f"{_operation}_", _operator_hook(_operation))


def _round_hook(self, n=None):
    if n in (None, 0):
        return self.data.program.operation("round", self.data, None)
    scale = 10.0 ** int(n)
    scaled = self.data.program.operation("mul", self.data, scale)
    rounded = self.data.program.operation("round", scaled, None)
    return self.data.program.operation("truediv", rounded, scale)


SSATensorOperations.round_ = _round_hook


def emit_ssa_tensor_backend_runtime(
    *,
    reference: SSATensorCodeReference | None = None,
    name: str = "ssa_tensor_backend_runtime",
):
    """Emit the finite SSA AbstractTensor basis as an ordinary Fortran module."""

    from ....compiler.ssa_fortran_backend import emit_module

    reference = reference or c_backend_repository_ssa_reference()
    functions = {
        function_name: function
        for function_name, function in reference.module.functions.items()
        if function_name not in SSA_TENSOR_FORTRAN_SOURCE_ONLY
    }
    module = IRModule(functions)
    return emit_module(
        module,
        name=name,
        extra_roots=tuple(functions),
    )


def compile_ssa_tensor_backend_runtime(
    directory,
    *,
    reference: SSATensorCodeReference | None = None,
    name: str = "ssa_tensor_backend_runtime",
    standalone: bool = False,
):
    """Compile the same source reference into symbols available at runtime."""

    from ....compiler.ssa_fortran_backend import compile_module

    emitted = emit_ssa_tensor_backend_runtime(reference=reference, name=name)
    return compile_module(emitted, directory=directory, standalone=standalone)


def replace_abstract_tensor_content_with_ssa(
    program: SSATensorProgram,
    value: Any,
    *,
    snapshot_content: bool = False,
    path: str = "input",
):
    """Recursively replace AbstractTensor leaves with SSA-owned tensors.

    Returned containers contain no original AbstractTensor leaves. This is the
    feed/ProcessGraph boundary: compound operations subsequently dispatch only
    through :class:`SSATensorOperations` and its referenced SSA source.
    """

    if isinstance(value, SSATensorOperations):
        if value.data.program is program:
            return value
        return SSATensorOperations.replace_abstract_tensor(
            program,
            value,
            snapshot_content=snapshot_content,
            input_name=path,
        )
    if isinstance(value, AbstractTensor):
        return SSATensorOperations.replace_abstract_tensor(
            program,
            value,
            snapshot_content=snapshot_content,
            input_name=path,
        )
    if isinstance(value, dict):
        return {
            key: replace_abstract_tensor_content_with_ssa(
                program,
                item,
                snapshot_content=snapshot_content,
                path=f"{path}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            replace_abstract_tensor_content_with_ssa(
                program,
                item,
                snapshot_content=snapshot_content,
                path=f"{path}.{index}",
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            replace_abstract_tensor_content_with_ssa(
                program,
                item,
                snapshot_content=snapshot_content,
                path=f"{path}.{index}",
            )
            for index, item in enumerate(value)
        ]
    return value




__all__ = [
    "SSA_TENSOR_FORTRAN_SOURCE_ONLY",
    "SSATensorOperations",
    "SSATensorProgram",
    "SSATensorValue",
    "replace_abstract_tensor_content_with_ssa",
    "compile_ssa_tensor_backend_runtime",
    "emit_ssa_tensor_backend_runtime",
]
