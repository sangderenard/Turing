"""One-boundary C execution of the established :class:`FusedProgram` IR.

``FusedProgram`` remains the semantic program.  This module only compiles its
equal-shape elementwise regions into a private slot plan matching the native C
ABI, then executes that plan through one CFFI call.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence

from ..fused_ir import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FusedProgram,
    Meta,
    OpStep,
    canonical_elementwise_op,
    ordered_feed_ids,
    primary_output_id,
)
from .c_backend import C, CTensor, ffi


_OP_NAMES = {
    "add": "CT_OP_ADD",
    "sub": "CT_OP_SUB",
    "mul": "CT_OP_MUL",
    "truediv": "CT_OP_DIV",
    "pow": "CT_OP_POW",
    "mod": "CT_OP_MOD",
    "floordiv": "CT_OP_FLOORDIV",
    "sqrt": "CT_OP_SQRT",
    "exp": "CT_OP_EXP",
    "log": "CT_OP_LOG",
    "tanh": "CT_OP_TANH",
    "sin": "CT_OP_SIN",
    "cos": "CT_OP_COS",
    "tan": "CT_OP_TAN",
    "asin": "CT_OP_ASIN",
    "acos": "CT_OP_ACOS",
    "atan": "CT_OP_ATAN",
    "sinh": "CT_OP_SINH",
    "cosh": "CT_OP_COSH",
    "asinh": "CT_OP_ASINH",
    "acosh": "CT_OP_ACOSH",
    "atanh": "CT_OP_ATANH",
    "neg": "CT_OP_NEG",
    "abs": "CT_OP_ABS",
    "round": "CT_OP_ROUND",
    "trunc": "CT_OP_TRUNC",
    "floor": "CT_OP_FLOOR",
    "ceil": "CT_OP_CEIL",
    "isfinite": "CT_OP_ISFINITE",
    "isnan": "CT_OP_ISNAN",
    "isinf": "CT_OP_ISINF",
    "logical_not": "CT_OP_LOGICAL_NOT",
    "less": "CT_OP_LT",
    "less_equal": "CT_OP_LE",
    "greater": "CT_OP_GT",
    "greater_equal": "CT_OP_GE",
    "equal": "CT_OP_EQ",
    "not_equal": "CT_OP_NE",
    "maximum": "CT_OP_MAXIMUM",
    "minimum": "CT_OP_MINIMUM",
}


@dataclass(frozen=True)
class _SlotInstruction:
    op: str
    out_slot: int
    left_slot: int
    right_slot: int | None = None
    right_scalar: float | None = None
    reverse: bool = False


@dataclass(frozen=True)
class _SlotPlan:
    """Private C ABI layout compiled from a FusedProgram."""

    instructions: tuple[_SlotInstruction, ...]
    feed_ids: tuple[int, ...]
    slot_count: int
    output_slot: int

    @property
    def feed_count(self) -> int:
        return len(self.feed_ids)

    def validate_feeds(self, feeds: Sequence[CTensor]):
        if len(feeds) != self.feed_count or not feeds:
            raise ValueError(f"expected {self.feed_count} non-empty feeds")
        shape = feeds[0].shape
        if any(feed.shape != shape for feed in feeds):
            raise ValueError("fused-program feeds must have identical shapes")
        return shape

    def native_instructions(self):
        native = ffi.new("CTensorPrimitiveInstruction[]", len(self.instructions))
        for index, instruction in enumerate(self.instructions):
            native[index].op = getattr(C, _OP_NAMES[instruction.op])
            native[index].out_slot = instruction.out_slot
            native[index].left_slot = instruction.left_slot
            if instruction.right_slot is not None:
                native[index].right_kind = C.CT_OPERAND_SLOT
                native[index].right_slot = instruction.right_slot
            elif instruction.right_scalar is not None:
                native[index].right_kind = C.CT_OPERAND_SCALAR
                native[index].right_scalar = instruction.right_scalar
            else:
                native[index].right_kind = C.CT_OPERAND_NONE
            native[index].reverse = int(instruction.reverse)
        return native

    def prepare(self, feeds: Sequence[CTensor]) -> "PreparedFusedProgram":
        shape = self.validate_feeds(feeds)
        slots = list(feeds) + [
            CTensor(shape) for _ in range(self.slot_count - self.feed_count)
        ]
        return PreparedFusedProgram(
            self, slots, self.native_instructions(), prod(shape)
        )

    def execute(self, feeds: Sequence[CTensor]) -> CTensor:
        shape = self.validate_feeds(feeds)
        native = self.native_instructions()
        element_count = prod(shape)
        feed_ptrs = ffi.new("const double*[]", [feed.as_c_ptr() for feed in feeds])
        workspace = ffi.new("double[]", self.slot_count * element_count)
        output = CTensor(shape)
        ok = C.ctensor_execute_primitive_program(
            native,
            len(self.instructions),
            feed_ptrs,
            self.feed_count,
            workspace,
            self.slot_count,
            element_count,
            self.output_slot,
            output.as_c_ptr(),
        )
        if not ok:
            raise ValueError("native fused-program validation failed")
        return output


class PreparedFusedProgram:
    """A compiled FusedProgram with stable native slots."""

    def __init__(self, plan, slots, native_instructions, element_count):
        self.plan = plan
        # Compatibility for the native calculator's prepared wrapper.
        self.program = plan
        self.slots = slots
        self.native_instructions = native_instructions
        self.element_count = element_count
        self.slot_ptrs = ffi.new(
            "double*[]", [slot.as_c_ptr() for slot in self.slots]
        )

    @property
    def output(self) -> CTensor:
        return self.slots[self.plan.output_slot]

    def execute(self) -> CTensor:
        ok = C.ctensor_execute_primitive_program_slots(
            self.native_instructions,
            len(self.plan.instructions),
            self.slot_ptrs,
            self.plan.slot_count,
            self.element_count,
        )
        if not ok:
            raise ValueError("native fused-program validation failed")
        return self.output


def compile_fused_program(program: FusedProgram) -> _SlotPlan:
    """Compile an equal-shape FusedProgram into the private native slot ABI."""

    feed_ids = ordered_feed_ids(program)
    slots = {value_id: index for index, value_id in enumerate(feed_ids)}
    instructions: list[_SlotInstruction] = []

    for step in program.steps:
        try:
            op, prefix_reverse = canonical_elementwise_op(step.op_name)
        except KeyError as exc:
            raise ValueError(
                f"{step.op_name} is not in the elementwise FusedProgram region"
            ) from exc
        attrs = dict(step.attrs)
        reverse = bool(attrs.pop("reverse", False)) ^ prefix_reverse
        scalar = attrs.pop("right_scalar", None)
        if attrs:
            raise ValueError(
                f"FusedProgram step {step.step_id} has unsupported attrs: "
                f"{', '.join(sorted(attrs))}"
            )
        if not step.input_ids or step.input_ids[0] not in slots:
            raise ValueError(
                f"FusedProgram step {step.step_id} reads an unavailable input"
            )
        left_slot = slots[step.input_ids[0]]
        right_slot = None
        if op in ELEMENTWISE_UNARY:
            if len(step.input_ids) != 1 or scalar is not None:
                raise ValueError(f"unary op {op} has an invalid operand layout")
        elif op in ELEMENTWISE_BINARY:
            if len(step.input_ids) == 2 and scalar is None:
                try:
                    right_slot = slots[step.input_ids[1]]
                except KeyError as exc:
                    raise ValueError(
                        f"FusedProgram step {step.step_id} reads an unavailable input"
                    ) from exc
            elif len(step.input_ids) != 1 or scalar is None:
                raise ValueError(f"binary op {op} has an invalid operand layout")
            scalar = None if scalar is None else float(scalar)
        out_slot = len(feed_ids) + len(instructions)
        if op not in _OP_NAMES:
            raise ValueError(
                f"C fused-program execution does not yet implement {op!r}"
            )
        slots[step.result_id] = out_slot
        instructions.append(
            _SlotInstruction(
                op,
                out_slot,
                left_slot,
                right_slot=right_slot,
                right_scalar=scalar,
                reverse=reverse,
            )
        )

    output_id = primary_output_id(program)
    if output_id not in slots:
        raise ValueError("FusedProgram output is not produced")
    return _SlotPlan(
        tuple(instructions), feed_ids, len(feed_ids) + len(instructions), slots[output_id]
    )


def _ordered_c_feeds(
    program: FusedProgram, feeds: Mapping[int, CTensor] | Sequence[CTensor]
) -> list[CTensor]:
    ids = ordered_feed_ids(program)
    if isinstance(feeds, Mapping):
        missing = set(ids) - set(feeds)
        if missing:
            raise ValueError(f"missing FusedProgram feeds: {sorted(missing)}")
        return [feeds[value_id] for value_id in ids]
    return list(feeds)


def execute_fused_program(
    program: FusedProgram, feeds: Mapping[int, CTensor] | Sequence[CTensor]
) -> CTensor:
    """Execute a FusedProgram through one CFFI call."""

    return compile_fused_program(program).execute(_ordered_c_feeds(program, feeds))


def prepare_fused_program(
    program: FusedProgram, feeds: Mapping[int, CTensor] | Sequence[CTensor]
) -> PreparedFusedProgram:
    """Prepare a FusedProgram for repeated one-boundary execution."""

    return compile_fused_program(program).prepare(_ordered_c_feeds(program, feeds))


@dataclass(frozen=True)
class CapturedFusedProgram:
    """A FusedProgram plus the tensor roots captured as its feed bindings."""

    program: FusedProgram
    feeds: Mapping[int, Any]

    def c_feeds(self) -> dict[int, CTensor]:
        return {
            feed_id: CTensor.from_list(value.tolist(), tuple(value.shape))
            for feed_id, value in self.feeds.items()
        }

    def execute_c(self) -> CTensor:
        return execute_fused_program(self.program, self.c_feeds())


def compile_elementwise_tape(
    tape,
    output: Any,
    *,
    dynamic_scalar_ids: Sequence[int] = (),
) -> CapturedFusedProgram:
    """Capture an eligible forward GradTape region as a FusedProgram.

    Reductions, broadcasting, indexing, and shape-changing operations are
    explicit boundaries.  No C-specific program representation escapes this
    lowering function. Tensor scalars normally become instruction literals;
    identities named by ``dynamic_scalar_ids`` instead remain runtime feeds.
    """

    dynamic_scalar_ids = {int(value) for value in dynamic_scalar_ids}
    dynamic_scalar_visiting: set[int] = set()
    feeds: dict[int, Any] = {}
    steps: list[OpStep] = []
    lowered: set[int] = set()
    metadata: dict[int, Meta] = {}

    def is_tensor(value):
        return hasattr(value, "data") and hasattr(value, "shape")

    def is_dynamic_scalar(value):
        if not is_tensor(value) or tuple(value.shape) not in ((), (1,)):
            return False
        identity = id(value)
        if identity in dynamic_scalar_ids:
            return True
        if identity in dynamic_scalar_visiting:
            return False
        node = tape._nodes.get(identity)
        if node is None:
            return False
        dynamic_scalar_visiting.add(identity)
        try:
            dynamic = any(
                is_dynamic_scalar(item)
                for item in node.ctx.get("inputs", ())
                if is_tensor(item)
            )
        finally:
            dynamic_scalar_visiting.remove(identity)
        if dynamic:
            dynamic_scalar_ids.add(identity)
        return dynamic

    def is_scalar(value):
        return isinstance(value, (int, float, bool)) or (
            is_tensor(value) and tuple(value.shape) in ((), (1,))
            and not is_dynamic_scalar(value)
        )

    def scalar_value(value):
        # Preserve scalar kind in the backend-neutral IR. C still consumes the
        # value through its double slot, while typed lowerers such as GLSL need
        # to distinguish ``3`` from ``3.0`` when selecting integer operations.
        return value.item() if is_tensor(value) else value

    def tensor_meta(value, *, node=None):
        dtype = None
        device = None
        if node is not None:
            dtype = node.ctx.get("result_dtype")
            device = node.ctx.get("result_device")
        graph_node = tape.graph.nodes.get(id(value), {})
        if dtype is None:
            dtype = graph_node.get("dtype", getattr(value, "dtype", None))
        if device is None:
            device = graph_node.get("device", getattr(value, "device", None))
        dtype_name = getattr(dtype, "name", None) or str(dtype)
        if "." in dtype_name:
            dtype_name = dtype_name.rsplit(".", 1)[-1]
        return Meta(
            shape=tuple(value.shape),
            dtype=None if dtype is None else dtype_name,
            device=None if device is None else str(device),
        )

    def lower(value):
        identity = id(value)
        if identity in lowered:
            return identity
        node = tape._nodes.get(identity)
        if node is None:
            if not is_tensor(value) or is_scalar(value):
                raise ValueError("non-tensor trace root cannot become a feed")
            feeds[identity] = value
            metadata[identity] = tensor_meta(value)
            lowered.add(identity)
            return identity

        original_op = node.op
        try:
            op, prefix_reverse = canonical_elementwise_op(original_op)
        except KeyError as exc:
            raise ValueError(
                f"{original_op} is not in the elementwise FusedProgram region"
            ) from exc
        inputs = node.ctx["inputs"]
        tensor_inputs = [
            item
            for item in inputs
            if is_tensor(item)
            and not is_scalar(item)
        ]
        if not tensor_inputs:
            raise ValueError(f"{op} has no tensor input")
        shape_inputs = [
            item for item in tensor_inputs if not is_dynamic_scalar(item)
        ] or tensor_inputs
        output_shape = tuple(value.shape)

        def broadcasts_to(input_shape, target_shape):
            if len(input_shape) > len(target_shape):
                return False
            padded = (1,) * (len(target_shape) - len(input_shape)) + tuple(
                input_shape
            )
            return all(
                source in (1, target)
                for source, target in zip(padded, target_shape)
            )

        if any(
            not broadcasts_to(tuple(item.shape), output_shape)
            for item in shape_inputs
        ):
            raise ValueError(
                f"{op} inputs do not broadcast to {output_shape}"
            )

        attrs: dict[str, Any] = {}
        input_ids: list[int]
        if op in ELEMENTWISE_UNARY and len(inputs) == 1:
            input_ids = [lower(inputs[0])]
        elif op in ELEMENTWISE_BINARY and len(inputs) == 2:
            left, right = inputs
            if not is_scalar(left) and not is_scalar(right):
                input_ids = [lower(left), lower(right)]
            elif not is_scalar(left) and is_scalar(right):
                input_ids = [lower(left)]
                attrs["right_scalar"] = scalar_value(right)
            elif is_scalar(left) and not is_scalar(right):
                input_ids = [lower(right)]
                attrs["right_scalar"] = scalar_value(left)
                attrs["reverse"] = True
            else:
                raise ValueError(f"{op} has an unsupported operand layout")
        else:
            raise ValueError(f"{op} has an unsupported operand layout")
        steps.append(
            OpStep(
                step_id=len(steps),
                op_name=op,
                input_ids=input_ids,
                attrs=attrs,
                result_id=identity,
            )
        )
        metadata[identity] = tensor_meta(value, node=node)
        lowered.add(identity)
        return identity

    if isinstance(output, Mapping):
        if not output:
            raise ValueError("a captured program needs at least one output")
        output_ids = {
            str(name): lower(value) for name, value in output.items()
        }
    else:
        output_ids = {"result": lower(output)}
    program = FusedProgram(
        version=1,
        feeds=set(feeds),
        steps=steps,
        outputs=output_ids,
        meta=metadata,
    )
    return CapturedFusedProgram(program, feeds)


def compile_recorded_elementwise_tape(
    tape,
    *,
    dynamic_scalar_ids: Sequence[int] = (),
) -> CapturedFusedProgram:
    """Compile every recorded operation without requiring result arguments."""

    produced_ids = set(tape._nodes)
    consumed_ids = {
        id(value)
        for node in tape._nodes.values()
        for value in node.ctx.get("inputs", ())
        if id(value) in produced_ids
    }
    terminal_ids = [
        result_id
        for result_id in tape._nodes
        if result_id not in consumed_ids
    ]
    if not terminal_ids:
        raise ValueError("recorded tape has no terminal operation")
    terminals = {
        f"result_{index}": tape._nodes[result_id].ctx["result"]
        for index, result_id in enumerate(terminal_ids)
    }
    return compile_elementwise_tape(
        tape,
        terminals,
        dynamic_scalar_ids=dynamic_scalar_ids,
    )


_CAPTURED_NATIVE_KERNELS = {
    "arange": "arange",
    "broadcast_to": "expand",
    "cat": "cat",
    "concat": "cat",
    "cumsum": "cumsum",
    "expand": "expand",
    "matmul": "matmul",
    "imatmul": "matmul",
    "rmatmul": "matmul",
    "max": "reduce",
    "mean": "reduce",
    "min": "reduce",
    "any": "reduce",
    "all": "reduce",
    "empty": "fill",
    "full": "fill",
    "gather": "index_select",
    "permute": "permute",
    "repeat": "repeat",
    "stack": "stack",
    "sum": "reduce",
    "ones": "fill",
    "zeros": "fill",
}


def _captured_meta(value: Any) -> Meta:
    device = getattr(value, "device", None)
    if device is None:
        try:
            device = value.get_device()
        except (AttributeError, NotImplementedError):
            device = "glsl" if type(value).__name__ == "GLChunk" else None
    return Meta(
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        device=None if device is None else str(device),
    )


def _compile_single_native_node(node, operation: str) -> CapturedFusedProgram:
    """Preserve one backend-native operation in the shared fused IR."""

    result = node.ctx["result"]
    feeds: dict[int, Any] = {}
    metadata: dict[int, Meta] = {}
    input_ids: list[int] = []
    for value in node.ctx.get("inputs", ()):
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            continue
        value_id = id(value)
        input_ids.append(value_id)
        feeds[value_id] = value
        metadata[value_id] = _captured_meta(value)

    result_id = id(result)
    metadata[result_id] = _captured_meta(result)
    attrs = dict(node.ctx.get("params") or {})
    kernel_kind = _CAPTURED_NATIVE_KERNELS.get(operation, operation)

    if operation == "rmatmul":
        input_ids.reverse()
    if kernel_kind == "reduce":
        attrs["reduce_op"] = operation
        if "axis" not in attrs and "dim" in attrs:
            attrs["axis"] = attrs.pop("dim")
    if kernel_kind == "permute":
        attrs["dims"] = attrs.pop("perm", attrs.get("dims"))
    if kernel_kind == "expand":
        attrs["shape"] = tuple(
            attrs.get("shape", tuple(result.shape))
        )
    if kernel_kind == "fill":
        attrs["shape"] = tuple(result.shape)
        attrs["fill_value"] = {
            "empty": 0,
            "zeros": 0,
            "ones": 1,
        }.get(operation, attrs.get("fill_value"))
    if kernel_kind == "arange" and attrs.get("end") is None:
        attrs["start"], attrs["end"] = 0, attrs.get("start")

    program = FusedProgram(
        version=1,
        feeds=set(feeds),
        steps=[
            OpStep(
                step_id=0,
                op_name=operation,
                input_ids=input_ids,
                attrs=attrs,
                result_id=result_id,
            )
        ],
        outputs={"result_0": result_id},
        meta=metadata,
        extras={"kernel_kind": kernel_kind},
    )
    return CapturedFusedProgram(program, feeds)


def compile_recorded_fused_tape(
    tape,
    *,
    dynamic_scalar_ids: Sequence[int] = (),
) -> CapturedFusedProgram:
    """Lower one recorded numerical region to the shared FusedProgram IR.

    Equal-shape arithmetic uses the established elementwise lowering. Layout
    regions retain their canonical operation and captured parameters so a
    backend can compile the entire region as one native dispatch.
    """

    nodes = list(tape._nodes.values())
    if not nodes:
        raise ValueError("recorded tape has no numerical operations")
    operations = tuple(str(node.op) for node in nodes)
    if all(
        operation in {"clone", "reshape", "view"}
        for operation in operations
    ):
        feeds: dict[int, Any] = {}
        metadata: dict[int, Meta] = {}
        steps: list[OpStep] = []
        outputs: dict[str, int] = {}
        for index, node in enumerate(nodes):
            source = node.ctx["inputs"][0]
            result = node.ctx["result"]
            source_id = id(source)
            result_id = id(result)
            feeds[source_id] = source
            metadata[source_id] = Meta(
                shape=tuple(source.shape),
                dtype=str(source.dtype),
                device=str(source.device),
            )
            metadata[result_id] = Meta(
                shape=tuple(result.shape),
                dtype=str(result.dtype),
                device=str(result.device),
            )
            # A reshape is a view. At a fused-program boundary, represent it
            # as one linear identity shader so the region still owns exactly
            # one backend dispatch and its output storage has the new shape.
            steps.append(
                OpStep(
                    step_id=index,
                    op_name="add",
                    input_ids=[source_id],
                    attrs={"right_scalar": 0},
                    result_id=result_id,
                )
            )
            outputs[f"result_{index}"] = result_id
        program = FusedProgram(
            version=1,
            feeds=set(feeds),
            steps=steps,
            outputs=outputs,
            meta=metadata,
            extras={"kernel_kind": "linear_reshape_copy"},
        )
        output_shapes = {
            tuple(metadata[value_id].shape or ())
            for value_id in outputs.values()
        }
        if len(output_shapes) != 1:
            raise ValueError(
                "one reshape region requires one common output shape"
            )
        program.glsl_linear_output_shape = next(iter(output_shapes))
        return CapturedFusedProgram(program, feeds)

    if len(nodes) == 1 and (
        operations[0] in _CAPTURED_NATIVE_KERNELS
        or operations[0] == "slice"
    ):
        node = nodes[0]
        if operations[0] != "slice":
            return _compile_single_native_node(node, operations[0])

        captured = _compile_single_native_node(node, operations[0])
        program = captured.program
        step = program.steps[0]
        attributes = dict(step.attrs)
        index = attributes.get("slices")
        if operations[0] == "slice":
            items = list(index) if isinstance(index, tuple) else [index]
            if Ellipsis in items:
                location = items.index(Ellipsis)
                missing = len(node.ctx["inputs"][0].shape) - (len(items) - 1)
                items[location:location + 1] = [slice(None)] * missing
            items.extend(
                [slice(None)]
                * (len(node.ctx["inputs"][0].shape) - len(items))
            )
            active = [
                (axis, item)
                for axis, item in enumerate(items)
                if not (
                    isinstance(item, slice)
                    and item.start is None
                    and item.stop is None
                    and item.step is None
                )
            ]
            if len(active) != 1:
                raise ValueError(
                    "one captured slice shader currently requires one "
                    "active index axis"
                )
            axis, item = active[0]
            if isinstance(item, int):
                axis_size = int(node.ctx["inputs"][0].shape[axis])
                start = item % axis_size
                attributes = {
                    "slice_kind": "axis",
                    "dim": axis,
                    "start": start,
                    "step": 1,
                    "count": 1,
                }
            elif hasattr(item, "shape") and hasattr(item, "dtype"):
                index_id = id(item)
                step.input_ids.append(index_id)
                captured.feeds[index_id] = item
                program.feeds.add(index_id)
                program.meta[index_id] = _captured_meta(item)
                attributes = {
                    "slice_kind": "index_select",
                    "dim": axis,
                }
            else:
                raise ValueError(
                    "captured slice index is not an integer or tensor"
                )

        step.attrs = attributes
        return captured

    return compile_recorded_elementwise_tape(
        tape,
        dynamic_scalar_ids=dynamic_scalar_ids,
    )
