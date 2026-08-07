"""One-boundary C execution of the established :class:`FusedProgram` IR.

``FusedProgram`` remains the semantic program.  This module only compiles its
equal-shape elementwise regions into a private slot plan matching the native C
ABI, then executes that plan through one CFFI call.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence

import numpy as np

from ..abstraction import tensor_identity
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


def _captured_storage_identity(value: Any):
    """Return the exact resident allocation range behind a tensor or storage."""

    storage = getattr(value, "data", value)
    physical = getattr(storage, "_storage", None)
    if physical is not None:
        return (
            id(physical),
            getattr(storage, "_offset", None),
            getattr(storage, "_count", None),
            str(getattr(storage, "_dtype", None)),
        )
    if storage is value and not hasattr(value, "shape"):
        return None
    return ("object", id(storage))


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
        try:
            return tuple(np.broadcast_shapes(*(feed.shape for feed in feeds)))
        except ValueError as error:
            raise ValueError(
                "fused-program feed shapes are not broadcast-compatible"
            ) from error

    def broadcast_feeds(
        self,
        feeds: Sequence[CTensor],
        shape: tuple[int, ...],
    ) -> list[CTensor]:
        broadcasted = []
        for feed in feeds:
            if feed.shape == shape:
                broadcasted.append(feed)
                continue
            source_shape = (1,) * (len(shape) - len(feed.shape)) + feed.shape
            expanded = CTensor(shape)
            C.broadcast_double(
                feed.as_c_ptr(),
                expanded.as_c_ptr(),
                ffi.new("int[]", source_shape),
                len(source_shape),
                ffi.new("int[]", shape),
                len(shape),
            )
            broadcasted.append(expanded)
        return broadcasted

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
        slots = self.broadcast_feeds(feeds, shape) + [
            CTensor(shape) for _ in range(self.slot_count - self.feed_count)
        ]
        return PreparedFusedProgram(
            self, slots, self.native_instructions(), prod(shape)
        )

    def execute(self, feeds: Sequence[CTensor]) -> CTensor:
        shape = self.validate_feeds(feeds)
        feeds = self.broadcast_feeds(feeds, shape)
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
    """A general captured program plus its backend-compilable stages.

    ``program`` retains the complete dependency graph and public outputs.
    ``stages`` is empty for an equal-shape elementwise program or one native
    kernel.  Mixed tapes carry a topologically ordered sequence whose
    elementwise runs remain fused and whose layout/reduction operations form
    explicit backend dispatch boundaries.
    """

    program: FusedProgram
    feeds: Mapping[int, Any]
    stages: tuple[FusedProgram, ...] = ()

    @property
    def execution_programs(self) -> tuple[FusedProgram, ...]:
        return self.stages or (self.program,)

    def c_feeds(self) -> dict[int, CTensor]:
        return {
            feed_id: CTensor.from_list(value.tolist(), tuple(value.shape))
            for feed_id, value in self.feeds.items()
        }

    def execute_c(self) -> CTensor:
        if self.stages:
            raise ValueError(
                "mixed captured programs require a staged backend executor"
            )
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
    lowering function. Python scalar operands may become instruction literals.
    AbstractTensor operands always remain runtime tensor dependencies,
    including tensors whose shape is ``()`` or ``(1,)``.  Compilation must
    never call ``.item()`` on a tensor and freeze its current value.
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
        identity = tensor_identity(value)
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
        if isinstance(value, (int, float, bool)):
            return True
        if not is_tensor(value) or tensor_identity(value) in dynamic_scalar_ids:
            return False
        node = tape._nodes.get(tensor_identity(value))
        return (
            node is not None
            and node.op == "tensor_from_list"
            and tuple(value.shape) in ((), (1,))
        )

    def scalar_value(value):
        if isinstance(value, (int, float, bool)):
            return value
        # A scalar created inside the observed expression is a compiler
        # constant, not a runtime scalar tensor. Read its authored constructor
        # payload rather than materializing the backend result with item().
        node = tape._nodes.get(tensor_identity(value))
        if node is None or node.op != "tensor_from_list":
            raise ValueError("runtime tensor cannot become a scalar literal")
        data = (node.ctx.get("params") or {}).get("data")
        while isinstance(data, (tuple, list)) and len(data) == 1:
            data = data[0]
        if not isinstance(data, (int, float, bool)):
            raise ValueError(
                "scalar tensor constructor has no scalar source literal"
            )
        return data

    def tensor_meta(value, *, node=None):
        dtype = None
        device = None
        if node is not None:
            dtype = node.ctx.get("result_dtype")
            device = node.ctx.get("result_device")
        graph_node = tape.graph.nodes.get(tensor_identity(value), {})
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
        identity = tensor_identity(value)
        if identity in lowered:
            return identity
        # Planner-declared region inputs are cut points into a whole-program
        # discovery tape.  They remain runtime feeds even when the same tape
        # contains their producer from an earlier region.  Without this stop,
        # extracting a later region recursively recompiles its predecessors
        # and can freeze observed intermediates into the new program.
        if identity in dynamic_scalar_ids:
            if not is_tensor(value):
                raise ValueError("a tape boundary feed must be a tensor")
            feeds[identity] = value
            metadata[identity] = tensor_meta(value)
            lowered.add(identity)
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
        if original_op == "tensor_from_list":
            # This is a recorded constructor instruction, not an external
            # program input.  Reject the elementwise-only fast path so the
            # general tape partitioner retains it as an explicit ``constant``
            # stage.  Treating it as a feed loses its producer when deployment
            # capture payloads are discarded; converting it back to a Python
            # scalar would also contradict AbstractTensor's type decision.
            raise ValueError(
                "tensor_from_list requires an explicit constant stage"
            )
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
            input_summary = tuple(
                {
                    "id": tensor_identity(item),
                    "type": type(item).__name__,
                    "shape": tuple(item.shape) if is_tensor(item) else None,
                    "dynamic": is_dynamic_scalar(item) if is_tensor(item) else False,
                    "recorded": tensor_identity(item) in tape._nodes,
                }
                for item in inputs
            )
            raise ValueError(
                f"{op} has no tensor input; inputs={input_summary!r}; "
                f"dynamic_scalar_ids={tuple(sorted(dynamic_scalar_ids))!r}"
            )
        attrs: dict[str, Any] = {}
        input_ids: list[int]
        if op in ELEMENTWISE_UNARY and len(inputs) == 1:
            input_ids = [lower(inputs[0])]
        elif (
            op in ELEMENTWISE_BINARY
            and len(inputs) == 1
            and "right_scalar" in (node.ctx.get("params") or {})
        ):
            input_ids = [lower(inputs[0])]
            attrs["right_scalar"] = node.ctx["params"]["right_scalar"]
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
            raise ValueError(
                f"{op} has an unsupported operand layout; "
                f"inputs={tuple((type(item).__name__, getattr(item, 'shape', None), is_scalar(item)) for item in inputs)!r}; "
                f"params={node.ctx.get('params')!r}"
            )
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
    output_shapes = {
        tuple(metadata[value_id].shape or ())
        for value_id in output_ids.values()
    }
    if len(output_shapes) != 1:
        raise ValueError(
            "one elementwise program requires one common output shape; "
            f"output_shapes={tuple(sorted(output_shapes))!r}"
        )
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
        tensor_identity(value)
        for node in tape._nodes.values()
        for value in node.ctx.get("inputs", ())
        if tensor_identity(value) in produced_ids
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
    "prod": "reduce",
    "any": "reduce",
    "all": "reduce",
    "empty": "fill",
    "empty_like": "fill",
    "full": "fill",
    "full_like": "fill",
    "gather": "index_select",
    "pad": "pad",
    "index_set": "index_set",
    "permute": "permute",
    "repeat": "repeat",
    "scatter": "scatter",
    "stack": "stack",
    "sum": "reduce",
    "tensor_from_list": "constant",
    "where": "where",
    "ones": "fill",
    "ones_like": "fill",
    "zeros": "fill",
    "zeros_like": "fill",
}

_CAPTURED_CAST_OPERATIONS = frozenset({
    "astype",
    "bool",
    "double",
    "float",
    "int",
    "long",
    "long_cast",
    "to_dtype",
})

_CAPTURED_COMPOSITE_OPERATIONS = frozenset({
    "clamp",
    "clamp_min",
    "clamp_max",
})


def _captured_dtype_kind(value: Any) -> str | None:
    """Return NumPy's dtype-kind code for a captured tensor value."""

    storage = getattr(value, "data", value)
    dtype = getattr(storage, "dtype", getattr(value, "dtype", None))
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return str(kind)
    name = str(dtype).rsplit(".", 1)[-1].lower()
    if "bool" in name:
        return "b"
    if "float" in name or "double" in name or "half" in name:
        return "f"
    if name.startswith("uint"):
        return "u"
    if name.startswith("int") or name.startswith("long"):
        return "i"
    return None


def _canonical_captured_cast(source: Any, result: Any) -> tuple[str, dict[str, Any]]:
    """Use the established AbstractTensor conversion vocabulary."""

    source_kind = _captured_dtype_kind(source)
    target_kind = _captured_dtype_kind(result)
    if source_kind is None or target_kind is None:
        raise TypeError(
            "captured dtype conversion requires recognizable source and "
            f"result dtypes; source={getattr(source, 'dtype', None)!r}, "
            f"result={getattr(result, 'dtype', None)!r}"
        )
    if target_kind == source_kind:
        return "add", {"right_scalar": 0}
    if target_kind == "b":
        return "not_equal", {"right_scalar": 0}
    if target_kind == "f":
        return ("uitofp" if source_kind in {"u", "b"} else "sitofp"), {}
    if target_kind == "u":
        return ("fptoui" if source_kind == "f" else "zext"), {}
    return ("fptosi" if source_kind == "f" else "sext"), {}


def _captured_meta(
    value: Any, *, shape_source_ids: tuple[int | None, ...] | None = None,
) -> Meta:
    device = getattr(value, "device", None)
    if device is None:
        try:
            device = value.get_device()
        except (AttributeError, NotImplementedError):
            device = "glsl" if type(value).__name__ == "GLChunk" else None
    storage = getattr(value, "data", value)
    dtype = (
        storage.dtype
        if isinstance(storage, np.ndarray)
        else value.dtype
    )
    return Meta(
        shape=tuple(value.shape),
        dtype=str(dtype),
        device=None if device is None else str(device),
        shape_source_ids=shape_source_ids,
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
        value_id = tensor_identity(value)
        input_ids.append(value_id)
        feeds[value_id] = value
        metadata[value_id] = _captured_meta(value)

    result_id = tensor_identity(result)
    shape_source_ids = node.ctx.get("shape_source_ids")
    metadata[result_id] = _captured_meta(
        result, shape_source_ids=shape_source_ids,
    )
    attrs = dict(node.ctx.get("params") or {})
    if shape_source_ids and any(
        source_id is not None for source_id in shape_source_ids
    ):
        # ``size`` was captured into ``attrs`` as a frozen literal by
        # ``_wrap_creation_fn`` -- the tape only tracks tensor-to-tensor
        # flow, so a plain size tuple was never going to survive there.
        # But its real dependency edge already exists, correctly, in the
        # ProcessGraph (that is exactly what shape_source_ids names).
        # Promote it to a genuine operand here instead of leaving the
        # frozen literal as the only record: use the ProcessGraph node id
        # directly as this feed's id (it is already resolved -- there is
        # no transient tape identity to remap through, unlike a real
        # tensor operand) so the existing feed/dependency machinery finds
        # it the same way it finds every other operand, with nothing
        # special-cased for this being a scalar rather than a tensor.
        size_origin_id = int(shape_source_ids[0])
        size_value = attrs.get("size")
        if size_value is not None and size_origin_id not in feeds:
            input_ids.append(size_origin_id)
            feeds[size_origin_id] = size_value
            metadata[size_origin_id] = Meta(
                shape=(len(tuple(size_value)),),
                dtype="int64",
                device=None,
            )
            attrs["dynamic_shape_input_id"] = size_origin_id
            # A frozen ``size``/``shape`` sitting beside the real operand
            # is not a harmless fallback -- it is a silently wrong answer
            # for any consumer that doesn't yet know to look for
            # ``dynamic_shape_input_id``, since it only ever holds what
            # this one discovery trace happened to observe.  A consumer
            # that isn't ready for a dynamic shape must fail loudly on
            # the missing key, not quietly compile a fixed-size buffer
            # that is wrong for every other real run.
            attrs.pop("size", None)
            attrs.pop("shape", None)
    kernel_kind = _CAPTURED_NATIVE_KERNELS.get(operation, operation)
    lowered_operation = operation

    if operation == "index_set":
        index = attrs.pop("idx", None)
        if index is None:
            raise ValueError("captured indexed assignment has no index")
        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]
        index_storage = (
            index
            if isinstance(index, np.ndarray)
            else getattr(index, "data", index)
        )
        index_array = np.asarray(index_storage)
        if index_array.dtype.kind == "b":
            if tuple(index_array.shape) != tuple(result.shape):
                raise ValueError(
                    "boolean indexed assignment requires a mask matching "
                    f"the destination shape; mask={index_array.shape!r}, "
                    f"destination={tuple(result.shape)!r}"
                )
            if len(input_ids) != 2:
                raise ValueError(
                    "captured boolean indexed assignment requires destination "
                    "and value tensor inputs"
                )
            mask_id = tensor_identity(index_storage)
            feeds[mask_id] = index_storage
            metadata[mask_id] = _captured_meta(index_storage)
            destination_id, value_id = input_ids
            value_meta = metadata[value_id]
            steps: list[OpStep] = []
            if tuple(value_meta.shape or ()) != tuple(result.shape):
                broadcast_id = -max(
                    1,
                    result_id,
                    destination_id,
                    value_id,
                    mask_id,
                )
                metadata[broadcast_id] = metadata[result_id]
                steps.append(OpStep(
                    step_id=0,
                    op_name="broadcast_to",
                    input_ids=[value_id],
                    attrs={"shape": tuple(result.shape)},
                    result_id=broadcast_id,
                ))
                value_id = broadcast_id
            steps.append(OpStep(
                step_id=len(steps),
                op_name="where",
                input_ids=[mask_id, value_id, destination_id],
                attrs={},
                result_id=result_id,
            ))
            return CapturedFusedProgram(
                FusedProgram(
                    version=1,
                    feeds=set(feeds),
                    steps=steps,
                    outputs={"result_0": result_id},
                    meta=metadata,
                    extras={"kernel_kind": "where"},
                ),
                feeds,
            )
        if len(input_ids) != 2:
            raise ValueError(
                "captured basic indexed assignment requires destination and "
                "value tensor inputs"
            )
        return CapturedFusedProgram(
            FusedProgram(
                version=1,
                feeds=set(feeds),
                steps=[OpStep(
                    step_id=0,
                    op_name="index_set",
                    input_ids=input_ids,
                    attrs={"slices": index},
                    result_id=result_id,
                )],
                outputs={"result_0": result_id},
                meta=metadata,
                extras={"kernel_kind": "index_set"},
            ),
            feeds,
        )

    if operation in _CAPTURED_COMPOSITE_OPERATIONS:
        if len(input_ids) != 1:
            raise ValueError(
                f"captured {operation!r} requires one tensor input"
            )
        lower_bound = attrs.get("min", attrs.get("min_val"))
        upper_bound = attrs.get("max", attrs.get("max_val"))
        if operation == "clamp_min" and lower_bound is None:
            lower_bound = attrs.get("value")
        if operation == "clamp_max" and upper_bound is None:
            upper_bound = attrs.get("value")
        specifications = []
        if lower_bound is not None:
            specifications.append(("maximum", lower_bound))
        if upper_bound is not None:
            specifications.append(("minimum", upper_bound))
        if not specifications:
            specifications.append(("add", 0))
        steps = []
        previous_id = input_ids[0]
        for index, (op_name, scalar) in enumerate(specifications):
            output_id = (
                result_id
                if index == len(specifications) - 1
                else -(result_id + index + 1)
            )
            metadata[output_id] = metadata[result_id]
            steps.append(OpStep(
                step_id=index,
                op_name=op_name,
                input_ids=[previous_id],
                attrs={"right_scalar": scalar},
                result_id=output_id,
            ))
            previous_id = output_id
        program = FusedProgram(
            version=1,
            feeds=set(feeds),
            steps=steps,
            outputs={"result_0": result_id},
            meta=metadata,
            extras={"kernel_kind": None},
        )
        return CapturedFusedProgram(program, feeds)

    if operation in _CAPTURED_CAST_OPERATIONS:
        if len(input_ids) != 1:
            raise ValueError(
                f"captured dtype conversion {operation!r} requires one "
                "tensor input"
            )
        source = node.ctx["inputs"][0]
        lowered_operation, attrs = _canonical_captured_cast(source, result)
        kernel_kind = None

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
        if "dynamic_shape_input_id" not in attrs:
            attrs["shape"] = tuple(result.shape)
        attrs["fill_value"] = {
            "empty": 0,
            "empty_like": 0,
            "zeros": 0,
            "zeros_like": 0,
            "ones": 1,
            "ones_like": 1,
        }.get(operation, attrs.get("fill_value"))
    if kernel_kind == "constant":
        attrs["shape"] = tuple(result.shape)
        storage = getattr(result, "data", result)
        dtype = (
            storage.dtype
            if isinstance(storage, np.ndarray)
            else result.dtype
        )
        attrs["values"] = tuple(
            np.asarray(attrs.pop("data"), dtype=dtype)
            .reshape(-1)
            .tolist()
        )
        attrs.pop("device", None)
    if kernel_kind == "arange" and attrs.get("end") is None:
        attrs["start"], attrs["end"] = 0, attrs.get("start")

    program = FusedProgram(
        version=1,
        feeds=set(feeds),
        steps=[
            OpStep(
                step_id=0,
                op_name=lowered_operation,
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
    outputs: Mapping[str, Any] | None = None,
    strict_outputs: bool = False,
) -> CapturedFusedProgram:
    """Lower one recorded numerical region to the shared FusedProgram IR.

    Equal-shape arithmetic uses the established elementwise lowering. Layout
    regions retain their canonical operation and captured parameters so a
    backend can compile the entire region as one native dispatch.
    """

    boundary_value_ids = {int(value) for value in dynamic_scalar_ids}
    requested_outputs = None if outputs is None else dict(outputs)
    passthrough_outputs: dict[str, Any] = {}
    if requested_outputs is not None:
        if not requested_outputs:
            raise ValueError("a captured program needs at least one output")
        required_ids: set[int] = set()
        storage_result_ids = {
            _captured_storage_identity(result): result_id
            for result_id, node in tape._nodes.items()
            for result in (node.ctx.get("result"),)
            if _captured_storage_identity(result) is not None
        }

        def recorded_dependencies(node):
            dependencies = set()

            def visit(value):
                value_id = tensor_identity(value)
                if value_id in tape._nodes:
                    dependencies.add(value_id)
                    return
                producer_id = storage_result_ids.get(
                    _captured_storage_identity(value)
                )
                if producer_id is not None:
                    dependencies.add(producer_id)
                    return
                if isinstance(value, (tuple, list)):
                    for item in value:
                        visit(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        visit(item)

            for value in node.ctx.get("inputs", ()):
                visit(value)
            visit(node.ctx.get("params") or {})
            return dependencies

        pending = [tensor_identity(value) for value in requested_outputs.values()]
        while pending:
            value_id = pending.pop()
            if value_id in required_ids:
                continue
            if (
                value_id in boundary_value_ids
                and all(
                    tensor_identity(value) != value_id
                    for value in requested_outputs.values()
                )
            ):
                continue
            node = tape._nodes.get(value_id)
            if node is None:
                matching = {
                    name: value
                    for name, value in requested_outputs.items()
                    if tensor_identity(value) == value_id
                }
                if strict_outputs:
                    raise ValueError(
                        "requested captured output is not produced by this tape"
                    )
                if not matching or not all(
                    hasattr(value, "shape") and hasattr(value, "data")
                    for value in matching.values()
                ):
                    raise ValueError(
                        "unproduced captured outputs must be tensor "
                        "pass-through values"
                    )
                passthrough_outputs.update(matching)
                continue
            required_ids.add(value_id)
            pending.extend(recorded_dependencies(node))
        nodes = [
            node
            for result_id, node in tape._nodes.items()
            if result_id in required_ids
        ]
    else:
        nodes = list(tape._nodes.values())
    if not nodes and passthrough_outputs:
        feeds = {
            tensor_identity(value): value for value in passthrough_outputs.values()
        }
        metadata = {
            value_id: _captured_meta(value)
            for value_id, value in feeds.items()
        }
        program = FusedProgram(
            version=1,
            feeds=set(feeds),
            steps=[],
            outputs={
                name: tensor_identity(value)
                for name, value in requested_outputs.items()
            },
            meta=metadata,
            extras={"kernel_kind": "passthrough"},
        )
        return CapturedFusedProgram(program, feeds)
    if not nodes:
        raise ValueError("recorded tape has no numerical operations")
    operations = tuple(str(node.op) for node in nodes)
    if all(
        operation in {"clone", "reshape", "view"}
        for operation in operations
    ) and len({
        tuple(node.ctx["result"].shape)
        for node in nodes
    }) == 1:
        feeds: dict[int, Any] = {}
        metadata: dict[int, Meta] = {}
        steps: list[OpStep] = []
        program_outputs: dict[str, int] = {}
        for index, node in enumerate(nodes):
            source = node.ctx["inputs"][0]
            result = node.ctx["result"]
            source_id = tensor_identity(source)
            result_id = tensor_identity(result)
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
            program_outputs[f"result_{index}"] = result_id
        if requested_outputs is not None:
            program_outputs = {
                name: tensor_identity(value)
                for name, value in requested_outputs.items()
            }
        program = FusedProgram(
            version=1,
            feeds=set(feeds),
            steps=steps,
            outputs=program_outputs,
            meta=metadata,
            extras={"kernel_kind": "linear_reshape_copy"},
        )
        output_shapes = tuple(
            tuple(metadata[value_id].shape or ())
            for value_id in program_outputs.values()
        )
        output_extents = {
            prod(shape) if shape else 1
            for shape in output_shapes
        }
        if len(output_extents) != 1:
            raise ValueError(
                "one reshape region requires one common linear output extent; "
                f"output_shapes={output_shapes!r}"
            )
        # Layout-only operations may expose the same storage through different
        # logical shapes (for example parallel unsqueeze operations).  The
        # fused shader iterates storage linearly, while each output retains its
        # own shape in ``program.meta``.
        program.glsl_linear_output_shape = output_shapes[0]
        return CapturedFusedProgram(program, feeds)

    if len(nodes) == 1 and (
        operations[0] in _CAPTURED_NATIVE_KERNELS
        or operations[0] in _CAPTURED_CAST_OPERATIONS
        or operations[0] in _CAPTURED_COMPOSITE_OPERATIONS
        or operations[0] == "slice"
    ):
        node = nodes[0]
        captured = _compile_single_native_node(node, operations[0])
        if requested_outputs is not None:
            captured.program.outputs = {
                name: tensor_identity(value)
                for name, value in requested_outputs.items()
            }
        if operations[0] != "slice":
            return captured

        program = captured.program
        step = program.steps[0]
        attributes = dict(step.attrs)
        index = attributes.get("slices")
        if operations[0] == "slice":
            items = list(index) if isinstance(index, tuple) else [index]
            location = next(
                (
                    item_index
                    for item_index, item in enumerate(items)
                    if item is Ellipsis
                ),
                None,
            )
            if location is not None:
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
            if len(active) > 1:
                source_shape = tuple(node.ctx["inputs"][0].shape)
                index_tensors = tuple(
                    (node.ctx.get("params") or {}).get(
                        "index_tensors", ()
                    )
                )
                if (
                    len(index_tensors) == len(active)
                    and all(
                        hasattr(item, "shape") and hasattr(item, "dtype")
                        for item in index_tensors
                    )
                ):
                    source_id = step.input_ids[0]
                    used_ids = {
                        source_id,
                        step.result_id,
                        *(tensor_identity(item) for item in index_tensors),
                    }
                    next_temp = -max(1, *(abs(value) for value in used_ids))

                    def temporary(meta):
                        nonlocal next_temp
                        while next_temp in used_ids:
                            next_temp -= 1
                        value_id = next_temp
                        used_ids.add(value_id)
                        next_temp -= 1
                        program.meta[value_id] = meta
                        return value_id

                    flat_source_id = temporary(Meta(
                        (int(prod(source_shape)),),
                        program.meta[source_id].dtype,
                        program.meta[source_id].device,
                    ))
                    advanced_steps = [OpStep(
                        step_id=0,
                        op_name="reshape",
                        input_ids=[source_id],
                        attrs={"shape": (int(prod(source_shape)),)},
                        result_id=flat_source_id,
                    )]
                    flat_index_id = None
                    for (axis, _item), index_tensor in zip(
                        active, index_tensors
                    ):
                        index_id = tensor_identity(index_tensor)
                        captured.feeds[index_id] = index_tensor
                        program.feeds.add(index_id)
                        program.meta[index_id] = _captured_meta(index_tensor)
                        stride = int(prod(source_shape[axis + 1:]))
                        term_id = index_id
                        if stride != 1:
                            term_id = temporary(program.meta[index_id])
                            advanced_steps.append(OpStep(
                                step_id=len(advanced_steps),
                                op_name="mul",
                                input_ids=[index_id],
                                attrs={"right_scalar": stride},
                                result_id=term_id,
                            ))
                        if flat_index_id is None:
                            flat_index_id = term_id
                        else:
                            combined_id = temporary(program.meta[index_id])
                            advanced_steps.append(OpStep(
                                step_id=len(advanced_steps),
                                op_name="add",
                                input_ids=[flat_index_id, term_id],
                                attrs={},
                                result_id=combined_id,
                            ))
                            flat_index_id = combined_id
                    advanced_steps.append(OpStep(
                        step_id=len(advanced_steps),
                        op_name="gather",
                        input_ids=[flat_source_id, flat_index_id],
                        attrs={"dim": 0},
                        result_id=step.result_id,
                    ))
                    program.steps = advanced_steps
                    program.extras["kernel_kind"] = "advanced_gather"
                    program.extras["synthetic_result_ids"] = tuple(
                        int(value_id)
                        for value_id in used_ids
                        if int(value_id) < 0
                    )
                    return captured
                if all(isinstance(item, slice) for _axis, item in active):
                    source_id = step.input_ids[0]
                    current_id = source_id
                    current_shape = list(source_shape)
                    used_ids = {source_id, step.result_id}
                    next_temp = -max(1, *(abs(value) for value in used_ids))
                    span_steps = []
                    synthetic_ids = []
                    for position, (axis, item) in enumerate(active):
                        start, stop, stride = item.indices(
                            current_shape[axis]
                        )
                        count = len(range(start, stop, stride))
                        next_shape = list(current_shape)
                        next_shape[axis] = count
                        output_id = step.result_id
                        if position != len(active) - 1:
                            while next_temp in used_ids:
                                next_temp -= 1
                            output_id = next_temp
                            used_ids.add(output_id)
                            synthetic_ids.append(output_id)
                            next_temp -= 1
                            program.meta[output_id] = Meta(
                                tuple(next_shape),
                                program.meta[source_id].dtype,
                                program.meta[source_id].device,
                            )
                        span_steps.append(OpStep(
                            step_id=len(span_steps),
                            op_name="slice",
                            input_ids=[current_id],
                            attrs={
                                "slice_kind": "axis",
                                "dim": axis,
                                "start": start,
                                "step": stride,
                                "count": count,
                            },
                            result_id=output_id,
                        ))
                        current_id = output_id
                        current_shape = next_shape
                    program.steps = span_steps
                    program.extras["kernel_kind"] = "multi_axis_span"
                    program.extras["synthetic_result_ids"] = tuple(
                        synthetic_ids
                    )
                    return captured
                last_axis = active[-1][0]
                prefix = items[:last_axis + 1]
                if not all(isinstance(item, int) for item in prefix):
                    raise ValueError(
                        "one captured multi-axis slice shader requires an "
                        "integer-index prefix"
                    )
                normalized = []
                for axis, item in enumerate(prefix):
                    axis_size = int(source_shape[axis])
                    value = int(item)
                    if value < -axis_size or value >= axis_size:
                        raise IndexError("tensor index out of range")
                    normalized.append(value % axis_size)
                strides = [
                    int(prod(source_shape[axis + 1:]))
                    for axis in range(len(prefix))
                ]
                start = sum(
                    value * stride
                    for value, stride in zip(normalized, strides)
                )
                count = int(prod(source_shape[last_axis + 1:]))
                attributes = {
                    "slice_kind": "flat",
                    "start": start,
                    "step": 1,
                    "count": count,
                    "source_count": int(prod(source_shape)),
                }
                step.attrs = attributes
                return captured
            if len(active) != 1:
                raise ValueError(
                    "one captured slice shader requires at least one active "
                    "index axis"
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
                index_id = tensor_identity(item)
                step.input_ids.append(index_id)
                captured.feeds[index_id] = item
                program.feeds.add(index_id)
                program.meta[index_id] = _captured_meta(item)
                attributes = {
                    "slice_kind": "index_select",
                    "dim": axis,
                }
            elif isinstance(item, slice):
                axis_size = int(node.ctx["inputs"][0].shape[axis])
                start, stop, stride = item.indices(axis_size)
                attributes = {
                    "slice_kind": "axis",
                    "dim": axis,
                    "start": start,
                    "step": stride,
                    "count": len(range(start, stop, stride)),
                }
            else:
                raise ValueError(
                    "captured slice index is not an integer or tensor"
                )

        step.attrs = attributes
        return captured

    elementwise_error: ValueError | None = None
    try:
        if requested_outputs is not None:
            return compile_elementwise_tape(
                tape,
                requested_outputs,
                dynamic_scalar_ids=dynamic_scalar_ids,
            )
        return compile_recorded_elementwise_tape(
            tape,
            dynamic_scalar_ids=dynamic_scalar_ids,
        )
    except ValueError as error:
        # A forward tape is a general AbstractTensor dependency graph, not
        # necessarily one equal-shape arithmetic expression.  Preserve maximal
        # elementwise runs, and lower each layout/reduction/native operation as
        # its own compiled stage.  No operation is executed as a fallback.
        elementwise_error = error

    class _TapeRegion:
        def __init__(self, source, region_nodes):
            self._nodes = {
                tensor_identity(node.ctx["result"]): node for node in region_nodes
            }
            self.graph = source.graph

    def is_elementwise(operation):
        try:
            canonical_elementwise_op(operation)
        except KeyError:
            return False
        return True

    groups: list[list[Any]] = []
    current: list[Any] = []
    for node, operation in zip(nodes, operations):
        if is_elementwise(operation):
            result_shape = tuple(node.ctx["result"].shape)
            if current and tuple(current[-1].ctx["result"].shape) != result_shape:
                groups.append(current)
                current = []
            current.append(node)
            continue
        if current:
            groups.append(current)
            current = []
        groups.append([node])
    if current:
        groups.append(current)

    stages: list[FusedProgram] = []
    stage_feeds: dict[int, Any] = {}
    metadata: dict[int, Meta] = {}
    produced: set[int] = set()
    node_by_result = {
        tensor_identity(node.ctx["result"]): node
        for node in nodes
    }
    result_by_storage = {
        _captured_storage_identity(node.ctx["result"]): result_id
        for result_id, node in node_by_result.items()
        if _captured_storage_identity(node.ctx.get("result")) is not None
    }
    consumers_by_result: dict[int, set[int]] = {
        result_id: set() for result_id in node_by_result
    }

    def consumed_results(value):
        value_id = tensor_identity(value)
        if value_id in node_by_result:
            yield value_id
            return
        producer_id = result_by_storage.get(
            _captured_storage_identity(value)
        )
        if producer_id is not None:
            yield producer_id
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                yield from consumed_results(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from consumed_results(item)

    for consumer in nodes:
        consumer_id = tensor_identity(consumer.ctx["result"])
        for value in (
            *consumer.ctx.get("inputs", ()),
            consumer.ctx.get("params") or {},
        ):
            for value_id in consumed_results(value):
                consumers_by_result[value_id].add(consumer_id)
    required_program_outputs = (
        {
            tensor_identity(value)
            for value in requested_outputs.values()
            if tensor_identity(value) in node_by_result
        }
        if requested_outputs is not None
        else {
            result_id
            for result_id, consumers in consumers_by_result.items()
            if not consumers
        }
    )
    for group in groups:
        region = _TapeRegion(tape, group)
        group_operations = tuple(str(node.op) for node in group)
        if all(is_elementwise(operation) for operation in group_operations):
            group_result_ids = {
                tensor_identity(node.ctx["result"]) for node in group
            }
            # A backend stage is a cut through one complete program, not an
            # independently discovered mini-program.  Any value consumed by a
            # later stage is a live-out even when it also has consumers inside
            # this stage.  Looking only for terminals of ``region`` drops such
            # fan-out producers and fabricates them as external feeds later.
            live_outs = {
                result_id
                for result_id in group_result_ids
                if (
                    result_id in required_program_outputs
                    or any(
                        consumer_id not in group_result_ids
                        for consumer_id in consumers_by_result[result_id]
                    )
                )
            }
            if not live_outs:
                raise ValueError(
                    "elementwise stage has no live output in the complete "
                    "captured program"
                )
            captured = compile_elementwise_tape(
                region,
                {
                    f"result_{index}": node_by_result[result_id].ctx[
                        "result"
                    ]
                    for index, result_id in enumerate(sorted(live_outs))
                },
                dynamic_scalar_ids=dynamic_scalar_ids,
            )
        else:
            if len(group) != 1:
                raise AssertionError("native tape regions contain one node")
            operation = group_operations[0]
            if (
                operation not in _CAPTURED_NATIVE_KERNELS
                and operation not in _CAPTURED_CAST_OPERATIONS
                and operation not in _CAPTURED_COMPOSITE_OPERATIONS
                and operation not in {
                    "slice",
                    "clone",
                    "reshape",
                    "view",
                }
            ):
                raise ValueError(
                    f"{operation} has no captured basic-operator lowering"
                ) from elementwise_error
            captured = compile_recorded_fused_tape(
                region,
                dynamic_scalar_ids=dynamic_scalar_ids,
            )
        stages.extend(captured.execution_programs)
        stage_feeds.update(captured.feeds)
        for stage in captured.execution_programs:
            produced.update(stage.outputs.values())
            metadata.update(stage.meta or {})

    # Native backend hooks sometimes retain an operand's raw storage object
    # (notably tensor indexing) while autograd identifies the producing
    # AbstractTensor wrapper.  Those are two references to one stage-local
    # value, not two program inputs.  Normalize the raw-storage IDs back to
    # their producer IDs before deciding which feeds are external.
    storage_producers = {
        _captured_storage_identity(result): tensor_identity(result)
        for node in nodes
        for result in (node.ctx.get("result"),)
        if _captured_storage_identity(result) is not None
    }
    storage_aliases = {}
    for feed_id, value in stage_feeds.items():
        storage_id = _captured_storage_identity(value)
        if storage_id is not None:
            producer_id = storage_producers.get(storage_id)
            if producer_id is not None:
                storage_aliases[feed_id] = producer_id
    if storage_aliases:
        remap = lambda value_id: storage_aliases.get(value_id, value_id)
        for stage in stages:
            stage.feeds = {remap(value_id) for value_id in stage.feeds}
            for step in stage.steps:
                step.input_ids = [
                    remap(value_id) for value_id in step.input_ids
                ]
            stage.outputs = {
                name: remap(value_id)
                for name, value_id in stage.outputs.items()
            }
            stage.meta = {
                remap(value_id): meta
                for value_id, meta in (stage.meta or {}).items()
            }
        stage_feeds = {
            remap(value_id): value
            for value_id, value in stage_feeds.items()
        }
        produced = {
            value_id
            for stage in stages
            for value_id in stage.outputs.values()
        }
        metadata = {
            value_id: meta
            for stage in stages
            for value_id, meta in (stage.meta or {}).items()
        }

    # Values produced by an earlier stage are routed on-device and are not
    # external roots of the complete captured program.
    in_place_feeds = {
        step.result_id
        for stage in stages
        for step in stage.steps
        if step.result_id in step.input_ids
    }
    external_feeds = {
        value_id: value
        for value_id, value in stage_feeds.items()
        if value_id not in produced or value_id in in_place_feeds
    }
    external_feeds.update(
        (tensor_identity(value), value) for value in passthrough_outputs.values()
    )
    metadata.update(
        {
            tensor_identity(value): _captured_meta(value)
            for value in passthrough_outputs.values()
        }
    )
    produced_ids = {tensor_identity(node.ctx["result"]) for node in nodes}
    consumed_ids = {
        tensor_identity(value)
        for node in nodes
        for value in node.ctx.get("inputs", ())
        if (
            tensor_identity(value) in produced_ids
            and tensor_identity(value) != tensor_identity(node.ctx["result"])
        )
    }
    if requested_outputs is None:
        terminal_ids = [
            result_id
            for result_id in produced_ids
            if result_id not in consumed_ids
        ]
        if not terminal_ids:
            raise ValueError("recorded tape has no terminal operation")
        program_outputs = {
            f"result_{index}": result_id
            for index, result_id in enumerate(terminal_ids)
        }
    else:
        program_outputs = {
            name: tensor_identity(value)
            for name, value in requested_outputs.items()
        }
    manifest = FusedProgram(
        version=1,
        feeds=set(external_feeds),
        steps=[
            step
            for stage in stages
            for step in stage.steps
        ],
        outputs=program_outputs,
        meta=metadata,
        extras={"kernel_kind": "mixed"},
    )
    return CapturedFusedProgram(
        manifest,
        external_feeds,
        tuple(stages),
    )
