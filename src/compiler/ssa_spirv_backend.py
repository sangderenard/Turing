"""Emit SPIR-V assembly from Turing SSA.

This is the first cut of the ``SSA -> SPIR-V`` route named in
``docs/C_NODUS_INTEROP_AND_FUSION.md`` ("typed SSA ... -> SPIR-V-compatible
operations -> SPIR-V module") and tracked as "not started" in
``docs/BACKEND_PERFORMANCE_HANDOFF.md``. It follows the same incremental
scoping this repo already used for ``ssa_fortran_backend.py``: land straight-
line scalar elementwise chains first, then array-shaped elementwise chains,
and report everything else (structural ops, reductions, multi-block control
flow, region calls) as an honest shortfall rather than a guess. Widening
scope further is follow-up work, not a redesign -- the block/shortfall
structure below already accommodates it.

SPIR-V differs from every other backend in this repo in one structural way
worth stating up front: ids and type/constant declarations are **module**
scoped, not function scoped. Two functions in the same ``IRModule`` must not
redeclare ``OpTypeFloat 64`` or fight over id numbers. ``_ModuleBuilder``
exists for exactly that reason -- one shared id/type/constant namespace,
threaded through every ``_FunctionEmitter`` in a module.

Array-shaped SSA values are represented as ``Function``-storage pointers to
nested ``OpTypeArray``s (dimension 0 outermost, matching this repo's SSA
dimension order directly -- no Fortran-style column-major reversal needed).
An array argument arrives as an ``OpTypePointer(Function, ...)`` parameter;
an array-shaped instruction result gets its own ``OpVariable`` declared at
the top of the entry block (SPIR-V requires every ``Function``-storage
``OpVariable`` to precede all other instructions in the block). Elementwise
ops over arrays are **unrolled** into one ``OpAccessChain``/``OpLoad``/
.../``OpStore`` sequence per flat index rather than looped -- shapes are
always statically known here, and a real loop needs structured control flow,
which is a separate, later milestone (see the handoff doc). This keeps every
function single-block, consistent with everything else in this module.

The buffer/descriptor binding ABI (SSBOs, descriptor sets, workgroup
dispatch) needed to hand real GPU-resident data to one of these functions is
still deliberately out of scope: it is its own milestone in the interop doc
("Specify a shared tensor descriptor ..."), not a prerequisite for proving
the op-level translation. Each SSA ``Function`` is emitted as an ordinary
SPIR-V library function (``OpFunction`` / ``OpFunctionParameter``s /
``OpReturnValue``), decorated ``LinkageAttributes ... Export`` so the module
is valid with no ``OpEntryPoint`` at all (the ``Linkage`` capability exists
for precisely this). Wiring a function into a real GLCompute/Vulkan entry
point with bound buffers is future work layered on top of this, not a
reason to block on it.
"""

from __future__ import annotations

import itertools
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue

DEFAULT_DTYPE = "float64"

_DTYPE_KIND: dict[str, str] = {
    "float64": "float",
    "double": "float",
    "float32": "float",
    "float": "float",
    "int64": "int",
    "int32": "int",
    "int": "int",
    "bool": "bool",
}

_DTYPE_WIDTH: dict[str, int] = {
    "float64": 64,
    "double": 64,
    "float32": 32,
    "float": 32,
    "int64": 64,
    "int32": 32,
    "int": 32,
}

# Native opcodes, selected by operand dtype bucket. Comparisons live here too
# -- their *result* type is bool, but which opcode applies is still a
# question about the *operand* type (ordered-float vs signed-int compare).
_BINARY_FLOAT: dict[str, str] = {
    "Add": "OpFAdd", "add": "OpFAdd",
    "Sub": "OpFSub", "sub": "OpFSub",
    "Mul": "OpFMul", "mul": "OpFMul",
    "Div": "OpFDiv", "truediv": "OpFDiv",
    "Mod": "OpFMod", "mod": "OpFMod",
    "Eq": "OpFOrdEqual", "equal": "OpFOrdEqual",
    "Ne": "OpFOrdNotEqual", "not_equal": "OpFOrdNotEqual",
    "Lt": "OpFOrdLessThan", "less": "OpFOrdLessThan",
    "Le": "OpFOrdLessThanEqual", "less_equal": "OpFOrdLessThanEqual",
    "Gt": "OpFOrdGreaterThan", "greater": "OpFOrdGreaterThan",
    "Ge": "OpFOrdGreaterThanEqual", "greater_equal": "OpFOrdGreaterThanEqual",
}

_BINARY_INT: dict[str, str] = {
    "Add": "OpIAdd", "add": "OpIAdd",
    "Sub": "OpISub", "sub": "OpISub",
    "Mul": "OpIMul", "mul": "OpIMul",
    "Div": "OpSDiv", "floordiv": "OpSDiv",
    "Mod": "OpSMod", "mod": "OpSMod",
    "And": "OpBitwiseAnd",
    "Or": "OpBitwiseOr",
    "Xor": "OpBitwiseXor",
    "Shl": "OpShiftLeftLogical",
    "Shr": "OpShiftRightArithmetic",
    "Eq": "OpIEqual", "equal": "OpIEqual",
    "Ne": "OpINotEqual", "not_equal": "OpINotEqual",
    "Lt": "OpSLessThan", "less": "OpSLessThan",
    "Le": "OpSLessThanEqual", "less_equal": "OpSLessThanEqual",
    "Gt": "OpSGreaterThan", "greater": "OpSGreaterThan",
    "Ge": "OpSGreaterThanEqual", "greater_equal": "OpSGreaterThanEqual",
}

_BOOL_BINARY: dict[str, str] = {
    "LAnd": "OpLogicalAnd", "logical_and": "OpLogicalAnd",
    "LOr": "OpLogicalOr", "logical_or": "OpLogicalOr",
}

_UNARY_NATIVE: dict[str, str] = {
    "Neg": "OpFNegate", "neg": "OpFNegate",
    "Not": "OpNot",
    "LNot": "OpLogicalNot", "logical_not": "OpLogicalNot",
}

# Integer-widening/narrowing and float<->int conversions map onto native
# OpXConvert opcodes rather than GLSL.std.450 -- these are core ops, not
# extended-instruction-set math.
_CAST_NATIVE: dict[str, str] = {
    "SExt": "OpSConvert",
    "ZExt": "OpUConvert",
    "FpToSi": "OpConvertFToS",
    "FpToUi": "OpConvertFToU",
    "SiToFp": "OpConvertSToF",
    "UiToFp": "OpConvertUToF",
    "FpExt": "OpFConvert",
    "FpTrunc": "OpFConvert",
}

# GLSL.std.450 extended-instruction names, looked up by canonical op name.
# ``trunc``/``Trunc`` deliberately do not collide: lowercase ``trunc`` is the
# FusedProgram "round toward zero" math op (GLSL.std.450 Trunc); PascalCase
# ``Trunc`` is the SSA integer-narrowing cast op (handled by _CAST_NATIVE,
# checked first).
_UNARY_EXTINST: dict[str, str] = {
    "sqrt": "Sqrt", "Sqrt": "Sqrt",
    "exp": "Exp", "log": "Log",
    "sin": "Sin", "cos": "Cos", "tan": "Tan",
    "asin": "Asin", "acos": "Acos", "atan": "Atan",
    "sinh": "Sinh", "cosh": "Cosh", "tanh": "Tanh",
    "asinh": "Asinh", "acosh": "Acosh", "atanh": "Atanh",
    "floor": "Floor", "Floor": "Floor",
    "ceil": "Ceil",
    "round": "RoundEven",
    "trunc": "Trunc",
    "abs": "FAbs", "Abs": "FAbs",
    "sign": "FSign",
}

_BINARY_EXTINST: dict[str, str] = {
    "pow": "Pow", "Pow": "Pow",
    "maximum": "FMax",
    "minimum": "FMin",
}

# GLSL.std.450 ``Fma`` (extended instruction 50) is the ternary
# single-rounding a*b+c. It reaches this backend as a plain ``Fma``
# instruction -- the precision pipeline's error-free transformations lower
# to it, and FMA_MANDATORY in ``ir_identities`` is the obligation that it be
# spelled as one operation, never as a mul+add that rounds twice. Unlike the
# trigonometric subset, ``Fma`` belongs to GLSL.std.450's width-unrestricted
# instructions, so float64 limbs are fine (it is deliberately absent from
# _EXTINST_FLOAT16_32_ONLY below). Note the extended instruction alone is
# not the whole obligation: per GLSL.std.450, single rounding is only
# guaranteed when the OpExtInst RESULT is decorated NoContraction, which the
# precision-section path in ``_apply`` takes care of.
_TERNARY_EXTINST: dict[str, str] = {
    "Fma": "Fma", "fma": "Fma",
}

# GLSL.std.450's "trigonometric instructions" subset (Sin/Cos/.../Pow) is
# specified for 16- or 32-bit float components only -- Result Type and every
# operand must share one of those widths. ``spirv-val`` rejects a 64-bit
# operand outright ("expected Result Type to be a 16 or 32-bit scalar or
# vector float type"), confirmed against the real validator while building
# this backend. float64 is this repo's default dtype everywhere else, so
# reaching one of these ops at 64-bit width is a real, expected case -- it is
# reported as a shortfall rather than silently narrowed to float32, which
# would change the computed result. The "common instructions" subset (Sqrt,
# FAbs, FSign, Floor, Ceil, Trunc, RoundEven, FMin, FMax) has no such
# restriction and is not in this set.
_EXTINST_FLOAT16_32_ONLY: frozenset[str] = frozenset(
    {
        "Sin", "Cos", "Tan", "Asin", "Acos", "Atan",
        "Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh",
        "Exp", "Log", "Pow",
    }
)

# The float arithmetic spellings whose results carry NoContraction inside a
# precision section. These are exactly the shapes a driver is otherwise
# licensed to contract or reassociate: the native float add/sub/mul/div/
# negate, and the GLSL.std.450 Fma itself (whose single rounding is only
# guaranteed under the decoration -- see _TERNARY_EXTINST). Everything else
# an instruction inside a section can emit (loads, stores, access chains,
# comparisons) has no contraction to withdraw, and SPIR-V permits
# NoContraction on arithmetic results only, so decorating more would trade a
# silent numerical failure for a validation failure.
_NOCONTRACTION_SPELLINGS: frozenset[str] = frozenset(
    {"OpFAdd", "OpFSub", "OpFMul", "OpFDiv", "OpFNegate", "Fma"}
)

# Whole-array-to-scalar reductions -- not in any op table above (they need
# accumulation across elements, not a single instruction). Over a static
# array they unroll in ``_FunctionEmitter._emit_reduction``.
_REDUCTION_OPS: frozenset[str] = frozenset(
    {"sum", "prod", "max", "min", "mean", "all", "any"}
)


class SPIRVEmissionError(ValueError):
    """Raised when an SSA construct has no honest SPIR-V spelling."""


@dataclass(frozen=True)
class SPIRVShortfall:
    """One SSA construct the emitter cannot express yet."""

    op: str
    block: str
    reason: str

    def format(self) -> str:
        return f"{self.op} [{self.block}]: {self.reason}"


@dataclass
class SPIRVFunction:
    """Generated SPIR-V body lines for one SSA function."""

    name: str
    body: tuple[str, ...] = ()
    shortfalls: tuple[SPIRVShortfall, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls


#: Product kinds a SPIR-V manifest names (docs/NODUS_SHADER_DEPLOYMENT_REQUEST.md).
#: A consumer must never reinterpret a library as an executable shader.
LINKABLE_SPIRV_LIBRARY = "linkable-spirv-library"
DEPLOYABLE_SHADER = "deployable-shader"


@dataclass
class SPIRVModule:
    """A complete, assembled SPIR-V text module (consumable by ``spirv-as``).

    This is the linkable library product: exported functions, no entry
    point, no execution mode, no descriptor interface.
    """

    name: str
    source: str
    functions: tuple[SPIRVFunction, ...] = ()
    product_kind = LINKABLE_SPIRV_LIBRARY

    @property
    def shortfalls(self) -> tuple[SPIRVShortfall, ...]:
        return tuple(s for fn in self.functions for s in fn.shortfalls)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.spvasm"
        path.write_text(self.source, encoding="utf-8")
        return path


class _ModuleBuilder:
    """Shared id/type/constant namespace for one SPIR-V module.

    SPIR-V ids and type/constant declarations are global to the module --
    ``OpTypeFloat 64`` may be declared exactly once and referenced by every
    function. Each ``_FunctionEmitter`` shares one of these so two functions
    never redeclare the same type or collide on an id.
    """

    def __init__(self, *, linkage: bool = True) -> None:
        self._next_id = 0
        # A library exports its functions through Linkage. A deployable
        # shader must not: Vulkan forbids the Linkage capability, and its
        # functions are reached from the entry point instead.
        self.linkage = bool(linkage)
        self.capabilities: set[str] = {"Shader", "Linkage"} if linkage else {"Shader"}
        self.needs_glsl_ext = False
        self.type_ids: dict[str, str] = {}
        self.const_ids: dict[tuple[str, Any], str] = {}
        self.type_lines: list[str] = []
        self.exported: list[str] = []
        # OpDecorate lines collected while emitting function bodies. SPIR-V's
        # module layout puts every annotation in one section BEFORE all types
        # and functions, so a decoration discovered mid-body (a NoContraction
        # on a precision-section result, say) cannot be written where it was
        # discovered -- it is parked here and _assemble threads it in next to
        # the linkage decorations. No capability is required for
        # NoContraction itself.
        self.decorations: list[str] = []

    def fresh(self, hint: str) -> str:
        self._next_id += 1
        return f"%{hint}{self._next_id}"

    def scalar_type(self, dtype: str | None) -> str:
        dtype = dtype or DEFAULT_DTYPE
        bucket = _DTYPE_KIND.get(dtype)
        if bucket is None:
            raise SPIRVEmissionError(f"no SPIR-V scalar type for dtype {dtype!r}")
        if bucket == "bool":
            key, id_, line = "bool", "%bool", "%bool = OpTypeBool"
        else:
            width = _DTYPE_WIDTH[dtype]
            key = f"{bucket}{width}"
            id_ = f"%{key}"
            if bucket == "float":
                if width == 64:
                    self.capabilities.add("Float64")
                line = f"{id_} = OpTypeFloat {width}"
            else:
                if width == 64:
                    self.capabilities.add("Int64")
                line = f"{id_} = OpTypeInt {width} 1"
        if key not in self.type_ids:
            self.type_ids[key] = id_
            self.type_lines.append(line)
        return self.type_ids[key]

    def void_type(self) -> str:
        if "void" not in self.type_ids:
            self.type_ids["void"] = "%void"
            self.type_lines.append("%void = OpTypeVoid")
        return self.type_ids["void"]

    def function_type(self, return_type: str, param_types: Sequence[str]) -> str:
        key = f"fn({return_type};{','.join(param_types)})"
        if key not in self.type_ids:
            id_ = self.fresh("fnty")
            params = (" " + " ".join(param_types)) if param_types else ""
            self.type_lines.append(f"{id_} = OpTypeFunction {return_type}{params}")
            self.type_ids[key] = id_
        return self.type_ids[key]

    def constant(self, dtype: str | None, value: Any) -> str:
        dtype = dtype or DEFAULT_DTYPE
        type_id = self.scalar_type(dtype)
        bucket = _DTYPE_KIND[dtype]
        key = (type_id, bucket, value)
        if key not in self.const_ids:
            id_ = self.fresh("c")
            if bucket == "bool":
                opcode = "OpConstantTrue" if value else "OpConstantFalse"
                self.type_lines.append(f"{id_} = {opcode} {type_id}")
            else:
                literal = repr(float(value)) if bucket == "float" else str(int(value))
                self.type_lines.append(f"{id_} = OpConstant {type_id} {literal}")
            self.const_ids[key] = id_
        return self.const_ids[key]

    def glsl_ext(self) -> str:
        self.needs_glsl_ext = True
        return "%glsl"

    def uint_type(self) -> str:
        if "uint" not in self.type_ids:
            self.type_ids["uint"] = "%uint"
            self.type_lines.append("%uint = OpTypeInt 32 0")
        return self.type_ids["uint"]

    def uint_constant(self, value: int) -> str:
        type_id = self.uint_type()
        key = (type_id, "uint", int(value))
        if key not in self.const_ids:
            id_ = self.fresh("cu")
            self.type_lines.append(f"{id_} = OpConstant {type_id} {int(value)}")
            self.const_ids[key] = id_
        return self.const_ids[key]

    def array_type(self, elem_type: str, dims: tuple[int, ...]) -> str:
        """A (possibly nested) ``OpTypeArray`` over ``dims``.

        Dimension 0 is outermost, matching this repo's SSA dimension order
        directly -- built inside-out so the innermost (fastest-varying)
        dimension nests first.
        """

        if not dims:
            raise SPIRVEmissionError("array_type requires at least one dimension")
        type_id = elem_type
        for size in reversed(dims):
            length_id = self.uint_constant(int(size))
            key = f"arr({type_id};{length_id})"
            if key not in self.type_ids:
                id_ = self.fresh("arrty")
                self.type_lines.append(f"{id_} = OpTypeArray {type_id} {length_id}")
                self.type_ids[key] = id_
            type_id = self.type_ids[key]
        return type_id

    def struct_type(self, members: Sequence[str]) -> str:
        """An ``OpTypeStruct`` over ``members``, declared once per member list."""

        key = f"struct({','.join(members)})"
        if key not in self.type_ids:
            id_ = self.fresh("structty")
            self.type_lines.append(
                f"{id_} = OpTypeStruct {' '.join(members)}"
            )
            self.type_ids[key] = id_
        return self.type_ids[key]

    def pointer_type(self, storage_class: str, pointee: str) -> str:
        key = f"ptr({storage_class};{pointee})"
        if key not in self.type_ids:
            id_ = self.fresh("ptrty")
            self.type_lines.append(f"{id_} = OpTypePointer {storage_class} {pointee}")
            self.type_ids[key] = id_
        return self.type_ids[key]


def _declared_rank(value: SSAValue) -> int:
    """Axes a value states: its static shape, or the declared span rank its
    accounting carries when the extents arrive at run time."""

    accounting = getattr(value, "accounting", None) or {}
    return max(
        len(tuple(value.shape or ())),
        int(accounting.get("program_abi_rank", 0) or 0),
    )


def _runtime_sized(value: SSAValue) -> bool:
    return _declared_rank(value) > len(tuple(value.shape or ()))


def _elementwise_runtime_rank(
    function: Function, outputs: Sequence[SSAValue],
) -> tuple[int | None, str | None]:
    """The rank R when ``function`` is elementwise over runtime-sized spans.

    Every value must be a runtime-sized span of one rank R or a rank-0
    scalar; every instruction that reads or writes a span must be an
    elementwise operation whose result is such a span; every output must
    be one. Then element ``k`` of every result depends only on element
    ``k`` of the span operands and on the scalars, and one invocation per
    element computes the function exactly. Returns ``(R, None)``, or
    ``(None, reason)``.
    """

    from .hierarchical_plan import TENSOR_OPERATION_SCALAR_SPELLING
    from .ssa_llvm_backend import scalar_likeness

    if len(function.blocks) != 1:
        return None, "multi-block control flow"
    block = next(iter(function.blocks.values()))
    values = (
        *function.args, *outputs,
        *(instr.res for instr in block.instrs if instr.res is not None),
    )
    ranks = {_declared_rank(value) for value in values if _runtime_sized(value)}
    if len(ranks) != 1:
        return None, (
            "runtime-sized spans of ranks "
            f"{sorted(ranks)}; element mode needs one rank"
        )
    rank = next(iter(ranks))
    shaped = next((value for value in values if value.shape), None)
    if shaped is not None:
        return None, (
            f"%t{shaped.id} has static shape {tuple(shaped.shape)} beside "
            "runtime-sized spans"
        )
    scalar_output = next(
        (value for value in outputs if not _runtime_sized(value)), None,
    )
    if scalar_output is not None:
        return None, (
            f"output %t{scalar_output.id} is a scalar beside runtime-sized "
            "outputs; element mode writes one element per invocation"
        )
    for instr in block.instrs:
        if instr.op in ("Ret", "ret", "Return", "return"):
            continue
        spans_in = any(_runtime_sized(argument) for argument in instr.args)
        span_out = instr.res is not None and _runtime_sized(instr.res)
        if not (spans_in or span_out):
            continue
        operation = str(
            instr.attributes.get("tensor_operation") or instr.op
        )
        operation = TENSOR_OPERATION_SCALAR_SPELLING.get(
            operation.casefold(), operation,
        )
        if not span_out or scalar_likeness(operation) is None:
            return None, (
                f"{instr.op} over a runtime-sized span is not elementwise"
            )
    return rank, None


@dataclass(frozen=True)
class _MapReducePlan:
    """``maps`` feed one runtime-sized span to ``reduction``; ``posts``
    finish its scalar into the one output."""

    maps: tuple[Instr, ...]
    reduction: Instr
    posts: tuple[Instr, ...]


def _runtime_reduction(
    function: Function, outputs: Sequence[SSAValue],
) -> tuple[_MapReducePlan | None, str | None]:
    """The map-reduce a function over runtime-sized spans computes, if any.

    Reduction mode applies when the function's one block is: elementwise
    ``maps`` over runtime-sized span feeds of one rank (scalar feeds and
    constants broadcast), then ONE whole-span ``sum``/``prod``/``max``/
    ``min``/``mean``/``all``/``any`` of a feed or a map's result (``sum``
    may carry a scalar ``start`` feed), then scalar ``posts`` over that
    total, scalar feeds and constants, ending in the one scalar output.
    Program: ``(x * y).sum() * 2.0``. Returns ``(plan, None)``,
    ``(None, reason)`` for a reduction it cannot carry, or ``(None, None)``
    when the function reduces no runtime-sized span.
    """

    if len(function.blocks) != 1:
        return None, None
    block = next(iter(function.blocks.values()))
    body = [
        instr for instr in block.instrs
        if instr.op not in ("Ret", "ret", "Return", "return")
    ]

    def operation(instr: Instr) -> str:
        return str(instr.attributes.get("tensor_operation") or instr.op)

    position = next((
        index for index, instr in enumerate(body)
        if operation(instr) in _REDUCTION_OPS and instr.args
        and _runtime_sized(instr.args[0])
        and not (operation(instr) in ("max", "min") and len(instr.args) > 1)
    ), None)
    if position is None:
        return None, None
    instr = body[position]
    op = operation(instr)
    source = instr.args[0]
    arguments = {int(value.id) for value in function.args}
    if instr.attributes.get("axis") is not None or instr.attributes.get("dim") is not None:
        return None, f"{op} over an axis of a runtime-sized span"
    rank = _declared_rank(source)
    spans = {int(value.id) for value in function.args if _runtime_sized(value)}
    scalars = {
        int(value.id) for value in function.args
        if not _runtime_sized(value) and not tuple(value.shape or ())
    }
    maps: list[Instr] = []
    for item in body[:position]:
        if operation(item) in ("Const", "const") and item.res is not None:
            if tuple(item.res.shape or ()) or _runtime_sized(item.res):
                return None, f"constant %t{item.res.id} is not a scalar"
            scalars.add(int(item.res.id))
            maps.append(item)
            continue
        if (
            item.res is None or not _runtime_sized(item.res)
            or _declared_rank(item.res) != rank
            or not item.args
            or any(
                int(value.id) not in spans | scalars for value in item.args
            )
            or any(
                int(value.id) in spans and _declared_rank(value) != rank
                for value in item.args
            )
        ):
            return None, (
                f"{operation(item)} before the reduction is not elementwise "
                f"over rank-{rank} runtime-sized spans"
            )
        spans.add(int(item.res.id))
        maps.append(item)
    if int(source.id) not in spans:
        return None, f"%t{source.id} is not a feed of {function.name}"
    if (
        instr.res is None or tuple(instr.res.shape or ())
        or _runtime_sized(instr.res)
    ):
        return None, f"{op} of %t{source.id} must be a scalar"
    if len(instr.args) > (2 if op == "sum" else 1) or any(
        int(value.id) not in arguments or value.shape or _runtime_sized(value)
        for value in instr.args[1:]
    ):
        return None, f"{op} takes one span (and for sum a scalar start feed)"
    scalars.add(int(instr.res.id))
    posts: list[Instr] = []
    for item in body[position + 1:]:
        if (
            item.res is None or tuple(item.res.shape or ())
            or _runtime_sized(item.res)
            or any(int(value.id) not in scalars for value in item.args)
        ):
            return None, (
                f"{operation(item)} after the reduction reads a span; only "
                "scalars may follow it"
            )
        scalars.add(int(item.res.id))
        posts.append(item)
    last = (posts[-1] if posts else instr).res
    if [int(value.id) for value in outputs] != [int(last.id)]:
        return None, f"{op} of %t{source.id} must lead to the one scalar output"
    if str(last.dtype or DEFAULT_DTYPE) == "bool":
        return None, f"a bool {op} result has no storage-buffer spelling"
    if op == "mean" and _DTYPE_KIND.get(str(instr.res.dtype or DEFAULT_DTYPE)) != "float":
        return None, "mean needs a float result"
    if op not in ("all", "any") and _DTYPE_KIND.get(str(source.dtype or DEFAULT_DTYPE)) == "bool":
        return None, f"a {op} over bool elements"
    return _MapReducePlan(tuple(maps), instr, tuple(posts)), None


class _FunctionEmitter:
    """Translate one straight-line SSA ``Function`` into SPIR-V.

    Scalar values pass by value; array-shaped values are represented as
    ``Function``-storage pointers (see module docstring). Control flow stays
    single-block -- array elementwise ops are unrolled, not looped.
    """

    def __init__(
        self,
        function: Function,
        builder: _ModuleBuilder,
        *,
        dtype: str = DEFAULT_DTYPE,
        outputs: Sequence[SSAValue] = (),
        carried_shortfalls: Sequence[SPIRVShortfall] = (),
        runtime_elements: bool = False,
    ):
        self.function = function
        # The deployable entry's element mode: runtime-sized spans of an
        # elementwise function arrive one element per invocation, so the
        # body computes on scalars (``_elementwise_runtime_rank``).
        self.runtime_elements = bool(runtime_elements)
        self.builder = builder
        self.dtype = dtype
        self.outputs = tuple(outputs)
        # Shortfalls discovered before emission started (precision-section
        # obligations the backend cannot honour) travel on the same channel
        # as the ones discovered per-instruction below.
        self.shortfalls: list[SPIRVShortfall] = list(carried_shortfalls)
        # Array-shaped SSA value id -> the Function-storage pointer holding
        # it (an OpFunctionParameter for arguments, an OpVariable for
        # computed temporaries). Scalars never appear here -- they pass by
        # value under their own %tN name.
        self._array_pointer: dict[int, str] = {}

    def _name(self, value: SSAValue) -> str:
        return f"%t{value.id}"

    def _shortfall(self, op: str, block: str, reason: str) -> None:
        self.shortfalls.append(SPIRVShortfall(op, block, reason))

    def _bail(self, op: str, block: str, reason: str) -> SPIRVFunction:
        self._shortfall(op, block, reason)
        return SPIRVFunction(self.function.name, (), tuple(self.shortfalls))

    def emit(self) -> SPIRVFunction:
        function = self.function
        if len(function.blocks) != 1:
            return self._bail(
                "<function>", "*",
                "multi-block control flow (Br/CondBr/Phi) is not yet "
                "supported by the SPIR-V backend",
            )
        block = next(iter(function.blocks.values()))

        # Program: ``x * y`` over spans declared ``storage: span, rank: 1``
        # with no ``shape``.  The rank rides the accounting and ``.shape`` is
        # empty, so this emitter would read each one as a scalar and return
        # x[0] * y[0].  Rule: a value whose declared rank exceeds its static
        # shape has runtime extents, which an unrolled single-block function
        # cannot address; it is refused by name.
        runtime_sized = next((
            value for value in (
                *function.args, *self.outputs,
                *(instr.res for instr in block.instrs if instr.res is not None),
            )
            if _runtime_sized(value)
        ), None)
        if runtime_sized is not None and not self.runtime_elements:
            return self._bail(
                "<function>", block.name,
                f"%t{runtime_sized.id} is a runtime-sized span (rank "
                f"{_declared_rank(runtime_sized)}, no static shape); the "
                "SPIR-V backend unrolls static shapes only",
            )

        # Several outputs leave as one struct, member m being output m; a
        # caller (the deployable entry) extracts each member.
        if len(self.outputs) > 1:
            return_type = self.builder.struct_type(
                [self._plain_type(out) for out in self.outputs]
            )
        elif self.outputs:
            return_type = self._plain_type(self.outputs[0])
        else:
            return_type = self.builder.void_type()
        self.return_type = return_type
        param_types = [self._value_type(a) for a in function.args]
        fn_type = self.builder.function_type(return_type, param_types)

        body: list[str] = []
        fn_id = f"%{function.name}"
        body.append(f"{fn_id} = OpFunction {return_type} None {fn_type}")
        for argument, param_type in zip(function.args, param_types):
            body.append(f"{self._name(argument)} = OpFunctionParameter {param_type}")
            if argument.shape:
                self._array_pointer[argument.id] = self._name(argument)
        body.append(f"%{function.name}_entry = OpLabel")

        # Every Function-storage OpVariable must precede all other
        # instructions in the block -- declare storage for every
        # array-shaped result up front, before emitting any computation.
        for instr in block.instrs:
            if instr.res is None or not instr.res.shape:
                continue
            if instr.res.id in self._array_pointer:
                continue
            # _value_type() returns the *pointer* type for a shaped value
            # (that's what a parameter/return needs); OpVariable's Result
            # Type must itself be that pointer type, not a pointer to it.
            ptr_type = self._value_type(instr.res)
            var_id = f"%t{instr.res.id}_arr"
            body.append(f"{var_id} = OpVariable {ptr_type} Function")
            self._array_pointer[instr.res.id] = var_id

        for instr in block.instrs:
            self._emit_instr(instr, block, body)

        body.append("OpFunctionEnd")
        self.builder.exported.append(function.name)
        return SPIRVFunction(function.name, tuple(body), tuple(self.shortfalls))

    def _plain_type(self, value: SSAValue) -> str:
        """The plain (non-pointer) SPIR-V type of ``value``.

        A scalar's own type, or the array type itself (not a pointer to it)
        -- what ``OpReturnValue`` and ``OpLoad`` both need as a Result Type.
        """

        elem_type = self.builder.scalar_type(value.dtype or self.dtype)
        if not value.shape:
            return elem_type
        return self.builder.array_type(
            elem_type, tuple(int(size) for size in value.shape)
        )

    def _value_type(self, value: SSAValue) -> str:
        """The SPIR-V type of ``value`` as it flows through the function.

        A scalar's own type; an array's storage pointer type (arrays never
        pass by value here except as a return value -- see the module
        docstring and ``_plain_type``).
        """

        elem_type = self.builder.scalar_type(value.dtype or self.dtype)
        if not value.shape:
            return elem_type
        array_type = self.builder.array_type(
            elem_type, tuple(int(size) for size in value.shape)
        )
        return self.builder.pointer_type("Function", array_type)

    @staticmethod
    def _resolve(op: str, operand_dtype: str | None, arity: int) -> tuple[str, str] | None:
        """The SPIR-V opcode/extended-instruction for ``op``, or ``None``.

        Resolution depends only on the op name, the operand dtype bucket,
        and how many operands there are -- not on whether any operand is an
        array, which is exactly what lets one lookup serve both the scalar
        path and the per-element array path below.
        """

        bucket = _DTYPE_KIND.get(operand_dtype or DEFAULT_DTYPE)
        binary_table = _BINARY_INT if bucket == "int" else _BINARY_FLOAT
        if arity == 2 and op in binary_table:
            return ("binary", binary_table[op])
        if arity == 2 and op in _BOOL_BINARY:
            return ("binary", _BOOL_BINARY[op])
        if arity == 2 and op in _BINARY_EXTINST:
            return ("extinst_binary", _BINARY_EXTINST[op])
        if arity == 1 and op in _CAST_NATIVE:
            return ("cast", _CAST_NATIVE[op])
        if arity == 1 and op in _UNARY_NATIVE:
            return ("unary", _UNARY_NATIVE[op])
        if arity == 1 and op in _UNARY_EXTINST:
            return ("extinst_unary", _UNARY_EXTINST[op])
        if arity == 3 and op in _TERNARY_EXTINST:
            return ("extinst_ternary", _TERNARY_EXTINST[op])
        if arity == 3 and op in ("Select", "where"):
            return ("select", "")
        return None

    def _apply(
        self,
        kind: str,
        payload: str,
        result_type: str,
        result_id: str,
        operand_ids: list[str],
        instr: Instr,
        body: list[str],
    ) -> None:
        """Emit exactly one SPIR-V instruction computing ``result_id``.

        Shared by the scalar path (operand_ids are plain SSA value names)
        and the array path (operand_ids are per-element loaded values) --
        neither knows or cares which.
        """

        if kind == "binary":
            ids = operand_ids
            if instr.attributes.get("reverse"):
                ids = [ids[1], ids[0]]
            body.append(f"    {result_id} = {payload} {result_type} {ids[0]} {ids[1]}")
        elif kind == "extinst_binary":
            ext = self.builder.glsl_ext()
            body.append(
                f"    {result_id} = OpExtInst {result_type} {ext} "
                f"{payload} {operand_ids[0]} {operand_ids[1]}"
            )
        elif kind in ("cast", "unary"):
            body.append(f"    {result_id} = {payload} {result_type} {operand_ids[0]}")
        elif kind == "extinst_unary":
            ext = self.builder.glsl_ext()
            body.append(
                f"    {result_id} = OpExtInst {result_type} {ext} "
                f"{payload} {operand_ids[0]}"
            )
        elif kind == "extinst_ternary":
            ext = self.builder.glsl_ext()
            body.append(
                f"    {result_id} = OpExtInst {result_type} {ext} "
                f"{payload} {operand_ids[0]} {operand_ids[1]} {operand_ids[2]}"
            )
        elif kind == "select":
            body.append(
                f"    {result_id} = OpSelect {result_type} "
                f"{operand_ids[0]} {operand_ids[1]} {operand_ids[2]}"
            )
        # Inside a precision section, contraction must be withdrawn from
        # every arithmetic result INCLUDING the declared Fma -- GLSL.std.450
        # only guarantees Fma's single rounding when its OpExtInst result is
        # decorated NoContraction, so without this line the FMA_MANDATORY
        # claim in BACKEND_PRECISION_CAPABILITIES would be a lie. Ordinary
        # instructions are deliberately left undecorated: contraction stays
        # available to them, exactly as SECTION_ISOLATION scopes it.
        if (
            instr.attributes.get("precision_section")
            and payload in _NOCONTRACTION_SPELLINGS
        ):
            self.builder.decorations.append(
                f"OpDecorate {result_id} NoContraction"
            )

    def _augment_scalar_attrs(
        self, instr: Instr, operand_ids: list[str], operand_dtype: str | None
    ) -> list[str]:
        """Fold ``right_scalar``/``left_scalar`` attributes into ``operand_ids``."""

        right = instr.attributes.get("right_scalar")
        left = instr.attributes.get("left_scalar")
        if right is not None and len(operand_ids) == 1:
            return [operand_ids[0], self.builder.constant(operand_dtype, right)]
        if left is not None and len(operand_ids) == 1:
            return [self.builder.constant(operand_dtype, left), operand_ids[0]]
        return operand_ids

    @staticmethod
    def _broadcast_index(
        operand_shape: tuple[int, ...],
        result_shape: tuple[int, ...],
        index: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        """numpy-style broadcast of a flat ``result_shape`` index onto ``operand_shape``.

        Dimensions align from the right; an extent-one operand dimension
        always reads index 0 regardless of the result index there.
        """

        rank = len(result_shape)
        if len(operand_shape) > rank:
            return None
        offset = rank - len(operand_shape)
        operand_index = []
        for position, size in enumerate(operand_shape):
            size = int(size)
            result_position = offset + position
            if size == 1:
                operand_index.append(0)
            elif size == int(result_shape[result_position]):
                operand_index.append(index[result_position])
            else:
                return None
        return tuple(operand_index)

    def _load_element(
        self,
        pointer: str,
        index: tuple[int, ...],
        elem_type: str,
        body: list[str],
    ) -> str:
        index_ids = [self.builder.uint_constant(i) for i in index]
        ptr_type = self.builder.pointer_type("Function", elem_type)
        access = self.builder.fresh("ap")
        body.append(
            f"    {access} = OpAccessChain {ptr_type} {pointer} "
            f"{' '.join(index_ids)}"
        )
        loaded = self.builder.fresh("v")
        body.append(f"    {loaded} = OpLoad {elem_type} {access}")
        return loaded

    def _store_element(
        self,
        pointer: str,
        index: tuple[int, ...],
        elem_type: str,
        value_id: str,
        body: list[str],
    ) -> None:
        index_ids = [self.builder.uint_constant(i) for i in index]
        ptr_type = self.builder.pointer_type("Function", elem_type)
        access = self.builder.fresh("ap")
        body.append(
            f"    {access} = OpAccessChain {ptr_type} {pointer} "
            f"{' '.join(index_ids)}"
        )
        body.append(f"    OpStore {access} {value_id}")

    def _max_step(
        self, rule: str, greatest: bool, running: str, value: str,
        value_type: str, floating: bool, body: list[str],
        result_id: str | None = None,
    ) -> str:
        """One ``max``/``min`` step under ``rule`` (see ``max_semantics``).

        ``PYTHON``: ``value`` replaces ``running`` only when strictly
        better, so a NaN never wins and a tie keeps ``running``. ``NUMPY``:
        a NaN on either side is the result (``running``'s first), otherwise
        the better, with a tie taking ``value``. ``FMax`` is not used: its
        NaN result is undefined by GLSL.std.450.
        """

        from .max_semantics import PYTHON

        bool_type = self.builder.scalar_type("bool")
        if floating:
            compare = "OpFOrdGreaterThan" if greatest else "OpFOrdLessThan"
        else:
            compare = "OpSGreaterThan" if greatest else "OpSLessThan"
        out = result_id or self.builder.fresh("v")
        if rule == PYTHON or not floating:
            wins = self.builder.fresh("v")
            body.append(f"    {wins} = {compare} {bool_type} {value} {running}")
            body.append(f"    {out} = OpSelect {value_type} {wins} {value} {running}")
            return out
        keeps, pick = self.builder.fresh("v"), self.builder.fresh("v")
        nan_value, after = self.builder.fresh("v"), self.builder.fresh("v")
        nan_running = self.builder.fresh("v")
        body.append(f"    {keeps} = {compare} {bool_type} {running} {value}")
        body.append(f"    {pick} = OpSelect {value_type} {keeps} {running} {value}")
        body.append(f"    {nan_value} = OpIsNan {bool_type} {value}")
        body.append(f"    {after} = OpSelect {value_type} {nan_value} {value} {pick}")
        body.append(f"    {nan_running} = OpIsNan {bool_type} {running}")
        body.append(f"    {out} = OpSelect {value_type} {nan_running} {running} {after}")
        return out

    def _convert(
        self, value_id: str, source: str, target: str, body: list[str]
    ) -> str | None:
        """``value_id`` of dtype ``source`` as dtype ``target``, or ``None``."""

        source_kind, target_kind = _DTYPE_KIND.get(source), _DTYPE_KIND.get(target)
        source_width, target_width = _DTYPE_WIDTH.get(source), _DTYPE_WIDTH.get(target)
        if (source_kind, source_width) == (target_kind, target_width):
            return value_id
        if source_kind == "int" and target_kind == "float":
            opcode = "OpConvertSToF"
        elif source_kind == target_kind == "float":
            opcode = "OpFConvert"
        elif source_kind == target_kind == "int":
            opcode = "OpSConvert"
        else:
            return None
        converted = self.builder.fresh("v")
        body.append(
            f"    {converted} = {opcode} {self.builder.scalar_type(target)} {value_id}"
        )
        return converted

    def _emit_max_pair(
        self, instr: Instr, block: BasicBlock, body: list[str],
        rule: str, greatest: bool,
    ) -> None:
        """A stamped two-operand ``max``/``min``, per element over arrays."""

        result_dtype = instr.res.dtype or self.dtype
        result_type = self.builder.scalar_type(result_dtype)
        floating = _DTYPE_KIND.get(result_dtype) == "float"
        result_shape = tuple(int(size) for size in instr.res.shape)
        if not result_shape:
            self._max_step(
                rule, greatest, self._name(instr.args[0]), self._name(instr.args[1]),
                result_type, floating, body, result_id=self._name(instr.res),
            )
            return
        pointer = self._array_pointer.get(instr.res.id)
        if pointer is None:
            self._shortfall(
                instr.op, block.name, "internal: no storage allocated for this array result"
            )
            return
        for index in itertools.product(*(range(size) for size in result_shape)):
            operands = []
            for argument in instr.args:
                shape = tuple(int(size) for size in argument.shape)
                if not shape:
                    operands.append(self._name(argument))
                    continue
                at = self._broadcast_index(shape, result_shape, index)
                source = self._array_pointer.get(argument.id)
                if at is None or source is None:
                    self._shortfall(
                        instr.op, block.name,
                        "operand shape does not broadcast to the result shape",
                    )
                    return
                operands.append(self._load_element(
                    source, at, self.builder.scalar_type(argument.dtype or self.dtype), body,
                ))
            value = self._max_step(
                rule, greatest, operands[0], operands[1], result_type, floating, body,
            )
            self._store_element(pointer, index, result_type, value, body)

    def _emit_axis_reduction(
        self, op: str, axis, instr: Instr, block: BasicBlock, body: list[str]
    ) -> None:
        """``y.sum(axis=k)`` (and prod/mean/max/min) over a static shape.

        Unrolled like the whole-array form: one result per lane of the
        kept axes, row-major, folded along axis k in index order; max/min
        take numpy's rule (a NaN in the lane is its result), as in C.
        """

        source = instr.args[0]
        shape = tuple(int(size) for size in source.shape or ())
        result_shape = tuple(int(size) for size in instr.res.shape or ())
        if not isinstance(axis, int) or not shape or not -len(shape) <= axis < len(shape):
            self._shortfall(
                op, block.name,
                f"{op} over axis {axis!r} needs one in-range integer axis of a "
                "static shape",
            )
            return
        axis %= len(shape)
        kept = shape[:axis] + shape[axis + 1:]
        if math.prod(result_shape) != math.prod(kept) or (
            shape[axis] == 0 and op in ("max", "min")
        ):
            self._shortfall(
                op, block.name,
                f"{op} over axis {axis} of {shape!r} into {result_shape!r} "
                "(an empty max/min axis raises in numpy)",
            )
            return
        pointer = self._array_pointer.get(source.id)
        target = self._array_pointer.get(instr.res.id)
        if pointer is None or target is None:
            self._shortfall(op, block.name, "array operand has no known storage")
            return
        source_dtype = source.dtype or self.dtype
        result_dtype = instr.res.dtype or self.dtype
        result_kind = _DTYPE_KIND.get(result_dtype)
        if "bool" in (_DTYPE_KIND.get(source_dtype), result_kind) or (
            op == "mean" and result_kind != "float"
        ):
            self._shortfall(
                op, block.name, f"{op} over an axis into {result_dtype}",
            )
            return
        source_type = self.builder.scalar_type(source_dtype)
        result_type = self.builder.scalar_type(result_dtype)
        lanes = list(itertools.product(*(range(size) for size in kept)))
        stored_shape = result_shape or (1,)
        for flat, lane in enumerate(lanes):
            elements = []
            for step in range(shape[axis]):
                index = lane[:axis] + (step,) + lane[axis:]
                value = self._load_element(pointer, index, source_type, body)
                converted = self._convert(value, source_dtype, result_dtype, body)
                if converted is None:
                    self._shortfall(
                        op, block.name,
                        f"no conversion from {source_dtype} to {result_dtype}",
                    )
                    return
                elements.append(converted)
            if op in ("max", "min"):
                from .max_semantics import NUMPY, max_step_rule

                accumulator = elements[0]
                for value in elements[1:]:
                    accumulator = self._max_step(
                        max_step_rule(instr) or NUMPY, op == "max", accumulator,
                        value, result_type, result_kind == "float", body,
                    )
            else:
                kind, payload = self._resolve(
                    "mul" if op == "prod" else "add", result_dtype, 2,
                )
                accumulator = self.builder.constant(
                    result_dtype, 1 if op == "prod" else 0,
                )
                for value in elements:
                    folded = self.builder.fresh("v")
                    self._apply(
                        kind, payload, result_type, folded,
                        [accumulator, value], instr, body,
                    )
                    accumulator = folded
                if op == "mean":
                    count = self.builder.constant(result_dtype, len(elements))
                    kind, payload = self._resolve("Div", result_dtype, 2)
                    averaged = self.builder.fresh("v")
                    self._apply(
                        kind, payload, result_type, averaged,
                        [accumulator, count], instr, body,
                    )
                    accumulator = averaged
            at = []
            remainder = flat
            for size in reversed(stored_shape):
                at.append(remainder % size)
                remainder //= size
            self._store_element(target, tuple(reversed(at)), result_type, accumulator, body)

    def _emit_reduction(
        self, op: str, instr: Instr, block: BasicBlock, body: list[str]
    ) -> None:
        """Whole-array ``sum``/``prod``/``max``/``min``/``mean``/``all``/``any``.

        Program: ``total = x.sum()`` over a static (2, 3) ``x`` in a shader.
        The shape is static, so the reduction unrolls like every other array
        op here: each element is loaded in row-major order and folded left to
        right, as Python adds.  ``sum(xs, start)`` starts from ``start``;
        ``max``/``min`` let a NaN in and never displace one (the numpy
        reference, and the C backend's rule); ``all``/``any`` count NaN as
        true.  An axis reduction, or ``max``/``min``/``mean`` of no elements,
        is refused by name.
        """

        source = instr.args[0]
        axis = instr.attributes.get("axis")
        if axis is None:
            axis = instr.attributes.get("dim")
        if axis is not None and op in ("sum", "prod", "mean", "max", "min"):
            self._emit_axis_reduction(op, axis, instr, block, body)
            return
        if (
            len(instr.args) > (2 if op == "sum" else 1)
            or any(a.shape for a in instr.args[1:])
            or axis is not None
        ):
            self._shortfall(
                op, block.name,
                "only whole-array reductions (no axis) are supported by the "
                "SPIR-V backend",
            )
            return
        pointer = self._array_pointer.get(source.id)
        if pointer is None:
            self._shortfall(op, block.name, "array operand has no known storage")
            return
        source_dtype = source.dtype or self.dtype
        result_dtype = instr.res.dtype or self.dtype
        source_kind = _DTYPE_KIND.get(source_dtype)
        result_kind = _DTYPE_KIND.get(result_dtype)
        source_type = self.builder.scalar_type(source_dtype)
        result_type = self.builder.scalar_type(result_dtype)
        shape = tuple(int(size) for size in source.shape)
        indices = list(itertools.product(*(range(size) for size in shape)))
        name = self._name(instr.res)

        if op in ("all", "any"):
            bool_type = self.builder.scalar_type("bool")
            truth = self.builder.constant("bool", op == "all")
            for index in indices:
                value = self._load_element(pointer, index, source_type, body)
                if source_kind == "float":
                    test = self.builder.fresh("v")
                    zero = self.builder.constant(source_dtype, 0.0)
                    body.append(f"    {test} = OpFUnordNotEqual {bool_type} {value} {zero}")
                elif source_kind == "int":
                    test = self.builder.fresh("v")
                    zero = self.builder.constant(source_dtype, 0)
                    body.append(f"    {test} = OpINotEqual {bool_type} {value} {zero}")
                else:
                    test = value
                folded = self.builder.fresh("v")
                logical = "OpLogicalAnd" if op == "all" else "OpLogicalOr"
                body.append(f"    {folded} = {logical} {bool_type} {truth} {test}")
                truth = folded
            if result_kind == "bool":
                body.append(f"    {name} = OpCopyObject {bool_type} {truth}")
            else:
                one = self.builder.constant(result_dtype, 1)
                zero = self.builder.constant(result_dtype, 0)
                body.append(f"    {name} = OpSelect {result_type} {truth} {one} {zero}")
            return

        if source_kind == "bool" or result_kind == "bool":
            self._shortfall(
                op, block.name,
                f"a {op} reduction over or into bool is not supported by the "
                "SPIR-V backend",
            )
            return
        if op == "mean" and result_kind != "float":
            self._shortfall(op, block.name, "mean needs a float result")
            return
        if not indices and op in ("max", "min", "mean"):
            self._shortfall(
                op, block.name, f"{op} of an empty array has no value",
            )
            return

        elements: list[str] = []
        for index in indices:
            value = self._load_element(pointer, index, source_type, body)
            converted = self._convert(value, source_dtype, result_dtype, body)
            if converted is None:
                self._shortfall(
                    op, block.name,
                    f"no conversion from {source_dtype} to {result_dtype}",
                )
                return
            elements.append(converted)

        if op in ("max", "min"):
            from .max_semantics import NUMPY, max_step_rule

            rule = max_step_rule(instr) or NUMPY
            accumulator = elements[0]
            for value in elements[1:]:
                accumulator = self._max_step(
                    rule, op == "max", accumulator, value, result_type,
                    result_kind == "float", body,
                )
            body.append(f"    {name} = OpCopyObject {result_type} {accumulator}")
            return

        fold = "mul" if op == "prod" else "add"
        kind, payload = self._resolve(fold, result_dtype, 2)
        if op == "sum" and len(instr.args) == 2:
            start = instr.args[1]
            accumulator = self._convert(
                self._name(start), start.dtype or self.dtype, result_dtype, body
            )
            if accumulator is None:
                self._shortfall(
                    op, block.name,
                    f"no conversion from {start.dtype} to {result_dtype}",
                )
                return
        else:
            accumulator = self.builder.constant(result_dtype, 1 if op == "prod" else 0)
        for value in elements:
            folded = self.builder.fresh("v")
            self._apply(kind, payload, result_type, folded, [accumulator, value], instr, body)
            accumulator = folded
        if op == "mean":
            count = self.builder.constant(result_dtype, len(elements))
            kind, payload = self._resolve("Div", result_dtype, 2)
            self._apply(kind, payload, result_type, name, [accumulator, count], instr, body)
            return
        body.append(f"    {name} = OpCopyObject {result_type} {accumulator}")

    def _emit_instr(self, instr: Instr, block: BasicBlock, body: list[str]) -> None:
        if instr.op in ("Br", "br", "CondBr", "condbr", "Phi", "phi"):
            self._shortfall(
                instr.op, block.name,
                "multi-block control flow is not yet supported by the "
                "SPIR-V backend",
            )
            return
        if instr.op in ("Ret", "ret", "Return", "return"):
            if len(self.outputs) > 1:
                members = []
                for out in self.outputs:
                    if not out.shape:
                        members.append(self._name(out))
                        continue
                    pointer = self._array_pointer.get(out.id)
                    if pointer is None:
                        self._shortfall(
                            instr.op, block.name,
                            f"output %t{out.id} has no known array storage",
                        )
                        return
                    loaded = self.builder.fresh("v")
                    body.append(
                        f"    {loaded} = OpLoad {self._plain_type(out)} {pointer}"
                    )
                    members.append(loaded)
                value_id = self.builder.fresh("v")
                body.append(
                    f"    {value_id} = OpCompositeConstruct {self.return_type} "
                    + " ".join(members)
                )
                body.append(f"    OpReturnValue {value_id}")
                return
            if self.outputs:
                out = self.outputs[0]
                if out.shape:
                    pointer = self._array_pointer.get(out.id)
                    if pointer is None:
                        self._shortfall(
                            instr.op, block.name,
                            "the function output has no known array storage",
                        )
                        return
                    array_type = self._plain_type(out)
                    value_id = self.builder.fresh("v")
                    body.append(f"    {value_id} = OpLoad {array_type} {pointer}")
                    body.append(f"    OpReturnValue {value_id}")
                else:
                    body.append(f"    OpReturnValue {self._name(out)}")
            else:
                body.append("    OpReturn")
            return

        op = instr.op
        if op in ("Call", "call"):
            # precompile_to_ssa wraps almost every tensor op in Handler.Call,
            # preserving the canonical op name under "tensor_operation" --
            # see ssa_fortran_backend.py's identical unwrap for why this
            # matters: without it, every op from that lowering path reports
            # as an unsupported "Call" instead of the op it actually is.
            tensor_operation = instr.attributes.get("tensor_operation")
            if tensor_operation is not None:
                op = str(tensor_operation)
            elif instr.attributes.get("result_convention") == "ssa.aggregate":
                self._shortfall(
                    instr.op, block.name,
                    "region calls are not yet supported by the SPIR-V "
                    "backend",
                )
                return

        if instr.res is None:
            self._shortfall(
                instr.op, block.name,
                "instructions with no result are not yet supported by the "
                "SPIR-V backend",
            )
            return

        if op in ("NoneValue", "nonevalue"):
            self._shortfall(
                op, block.name,
                "SPIR-V shader values have no untyped absence value; lower "
                "an optional to explicit presence and payload values first",
            )
            return

        result_dtype = instr.res.dtype or self.dtype
        result_shape = tuple(int(size) for size in instr.res.shape)
        elem_type = self.builder.scalar_type(result_dtype)

        if op in ("Const", "const"):
            if result_shape:
                self._shortfall(
                    op, block.name,
                    "array constants are not yet supported by the SPIR-V "
                    "backend",
                )
                return
            value = instr.attributes.get("constant")
            if value is None:
                value = instr.attributes.get("value")
            if value is None:
                self._shortfall(
                    op, block.name,
                    "no literal value is recorded for this constant",
                )
                return
            const_id = self.builder.constant(result_dtype, value)
            body.append(f"    {self._name(instr.res)} = OpCopyObject {elem_type} {const_id}")
            return

        if not result_shape and any(a.shape for a in instr.args) and op in _REDUCTION_OPS:
            self._emit_reduction(op, instr, block, body)
            return
        if (
            op in ("sum", "prod", "mean", "max", "min")
            and instr.args and instr.args[0].shape
            and (
                instr.attributes.get("axis") is not None
                or instr.attributes.get("dim") is not None
            )
        ):
            self._emit_axis_reduction(
                op,
                instr.attributes.get("axis")
                if instr.attributes.get("axis") is not None
                else instr.attributes.get("dim"),
                instr, block, body,
            )
            return

        from .max_semantics import is_max, max_step_rule

        greatest = is_max(op)
        rule = max_step_rule(instr)
        if (
            greatest is not None and rule is not None and len(instr.args) == 2
            and not any(
                instr.attributes.get(key) is not None
                for key in ("right_scalar", "left_scalar")
            )
        ):
            self._emit_max_pair(instr, block, body, rule, greatest)
            return

        operand_dtype = instr.args[0].dtype if instr.args else result_dtype
        has_scalar_attr = (
            instr.attributes.get("right_scalar") is not None
            or instr.attributes.get("left_scalar") is not None
        )
        arity = len(instr.args) + (1 if has_scalar_attr else 0)
        resolved = self._resolve(op, operand_dtype, arity)
        if resolved is None:
            self._shortfall(
                op, block.name,
                "no SPIR-V opcode or GLSL.std.450 extended instruction is "
                "registered",
            )
            return
        kind, payload = resolved
        if kind in ("extinst_binary", "extinst_unary"):
            if payload in _EXTINST_FLOAT16_32_ONLY and _DTYPE_WIDTH.get(result_dtype) == 64:
                self._shortfall(
                    op, block.name,
                    f"GLSL.std.450 {payload} is restricted to 16/32-bit "
                    "float operands; this value is float64",
                )
                return

        array_operands = any(a.shape for a in instr.args)

        if not result_shape and not array_operands:
            operand_ids = self._augment_scalar_attrs(
                instr, [self._name(a) for a in instr.args], operand_dtype
            )
            self._apply(kind, payload, elem_type, self._name(instr.res), operand_ids, instr, body)
            return

        if not result_shape:
            # An array operand feeding a scalar result outside the Const/
            # aggregate-call cases above is a reduction -- not yet supported.
            self._shortfall(
                op, block.name,
                "reductions over arrays are not yet supported by the "
                "SPIR-V backend",
            )
            return

        pointer = self._array_pointer.get(instr.res.id)
        if pointer is None:
            self._shortfall(
                op, block.name, "internal: no storage allocated for this array result"
            )
            return

        for index in itertools.product(*(range(size) for size in result_shape)):
            operand_ids: list[str] = []
            for argument in instr.args:
                argument_shape = tuple(int(size) for size in argument.shape)
                if not argument_shape:
                    operand_ids.append(self._name(argument))
                    continue
                argument_index = self._broadcast_index(argument_shape, result_shape, index)
                if argument_index is None:
                    self._shortfall(
                        op, block.name,
                        "operand shape does not broadcast to the result shape",
                    )
                    return
                argument_pointer = self._array_pointer.get(argument.id)
                if argument_pointer is None:
                    self._shortfall(
                        op, block.name, "array operand has no known storage"
                    )
                    return
                argument_elem_type = self.builder.scalar_type(argument.dtype or self.dtype)
                operand_ids.append(
                    self._load_element(argument_pointer, argument_index, argument_elem_type, body)
                )
            operand_ids = self._augment_scalar_attrs(instr, operand_ids, operand_dtype)
            elem_result_id = self.builder.fresh("v")
            self._apply(kind, payload, elem_type, elem_result_id, operand_ids, instr, body)
            self._store_element(pointer, index, elem_type, elem_result_id, body)


def _emit_function(
    function: Function,
    builder: _ModuleBuilder,
    *,
    dtype: str,
    outputs: Sequence[SSAValue],
    carried_shortfalls: Sequence[SPIRVShortfall] = (),
) -> SPIRVFunction:
    return _FunctionEmitter(
        function,
        builder,
        dtype=dtype,
        outputs=outputs,
        carried_shortfalls=carried_shortfalls,
    ).emit()


def _precision_shortfalls(
    module: Any, function_names: Sequence[str],
) -> dict[str, tuple[SPIRVShortfall, ...]]:
    """Precision-section obligations this backend cannot honour, per function.

    ``BACKEND_PRECISION_CAPABILITIES`` declares "spirv" delivers both
    FMA_MANDATORY (GLSL.std.450 Fma) and SECTION_ISOLATION (NoContraction on
    every section result), so with today's obligations this returns nothing
    -- the call exists to keep that table honest: if a section ever carries
    an obligation the table does not claim, it is refused here, on the same
    shortfall channel as an unregistered opcode, rather than emitted wrong.
    The receipt lives on the IRModule's metadata; a bare Function or plain
    mapping simply has none, and the check passes vacuously.
    """

    from .ir_identities import precision_backend_shortfalls

    grouped: dict[str, tuple[SPIRVShortfall, ...]] = {}
    for item in precision_backend_shortfalls(module, "spirv", function_names):
        function_name = str(item["function"])
        grouped[function_name] = grouped.get(function_name, ()) + (
            SPIRVShortfall(
                "precision_section", "*",
                "backend cannot honour precision obligations "
                + repr(item["missing"]),
            ),
        )
    return grouped


def _assemble(name: str, builder: _ModuleBuilder, functions: tuple[SPIRVFunction, ...]) -> SPIRVModule:
    lines: list[str] = []
    for capability in sorted(builder.capabilities):
        lines.append(f"OpCapability {capability}")
    if builder.needs_glsl_ext:
        lines.append('%glsl = OpExtInstImport "GLSL.std.450"')
    lines.append("OpMemoryModel Logical GLSL450")
    for exported_name in (builder.exported if builder.linkage else ()):
        lines.append(f'OpDecorate %{exported_name} LinkageAttributes "{exported_name}" Export')
    # Decorations parked by the function emitters (NoContraction on
    # precision-section results) belong in the same annotations section as
    # the linkage decorations above -- SPIR-V's module layout requires every
    # OpDecorate before the types and functions that follow.
    lines.extend(builder.decorations)
    lines.extend(builder.type_lines)
    for fn in functions:
        if not fn.body:
            continue
        lines.append("")
        lines.extend(fn.body)
    source = "\n".join(lines) + "\n"
    return SPIRVModule(name, source, functions)


def emit_function(
    function: Function,
    *,
    dtype: str = DEFAULT_DTYPE,
    outputs: Sequence[SSAValue] = (),
    module: IRModule | None = None,
) -> SPIRVModule:
    """Translate one SSA function into a standalone SPIR-V module.

    ``outputs`` names the SSA value that leaves the function. SSA itself
    records only arguments, so a result would otherwise never become the
    function's return value. ``module`` is optional and only consulted for
    the precision-pipeline receipt in its metadata -- a caller holding the
    IRModule the function came from should pass it so precision-section
    obligations are checked; without it the function's own metadata is the
    only carrier available.
    """

    from .machine_dialect_ssa import (
        format_machine_dialect_occurrences,
        module_machine_dialect_occurrences,
    )
    machine_residuals = module_machine_dialect_occurrences({
        function.name: function,
    })
    if machine_residuals:
        raise SPIRVEmissionError(
            "SPIR-V emission requires legalized repository SSA; retained "
            "machine-state SSA remains: "
            + format_machine_dialect_occurrences(machine_residuals)
        )
    precision = _precision_shortfalls(
        module if module is not None else function, (function.name,)
    )
    builder = _ModuleBuilder()
    fn = _emit_function(
        function,
        builder,
        dtype=dtype,
        outputs=outputs,
        carried_shortfalls=precision.get(function.name, ()),
    )
    return _assemble(function.name, builder, (fn,))


def emit_module(
    module: IRModule | Mapping[str, Function],
    *,
    name: str = "turing_ssa",
    dtype: str = DEFAULT_DTYPE,
    outputs: Mapping[str, Sequence[SSAValue]] | None = None,
) -> SPIRVModule:
    """Translate an SSA module into one SPIR-V module.

    ``outputs`` maps a function name to the single SSA value it returns.
    Every function shares one id/type/constant namespace, so two functions
    using the same dtype never redeclare the same SPIR-V type.
    """

    functions = module.functions if isinstance(module, IRModule) else dict(module)
    from .machine_dialect_ssa import (
        format_machine_dialect_occurrences,
        module_machine_dialect_occurrences,
    )
    machine_residuals = module_machine_dialect_occurrences(functions)
    if machine_residuals:
        raise SPIRVEmissionError(
            "SPIR-V emission requires legalized repository SSA; retained "
            "machine-state SSA remains: "
            + format_machine_dialect_occurrences(machine_residuals)
        )
    named_outputs = dict(outputs or {})
    # Precision-section obligations are checked against the module BEFORE
    # anything is emitted, mirroring the wasm/llvm lanes -- the receipt
    # travels on the IRModule's metadata (a plain mapping has none and
    # passes vacuously). See _precision_shortfalls for why this is expected
    # to find nothing today.
    precision = _precision_shortfalls(module, tuple(functions))
    builder = _ModuleBuilder()
    fns = tuple(
        _emit_function(
            function,
            builder,
            dtype=dtype,
            outputs=named_outputs.get(function_name, ()),
            carried_shortfalls=precision.get(str(function_name), ()),
        )
        for function_name, function in functions.items()
    )
    return _assemble(name, builder, fns)


#: Element spelling and byte stride of a storage-buffer element, by SSA dtype.
#: The spelling is the logical-port vocabulary the WGSL and GLSL components
#: already publish, so ports compare across backends.
_STORAGE_ELEMENT: dict[str, tuple[str, int]] = {
    "float32": ("f32", 4),
    "float": ("f32", 4),
    "float64": ("f64", 8),
    "double": ("f64", 8),
    "int32": ("i32", 4),
    "int64": ("i64", 8),
}


@dataclass(frozen=True)
class SPIRVStorageBinding:
    """One descriptor of a deployable shader, as the module declares it."""

    binding: int
    name: str
    role: str  # "feed" | "output" | "control"
    dtype: str
    element_bytes: int
    access: str  # "read" | "read_write"
    value_id: int | None = None
    #: The static shape of ONE instance of this value (``()`` for a
    #: scalar): the buffer holds ``element_count`` instances back to back,
    #: each ``prod(instance_shape)`` elements, row-major.
    instance_shape: tuple[int, ...] = ()
    #: Element mode: the rank of a runtime-sized span read or written one
    #: element per invocation (0 otherwise).
    runtime_rank: int = 0
    #: Element mode: a rank-0 feed every invocation reads at element 0.
    broadcast: bool = False


@dataclass
class SPIRVDeployableShader:
    """A GLCompute SPIR-V module with an entry point and a descriptor ABI.

    A separate product from :class:`SPIRVModule`: the library function is
    called once per invocation by ``main``, which reads instance
    ``GlobalInvocationId.x`` of each feed buffer, writes that instance of
    every output buffer, and guards on the control buffer's
    ``element_count`` word (the instance count). An instance of a value of
    static shape ``S`` is ``prod(S)`` consecutive elements, row-major; a
    scalar's is one element. Every resource is a storage buffer in
    descriptor set 0 at a contiguous binding (feeds in argument order, then
    the outputs in order, then the control buffer).
    """

    name: str
    source: str
    function: str
    local_size: tuple[int, int, int]
    bindings: tuple[SPIRVStorageBinding, ...]
    shortfalls: tuple[SPIRVShortfall, ...] = ()
    target_env: str = "vulkan1.1"
    spirv_version: str = "1.3"
    entrypoint: str = "main"
    capabilities: tuple[str, ...] = ()
    #: The logical ``turing.shader-component.v1`` record and the API sidecar
    #: that carries it; ``None`` when the function was refused.
    component_abi: Any = None
    api: Any = None
    #: Reduction mode: the whole-span reduction one invocation computes by
    #: looping over ``element_count`` elements (``None`` otherwise).
    reduction: str | None = None
    product_kind = DEPLOYABLE_SHADER

    @property
    def complete(self) -> bool:
        return not self.shortfalls and bool(self.source)

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.spvasm"
        path.write_text(self.source, encoding="utf-8", newline="\n")
        return path


def emit_deployable_compute_shader(
    function: Function,
    *,
    outputs: Sequence[SSAValue],
    dtype: str = DEFAULT_DTYPE,
    local_size: int = 64,
    name: str | None = None,
    module: IRModule | None = None,
) -> SPIRVDeployableShader:
    """Wrap one SSA function in a GLCompute entry point.

    The function body is the same translation the library product uses;
    only the interface differs. Each invocation is one INSTANCE of the
    function: invocation ``i`` reads instance ``i`` of every feed buffer,
    calls the function once, and writes instance ``i`` of every output
    buffer. A scalar value occupies one element per instance; a value of
    static shape ``S`` occupies ``prod(S)`` consecutive elements, row-major,
    so instance ``i`` starts at element ``i * prod(S)``. Several outputs are
    several output buffers, in order.

    An elementwise function over runtime-sized spans of one rank (rank
    stated, no static shape) runs in ELEMENT mode instead: invocation ``i``
    computes element ``i`` of every span, rank-0 feeds broadcast (each
    invocation reads their element 0), and ``element_count`` is the product
    of the host's extents. A lone whole-span reduction of a runtime-sized
    span runs in REDUCTION mode: one invocation loops over the host's
    element count (``_emit_runtime_reduction_shader``). What no mode can
    carry (mixed span ranks, a non-elementwise operation over a span, a
    boolean element, no result) is a shortfall, never a guessed layout.
    """

    name = str(name or function.name)
    local = (int(local_size), 1, 1)
    if local[0] <= 0:
        raise SPIRVEmissionError("local size must be positive")
    shortfalls: list[SPIRVShortfall] = []

    def refused() -> SPIRVDeployableShader:
        return SPIRVDeployableShader(
            name, "", function.name, local, (), tuple(shortfalls),
        )

    outputs = tuple(outputs)
    if not outputs:
        shortfalls.append(SPIRVShortfall(
            "<entry>", "*",
            f"a deployable compute entry writes at least one result per "
            f"invocation; {function.name} names none",
        ))
        return refused()
    # Reduction mode: one whole-span reduction of a runtime-sized span runs
    # in ONE invocation, which loops over the host's element count.
    reduction, reason = _runtime_reduction(function, outputs)
    if reason is not None:
        shortfalls.append(SPIRVShortfall(
            "<entry>", "*",
            f"{function.name} reduces a runtime-sized span but cannot run "
            f"as one looping invocation: {reason}",
        ))
        return refused()
    if reduction is not None:
        return _emit_runtime_reduction_shader(
            function, reduction.reduction, dtype=dtype, local=local, name=name,
            module=module, maps=reduction.maps, posts=reduction.posts,
        )
    # Element mode: an elementwise function over runtime-sized spans of one
    # rank runs one ELEMENT per invocation instead of one instance; its
    # rank-0 feeds broadcast (every invocation reads element 0).
    element_rank = None
    if any(_runtime_sized(value) for value in (*function.args, *outputs)):
        element_rank, reason = _elementwise_runtime_rank(function, outputs)
        if element_rank is None:
            shortfalls.append(SPIRVShortfall(
                "<entry>", "*",
                f"{function.name} reads or writes runtime-sized spans but "
                f"cannot run one element per invocation: {reason}",
            ))
            return refused()
    for value in (*function.args, *outputs):
        if element_rank is not None:
            pass
        elif not all(
            isinstance(size, int) and size > 0
            for size in tuple(value.shape or ())
        ):
            shortfalls.append(SPIRVShortfall(
                "<entry>", "*",
                f"%t{value.id} has shape {tuple(value.shape)}, which is not "
                "a static positive extent per axis",
            ))
        if _STORAGE_ELEMENT.get(str(value.dtype or dtype)) is None:
            shortfalls.append(SPIRVShortfall(
                "<entry>", "*",
                f"%t{value.id} has dtype {value.dtype or dtype!r}, which has "
                "no storage-buffer element spelling",
            ))
    if shortfalls:
        return refused()

    from .machine_dialect_ssa import (
        format_machine_dialect_occurrences,
        module_machine_dialect_occurrences,
    )
    machine_residuals = module_machine_dialect_occurrences({
        function.name: function,
    })
    if machine_residuals:
        raise SPIRVEmissionError(
            "SPIR-V emission requires legalized repository SSA; retained "
            "machine-state SSA remains: "
            + format_machine_dialect_occurrences(machine_residuals)
        )
    precision = _precision_shortfalls(
        module if module is not None else function, (function.name,)
    )
    builder = _ModuleBuilder(linkage=False)
    emitter = _FunctionEmitter(
        function, builder, dtype=dtype, outputs=outputs,
        carried_shortfalls=precision.get(function.name, ()),
        runtime_elements=element_rank is not None,
    )
    library = emitter.emit()
    if library.shortfalls:
        shortfalls.extend(library.shortfalls)
        return refused()

    uint = builder.uint_type()
    boolean = builder.scalar_type("bool")
    void = builder.void_type()
    zero = builder.uint_constant(0)
    lines: list[str] = []
    decorations: list[str] = []

    def declare(line: str) -> None:
        builder.type_lines.append(line)

    v3uint = "%v3uint"
    declare(f"{v3uint} = OpTypeVector {uint} 3")
    gid_pointer = builder.pointer_type("Input", v3uint)
    declare(f"%gid = OpVariable {gid_pointer} Input")
    decorations.append("OpDecorate %gid BuiltIn GlobalInvocationId")

    block_types: dict[str, tuple[str, str]] = {}

    def block_for(element_type: str, stride: int) -> tuple[str, str]:
        """The Block struct over a runtime array of ``element_type``."""

        if element_type not in block_types:
            runtime = builder.fresh("rtarr")
            block = builder.fresh("blk")
            declare(f"{runtime} = OpTypeRuntimeArray {element_type}")
            declare(f"{block} = OpTypeStruct {runtime}")
            decorations.append(f"OpDecorate {runtime} ArrayStride {stride}")
            decorations.append(f"OpDecorate {block} Block")
            decorations.append(f"OpMemberDecorate {block} 0 Offset 0")
            block_types[element_type] = (
                block, builder.pointer_type("StorageBuffer", element_type),
            )
        return block_types[element_type]

    bindings: list[SPIRVStorageBinding] = []
    # (variable, element type, storage element pointer, value)
    variables: list[tuple[str, str, str, SSAValue]] = []
    roles = [("feed", value) for value in function.args] + [
        ("output", value) for value in outputs
    ]
    output_position = 0
    for binding, (role, value) in enumerate(roles):
        value_dtype = str(value.dtype or dtype)
        spelling, stride = _STORAGE_ELEMENT[value_dtype]
        element_type = builder.scalar_type(value_dtype)
        block, element_pointer = block_for(element_type, stride)
        variable = f"%{role}_{binding}"
        declare(
            f"{variable} = OpVariable "
            f"{builder.pointer_type('StorageBuffer', block)} StorageBuffer"
        )
        decorations.append(f"OpDecorate {variable} DescriptorSet 0")
        decorations.append(f"OpDecorate {variable} Binding {binding}")
        if role == "feed":
            decorations.append(f"OpDecorate {variable} NonWritable")
        variables.append((variable, element_type, element_pointer, value))
        shape = tuple(int(size) for size in tuple(value.shape or ()))
        bindings.append(SPIRVStorageBinding(
            binding,
            f"feed_{value.id}" if role == "feed" else f"output_{output_position}",
            role, spelling, stride,
            "read" if role == "feed" else "read_write",
            int(value.id),
            instance_shape=shape,
            runtime_rank=(
                _declared_rank(value)
                if element_rank is not None and _runtime_sized(value) else 0
            ),
            broadcast=element_rank is not None and not _runtime_sized(value),
        ))
        if role == "output":
            output_position += 1

    control_binding = len(roles)
    control_block = builder.fresh("ctl")
    declare(f"{control_block} = OpTypeStruct {uint}")
    decorations.append(f"OpDecorate {control_block} Block")
    decorations.append(f"OpMemberDecorate {control_block} 0 Offset 0")
    declare(
        f"%control = OpVariable "
        f"{builder.pointer_type('StorageBuffer', control_block)} StorageBuffer"
    )
    decorations.append("OpDecorate %control DescriptorSet 0")
    decorations.append(f"OpDecorate %control Binding {control_binding}")
    decorations.append("OpDecorate %control NonWritable")
    bindings.append(SPIRVStorageBinding(
        control_binding, "control", "control", "u32", 4, "read",
    ))
    uint_pointer = builder.pointer_type("StorageBuffer", uint)
    main_type = builder.function_type(void, ())

    def flat_indices(shape: tuple[int, ...]):
        return itertools.product(*(range(size) for size in shape))

    feeds = variables[:len(function.args)]
    # Function-storage variables must open the entry block.
    staged = {}
    body = [
        f"%main = OpFunction {void} None {main_type}",
        "%main_entry = OpLabel",
    ]
    for position, (_variable, element_type, _pointer, value) in enumerate(feeds):
        if value.shape:
            local_array = emitter._value_type(value)
            body.append(f"%feed_arr_{position} = OpVariable {local_array} Function")
            staged[position] = f"%feed_arr_{position}"
    body.extend([
        f"%gid_v = OpLoad {v3uint} %gid",
        f"%index = OpCompositeExtract {uint} %gid_v 0",
        f"%count_ptr = OpAccessChain {uint_pointer} %control {zero}",
        f"%count = OpLoad {uint} %count_ptr",
        f"%inside = OpULessThan {boolean} %index %count",
        "OpSelectionMerge %main_merge None",
        "OpBranchConditional %inside %main_body %main_merge",
        "%main_body = OpLabel",
    ])

    def instance_base(tag: str, size: int) -> str:
        if size == 1:
            return "%index"
        body.append(
            f"%base_{tag} = OpIMul {uint} %index {builder.uint_constant(size)}"
        )
        return f"%base_{tag}"

    def element_at(tag: str, base: str, flat: int) -> str:
        if flat == 0:
            return base
        body.append(
            f"%at_{tag}_{flat} = OpIAdd {uint} {base} {builder.uint_constant(flat)}"
        )
        return f"%at_{tag}_{flat}"

    loaded: list[str] = []
    for position, (variable, element_type, element_pointer, value) in enumerate(feeds):
        if not value.shape:
            broadcast = element_rank is not None and not _runtime_sized(value)
            body.append(
                f"%feed_ptr_{position} = OpAccessChain {element_pointer} "
                f"{variable} {zero} {zero if broadcast else '%index'}"
            )
            body.append(f"%feed_val_{position} = OpLoad {element_type} %feed_ptr_{position}")
            loaded.append(f"%feed_val_{position}")
            continue
        shape = tuple(int(size) for size in value.shape)
        tag = f"f{position}"
        base = instance_base(tag, math.prod(shape))
        local_pointer = builder.pointer_type("Function", element_type)
        for flat, nested in enumerate(flat_indices(shape)):
            at = element_at(tag, base, flat)
            body.append(
                f"%{tag}_src_{flat} = OpAccessChain {element_pointer} "
                f"{variable} {zero} {at}"
            )
            body.append(f"%{tag}_val_{flat} = OpLoad {element_type} %{tag}_src_{flat}")
            body.append(
                f"%{tag}_dst_{flat} = OpAccessChain {local_pointer} "
                f"{staged[position]} "
                + " ".join(builder.uint_constant(item) for item in nested)
            )
            body.append(f"OpStore %{tag}_dst_{flat} %{tag}_val_{flat}")
        loaded.append(staged[position])
    body.append(
        f"%result = OpFunctionCall {emitter.return_type} %{function.name}"
        + "".join(f" {item}" for item in loaded)
    )
    for member, (variable, element_type, element_pointer, value) in enumerate(
        variables[len(function.args):]
    ):
        tag = f"o{member}"
        if len(outputs) > 1:
            body.append(
                f"%{tag}_value = OpCompositeExtract "
                f"{emitter._plain_type(value)} %result {member}"
            )
            produced = f"%{tag}_value"
        else:
            produced = "%result"
        if not value.shape:
            body.append(
                f"%{tag}_ptr = OpAccessChain {element_pointer} {variable} "
                f"{zero} %index"
            )
            body.append(f"OpStore %{tag}_ptr {produced}")
            continue
        shape = tuple(int(size) for size in value.shape)
        base = instance_base(tag, math.prod(shape))
        for flat, nested in enumerate(flat_indices(shape)):
            at = element_at(tag, base, flat)
            body.append(
                f"%{tag}_val_{flat} = OpCompositeExtract {element_type} "
                f"{produced} " + " ".join(str(item) for item in nested)
            )
            body.append(
                f"%{tag}_dst_{flat} = OpAccessChain {element_pointer} "
                f"{variable} {zero} {at}"
            )
            body.append(f"OpStore %{tag}_dst_{flat} %{tag}_val_{flat}")
    body.extend([
        "OpBranch %main_merge",
        "%main_merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ])

    for capability in sorted(builder.capabilities):
        lines.append(f"OpCapability {capability}")
    if builder.needs_glsl_ext:
        lines.append('%glsl = OpExtInstImport "GLSL.std.450"')
    lines.append("OpMemoryModel Logical GLSL450")
    # SPIR-V 1.3 lists only Input/Output variables on the entry point.
    lines.append('OpEntryPoint GLCompute %main "main" %gid')
    lines.append(f"OpExecutionMode %main LocalSize {local[0]} {local[1]} {local[2]}")
    lines.extend(decorations)
    lines.extend(builder.decorations)
    lines.extend(builder.type_lines)
    lines.append("")
    lines.extend(body)
    lines.append("")
    lines.extend(library.body)
    from .compiled_program_api import CompiledProgramAPI, EntryPoint
    from .shader_component_abi import component_abi_from_layout
    from .shader_stages import COMPUTE, BufferBinding, ShaderIOLayout

    def port(item: SPIRVStorageBinding) -> BufferBinding:
        return BufferBinding(item.name, item.role, item.dtype, item.binding, item.value_id)

    layout = ShaderIOLayout(
        COMPUTE.name,
        feeds=tuple(port(item) for item in bindings if item.role == "feed"),
        outputs=tuple(port(item) for item in bindings if item.role == "output"),
    )
    component_abi = component_abi_from_layout(
        name, "spirv", layout,
        decorations={
            "execution_model": "GLCompute",
            "workgroup_size": local,
            "sentinel_policy": "validate-before-dispatch/publish-after-dispatch",
        },
    )
    api = CompiledProgramAPI(
        module=name,
        language="spirv",
        entry="main",
        entry_points=(EntryPoint("main", "main", "numerical"),),
        metadata={
            "stage": COMPUTE.name,
            "io_layout": layout.to_mapping(),
            "component_abi": component_abi.to_mapping(),
        },
    )
    return SPIRVDeployableShader(
        name, "\n".join(lines) + "\n", function.name, local, tuple(bindings),
        capabilities=tuple(sorted(builder.capabilities)),
        component_abi=component_abi, api=api,
    )


def _emit_runtime_reduction_shader(
    function: Function,
    instr: Instr,
    *,
    dtype: str,
    local: tuple[int, int, int],
    name: str,
    module: IRModule | None,
    maps: Sequence[Instr] = (),
    posts: Sequence[Instr] = (),
) -> SPIRVDeployableShader:
    """Reduction mode: one invocation folds a runtime-sized span.

    Program: ``x.sum()`` with ``x`` declared ``storage: span, rank: 2`` and
    no shape. The unrolled library has no count to unroll, and element mode
    writes one element per invocation, so the region was refused. Rule:
    invocation 0 loops ``i`` over ``[0, element_count)`` in a structured
    loop, folding element ``i`` left to right exactly as the unrolled form
    does (NaN enters ``max``/``min`` and stays; NaN is true for
    ``all``/``any``), and writes element 0 of ``output_0``. The launch is one
    group. ``max``, ``min`` and ``mean`` of no elements have no value: the
    shader then writes nothing, and the record says ``minimum`` 1.

    Map-reduce: the ``maps`` (see ``_MapReducePlan``) run per element inside
    the loop, so ``(x * y).sum()`` reads ``x[i]`` and ``y[i]`` and folds
    their product without storing the product span; scalar feeds and
    constants broadcast. The ``posts`` then finish the total into the one
    output, in order.
    """

    op = str(instr.attributes.get("tensor_operation") or instr.op)
    source, result = instr.args[0], instr.res
    last = (list(posts)[-1] if posts else instr).res
    shortfalls = list(_precision_shortfalls(
        module if module is not None else function, (function.name,)
    ).get(function.name, ()))
    if shortfalls:
        return SPIRVDeployableShader(
            name, "", function.name, local, (), tuple(shortfalls),
        )
    builder = _ModuleBuilder(linkage=False)
    folding = _FunctionEmitter(
        function, builder, dtype=dtype, outputs=(last,), runtime_elements=True,
    )
    uint = builder.uint_type()
    boolean = builder.scalar_type("bool")
    void = builder.void_type()
    zero, one = builder.uint_constant(0), builder.uint_constant(1)
    source_dtype = str(source.dtype or dtype)
    result_dtype = str(result.dtype or dtype)
    source_kind = _DTYPE_KIND.get(source_dtype)
    result_kind = _DTYPE_KIND.get(result_dtype)
    source_type = builder.scalar_type(source_dtype)
    result_type = builder.scalar_type(result_dtype)
    decorations: list[str] = []

    def declare(line: str) -> None:
        builder.type_lines.append(line)

    declare(f"%v3uint = OpTypeVector {uint} 3")
    declare(f"%gid = OpVariable {builder.pointer_type('Input', '%v3uint')} Input")
    decorations.append("OpDecorate %gid BuiltIn GlobalInvocationId")
    block_types: dict[str, tuple[str, str]] = {}

    def block_for(element_type: str, stride: int) -> tuple[str, str]:
        if element_type not in block_types:
            runtime, block = builder.fresh("rtarr"), builder.fresh("blk")
            declare(f"{runtime} = OpTypeRuntimeArray {element_type}")
            declare(f"{block} = OpTypeStruct {runtime}")
            decorations.append(f"OpDecorate {runtime} ArrayStride {stride}")
            decorations.append(f"OpDecorate {block} Block")
            decorations.append(f"OpMemberDecorate {block} 0 Offset 0")
            block_types[element_type] = (
                block, builder.pointer_type("StorageBuffer", element_type),
            )
        return block_types[element_type]

    bindings: list[SPIRVStorageBinding] = []
    variables: dict[int, tuple[str, str, str]] = {}
    for binding, (role, value) in enumerate(
        [("feed", value) for value in function.args] + [("output", last)]
    ):
        value_dtype = str(value.dtype or dtype)
        spelling, stride = _STORAGE_ELEMENT[value_dtype]
        element_type = builder.scalar_type(value_dtype)
        block, element_pointer = block_for(element_type, stride)
        variable = f"%{role}_{binding}"
        declare(
            f"{variable} = OpVariable "
            f"{builder.pointer_type('StorageBuffer', block)} StorageBuffer"
        )
        decorations.append(f"OpDecorate {variable} DescriptorSet 0")
        decorations.append(f"OpDecorate {variable} Binding {binding}")
        if role == "feed":
            decorations.append(f"OpDecorate {variable} NonWritable")
            variables[int(value.id)] = (variable, element_type, element_pointer)
        else:
            variables[-1] = (variable, element_type, element_pointer)
        spans = role == "feed" and _runtime_sized(value)
        bindings.append(SPIRVStorageBinding(
            binding, f"feed_{value.id}" if role == "feed" else "output_0",
            role, spelling, stride,
            "read" if role == "feed" else "read_write", int(value.id),
            runtime_rank=_declared_rank(value) if spans else 0,
            broadcast=not spans,
        ))
    control_binding = len(bindings)
    control_block = builder.fresh("ctl")
    declare(f"{control_block} = OpTypeStruct {uint}")
    decorations.append(f"OpDecorate {control_block} Block")
    decorations.append(f"OpMemberDecorate {control_block} 0 Offset 0")
    declare(
        f"%control = OpVariable "
        f"{builder.pointer_type('StorageBuffer', control_block)} StorageBuffer"
    )
    decorations.append("OpDecorate %control DescriptorSet 0")
    decorations.append(f"OpDecorate %control Binding {control_binding}")
    decorations.append("OpDecorate %control NonWritable")
    bindings.append(SPIRVStorageBinding(
        control_binding, "control", "control", "u32", 4, "read",
    ))

    logical = op in ("all", "any")
    accumulator_type = boolean if logical else result_type
    from .max_semantics import NUMPY, is_max, max_step_rule

    def operation(item: Instr) -> str:
        return str(item.attributes.get("tensor_operation") or item.op)

    def compute(item: Instr, values: dict[int, str], tag: str) -> str:
        """``item`` over the scalar ids in ``values``, as a fresh id."""

        item_dtype = str(item.res.dtype or dtype)
        item_type = builder.scalar_type(item_dtype)
        if operation(item) in ("Const", "const"):
            literal = item.attributes.get("constant")
            if literal is None:
                literal = item.attributes.get("value")
            return builder.constant(item_dtype, literal)
        operand_dtype = str(item.args[0].dtype or item_dtype)
        ids = [values[int(value.id)] for value in item.args]
        target = f"%{tag}_{item.res.id}"
        greatest = is_max(operation(item))
        if greatest is not None and len(ids) == 2:
            folding._max_step(
                max_step_rule(item) or NUMPY, greatest, ids[0], ids[1],
                item_type, _DTYPE_KIND.get(item_dtype) == "float", body,
                result_id=target,
            )
            return target
        ids = folding._augment_scalar_attrs(item, ids, operand_dtype)
        resolved = folding._resolve(operation(item), operand_dtype, len(ids))
        if resolved is None:
            raise SPIRVEmissionError(
                f"no SPIR-V spelling for {operation(item)} in a map-reduce"
            )
        folding._apply(*resolved, item_type, target, ids, item, body)
        return target

    def load(value: SSAValue, at: str, tag: str) -> str:
        variable, value_type, pointer = variables[int(value.id)]
        body.append(f"%{tag}_ptr = OpAccessChain {pointer} {variable} {zero} {at}")
        body.append(f"%{tag} = OpLoad {value_type} %{tag}_ptr")
        return f"%{tag}"

    broadcast: dict[int, str] = {}
    body = [
        f"%main = OpFunction {void} None {builder.function_type(void, ())}",
        "%main_entry = OpLabel",
        f"%i_var = OpVariable {builder.pointer_type('Function', uint)} Function",
        f"%acc_var = OpVariable {builder.pointer_type('Function', accumulator_type)} Function",
        "%gid_v = OpLoad %v3uint %gid",
        f"%index = OpCompositeExtract {uint} %gid_v 0",
        f"%count_ptr = OpAccessChain {builder.pointer_type('StorageBuffer', uint)} %control {zero}",
        f"%count = OpLoad {uint} %count_ptr",
        f"%first = OpIEqual {boolean} %index {zero}",
    ]
    nonempty = op in ("max", "min", "mean")
    if nonempty:
        body.extend([
            f"%some = OpUGreaterThan {boolean} %count {zero}",
            f"%run = OpLogicalAnd {boolean} %first %some",
        ])
    run = "%run" if nonempty else "%first"
    body.extend([
        "OpSelectionMerge %main_merge None",
        f"OpBranchConditional {run} %main_body %main_merge",
        "%main_body = OpLabel",
    ])

    for value in function.args:
        if not _runtime_sized(value):
            broadcast[int(value.id)] = load(value, zero, f"s{value.id}")

    def element(at: str, tag: str) -> str:
        values = dict(broadcast)
        for value in function.args:
            if _runtime_sized(value):
                values[int(value.id)] = load(value, at, f"{tag}_f{value.id}")
        for item in maps:
            values[int(item.res.id)] = compute(item, values, tag)
        body.append(f"%{tag} = OpCopyObject {source_type} {values[int(source.id)]}")
        if logical:
            if source_kind == "float":
                test = builder.constant(source_dtype, 0.0)
                body.append(f"%{tag}_t = OpFUnordNotEqual {boolean} %{tag} {test}")
            elif source_kind == "int":
                test = builder.constant(source_dtype, 0)
                body.append(f"%{tag}_t = OpINotEqual {boolean} %{tag} {test}")
            else:
                return f"%{tag}"
            return f"%{tag}_t"
        converted = folding._convert(f"%{tag}", source_dtype, result_dtype, body)
        if converted is None:
            raise SPIRVEmissionError(
                f"no conversion from {source_dtype} to {result_dtype}"
            )
        return converted

    if logical:
        initial = builder.constant("bool", op == "all")
    elif op in ("max", "min"):
        initial = element(zero, "e0")
    elif op == "sum" and len(instr.args) == 2:
        start = instr.args[1]
        initial = folding._convert(
            broadcast[int(start.id)], str(start.dtype or dtype), result_dtype, body,
        )
        if initial is None:
            raise SPIRVEmissionError(
                f"no conversion from {start.dtype} to {result_dtype}"
            )
    else:
        initial = builder.constant(result_dtype, 1 if op == "prod" else 0)
    body.extend([
        f"OpStore %acc_var {initial}",
        f"OpStore %i_var {zero}",
        "OpBranch %loop_head",
        "%loop_head = OpLabel",
        "OpLoopMerge %loop_merge %loop_continue None",
        "OpBranch %loop_test",
        "%loop_test = OpLabel",
        f"%iv = OpLoad {uint} %i_var",
        f"%more = OpULessThan {boolean} %iv %count",
        "OpBranchConditional %more %loop_body %loop_merge",
        "%loop_body = OpLabel",
        f"%acc = OpLoad {accumulator_type} %acc_var",
    ])
    value = element("%iv", "ev")
    if logical:
        folded = "OpLogicalAnd" if op == "all" else "OpLogicalOr"
        body.append(f"%next_acc = {folded} {boolean} %acc {value}")
    elif op in ("max", "min"):
        folding._max_step(
            max_step_rule(instr) or NUMPY, op == "max", "%acc", value,
            result_type, result_kind == "float", body, result_id="%next_acc",
        )
    else:
        kind, payload = folding._resolve(
            "mul" if op == "prod" else "add", result_dtype, 2,
        )
        folding._apply(kind, payload, result_type, "%next_acc", ["%acc", value], instr, body)
    output_variable, _output_type, output_pointer = variables[-1]
    body.extend([
        "OpStore %acc_var %next_acc",
        "OpBranch %loop_continue",
        "%loop_continue = OpLabel",
        f"%iv_next = OpIAdd {uint} %iv {one}",
        "OpStore %i_var %iv_next",
        "OpBranch %loop_head",
        "%loop_merge = OpLabel",
        f"%total = OpLoad {accumulator_type} %acc_var",
    ])
    final = "%total"
    if logical and result_kind != "bool":
        body.append(
            f"%final = OpSelect {result_type} %total "
            f"{builder.constant(result_dtype, 1)} {builder.constant(result_dtype, 0)}"
        )
        final = "%final"
    elif op == "mean":
        body.append(f"%count_f = OpConvertUToF {result_type} %count")
        kind, payload = folding._resolve("Div", result_dtype, 2)
        folding._apply(kind, payload, result_type, "%final", ["%total", "%count_f"], instr, body)
        final = "%final"
    finished = dict(broadcast)
    finished[int(result.id)] = final
    for item in posts:
        finished[int(item.res.id)] = compute(item, finished, "post")
    final = finished[int(last.id)]
    body.extend([
        f"%out_ptr = OpAccessChain {output_pointer} {output_variable} {zero} {zero}",
        f"OpStore %out_ptr {final}",
        "OpBranch %main_merge",
        "%main_merge = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ])

    lines = [f"OpCapability {item}" for item in sorted(builder.capabilities)]
    if builder.needs_glsl_ext:
        lines.append('%glsl = OpExtInstImport "GLSL.std.450"')
    lines.append("OpMemoryModel Logical GLSL450")
    lines.append('OpEntryPoint GLCompute %main "main" %gid')
    lines.append(f"OpExecutionMode %main LocalSize {local[0]} {local[1]} {local[2]}")
    lines.extend(decorations)
    lines.extend(builder.decorations)
    lines.extend(builder.type_lines)
    lines.append("")
    lines.extend(body)
    from .compiled_program_api import CompiledProgramAPI, EntryPoint
    from .shader_component_abi import component_abi_from_layout
    from .shader_stages import COMPUTE, BufferBinding, ShaderIOLayout

    def port(item: SPIRVStorageBinding) -> BufferBinding:
        return BufferBinding(item.name, item.role, item.dtype, item.binding, item.value_id)

    layout = ShaderIOLayout(
        COMPUTE.name,
        feeds=tuple(port(item) for item in bindings if item.role == "feed"),
        outputs=tuple(port(item) for item in bindings if item.role == "output"),
    )
    component_abi = component_abi_from_layout(
        name, "spirv", layout,
        decorations={
            "execution_model": "GLCompute",
            "workgroup_size": local,
            "sentinel_policy": "validate-before-dispatch/publish-after-dispatch",
        },
    )
    api = CompiledProgramAPI(
        module=name, language="spirv", entry="main",
        entry_points=(EntryPoint("main", "main", "numerical"),),
        metadata={
            "stage": COMPUTE.name,
            "io_layout": layout.to_mapping(),
            "component_abi": component_abi.to_mapping(),
        },
    )
    return SPIRVDeployableShader(
        name, "\n".join(lines) + "\n", function.name, local, tuple(bindings),
        capabilities=tuple(sorted(builder.capabilities)),
        component_abi=component_abi, api=api, reduction=op,
    )


def spirv_assembler() -> str | None:
    """Return an available ``spirv-as`` binary, or ``None``.

    Emission never requires one. ``TURING_SPIRV_AS`` overrides the search.
    """

    import os

    override = os.environ.get("TURING_SPIRV_AS")
    if override and Path(override).exists():
        return override
    return shutil.which("spirv-as")


def compile_module(module: SPIRVModule, *, directory: str | Path | None = None) -> Path:
    """Assemble generated SPIR-V text into a ``.spv`` binary via ``spirv-as``.

    Raises ``SPIRVEmissionError`` when no assembler is present; callers that
    only need source should not call this.
    """

    tool = spirv_assembler()
    if tool is None:
        raise SPIRVEmissionError(
            "no spirv-as found; emission does not require one, but "
            "compile_module does"
        )
    workdir = Path(directory or tempfile.mkdtemp(prefix="turing_spirv_"))
    workdir.mkdir(parents=True, exist_ok=True)
    source = module.write(workdir)
    binary = workdir / f"{module.name}.spv"
    completed = subprocess.run(
        [tool, str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        cwd=str(workdir),
    )
    if completed.returncode != 0:
        raise SPIRVEmissionError(f"spirv-as failed:\n{completed.stderr}")
    return binary


__all__ = [
    "DEFAULT_DTYPE",
    "DEPLOYABLE_SHADER",
    "LINKABLE_SPIRV_LIBRARY",
    "SPIRVDeployableShader",
    "SPIRVEmissionError",
    "SPIRVStorageBinding",
    "SPIRVFunction",
    "SPIRVModule",
    "SPIRVShortfall",
    "compile_module",
    "emit_deployable_compute_shader",
    "emit_function",
    "emit_module",
    "spirv_assembler",
]
