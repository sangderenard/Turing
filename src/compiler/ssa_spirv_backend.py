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
# accumulation across elements, not a single instruction), so they would
# otherwise fall through to the generic "no opcode registered" shortfall.
# Naming them lets that case report what it actually is.
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


@dataclass
class SPIRVModule:
    """A complete, assembled SPIR-V text module (consumable by ``spirv-as``)."""

    name: str
    source: str
    functions: tuple[SPIRVFunction, ...] = ()

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

    def __init__(self) -> None:
        self._next_id = 0
        self.capabilities: set[str] = {"Shader", "Linkage"}
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

    def pointer_type(self, storage_class: str, pointee: str) -> str:
        key = f"ptr({storage_class};{pointee})"
        if key not in self.type_ids:
            id_ = self.fresh("ptrty")
            self.type_lines.append(f"{id_} = OpTypePointer {storage_class} {pointee}")
            self.type_ids[key] = id_
        return self.type_ids[key]


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
    ):
        self.function = function
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

        if len(self.outputs) > 1:
            return self._bail(
                "<function>", block.name,
                "multiple outputs are not yet supported by the SPIR-V "
                "backend (region-call aggregate unbundling is future work)",
            )

        return_type = (
            self._plain_type(self.outputs[0])
            if self.outputs
            else self.builder.void_type()
        )
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

    def _emit_instr(self, instr: Instr, block: BasicBlock, body: list[str]) -> None:
        if instr.op in ("Br", "br", "CondBr", "condbr", "Phi", "phi"):
            self._shortfall(
                instr.op, block.name,
                "multi-block control flow is not yet supported by the "
                "SPIR-V backend",
            )
            return
        if instr.op in ("Ret", "ret", "Return", "return"):
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
            self._shortfall(
                op, block.name,
                "reductions over arrays are not yet supported by the "
                "SPIR-V backend",
            )
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
    for exported_name in builder.exported:
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
    "SPIRVEmissionError",
    "SPIRVFunction",
    "SPIRVModule",
    "SPIRVShortfall",
    "compile_module",
    "emit_function",
    "emit_module",
    "spirv_assembler",
]
