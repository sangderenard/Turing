"""Emit SPIR-V assembly from Turing SSA.

This is the first cut of the ``SSA -> SPIR-V`` route named in
``docs/C_NODUS_INTEROP_AND_FUSION.md`` ("typed SSA ... -> SPIR-V-compatible
operations -> SPIR-V module") and tracked as "not started" in
``docs/BACKEND_PERFORMANCE_HANDOFF.md``. It follows the same incremental
scoping this repo already used for ``ssa_fortran_backend.py``: land straight-
line scalar elementwise chains first, and report everything else (array
operands, structural ops, multi-block control flow, region calls) as an
honest shortfall rather than a guess. Widening scope is follow-up work, not a
redesign -- the block/shortfall structure below already accommodates it.

SPIR-V differs from every other backend in this repo in one structural way
worth stating up front: ids and type/constant declarations are **module**
scoped, not function scoped. Two functions in the same ``IRModule`` must not
redeclare ``OpTypeFloat 64`` or fight over id numbers. ``_ModuleBuilder``
exists for exactly that reason -- one shared id/type/constant namespace,
threaded through every ``_FunctionEmitter`` in a module.

The array/buffer binding ABI (SSBOs, descriptor sets, workgroup dispatch) is
deliberately out of scope here: it is its own milestone in the interop doc
("Specify a shared tensor descriptor ..."), not a prerequisite for proving
the op-level translation. Each SSA ``Function`` is emitted as an ordinary
SPIR-V library function (``OpFunction`` / scalar ``OpFunctionParameter``s /
``OpReturnValue``), decorated ``LinkageAttributes ... Export`` so the module
is valid with no ``OpEntryPoint`` at all (the ``Linkage`` capability exists
for precisely this). Wiring a function into a real GLCompute/Vulkan entry
point with bound buffers is future work layered on top of this, not a
reason to block on it.
"""

from __future__ import annotations

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


class _FunctionEmitter:
    """Translate one straight-line scalar SSA ``Function`` into SPIR-V."""

    def __init__(
        self,
        function: Function,
        builder: _ModuleBuilder,
        *,
        dtype: str = DEFAULT_DTYPE,
        outputs: Sequence[SSAValue] = (),
    ):
        self.function = function
        self.builder = builder
        self.dtype = dtype
        self.outputs = tuple(outputs)
        self.shortfalls: list[SPIRVShortfall] = []

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

        for argument in function.args:
            if argument.shape:
                return self._bail(
                    "<function>", block.name,
                    "array-shaped arguments are not yet supported by the "
                    "SPIR-V backend (the buffer/binding ABI is separate, "
                    "later work)",
                )
        for value in self.outputs:
            if value.shape:
                return self._bail(
                    "<function>", block.name,
                    "array-shaped outputs are not yet supported by the "
                    "SPIR-V backend (the buffer/binding ABI is separate, "
                    "later work)",
                )
        if len(self.outputs) > 1:
            return self._bail(
                "<function>", block.name,
                "multiple outputs are not yet supported by the SPIR-V "
                "backend (region-call aggregate unbundling is future work)",
            )

        return_type = (
            self.builder.scalar_type(self.outputs[0].dtype or self.dtype)
            if self.outputs
            else self.builder.void_type()
        )
        param_types = [
            self.builder.scalar_type(a.dtype or self.dtype) for a in function.args
        ]
        fn_type = self.builder.function_type(return_type, param_types)

        body: list[str] = []
        fn_id = f"%{function.name}"
        body.append(f"{fn_id} = OpFunction {return_type} None {fn_type}")
        for argument in function.args:
            type_id = self.builder.scalar_type(argument.dtype or self.dtype)
            body.append(f"{self._name(argument)} = OpFunctionParameter {type_id}")
        body.append(f"%{function.name}_entry = OpLabel")

        for instr in block.instrs:
            self._emit_instr(instr, block, body)

        body.append("OpFunctionEnd")
        self.builder.exported.append(function.name)
        return SPIRVFunction(function.name, tuple(body), tuple(self.shortfalls))

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
                body.append(f"    OpReturnValue {self._name(self.outputs[0])}")
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

        result_id = self._name(instr.res)
        result_dtype = instr.res.dtype or self.dtype
        result_type = self.builder.scalar_type(result_dtype)

        if op in ("Const", "const"):
            value = instr.attributes.get("constant")
            if value is None:
                value = instr.attributes.get("value")
            if value is None:
                self._shortfall(
                    op, block.name,
                    "array constants are not yet supported by the SPIR-V "
                    "backend",
                )
                return
            const_id = self.builder.constant(result_dtype, value)
            body.append(f"    {result_id} = OpCopyObject {result_type} {const_id}")
            return

        args = [self._name(a) for a in instr.args]

        # Scalar operands recorded as attributes rather than SSA values.
        right = instr.attributes.get("right_scalar")
        left = instr.attributes.get("left_scalar")
        operand_dtype = instr.args[0].dtype if instr.args else result_dtype
        if right is not None and len(args) == 1:
            args = [args[0], self.builder.constant(operand_dtype, right)]
        elif left is not None and len(args) == 1:
            args = [self.builder.constant(operand_dtype, left), args[0]]

        bucket = _DTYPE_KIND.get(operand_dtype or self.dtype)
        binary_table = _BINARY_INT if bucket == "int" else _BINARY_FLOAT

        if op in binary_table and len(args) == 2:
            if instr.attributes.get("reverse"):
                args = [args[1], args[0]]
            body.append(f"    {result_id} = {binary_table[op]} {result_type} {args[0]} {args[1]}")
            return

        if op in _BOOL_BINARY and len(args) == 2:
            body.append(f"    {result_id} = {_BOOL_BINARY[op]} {result_type} {args[0]} {args[1]}")
            return

        if op in _BINARY_EXTINST and len(args) == 2:
            extinst = _BINARY_EXTINST[op]
            if extinst in _EXTINST_FLOAT16_32_ONLY and _DTYPE_WIDTH.get(result_dtype) == 64:
                self._shortfall(
                    op, block.name,
                    f"GLSL.std.450 {extinst} is restricted to 16/32-bit "
                    "float operands; this value is float64",
                )
                return
            ext = self.builder.glsl_ext()
            body.append(
                f"    {result_id} = OpExtInst {result_type} {ext} "
                f"{extinst} {args[0]} {args[1]}"
            )
            return

        if op in _CAST_NATIVE and len(args) == 1:
            body.append(f"    {result_id} = {_CAST_NATIVE[op]} {result_type} {args[0]}")
            return

        if op in _UNARY_NATIVE and len(args) == 1:
            body.append(f"    {result_id} = {_UNARY_NATIVE[op]} {result_type} {args[0]}")
            return

        if op in _UNARY_EXTINST and len(args) == 1:
            extinst = _UNARY_EXTINST[op]
            if extinst in _EXTINST_FLOAT16_32_ONLY and _DTYPE_WIDTH.get(result_dtype) == 64:
                self._shortfall(
                    op, block.name,
                    f"GLSL.std.450 {extinst} is restricted to 16/32-bit "
                    "float operands; this value is float64",
                )
                return
            ext = self.builder.glsl_ext()
            body.append(
                f"    {result_id} = OpExtInst {result_type} {ext} "
                f"{extinst} {args[0]}"
            )
            return

        if op in ("Select", "where") and len(args) == 3:
            body.append(f"    {result_id} = OpSelect {result_type} {args[0]} {args[1]} {args[2]}")
            return

        self._shortfall(
            op, block.name,
            "no SPIR-V opcode or GLSL.std.450 extended instruction is "
            "registered",
        )


def _emit_function(
    function: Function,
    builder: _ModuleBuilder,
    *,
    dtype: str,
    outputs: Sequence[SSAValue],
) -> SPIRVFunction:
    return _FunctionEmitter(function, builder, dtype=dtype, outputs=outputs).emit()


def _assemble(name: str, builder: _ModuleBuilder, functions: tuple[SPIRVFunction, ...]) -> SPIRVModule:
    lines: list[str] = []
    for capability in sorted(builder.capabilities):
        lines.append(f"OpCapability {capability}")
    if builder.needs_glsl_ext:
        lines.append('%glsl = OpExtInstImport "GLSL.std.450"')
    lines.append("OpMemoryModel Logical GLSL450")
    for exported_name in builder.exported:
        lines.append(f'OpDecorate %{exported_name} LinkageAttributes "{exported_name}" Export')
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
) -> SPIRVModule:
    """Translate one SSA function into a standalone SPIR-V module.

    ``outputs`` names the SSA value that leaves the function. SSA itself
    records only arguments, so a result would otherwise never become the
    function's return value.
    """

    builder = _ModuleBuilder()
    fn = _emit_function(function, builder, dtype=dtype, outputs=outputs)
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
    named_outputs = dict(outputs or {})
    builder = _ModuleBuilder()
    fns = tuple(
        _emit_function(
            function,
            builder,
            dtype=dtype,
            outputs=named_outputs.get(function_name, ()),
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
