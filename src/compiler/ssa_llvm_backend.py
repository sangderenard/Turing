"""Deterministic likeness between Turing SSA operations and LLVM SSA.

The LLVM sibling of :mod:`ssa_fortran_backend`'s operation tables, and
nothing more: one table entry per SSA operation, giving the exact LLVM
spelling that operation reduces to. ``{0}`` and ``{1}`` are the operand
registers, ``{out}`` the result register. Anything absent is reported as an
unsupported operation by consumers rather than guessed at -- the same
contract every other backend table in this repository carries.

Scalar operations are pure LLVM instructions. Tensor operations name the
authored C-kernel LLVM definitions (``c_backend_llvm_ssa.TRANSLATIONS``);
their likeness is the kernel symbol, and their calling convention is the
authored signature -- both already catalogued there. This module does not
re-state kernel bodies or signatures; it states which symbol each SSA tensor
operation is deterministically alike to.
"""

from __future__ import annotations


# --- scalar likeness: SSA opcode -> LLVM instruction template ---------------
# Floating (double) domain first; the integer columns join as the type
# vocabulary fills out, exactly as everywhere else in the matrix.
_BINARY: dict[str, str] = {
    "Add": "{out} = fadd double {0}, {1}",
    "Sub": "{out} = fsub double {0}, {1}",
    "Mul": "{out} = fmul double {0}, {1}",
    "Div": "{out} = fdiv double {0}, {1}",
    # Python's % is FLOORED for floats too; frem alone is C sign semantics.
    "Mod": (
        "{out}.rem = frem double {0}, {1}\n"
        "{out}.mix = fmul double {out}.rem, {1}\n"
        "{out}.opposed = fcmp olt double {out}.mix, 0.0\n"
        "{out}.adjust = select i1 {out}.opposed, double {1}, double 0.0\n"
        "{out} = fadd double {out}.rem, {out}.adjust"
    ),
    "Pow": "{out} = call double @llvm.pow.f64(double {0}, double {1})",
    "FloorDiv": (
        "{out}.q = fdiv double {0}, {1}\n"
        "{out} = call double @llvm.floor.f64(double {out}.q)"
    ),
    "Eq": "{out} = fcmp oeq double {0}, {1}",
    "Ne": "{out} = fcmp one double {0}, {1}",
    "Lt": "{out} = fcmp olt double {0}, {1}",
    "Le": "{out} = fcmp ole double {0}, {1}",
    "Gt": "{out} = fcmp ogt double {0}, {1}",
    "Ge": "{out} = fcmp oge double {0}, {1}",
    # Unsigned bit-sequence comparisons (the Fortran table's blt/ble): the
    # operands are integer-typed values, compared as unsigned.
    "ULt": "{out} = icmp ult i64 {0}, {1}",
    "ULe": "{out} = icmp ule i64 {0}, {1}",
    "And": "{out} = and i1 {0}, {1}",
    "Or": "{out} = or i1 {0}, {1}",
    "Xor": "{out} = xor i1 {0}, {1}",
    "LAnd": "{out} = and i1 {0}, {1}",
    "LOr": "{out} = or i1 {0}, {1}",
    "BitAnd": "{out} = and i64 {0}, {1}",
    "BitOr": "{out} = or i64 {0}, {1}",
    "BitXor": "{out} = xor i64 {0}, {1}",
    "Shl": "{out} = shl i64 {0}, {1}",
    "Shr": "{out} = lshr i64 {0}, {1}",
    "Min": "{out} = call double @llvm.minnum.f64(double {0}, double {1})",
    "Max": "{out} = call double @llvm.maxnum.f64(double {0}, double {1})",
}

_UNARY: dict[str, str] = {
    "Neg": "{out} = fneg double {0}",
    "Abs": "{out} = call double @llvm.fabs.f64(double {0})",
    "Sqrt": "{out} = call double @llvm.sqrt.f64(double {0})",
    "Exp": "{out} = call double @llvm.exp.f64(double {0})",
    "Log": "{out} = call double @llvm.log.f64(double {0})",
    "Sin": "{out} = call double @llvm.sin.f64(double {0})",
    "Cos": "{out} = call double @llvm.cos.f64(double {0})",
    "Floor": "{out} = call double @llvm.floor.f64(double {0})",
    "Ceil": "{out} = call double @llvm.ceil.f64(double {0})",
    "Trunc": "{out} = call double @llvm.trunc.f64(double {0})",
    "Round": "{out} = call double @llvm.round.f64(double {0})",
    "Not": "{out} = xor i1 {0}, true",
    "LNot": "{out} = xor i1 {0}, true",
    "Invert": "{out} = xor i64 {0}, -1",
    "SIToFP": "{out} = sitofp i32 {0} to double",
    "FPToSI": "{out} = fptosi double {0} to i32",
    "SiToFp": "{out} = sitofp i32 {0} to double",
    "UiToFp": "{out} = uitofp i32 {0} to double",
    "FpToSi": "{out} = fptosi double {0} to i32",
    "FpToUi": "{out} = fptoui double {0} to i32",
    "SExt": "{out} = sext i32 {0} to i64",
    "ZExt": "{out} = zext i32 {0} to i64",
}

def _intrinsic_declarations_from_templates(
    *tables: dict[str, str],
) -> dict[str, str]:
    """Derive `declare` lines for the LLVM intrinsics the tables actually call.

    A target intrinsic belongs to the LLVM language itself, not to the
    repository's authored kernels, so it has no extractable definition. Its
    signature is nevertheless already stated exactly by the authored call
    template above, so the declaration is read back from that same text rather
    than restated as a second, drift-prone signature table.
    """

    declarations: dict[str, str] = {}
    call = _re.compile(
        r"call\s+(?P<ret>[\w.]+)\s+@(?P<symbol>llvm\.[\w.]+)\((?P<args>[^)]*)\)"
    )
    for table in tables:
        for template in table.values():
            for match in call.finditer(template):
                symbol = match.group("symbol")
                operands = []
                for operand in match.group("args").split(","):
                    operand = operand.strip()
                    if not operand:
                        continue
                    # "double {0}" / "double {out}.q" -> the declared type only.
                    operands.append(operand.split()[0])
                declarations[symbol] = (
                    f"declare {match.group('ret')} @{symbol}"
                    f"({', '.join(operands)})"
                )
    return declarations


# --- tensor likeness: SSA tensor operation -> authored kernel symbol --------
# The symbol is the deterministic likeness; body and signature live with the
# authored kernels in c_backend_llvm_ssa and are not restated here.
_TENSOR: dict[str, str] = {
    "add": "binary_double",
    "sub": "binary_double",
    "mul": "binary_double",
    "truediv": "binary_double",
    "pow": "binary_double",
    "mod": "binary_double",
    "floordiv": "binary_double",
    "eq": "binary_double",
    "ne": "binary_double",
    "lt": "binary_double",
    "le": "binary_double",
    "gt": "binary_double",
    "ge": "binary_double",
    "maximum": "binary_double",
    "minimum": "binary_double",
    "add_scalar": "binary_scalar_double",
    "abs": "unary_double",
    "neg": "unary_double",
    "sqrt": "unary_double",
    "exp": "unary_double",
    "log": "unary_double",
    "floor": "unary_double",
    "ceil": "unary_double",
    "round": "unary_double",
    "trunc": "unary_double",
    "isfinite": "unary_double",
    "isnan": "unary_double",
    "isinf": "unary_double",
    "logical_not": "unary_double",
    "tanh": "unary_double",
    "sigmoid": "unary_double",
    "sin": "unary_double",
    "cos": "unary_double",
    "tan": "unary_double",
    "asin": "unary_double",
    "acos": "unary_double",
    "atan": "unary_double",
    "sinh": "unary_double",
    "cosh": "unary_double",
    "asinh": "unary_double",
    "acosh": "unary_double",
    "atanh": "unary_double",
    "sign": "sign_double",
    "matmul": "matmul_double",
    "unfold2d": "unfold2d_double",
    "fold2d": "fold2d_double",
    "transpose": "transpose_double",
    "swapaxes": "transpose_double",
    "permute": "transpose_double",
    "sum": "sum_double",
    "mean": "sum_double",          # flat mean = sum likeness + scalar Div
    "sum_dim": "reduce_dim_double",
    "prod": "reduce_dim_double",
    "min": "reduce_dim_double",
    "max": "reduce_dim_double",
    "any": "reduce_dim_double",
    "all": "reduce_dim_double",
    "cumsum": "cumsum_dim_double",
    "where": "where_double",
    "broadcast": "broadcast_double",
    "broadcast_to": "broadcast_double",
    "expand": "broadcast_double",
    "gather": "index_select_double",
    "scatter": "index_assign_double",
    "pad": "pad_double_nd",
    "stack": "stack_double",
    "cat": "cat_double",
    "concat": "cat_double",
    "concatenate": "cat_double",
    "fill": "fill_double",
    "zeros": "fill_double",
    "zeros_like": "fill_double",
    "empty": "fill_double",
    "ones": "fill_double",
    "ones_like": "fill_double",
    "full": "fill_double",
    "full_like": "fill_double",
    # Value-precision casts; reference is the numpy backend's ``_cast_`` map.
    # ``double`` is float64 values -- a copying identity under the double
    # working type, never the narrowing kernel.
    "float": "cast_double_to_float_values",
    "double": "cast_double_to_double_values",
    "bool": "cast_double_to_bool_values",
    "long": "cast_double_to_int_values",
    "int": "cast_double_to_int_values",
    "arange": "create_arange",
    "extent": "extent",            # runtime metadata read; each target's own
}

# --- shape-only operations: no runtime existence, alias in every target -----
_SHAPE_ONLY = frozenset({
    "reshape", "view", "flatten", "unsqueeze", "squeeze", "contiguous",
})


def supported_scalar_operations() -> frozenset[str]:
    return frozenset(_BINARY) | frozenset(_UNARY)


def supported_tensor_operations() -> frozenset[str]:
    from .ssa_numeric_operators import TENSOR_SSA_OPERATORS

    scalar = supported_scalar_operations()
    direct = {
        row.name
        for row in TENSOR_SSA_OPERATORS
        if row.is_direct and row.handler.value in scalar
    }
    # Const is a first-class repository instruction handled by the emitter,
    # not a scalar likeness-table entry.  tensor_from_list is its canonical
    # tensor spelling.
    direct.add("tensor_from_list")
    return frozenset(_TENSOR) | _SHAPE_ONLY | frozenset(direct)


# Multiply-add contraction is opt-in and OFF by default: an fma rounds once
# where fmul+fadd round twice, so contracted results differ bitwise from every
# other backend's and from the reference evaluator's. `contract` alone only
# PERMITS fusion; the emitted module names no target, so the toolchain also
# needs `-march=native` (added under the same switch at the zig cc invocation)
# before the host's FMA units are reachable. Audit 2026-08-19: every float op
# in the emitted kernel reads its operands from memory slots (`%load.*`), so
# there are no textual producer->consumer chains to fuse at emission time --
# contraction is decided entirely inside LLVM after mem2reg, which is why the
# switch is a flag on the instruction and not a rewrite of the SSA.
_CONTRACT_ELIGIBLE = frozenset({"Add", "Sub", "Mul"})


def _fma_contract_enabled() -> bool:
    from .work_contract import active_contract

    return active_contract().contract_multiply_add


_HOST_TARGET_LINES: tuple[str, ...] | None = None


def _host_target_lines() -> tuple[str, ...]:
    """``target datalayout`` + ``target triple`` for the host, asked of the
    toolchain itself.

    Emitted modules previously named no target at all, so LLVM optimized
    with a generic cost model -- vector widths, alignments and addressing
    costs all guessed. The lines are read from what the SAME bundled
    clang produces for a trivial C file (dynamic reference, no
    hand-mirrored platform strings), probed once per process and cached.
    An unavailable toolchain degrades to the previous behavior: no target
    lines, generic model.
    """

    global _HOST_TARGET_LINES
    if _HOST_TARGET_LINES is not None:
        return _HOST_TARGET_LINES
    import subprocess as _probe_subprocess
    import sys as _probe_sys
    import tempfile as _probe_tempfile
    from pathlib import Path as _ProbePath

    lines: tuple[str, ...] = ()
    try:
        with _probe_tempfile.TemporaryDirectory() as scratch:
            probe_source = _ProbePath(scratch) / "probe.c"
            probe_source.write_text("int turing_target_probe;\n")
            probe_output = _ProbePath(scratch) / "probe.ll"
            completed = _probe_subprocess.run(
                [_probe_sys.executable, "-m", "ziglang", "cc",
                 "-S", "-emit-llvm",
                 "-o", str(probe_output), str(probe_source)],
                capture_output=True, text=True, timeout=180, check=False,
            )
            if completed.returncode == 0 and probe_output.is_file():
                lines = tuple(
                    line
                    for line in probe_output.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.startswith("target datalayout")
                    or line.startswith("target triple")
                )
    except Exception:
        lines = ()
    _HOST_TARGET_LINES = lines
    return lines


import re as _noalias_re

_FORMAL_ACTUAL = _noalias_re.compile(r"^%(arg|out)\.(\d+)$")
_WRAPPER_STORAGE = _noalias_re.compile(r"^%(public|root\.frame)\.\d+$")
_DEFINE_LINE = _noalias_re.compile(
    r"^define void @(?P<symbol>[\w$.]+)\((?P<params>[^)]*)\)"
)


def _annotate_noalias(
    emitted_functions: list[str],
    internal_call_records: list[tuple[str | None, str, tuple[str, ...]]],
) -> list[str]:
    """Add ``noalias`` to parameter positions proven distinct BY STORAGE.

    The authority is the EMITTED POINTER STRING at every internal call
    site, never the SSA value id: aliased IO through the pointer ABI is a
    design feature here (a carried value's initial and updated ids share
    one buffer across a call boundary), so distinct ids do not imply
    distinct storage -- an id-based version of this pass miscompiled a
    five-element accumulation to its last element. Strings are judged
    conservatively, whitelist only:

    * the wrapper's actuals (``%public.N`` per public value id,
      ``%root.frame.N`` allocas) are terminal storage -- distinct names
      are distinct allocations, so a position is safe there when its
      string is unique in the call. Externally, authored array parameters
      follow the BLAS restrict convention (a caller may alias two
      READ-ONLY arrays; in-place input/output overlap arrives as the same
      ``%public.N`` on both positions and is caught by the dup check);
    * a caller's own formal (``%arg.K`` / ``%out.K``) inherits that
      caller position's verdict, to a fixed point over the call graph;
    * anything else -- derived addresses, aggregate tables, locals whose
      provenance this pass has not proven -- disqualifies its position.

    A position must be safe at EVERY call site to earn the attribute; a
    function nobody calls (dead but emitted) earns nothing.
    """

    position_count: dict[str, int] = {}
    sites: dict[str, list[tuple[str | None, tuple[str, ...]]]] = {}
    for caller, callee, actuals in internal_call_records:
        sites.setdefault(callee, []).append((caller, actuals))
        count = position_count.get(callee)
        if count is None or len(actuals) < count:
            position_count[callee] = len(actuals)

    safe: dict[str, set[int]] = {
        symbol: set(range(count))
        for symbol, count in position_count.items()
    }
    changed = True
    while changed:
        changed = False
        for symbol, call_sites in sites.items():
            for caller, actuals in call_sites:
                for position in tuple(safe[symbol]):
                    if position >= len(actuals):
                        safe[symbol].discard(position)
                        changed = True
                        continue
                    actual = actuals[position]
                    if actuals.count(actual) > 1:
                        safe[symbol].discard(position)
                        changed = True
                        continue
                    if _FORMAL_ACTUAL.match(actual) is not None:
                        # A caller formal: verdict inherited in the second
                        # fixed point below, once direct disqualifiers are
                        # settled everywhere.
                        continue
                    if caller is None and _WRAPPER_STORAGE.match(actual):
                        continue
                    safe[symbol].discard(position)
                    changed = True

    # Second fixed point for formal inheritance, now that direct
    # disqualifiers are settled: a caller formal's flat position is its
    # index among (args..., outs...), which is exactly how the actual
    # strings %arg.K / %out.K were numbered at emission -- but %out.K's
    # flat position needs the caller's arg count, recovered from its own
    # define line.
    argument_counts: dict[str, int] = {}
    for text in emitted_functions:
        match = _DEFINE_LINE.match(text.splitlines()[0])
        if match is None:
            continue
        params = match.group("params").split(", ") if match.group("params") else []
        argument_counts[match.group("symbol")] = sum(
            1 for parameter in params if "%arg." in parameter
        )
    changed = True
    while changed:
        changed = False
        for symbol, call_sites in sites.items():
            for caller, actuals in call_sites:
                if caller is None:
                    continue
                for position in tuple(safe[symbol]):
                    if position >= len(actuals):
                        continue
                    formal = _FORMAL_ACTUAL.match(actuals[position])
                    if formal is None:
                        continue
                    flat = int(formal.group(2)) + (
                        argument_counts.get(caller, 0)
                        if formal.group(1) == "out" else 0
                    )
                    if flat not in safe.get(caller, set()):
                        safe[symbol].discard(position)
                        changed = True

    annotated: list[str] = []
    for text in emitted_functions:
        lines = text.splitlines()
        match = _DEFINE_LINE.match(lines[0])
        if match is None or match.group("symbol") not in safe:
            annotated.append(text)
            continue
        symbol = match.group("symbol")
        parameters = (
            match.group("params").split(", ")
            if match.group("params") else []
        )
        rewritten = []
        for position, parameter in enumerate(parameters):
            if (
                position in safe[symbol]
                and parameter.startswith("ptr %")
                and "%extents" not in parameter
            ):
                parameter = parameter.replace("ptr %", "ptr noalias %", 1)
            rewritten.append(parameter)
        lines[0] = (
            f"define void @{symbol}({', '.join(rewritten)})"
            + lines[0][match.end():]
        )
        annotated.append("\n".join(lines))
    return annotated


def scalar_likeness(operation: str) -> str | None:
    template = _BINARY.get(operation) or _UNARY.get(operation)
    if (
        template is not None
        and operation in _CONTRACT_ELIGIBLE
        and _fma_contract_enabled()
    ):
        template = template.replace(" double ", " contract double ", 1)
    return template


# Integer-domain scalar emission. The templates above are the double column of
# the same table; an integer-typed value must not be quietly widened to double
# and stored back into a narrower ABI slot, so the integer spelling lives here
# as one shared rule both module emitters consult.
_INTEGER_BINARY: dict[str, str] = {
    "Add": "add", "Sub": "sub", "Mul": "mul",
    "Div": "sdiv", "Mod": "srem",
    "And": "and", "Or": "or", "Xor": "xor",
    "LAnd": "and", "LOr": "or",
    "BitAnd": "and", "BitOr": "or", "BitXor": "xor",
    "Shl": "shl", "Shr": "lshr",
}
_INTEGER_COMPARISON: dict[str, str] = {
    "Eq": "eq", "Ne": "ne", "Lt": "slt",
    "Le": "sle", "Gt": "sgt", "Ge": "sge",
    "ULt": "ult", "ULe": "ule",
}
# The double column reaches minnum/maxnum, which have no integer form.
# Compare-and-select is the exact integer equivalent and needs no intrinsic.
# Unsigned selection stays absent until an unsigned opcode names itself,
# exactly as ULt/ULe already do for comparison.
_INTEGER_SELECTION: dict[str, str] = {"Max": "sgt", "Min": "slt"}


def integer_scalar_lines(
    operation: str,
    operand_type: str,
    operands: list[str],
    register: str,
) -> tuple[list[str], str] | None:
    """(lines, result type) for an integer-domain scalar op, else ``None``.

    ``None`` means this opcode has no exact integer spelling; the caller
    decides whether that is a shortfall or a documented double-domain
    evaluation converted back to the declared integer type.
    """

    if operation == "Mod" and len(operands) == 2:
        # Python ``%`` is FLOORED: the result carries the divisor's sign.
        # ``srem`` is C semantics -- ``-1 % 16`` gave -1 instead of 15, and a
        # periodic wrap like ``(row - 1) % height`` addressed sixteen doubles
        # BEFORE its span for the whole first row.
        return ([
            f"{register}.rem = srem {operand_type} "
            f"{operands[0]}, {operands[1]}",
            f"{register}.mix = xor {operand_type} "
            f"{register}.rem, {operands[1]}",
            f"{register}.opposed = icmp slt {operand_type} "
            f"{register}.mix, 0",
            f"{register}.nonzero = icmp ne {operand_type} "
            f"{register}.rem, 0",
            f"{register}.fix = and i1 {register}.opposed, "
            f"{register}.nonzero",
            f"{register}.adjust = select i1 {register}.fix, "
            f"{operand_type} {operands[1]}, {operand_type} 0",
            f"{register} = add {operand_type} {register}.rem, "
            f"{register}.adjust",
        ], operand_type)
    if operation in _INTEGER_BINARY and len(operands) == 2:
        return ([
            f"{register} = {_INTEGER_BINARY[operation]} "
            f"{operand_type} {operands[0]}, {operands[1]}"
        ], operand_type)
    if operation in _INTEGER_SELECTION and len(operands) == 2:
        return ([
            f"{register}.cmp = icmp {_INTEGER_SELECTION[operation]} "
            f"{operand_type} {operands[0]}, {operands[1]}",
            f"{register} = select i1 {register}.cmp, "
            f"{operand_type} {operands[0]}, {operand_type} {operands[1]}",
        ], operand_type)
    if operation in _INTEGER_COMPARISON and len(operands) == 2:
        return ([
            f"{register} = icmp {_INTEGER_COMPARISON[operation]} "
            f"{operand_type} {operands[0]}, {operands[1]}"
        ], "i1")
    if operation == "Neg" and len(operands) == 1:
        return ([f"{register} = sub {operand_type} 0, {operands[0]}"],
                operand_type)
    if operation in {"Not", "LNot"} and len(operands) == 1:
        return ([f"{register} = xor {operand_type} {operands[0]}, 1"],
                operand_type)
    return None


def tensor_likeness(operation: str) -> str | None:
    return _TENSOR.get(operation)


# --- the table wired as an emitter ------------------------------------------
#
# The emitter renders one SSA function through the likeness tables above:
# every instruction either has a table entry or becomes a named shortfall.
# Kernel calling conventions are parsed from the authored kernel definitions
# themselves (c_backend_llvm_ssa), never restated. Compilation is a separate
# step through an LLVM compiler ahead of time -- Zig's embedded clang, the
# toolchain this repository already builds C with. No JIT.

import ctypes as _ctypes
import re as _re
import struct as _struct
import subprocess as _subprocess
import tempfile as _tempfile
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path
from typing import (
    Any as _Any, Mapping as _Mapping, Sequence as _Sequence,
)

from ..transmogrifier.ssa import IRModule as _IRModule, SSAValue as _SSAValue

# Synthetic buffer ids for history watches. They are deliberately far above
# any real SSA numbering so a history slot can never be mistaken for, or
# collide with, a value the program actually owns -- the id-collision class
# of bug this tree has already paid for more than once.
_HISTORY_RING_BASE = 1_000_000_000
_HISTORY_COUNT_BASE = 2_000_000_000


def history_ids(value_id: int) -> tuple[int, int]:
    """Buffer ids carrying ``value_id``'s history ring and its sample count."""
    return (
        _HISTORY_RING_BASE + int(value_id),
        _HISTORY_COUNT_BASE + int(value_id),
    )
from .output_publication import (
    function_output_publications,
    publication_surface_plan,
)


# Target intrinsics reached by the scalar tables, plus the block-copy the
# arena/aggregate paths emit directly. This is the one home for intrinsic
# declarations; both module emitters seed from it.
_LLVM_INTRINSIC_DECLARATIONS: dict[str, str] = {
    **_intrinsic_declarations_from_templates(_BINARY, _UNARY),
    "llvm.memcpy.p0.p0.i64": (
        "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)"
    ),
    # Zero fills reach this from the BACKWARD side: the derivative of any
    # step-like operation is zero almost everywhere, and the authored rule
    # spells that ``zeros_like``.
    "llvm.memset.p0.i64": (
        "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)"
    ),
}


@_dataclass(frozen=True)
class LLVMEmissionShortfall:
    function: str
    operation: str
    reason: str


def _kernel_signature(symbol: str) -> tuple[str, tuple[str, ...]]:
    """(return type, argument types) parsed from the authored definition."""

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        extract_llvm_function,
    )

    text = extract_llvm_function(symbol)
    match = _re.search(
        r"define\s+([\w<>\s\*]+?)\s*@" + _re.escape(symbol) + r"\((.*?)\)",
        text, _re.DOTALL,
    )
    if match is None:
        raise ValueError(f"authored kernel {symbol!r} has no parseable define")
    returns = match.group(1).strip()
    arguments = tuple(
        parameter.strip().split()[0]
        for parameter in match.group(2).split(",")
        if parameter.strip()
    )
    return returns, arguments


def _double_literal(value: _Any) -> str:
    bits = _struct.unpack(">Q", _struct.pack(">d", float(value)))[0]
    return f"0x{bits:016X}"


#: Bytes one element of each LLVM ABI type occupies in a buffer. This is the
#: SAME set ``prepare_artifact_execution`` allocates from -- double/f64,
#: i32, i64, i1 as one byte (numpy bool_), ptr as a machine word -- so a block
#: copy or fill sized from here matches what the runtime actually allocated.
#:
#: Sizing bulk memory as "elements * 8" is only correct while every buffer is
#: a double. It silently writes FOUR TIMES the allocation for an i32 buffer
#: and eight times for an i1 one, past the end, into whatever follows.
_LLVM_TYPE_BYTES: dict[str, int] = {
    "double": 8, "i64": 8, "ptr": 8, "i32": 4, "i1": 1,
}


def _value_bytes(value: _Any) -> int:
    """Bytes one element of ``value`` occupies, from its declared type."""

    llvm_type = _value_llvm_type(value)
    try:
        return _LLVM_TYPE_BYTES[llvm_type]
    except KeyError as error:
        raise ValueError(
            f"no ABI byte width declared for LLVM type {llvm_type!r}"
        ) from error


def _value_block_bytes(value: _Any) -> int:
    """Bytes the whole buffer behind ``value`` occupies."""

    return _value_element_count(value) * _value_bytes(value)


def _align(llvm_type: _Any) -> int:
    """Natural alignment of one LLVM ABI value, in bytes.

    ``align 8`` is a PROMISE to LLVM, not a formatting detail. It is true for
    double, i64 and ptr, and false for an element inside an i32 or i1 buffer:
    element one of an i32 array sits at offset 4. LLVM is entitled to act on
    the claim -- widening or vectorising the access -- so an over-stated
    alignment on an under-aligned address is a miscompile waiting for the
    right optimisation pass, not a cosmetic wart.

    Over-aligning an ``alloca`` is a different case and stays at 8: there the
    number is a request for the allocation, and asking for more is safe.
    """

    return _LLVM_TYPE_BYTES.get(str(llvm_type), 8)


def _value_llvm_type(value: _Any) -> str:
    accounting = getattr(value, "accounting", {}) or {}
    if tuple(
        accounting.get(
            "ssa_aggregate_outputs", ()
        )
    ):
        return "ptr"
    dtype = str(
        accounting.get("physical_dtype")
        or getattr(value, "dtype", None)
        or "float64"
    ).lower()
    if dtype in {"bool", "i1"}:
        return "i1"
    if dtype in {"int", "int32", "i32"}:
        return "i32"
    if dtype in {"int64", "i64", "long"}:
        return "i64"
    if dtype == "opaque_ref":
        return "i64"
    return "double"


def _declared_span_rank(value: _Any) -> int:
    """How many axes this value has, by the same authority Fortran uses.

    `.shape` alone is not that authority: these arrays are sized at run
    time, so a rank-2 record field routinely carries an EMPTY static shape
    while its accounting states `program_abi_rank: 2`. The Fortran backend
    reads the accounting and therefore declares `t(e1, e2)` and copies the
    whole array; a backend consulting `.shape` sees a scalar.
    """
    accounting = getattr(value, "accounting", None) or {}
    return max(
        len(tuple(getattr(value, "shape", ()) or ())),
        int(accounting.get("program_abi_rank", 0) or 0),
        int(accounting.get("ssa_call_rank", 0) or 0),
    )


def _span_element_count(
    function_name: str,
    value: _Any,
    tag: str,
    body: list,
    public_span_value: _Any,
    module_extent_slot: _Any,
) -> str | None:
    """An i32 register (or literal) holding this span's element count.

    Static extents fold to a literal; dynamic ones are measured from the
    artifact's own extents vector, the same mechanism multi-axis addressing
    already uses. Returns None when neither is available, so the caller can
    refuse rather than invent a size.
    """
    static = tuple(getattr(value, "shape", ()) or ())
    rank = _declared_span_rank(value)
    if static and all(isinstance(item, int) for item in static):
        total = 1
        for extent in static:
            total *= int(extent)
        return str(total)
    public_id = public_span_value(function_name, int(value.id))
    if public_id is None or not rank:
        return None
    running: str | None = None
    for axis in range(rank):
        slot = module_extent_slot(public_id, axis)
        address = f"%ew.extent.addr.{tag}.{axis}"
        register = f"%ew.extent.{tag}.{axis}"
        body.append(
            f"  {address} = getelementptr i32, ptr %extents, i64 {slot}"
        )
        body.append(f"  {register} = load i32, ptr {address}, align 4")
        if running is None:
            running = register
        else:
            product = f"%ew.extent.prod.{tag}.{axis}"
            body.append(f"  {product} = mul i32 {running}, {register}")
            running = product
    return running


def _value_element_count(value: _Any) -> int:
    from math import prod

    aggregate = tuple(
        (getattr(value, "accounting", {}) or {}).get(
            "ssa_aggregate_outputs", ()
        )
    )
    if aggregate:
        return len(aggregate)
    shape = tuple(getattr(value, "shape", ()) or ())
    return max(1, int(prod(map(int, shape)))) if shape else 1


def _internal_call_closure(
    module: _IRModule, root: str,
) -> tuple[tuple[str, ...], set[str]]:
    """Repository functions (module order) and authored kernel leaves.

    Which calls are edges is this backend's policy -- a callee with an
    authored kernel signature is a leaf, not an edge -- but the ORDER of
    the result is the module's own, via ``IRModule.reachable_functions``.
    """

    kernels: set[str] = set()

    def follow(instruction: _Any) -> str | None:
        callee = instruction.attributes.get("callee")
        if callee is None:
            return None
        symbol = str(callee)
        try:
            _kernel_signature(symbol)
        except (KeyError, ValueError):
            return symbol if symbol in module.functions else None
        kernels.add(symbol)
        return None

    return module.reachable_functions(str(root), follow_call=follow), kernels


from .hierarchical_plan import PREDICATE_OPERATIONS  # noqa: E402


def _emit_repository_call_module(
    module: _IRModule,
    function_name: str,
    *,
    entry_name: str,
    text_sink: bool,
    watch: _Sequence[int] = (),
    history: int = 0,
) -> "LLVMFunctionArtifact":
    """Emit a repository-SSA call closure with a pointer-only internal ABI.

    Every internal function receives one pointer per SSA argument followed by
    one pointer per result.  Aggregate call results remain explicit arrays of
    pointers, so the repository's GetElementPtr/Load projections retain their
    meaning without flattening or call-site substitution.
    """

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        extract_llvm_declaration,
        extract_llvm_function,
    )

    reachable, kernels_used = _internal_call_closure(module, function_name)
    shortfalls: list[LLVMEmissionShortfall] = []

    values_by_function: dict[str, dict[int, _Any]] = {}
    for name in reachable:
        function = module.functions[name]
        values: dict[int, _Any] = {int(value.id): value for value in function.args}
        for block in function.blocks.values():
            for instruction in block.instrs:
                for value in instruction.args:
                    current = values.get(int(value.id))
                    if current is None or (
                        not tuple(getattr(current, "shape", ()) or ())
                        and tuple(getattr(value, "shape", ()) or ())
                    ):
                        values[int(value.id)] = value
                if instruction.res is not None:
                    value = instruction.res
                    current = values.get(int(value.id))
                    if current is None or (
                        not tuple(getattr(current, "shape", ()) or ())
                        and tuple(getattr(value, "shape", ()) or ())
                    ):
                        values[int(value.id)] = value
        values_by_function[name] = values

    # Determine which aggregate projections are actually consumed.  This
    # shrinks planned-region ABIs and removes descriptor getters made dead by
    # structural specialization before target emission.
    aggregate_output_positions: dict[str, list[int]] = {}
    aggregate_output_ids: dict[str, list[int]] = {}
    aggregate_output_values: dict[str, dict[int, _Any]] = {}
    aggregate_escapes_whole: set[tuple[str, int]] = set()
    aggregate_positions: dict[tuple[str, int], tuple[int, ...]] = {}
    for caller_name in reachable:
        function = module.functions[caller_name]
        uses: dict[int, int] = {}
        instructions = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
        ]
        for instruction in instructions:
            for argument in instruction.args:
                uses[int(argument.id)] = uses.get(int(argument.id), 0) + 1
        for instruction in instructions:
            if (
                instruction.op not in {"Call", "call"}
                or instruction.res is None
                or instruction.attributes.get("result_convention")
                != "ssa.aggregate"
            ):
                continue
            callee = str(instruction.attributes.get("callee") or "")
            declared = tuple(map(int, instruction.attributes.get("output_ids", ())))
            live_positions: list[int] = []
            address_position: dict[int, int] = {}
            projected_values: dict[int, _Any] = {}
            for follower in instructions:
                if (
                    follower.op in {"GetElementPtr", "getelementptr"}
                    and follower.res is not None
                    and follower.args
                    and int(follower.args[0].id) == int(instruction.res.id)
                ):
                    position = follower.attributes.get("aggregate_index")
                    if position is not None:
                        address_position[int(follower.res.id)] = int(position)
                elif (
                    follower.op in {"Load", "load"}
                    and follower.res is not None
                    and follower.args
                    and int(follower.args[0].id) in address_position
                    and uses.get(int(follower.res.id), 0) > 0
                ):
                    projected_position = address_position[int(follower.args[0].id)]
                    live_positions.append(projected_position)
                    projected_values[projected_position] = follower.res
            consumed_whole = any(
                follower is not instruction
                and follower.op not in {"GetElementPtr", "getelementptr"}
                and any(
                    int(argument.id) == int(instruction.res.id)
                    for argument in follower.args
                )
                for follower in instructions
            )
            # The authored program declares these outputs; ALL of them are
            # published.  Shrinking the ABI to a use-count snapshot created
            # a 7-of-11 pairing problem every later pass tripped over -- a
            # dead output costs one store to a cell nobody reads, which is
            # nothing.  Removal belongs to the planner's proof, never to an
            # emitter-local count.
            selected = tuple(range(len(declared)))
            if consumed_whole:
                aggregate_escapes_whole.add(
                    (caller_name, int(instruction.res.id))
                )
            aggregate_positions[(caller_name, int(instruction.res.id))] = selected
            if callee in reachable and declared:
                existing = aggregate_output_positions.setdefault(callee, [])
                existing_ids = aggregate_output_ids.setdefault(callee, [])
                typed = aggregate_output_values.setdefault(callee, {})
                for position in selected:
                    if position not in existing:
                        existing.append(position)
                        existing_ids.append(declared[position])
                    value = projected_values.get(position)
                    if value is None:
                        value = values_by_function[caller_name].get(
                            declared[position]
                        )
                    if value is not None:
                        typed[declared[position]] = value

    # Which of a callee's OWN formal parameters does its body treat as a
    # pointer TABLE (a `ptr[N]` of other buffers' addresses), rather than a
    # single tensor/scalar buffer? A parameter is unpacked this way when the
    # callee's own instructions GetElementPtr off it at a nonzero constant
    # offset (offset 0 is indistinguishable from ordinary single-value
    # indexing, so it is not by itself evidence of aggregate use) AND the
    # loaded slot is then used as an ADDRESS. Both halves matter: a flat
    # array indexed at a compile-time-constant offset -- the shape every
    # size-specialized kernel takes once its strides are baked -- produces
    # the same GEP+Const, but what it loads is a NUMBER that flows into
    # arithmetic, never a pointer that gets dereferenced. This is a purely
    # structural, callee-local fact -- independent of any one call site --
    # used below to check that whoever calls in with that position actually
    # built a table there.
    def _gep_result_is_dereferenced(
        callee_function: _Any, gep_result_id: int
    ) -> bool:
        loaded = [
            instruction.res
            for block in callee_function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Load", "load"}
            and instruction.args
            and int(instruction.args[0].id) == gep_result_id
            and instruction.res is not None
        ]
        if not loaded:
            return False
        # A table slot holds another buffer's ADDRESS, so loading it yields
        # a whole tensor -- a NON-scalar value whose storage is the loaded
        # pointer (measured: the MSE-family unpacks load (2,2)/(2,)/(3,2)
        # values). A flat array indexed at a constant offset loads a single
        # NUMBER (shape ()) that flows into arithmetic.
        if any(tuple(value.shape or ()) != () for value in loaded):
            return True
        loaded_ids = {int(value.id) for value in loaded}
        for block in callee_function.blocks.values():
            for instruction in block.instrs:
                if not instruction.args:
                    continue
                if (
                    instruction.op in {
                        "GetElementPtr", "getelementptr", "Load", "load",
                    }
                    and int(instruction.args[0].id) in loaded_ids
                ):
                    return True
                if (
                    instruction.op in {"Store", "store"}
                    and len(instruction.args) == 2
                    and int(instruction.args[1].id) in loaded_ids
                ):
                    return True
        return False

    callee_aggregate_parameter_positions: dict[str, set[int]] = {}
    for callee_name in reachable:
        callee_function = module.functions[callee_name]
        parameter_ids = [int(parameter.id) for parameter in callee_function.args]
        if not parameter_ids:
            continue
        callee_const_values: dict[int, _Any] = {}
        for block in callee_function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op in {"Const", "const"}
                    and instruction.res is not None
                ):
                    callee_const_values[int(instruction.res.id)] = (
                        instruction.attributes.get("value")
                    )
        positions: set[int] = set()
        for block in callee_function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op not in {"GetElementPtr", "getelementptr"}
                    or not instruction.args
                ):
                    continue
                base_id = int(instruction.args[0].id)
                if base_id not in parameter_ids:
                    continue
                offset = instruction.attributes.get("aggregate_index")
                if offset is None and len(instruction.args) > 1:
                    # No explicit aggregate_index: a constant index operand
                    # is only evidence of TABLE unpacking when the loaded
                    # slot is itself dereferenced as an address (the memcpy
                    # -through-loaded-pointer shape this guard was built
                    # for).  A flat array indexed at a compile-time-constant
                    # offset -- which is what every size-specialized kernel
                    # produces once its strides are baked to literals --
                    # emits the same GEP+Const shape but LOADS A NUMBER,
                    # not a pointer, so it must not be classified as an
                    # aggregate parameter (that misclassification refused
                    # every size-baked nested-loop kernel).
                    constant_offset = callee_const_values.get(
                        int(instruction.args[1].id)
                    )
                    if (
                        constant_offset is not None
                        and int(constant_offset) >= 1
                        and instruction.res is not None
                        and _gep_result_is_dereferenced(
                            callee_function, int(instruction.res.id)
                        )
                    ):
                        positions.add(parameter_ids.index(base_id))
                    continue
                if offset is not None and int(offset) >= 1:
                    positions.add(parameter_ids.index(base_id))
        if positions:
            callee_aggregate_parameter_positions[callee_name] = positions

    function_outputs: dict[str, tuple[_Any, ...]] = {}
    for name in reachable:
        function = module.functions[name]
        returned = next((
            tuple(instruction.args)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        ), ())
        physical_returned = returned
        if len(returned) == 1:
            aggregate_ids = tuple(map(
                int,
                (returned[0].accounting or {}).get(
                    "ssa_aggregate_outputs", ()
                ),
            ))
            if aggregate_ids and all(
                value_id in values_by_function[name]
                for value_id in aggregate_ids
            ):
                physical_returned = tuple(
                    values_by_function[name][value_id]
                    for value_id in aggregate_ids
                )
        positions = aggregate_output_positions.get(name)
        if physical_returned:
            function_outputs[name] = (
                tuple(physical_returned[position] for position in positions)
                if positions is not None
                and all(
                    position < len(physical_returned)
                    for position in positions
                )
                else physical_returned
            )
        else:
            declared_ids = aggregate_output_ids.get(name, ())
            function_outputs[name] = tuple(
                (
                    values_by_function[name][value_id]
                    if value_id in values_by_function[name]
                    else aggregate_output_values[name][value_id]
                )
                for value_id in declared_ids
                if (
                    value_id in values_by_function[name]
                    or value_id in aggregate_output_values.get(name, {})
                )
            )

    # A source wrapper may return a callee's aggregate unchanged (the
    # canonical backward ``bw_matmul -> matmul_vjp`` shape).  Returning an
    # array of pointers to callee temporaries is not a valid native ABI: those
    # pointees die with the wrapper's stack frame.  Spell the wrapper as the
    # same multiple-output ABI as its callee, so its caller owns every output
    # buffer and no ephemeral pointer escapes.
    forwarded_aggregate_calls: dict[str, tuple[int, str]] = {}
    changed = True
    while changed:
        changed = False
        for name in reachable:
            function = module.functions[name]
            returned = next((
                tuple(instruction.args)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Ret", "ret", "Return", "return"}
            ), ())
            if len(returned) != 1 or not tuple(
                (returned[0].accounting or {}).get(
                    "ssa_aggregate_outputs", ()
                )
            ):
                continue
            producer = next((
                instruction
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Call", "call"}
                and instruction.res is not None
                and int(instruction.res.id) == int(returned[0].id)
                and str(instruction.attributes.get("callee") or "")
                in reachable
            ), None)
            if producer is None:
                continue
            callee = str(producer.attributes["callee"])
            callee_outputs = function_outputs.get(callee, ())
            if len(callee_outputs) <= 1:
                continue
            forwarded_aggregate_calls[name] = (
                int(producer.res.id), callee,
            )
            if function_outputs.get(name) != callee_outputs:
                function_outputs[name] = callee_outputs
                changed = True

    # --- diagnostic watches ------------------------------------------------
    # Appended AFTER the output fixed point has settled, so a watch can never
    # influence what the program decided its real outputs are, and BEFORE
    # emission, so the extra slots are allocated by the ordinary path rather
    # than by a second mechanism that could disagree with it. Everything
    # downstream -- the internal signature, the public wrapper, buffer_order
    # -- then treats a watched value exactly like any other output.
    watched_ids: list[int] = []
    watch_shortfalls: list[tuple[int, str]] = []
    phi_backed: dict[int, tuple[int, ...]] = {}
    if watch:
        root_values = values_by_function.get(function_name, {})
        already = {int(value.id) for value in function_outputs[function_name]}
        root_function = module.functions[function_name]
        # A value whose storage is selected by a Phi lives in an SSA register
        # defined inside the loop, not in a frame slot, so a return-site copy
        # would reference a register that does not dominate the return --
        # which LLVM rejects, correctly. Those get a shadow frame slot at
        # emission instead (see `watch_shadows`), which is what makes a
        # loop-carried accumulator watchable at all: the slot keeps whatever
        # the phi last selected, i.e. the converged value at loop exit.
        phi_backed.update({
            int(instruction.res.id): tuple(
                int(argument.id) for argument in instruction.args
            )
            for block in root_function.blocks.values()
            for instruction in block.instrs
            if str(instruction.op) in {"Phi", "phi"}
            and instruction.res is not None
        })
        appended = []
        for raw_id in dict.fromkeys(int(item) for item in watch):
            if raw_id in already:
                # Already public. Not an error: the caller gets what they
                # asked for, and saying so beats silently doing nothing.
                watched_ids.append(raw_id)
                continue
            value = root_values.get(raw_id)
            if value is None:
                watch_shortfalls.append((
                    raw_id,
                    f"no value {raw_id} in {function_name}'s own frame; "
                    "region-local ids are a different numbering space",
                ))
                continue
            appended.append(value)
            watched_ids.append(raw_id)
        if appended:
            function_outputs[function_name] = (
                *function_outputs[function_name], *appended,
            )
    # Phi-backed watches need a frame slot of their own; see `watch_shadows`
    # at the emission site. Recorded here so emission knows which ones.
    phi_watch_ids = frozenset(
        raw_id for raw_id in watched_ids
        if raw_id in (phi_backed if watch else {})
    )
    # A history watch keeps every value the phi selected, not just the last
    # one, in a ring. A final value proves an accumulator ended wrong; only
    # the per-iteration series shows WHICH step it went wrong on, which is
    # the difference between knowing there is a bug and knowing where.
    #
    # The ring is written with `urem`, so it is branch-free and cannot run
    # off its end no matter how many iterations execute. The companion
    # counter records how many samples were taken, so a reader can tell a
    # full ring from a partly-filled one and can recover ordering when the
    # count exceeds the depth.
    history_slots: dict[int, tuple[int, int]] = {}
    if watch and history:
        depth = max(1, int(history))
        root_values = values_by_function.get(function_name, {})
        appended_history = []
        # Every watched value, not only the phi-backed ones. A phi is where a
        # loop-carried accumulator is visible, but the value that CORRUPTED it
        # is usually an ordinary result computed in the loop body, and a ring
        # on the accumulator alone can only say "it was already wrong by the
        # end". Ordinary results live in frame slots that are rewritten each
        # iteration, so they are ringed at the block terminator -- once per
        # execution of the block that defines them, which is once per
        # iteration, the same cadence the phi ring records at.
        for raw_id in sorted(watched_ids):
            value = root_values.get(raw_id)
            if value is None:
                continue
            ring_id = _HISTORY_RING_BASE + raw_id
            count_id = _HISTORY_COUNT_BASE + raw_id
            appended_history.append(_SSAValue(
                ring_id,
                dtype=getattr(value, "dtype", None) or "float64",
                shape=(depth,),
            ))
            appended_history.append(_SSAValue(count_id, dtype="int64"))
            history_slots[raw_id] = (ring_id, count_id)
        if appended_history:
            function_outputs[function_name] = (
                *function_outputs[function_name], *appended_history,
            )

    internal_symbols = {
        name: "__ssa_" + _re.sub(r"[^A-Za-z0-9_$.-]", "_", name)
        for name in reachable
    }

    def literal(payload: _Any, llvm_type: str) -> str:
        if llvm_type == "double":
            return _double_literal(0.0 if payload is None else payload)
        if llvm_type == "i1":
            return "true" if bool(payload) else "false"
        return str(int(0 if payload is None else payload))

    # --- runtime span extents across the internal call frames ---------------
    # A rank-N span is addressed inside a region, but only the public boundary
    # knows how large the real buffer is. Resolve each internal span parameter
    # back to the public value it came from by walking the argument bindings
    # the caller already computed -- the same binding the record-field identity
    # travels on -- then measure that buffer's axis once through the artifact's
    # existing extents vector. Nothing is inferred from names or positions.
    span_origin: dict[tuple[str, int], tuple[str, int]] = {}
    internal_callers: dict[str, set[str]] = {}
    for caller_name in reachable:
        for block in module.functions[caller_name].blocks.values():
            for instruction in block.instrs:
                if instruction.op != "Call":
                    continue
                callee_name = str(instruction.attributes.get("callee") or "")
                callee_function = module.functions.get(callee_name)
                if callee_function is None:
                    continue
                internal_callers.setdefault(callee_name, set()).add(caller_name)
                if len(callee_function.args) != len(instruction.args):
                    continue
                for fed, formal in zip(instruction.args, callee_function.args):
                    span_origin[(callee_name, int(formal.id))] = (
                        caller_name, int(fed.id),
                    )

    def public_span_value(name: str, value_id: int) -> int | None:
        """The root's own value id for a span reached through call frames."""

        seen: set[tuple[str, int]] = set()
        current = (str(name), int(value_id))
        while current[0] != function_name:
            if current in seen:
                return None
            seen.add(current)
            origin = span_origin.get(current)
            if origin is None:
                return None
            current = origin
        return current[1]

    # Functions that address a span across more than one axis need the extents
    # vector, and so does every function that calls one.
    extent_users: set[str] = {
        name for name in reachable
        for block in module.functions[name].blocks.values()
        for instruction in block.instrs
        if instruction.op in {"GetElementPtr", "getelementptr"}
        and len(instruction.args) > 2
    }
    # An elementwise operation whose result is a span expands into a loop
    # over that span's elements, and the trip count is measured from the
    # extents vector. Such a function needs %extents even when it never
    # addresses an element itself -- region_4 does exactly this: four
    # whole-array assignments and not one GetElementPtr.
    extent_users.update(
        name for name in reachable
        for block in module.functions[name].blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
        and _declared_span_rank(instruction.res) > 0
        and scalar_likeness(str(instruction.op)) is not None
        and str(instruction.op) not in PREDICATE_OPERATIONS
    )
    growing = True
    while growing:
        growing = False
        for callee_name in tuple(extent_users):
            for caller_name in internal_callers.get(callee_name, ()):  # noqa: B007
                if caller_name in reachable and caller_name not in extent_users:
                    extent_users.add(caller_name)
                    growing = True

    module_extent_order: list[tuple[int, str, int | None]] = []
    module_extent_slots: dict[tuple[int, str, int | None], int] = {}

    def module_extent_slot(public_id: int, axis: int) -> int:
        key = (int(public_id), "dim", int(axis))
        slot = module_extent_slots.get(key)
        if slot is None:
            slot = len(module_extent_order)
            module_extent_order.append(key)
            module_extent_slots[key] = slot
        return slot

    # (caller internal symbol or None for the wrapper, callee internal
    # symbol, actual pointer strings in position order). Collected at
    # every internal call emission; consumed by _annotate_noalias after
    # all bodies exist -- pointer STRINGS are the storage authority here,
    # value ids are not (a carried value's initial and updated ids share
    # one buffer across a call boundary; marking on ids miscompiled a
    # five-element accumulation to its last element).
    internal_call_records: list[tuple[str | None, str, tuple[str, ...]]] = []
    emitted_functions: list[str] = []
    for name in reachable:
        function = module.functions[name]
        outputs = function_outputs[name]
        parameters = [
            *(f"ptr %arg.{index}" for index in range(len(function.args))),
            *(f"ptr %out.{index}" for index in range(len(outputs))),
            *(("ptr %extents",) if name in extent_users else ()),
        ]
        body: list[str] = []
        entry_allocas: list[str] = []
        pointers: dict[int, str] = {
            int(value.id): f"%arg.{index}"
            for index, value in enumerate(function.args)
        }
        aggregate_members: dict[int, dict[int, str]] = {}
        address_members: dict[int, str] = {}
        address_slots: dict[int, str] = {}
        span_addresses: dict[int, str] = {}
        span_address_types: dict[int, str] = {}
        allocated: set[int] = set()
        output_pointer = {
            int(value.id): f"%out.{index}"
            for index, value in enumerate(outputs)
        }
        # A watched value whose storage is a Phi register cannot be copied at
        # the return -- the register does not dominate it. Give it a slot in
        # the entry frame (which dominates everything) and copy the phi's
        # CONTENT into that slot each time the phi executes. The slot then
        # holds whatever the phi last selected, which for a loop-carried
        # accumulator is exactly its converged value at loop exit.
        watch_shadows: dict[int, str] = {}
        if name == function_name:
            for watched_id in sorted(phi_watch_ids):
                watched_value = values_by_function[name].get(watched_id)
                if watched_value is None:
                    continue
                llvm_type = _value_llvm_type(watched_value)
                count = _value_element_count(watched_value)
                slot = f"%watch.{watched_id}"
                entry_allocas.append(
                    f"  {slot} = alloca {llvm_type}, i64 {count}, align 8"
                )
                watch_shadows[watched_id] = slot
        # The ring and its counter are ordinary outputs, so they already have
        # %out slots; bind them by id here so the phi capture can address them.
        history_pointers: dict[int, tuple[str, str, int]] = {}
        if name == function_name and history_slots:
            depth = max(1, int(history))
            for watched_id, (ring_id, count_id) in history_slots.items():
                ring_pointer = output_pointer.get(ring_id)
                count_pointer = output_pointer.get(count_id)
                if ring_pointer is None or count_pointer is None:
                    continue
                history_pointers[watched_id] = (
                    ring_pointer, count_pointer, depth,
                )

        # Which ordinary (non-phi) history values each block defines. Phis are
        # excluded because the phi path already rings them at the phi itself.
        history_by_block: dict[str, list[int]] = {}
        if history_pointers:
            for block_name_, block_ in function.blocks.items():
                for instruction_ in block_.instrs:
                    if instruction_.res is None:
                        continue
                    if str(instruction_.op) in {"Phi", "phi"}:
                        continue
                    result_id_ = int(instruction_.res.id)
                    if result_id_ in history_pointers:
                        history_by_block.setdefault(
                            str(block_name_), [],
                        ).append(result_id_)

        def pointer(value: _Any) -> str:
            value_id = int(value.id)
            known = pointers.get(value_id)
            if known is not None:
                return known
            known = output_pointer.get(value_id)
            if known is not None:
                pointers[value_id] = known
                return known
            llvm_type = _value_llvm_type(value)
            count = _value_element_count(value)
            register = f"%value.{value_id}"
            if value_id not in allocated:
                entry_allocas.append(
                    f"  {register} = alloca {llvm_type}, i64 {count}, align 8"
                )
                allocated.add(value_id)
            pointers[value_id] = register
            return register

        # Same-block register reuse, keyed by SLOT NAME, not value id. The
        # storage design composes operations in place over pre-allotted pool
        # slots, so two value ids may legitimately view one slot (a carried
        # value read under one id and rewritten under another); the slot name
        # is the one honest currency for "same storage". A slot's key holds
        # the register (or literal) known to be its current content; a store
        # to that slot replaces its key, so an in-place write is coherent by
        # construction and every OTHER key stays valid. The stores themselves
        # always still happen -- slot contents, the ABI, watches and the
        # in-place discipline are byte-identical; only redundant loads
        # evaporate. Cacheable keys are the static scalar homes (%value.N,
        # %out.N, %arg.N); loads through phi-selected pointers or span/array
        # addresses are never cached (dynamic targets). All keys drop at
        # every scheduled block label (no cross-block dominance question can
        # arise), at every call (a callee may write caller storage through
        # the pointer ABI), and at every scatter Store (array writes stay
        # ordered against every read).
        # The "prove" work contract turns reuse off wholesale: every read
        # comes from real storage, which is the shape two backends are
        # diffed over value by value.
        from .work_contract import active_contract

        _reuse_registers = active_contract().register_reuse
        register_cache: dict[str, tuple[str, str]] = {}
        _CACHEABLE_SLOT = ("%value.", "%out.", "%arg.")

        def load_as(value: _Any, wanted: str, tag: str) -> str:
            slot_home = pointer(value)
            cached = (
                register_cache.get(slot_home) if _reuse_registers else None
            )
            if cached is not None:
                loaded, source_type = cached
            else:
                source_type = _value_llvm_type(value)
                loaded = f"%load.{tag}"
                body.append(
                    f"  {loaded} = load {source_type}, ptr {slot_home}, "
                    f"align {_align(source_type)}"
                )
                if _reuse_registers and slot_home.startswith(_CACHEABLE_SLOT):
                    register_cache[slot_home] = (loaded, source_type)
            if source_type == wanted:
                return loaded
            converted = f"%convert.{tag}"
            if wanted == "double" and source_type in {"i1", "i32", "i64"}:
                opcode = "uitofp" if source_type == "i1" else "sitofp"
                body.append(
                    f"  {converted} = {opcode} {source_type} {loaded} to double"
                )
                return converted
            if wanted in {"i32", "i64"} and source_type == "double":
                body.append(
                    f"  {converted} = fptosi double {loaded} to {wanted}"
                )
                return converted
            # Integer width coercion.  An i32 induction against an int64 ABI
            # cell (a keyed mapping's length slot) previously emitted a mixed
            # icmp the LLVM verifier rejects.
            if wanted == "i64" and source_type == "i32":
                body.append(f"  {converted} = sext i32 {loaded} to i64")
                return converted
            if wanted == "i32" and source_type == "i64":
                body.append(f"  {converted} = trunc i64 {loaded} to i32")
                return converted
            if wanted == "i1":
                zero = "0.0" if source_type == "double" else "0"
                comparison = "fcmp one" if source_type == "double" else "icmp ne"
                body.append(
                    f"  {converted} = {comparison} {source_type} {loaded}, {zero}"
                )
                return converted
            return loaded

        scheduled_instructions = [
            (block_name, instruction)
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
        ]
        # Phi registers must be visible to every block regardless of render
        # order: a loop exit may render before a nested header (dict order),
        # and an outer phi's latch operand references the inner phi.  Left
        # unregistered, pointer() fabricated a dead alloca for the
        # not-yet-rendered phi and cached it -- the outer phi then read a
        # cell nothing ever wrote, which zeroed every nested carried
        # reduction.  LLVM accepts forward references; pre-register them all.
        for _block_name, instruction in scheduled_instructions:
            if (
                str(instruction.op) in {"Phi", "phi"}
                and instruction.res is not None
            ):
                pointers[int(instruction.res.id)] = (
                    f"%phi.{int(instruction.res.id)}"
                )
        # A dead unpack pair -- GEP(aggregate_index) plus the Load of its
        # address whose result nothing consumes -- reads a slot the callee
        # never publishes (only live positions have out cells).  Skip both;
        # a CONSUMED projection without a member stays a named shortfall.
        function_use_counts: dict[int, int] = {}
        for _pre_block, pre_instruction in scheduled_instructions:
            for argument in pre_instruction.args:
                function_use_counts[int(argument.id)] = (
                    function_use_counts.get(int(argument.id), 0) + 1
                )
        dead_unpack_results: set[int] = set()
        address_consumers: dict[int, list[_Any]] = {}
        for _pre_block, pre_instruction in scheduled_instructions:
            if (
                str(pre_instruction.op) in {"Load", "load"}
                and pre_instruction.args
                and pre_instruction.res is not None
            ):
                address_consumers.setdefault(
                    int(pre_instruction.args[0].id), []
                ).append(pre_instruction)
        for _pre_block, pre_instruction in scheduled_instructions:
            if (
                str(pre_instruction.op) in {"GetElementPtr", "getelementptr"}
                and pre_instruction.res is not None
                and pre_instruction.attributes.get("aggregate_index")
                is not None
            ):
                gep_result = int(pre_instruction.res.id)
                paired_loads = address_consumers.get(gep_result, ())
                if all(
                    function_use_counts.get(int(load.res.id), 0) == 0
                    for load in paired_loads
                    if load.res is not None
                ) and function_use_counts.get(gep_result, 0) == len(
                    tuple(paired_loads)
                ):
                    dead_unpack_results.add(gep_result)
                    dead_unpack_results.update(
                        int(load.res.id)
                        for load in paired_loads
                        if load.res is not None
                    )
        projection_values: dict[int, dict[int, _Any]] = {}
        projection_addresses: dict[int, tuple[int, int]] = {}
        constant_values = {
            int(projected.res.id): (
                projected.attributes.get("constant")
                if projected.attributes.get("constant") is not None
                else projected.attributes.get("value")
            )
            for _block_name, projected in scheduled_instructions
            if projected.op == "Const" and projected.res is not None
        }
        for _block_name, projected in scheduled_instructions:
            if (
                projected.op in {"GetElementPtr", "getelementptr"}
                and projected.res is not None
                and projected.args
                and projected.attributes.get("aggregate_index") is not None
            ):
                projection_addresses[int(projected.res.id)] = (
                    int(projected.args[0].id),
                    int(projected.attributes["aggregate_index"]),
                )
            elif (
                projected.op in {"Load", "load"}
                and projected.res is not None
                and projected.args
                and int(projected.args[0].id) in projection_addresses
            ):
                aggregate_id, position = projection_addresses[
                    int(projected.args[0].id)
                ]
                projection_values.setdefault(aggregate_id, {})[position] = (
                    projected.res
                )

        def emit_return_values(
            returned_values: tuple = (),
        ) -> None:
            for output_index, output in enumerate(outputs):
                destination = f"%out.{output_index}"
                # The Ret instruction's own arguments are the authoritative
                # publication objects: a carried reduction returns its PHI,
                # whose id differs from the declared output id (the port /
                # field slot).  Publishing by declared id alone read the
                # port's unwritten cell and returned zero for every carried
                # maximum.
                source = None
                if output_index < len(returned_values):
                    source = pointers.get(
                        int(returned_values[output_index].id)
                    )
                if source is None:
                    # A watch shadow, when one exists, is the ONLY readable
                    # storage for that value at the return; prefer it over
                    # the phi register it mirrors.
                    source = watch_shadows.get(int(output.id))
                if source is None:
                    source = pointers.get(int(output.id))
                if source is None:
                    tensor_table = getattr(module, "tensor_tables", {}).get(name)
                    descriptor = (
                        tensor_table.by_id(int(output.id))
                        if tensor_table is not None else None
                    )
                    if descriptor is not None:
                        source = pointers.get(int(descriptor.data_value_id))
                if source is None or source == destination:
                    continue
                llvm_type = _value_llvm_type(output)
                count = _value_element_count(output)
                if count == 1:
                    loaded = f"%return.load.{output_index}.{len(body)}"
                    body.append(
                        f"  {loaded} = load {llvm_type}, ptr {source}, "
                        f"align {_align(llvm_type)}"
                    )
                    body.append(
                        f"  store {llvm_type} {loaded}, ptr {destination}, "
                        f"align {_align(llvm_type)}"
                    )
                else:
                    body.append(
                        "  call void @llvm.memcpy.p0.p0.i64("
                        f"ptr {destination}, ptr {source}, "
                        f"i64 {count * _LLVM_TYPE_BYTES[llvm_type]}, i1 false)"
                    )
        active_block: str | None = None
        # An elementwise op over a span expands into its own loop, which ends
        # the SSA block early and continues in a new LLVM label. A later phi
        # naming this SSA block as a predecessor must name the label control
        # actually arrives from, so record where each SSA block currently
        # exits. Identity until something splits the block.
        block_exit_label: dict[str, str] = {}
        emitted_return = False
        # Watch-shadow copies wait here until the phi group they follow has
        # been emitted in full; see the Phi branch below.
        pending_shadow: list[str] = []

        def capture_block_history(block_name: str, tag: str) -> None:
            """Ring every ordinary history value this block defines.

            Emitted at the terminator, so each execution of the block
            contributes exactly one sample -- the value's final content for
            that pass. Reading the slot rather than a register keeps this
            free of dominance concerns: the slot is in the entry frame.
            """
            for ringed_id in history_by_block.get(block_name, ()):
                ringed = values_by_function[name].get(ringed_id)
                if ringed is None:
                    continue
                ring_pointer, count_pointer, depth = history_pointers[ringed_id]
                ringed_type = _value_llvm_type(ringed)
                mark = f"{tag}.{ringed_id}"
                body.extend((
                    f"  %hist.val.{mark} = load {ringed_type}, ptr "
                    f"{pointer(ringed)}, align 8",
                    f"  %hist.seen.{mark} = load i64, ptr {count_pointer}, "
                    "align 8",
                    f"  %hist.slot.{mark} = urem i64 %hist.seen.{mark}, {depth}",
                    f"  %hist.at.{mark} = getelementptr {ringed_type}, ptr "
                    f"{ring_pointer}, i64 %hist.slot.{mark}",
                    f"  store {ringed_type} %hist.val.{mark}, ptr "
                    f"%hist.at.{mark}, align 8",
                    f"  %hist.next.{mark} = add i64 %hist.seen.{mark}, 1",
                    f"  store i64 %hist.next.{mark}, ptr {count_pointer}, "
                    "align 8",
                ))

        for instruction_index, (block_name, instruction) in enumerate(
            scheduled_instructions
        ):
            if block_name != active_block:
                if pending_shadow:
                    body.extend(pending_shadow)
                    pending_shadow.clear()
                body.append(f"{block_name}:")
                active_block = block_name
                block_exit_label[block_name] = block_name
                register_cache.clear()
            operation = str(instruction.op)
            if pending_shadow and operation not in {"Phi", "phi"}:
                body.extend(pending_shadow)
                pending_shadow.clear()
            result = instruction.res
            result_id = int(result.id) if result is not None else None
            tag = f"{instruction_index}.{result_id if result_id is not None else 'v'}"

            if operation in {"Const", "StaticRef"} and result is not None:
                if operation == "StaticRef":
                    payload = int(instruction.attributes["reference_handle"])
                else:
                    payload = instruction.attributes.get("constant")
                if payload is None and "values" in instruction.attributes:
                    payload = instruction.attributes.get("values")
                if payload is None and "value" in instruction.attributes:
                    payload = instruction.attributes.get("value")
                target = pointer(result)
                if isinstance(payload, (tuple, list)):
                    for index, item in enumerate(payload):
                        slot = f"%const.slot.{tag}.{index}"
                        body.append(
                            f"  {slot} = getelementptr i32, ptr {target}, i64 {index}"
                        )
                        body.append(f"  store i32 {int(item)}, ptr {slot}, align 4")
                else:
                    llvm_type = _value_llvm_type(result)
                    body.append(
                        f"  store {llvm_type} {literal(payload, llvm_type)}, ptr {target}, "
                        f"align {_align(llvm_type)}"
                    )
                    if _reuse_registers and target.startswith(_CACHEABLE_SLOT):
                        register_cache[target] = (
                            literal(payload, llvm_type), llvm_type,
                        )
                    else:
                        register_cache.clear()
                continue

            if operation in {"Phi", "phi"} and result is not None:
                incoming_blocks = tuple(
                    instruction.attributes.get("incoming_blocks") or ()
                )
                incoming = tuple(instruction.attributes.get("incoming") or ())
                if incoming:
                    incoming_blocks = tuple(str(item[0]) for item in incoming)
                    incoming_values = tuple(item[1] for item in incoming)
                else:
                    incoming_values = tuple(instruction.args)
                if len(incoming_blocks) != len(incoming_values):
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation,
                        "phi incoming blocks do not match incoming values",
                    ))
                    continue
                register = f"%phi.{result_id}"
                body.append(
                    f"  {register} = phi ptr "
                    + ", ".join(
                        f"[ {pointer(value)}, "
                        f"%{block_exit_label.get(predecessor, predecessor)} ]"
                        for predecessor, value in zip(
                            incoming_blocks, incoming_values
                        )
                    )
                )
                pointers[result_id] = register
                shadow = watch_shadows.get(result_id)
                if shadow is not None:
                    # Diagnostic copy only: reads what the phi already
                    # selected and writes a slot nothing else observes.
                    # DEFERRED, not emitted here: LLVM requires every phi in
                    # a block to be grouped at its top, and a block with two
                    # carried values has two phis. These flush at the first
                    # non-phi instruction of the same block.
                    shadow_type = _value_llvm_type(result)
                    captured = f"%watch.load.{tag}"
                    pending_shadow.append(
                        f"  {captured} = load {shadow_type}, ptr {register}, "
                        "align 8"
                    )
                    pending_shadow.append(
                        f"  store {shadow_type} {captured}, ptr {shadow}, "
                        "align 8"
                    )
                    ring = history_pointers.get(result_id)
                    if ring is not None:
                        ring_pointer, count_pointer, depth = ring
                        seen = f"%watch.seen.{tag}"
                        slot_index = f"%watch.slot.{tag}"
                        slot_address = f"%watch.at.{tag}"
                        advanced = f"%watch.next.{tag}"
                        pending_shadow.extend((
                            f"  {seen} = load i64, ptr {count_pointer}, "
                            "align 8",
                            # urem, not a bounds branch: straight-line code
                            # cannot run off the ring and cannot disturb the
                            # block structure the phi group depends on.
                            f"  {slot_index} = urem i64 {seen}, {depth}",
                            f"  {slot_address} = getelementptr {shadow_type}, "
                            f"ptr {ring_pointer}, i64 {slot_index}",
                            f"  store {shadow_type} {captured}, ptr "
                            f"{slot_address}, align 8",
                            f"  {advanced} = add i64 {seen}, 1",
                            f"  store i64 {advanced}, ptr {count_pointer}, "
                            "align 8",
                        ))
                continue

            if operation in {"Br", "br"}:
                target = str(instruction.attributes.get("target") or "")
                if target not in function.blocks:
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation, f"unknown branch target {target!r}",
                    ))
                    continue
                capture_block_history(block_name, tag)
                body.append(f"  br label %{target}")
                continue

            if operation in {"CondBr", "condbr"} and instruction.args:
                true_target = str(
                    instruction.attributes.get("true")
                    or instruction.attributes.get("true_target")
                    or ""
                )
                false_target = str(
                    instruction.attributes.get("false")
                    or instruction.attributes.get("false_target")
                    or ""
                )
                if true_target not in function.blocks or false_target not in function.blocks:
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation, "conditional branch has an unknown target",
                    ))
                    continue
                condition = load_as(instruction.args[0], "i1", f"{tag}.condition")
                capture_block_history(block_name, tag)
                body.append(
                    f"  br i1 {condition}, label %{true_target}, label %{false_target}"
                )
                continue

            if operation == "extent" and result is not None and instruction.args:
                source = instruction.args[0]
                shape = tuple(getattr(source, "shape", ()) or ())
                try:
                    concrete_shape = tuple(int(extent) for extent in shape)
                except (TypeError, ValueError):
                    shortfalls.append(LLVMEmissionShortfall(
                        name,
                        operation,
                        "internal dynamic extent has no static repository "
                        "shape contract",
                    ))
                    continue
                kind = str(instruction.attributes.get("extent_kind") or "")
                target = pointer(result)
                if kind == "shape":
                    for axis, extent in enumerate(concrete_shape):
                        slot = f"%extent.shape.slot.{tag}.{axis}"
                        body.append(
                            f"  {slot} = getelementptr i32, ptr {target}, "
                            f"i64 {axis}"
                        )
                        body.append(
                            f"  store i32 {extent}, ptr {slot}, align 4"
                        )
                else:
                    if kind in {"element_count", "numel"}:
                        from math import prod as _shape_product

                        extent_value = int(_shape_product(concrete_shape))
                    elif kind == "rank":
                        extent_value = len(concrete_shape)
                    elif kind == "dim":
                        axis = int(instruction.attributes.get(
                            "axis", instruction.attributes.get("dim", 0)
                        ))
                        extent_value = concrete_shape[axis]
                    else:
                        shortfalls.append(LLVMEmissionShortfall(
                            name, operation, f"unknown extent kind {kind!r}",
                        ))
                        continue
                    llvm_type = _value_llvm_type(result)
                    body.append(
                        f"  store {llvm_type} "
                        f"{literal(extent_value, llvm_type)}, ptr {target}, "
                        "align 8"
                    )
                continue

            if operation in {"Ret", "ret", "Return", "return"}:
                emit_return_values(tuple(instruction.args))
                body.append("  ret void")
                emitted_return = True
                continue

            if operation in _SHAPE_ONLY and result is not None and instruction.args:
                source_pointer = pointer(instruction.args[0])
                destination = output_pointer.get(result_id)
                if destination is not None and destination != source_pointer:
                    body.append(
                        "  call void @llvm.memcpy.p0.p0.i64("
                        f"ptr {destination}, ptr {source_pointer}, "
                        f"i64 {_value_block_bytes(result)}, i1 false)"
                    )
                    pointers[result_id] = destination
                else:
                    pointers[result_id] = source_pointer
                continue

            if operation in {"zeros_like", "zeros"} and result is not None:
                # A zero fill is the whole emission. This arrives from the
                # BACKWARD side: the derivative of a step-like operation --
                # `floor`, `ceil`, `round`, any comparison -- is zero almost
                # everywhere, and the authored rule states that as
                # ``zeros_like(x)``. Without this the reverse of every such
                # operation stops with "no repository LLVM emission" while its
                # forward compiles perfectly, which is exactly the asymmetry
                # that made `floor` uncompilable backwards.
                destination = output_pointer.get(result_id) or pointer(result)
                body.append(
                    "  call void @llvm.memset.p0.i64("
                    f"ptr {destination}, i8 0, "
                    f"i64 {_value_block_bytes(result)}, i1 false)"
                )
                pointers[result_id] = destination
                continue

            if operation in {"GetElementPtr", "getelementptr"} and result is not None:
                base_id = int(instruction.args[0].id) if instruction.args else -1
                members = aggregate_members.get(base_id)
                position = instruction.attributes.get("aggregate_index")
                if position is None and len(instruction.args) > 1:
                    position = constant_values.get(int(instruction.args[1].id))
                if members is not None and position is not None and int(position) in members:
                    address_members[result_id] = members[int(position)]
                    continue
                if members is not None and position is not None:
                    # The projection map is the whole truth about this
                    # aggregate.  A projection it does not answer is a
                    # compile-time contradiction and must refuse by name --
                    # the raw-slot fallback read past the table and returned
                    # different garbage every run.
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation,
                        f"aggregate projection {int(position)} of "
                        f"%t{base_id} has no compile-time member "
                        f"(known: {sorted(members)})",
                    ))
                    continue
                if (
                    position is not None
                    and instruction.args
                    and len(instruction.args) <= 2
                    and int(instruction.args[0].id) in pointers
                ):
                    slot = f"%aggregate.slot.{tag}"
                    body.append(
                        f"  {slot} = getelementptr ptr, ptr {pointer(instruction.args[0])}, i64 {int(position)}"
                    )
                    address_slots[result_id] = slot
                    continue
                # A single index is already the offset and touches no extent
                # slot, so it must not require membership in extent_users --
                # that set names functions with MULTI-axis addresses.  A
                # rank-1 span walk (a keyed lookup helper) is single-index.
                if len(instruction.args) > 1 and (
                    name in extent_users or len(instruction.args) == 2
                ):
                    base = instruction.args[0]
                    indices = [
                        load_as(argument, "i32", f"{tag}.{position}")
                        for position, argument in enumerate(
                            instruction.args[1:]
                        )
                    ]
                    declared_rank = int(
                        (getattr(base, "accounting", None) or {}).get(
                            "program_abi_rank"
                        ) or 0
                    )
                    static = tuple(base.shape or ())
                    rank = len(static) or declared_rank
                    public_id = public_span_value(name, int(base.id))
                    # A single index is already the offset; only a multi-axis
                    # address has to know the axis it is striding over.
                    if len(indices) == 1 or (
                        rank == len(indices)
                        and (
                            public_id is not None
                            # Vacuously-true all() over an empty static shape
                            # walked a declared-rank span into extent slots
                            # for a public origin that does not exist.
                            or (
                                len(static) >= len(indices)
                                and all(
                                    isinstance(item, int) for item in static
                                )
                            )
                        )
                    ):
                        offset = indices[0]
                        for axis, index in enumerate(indices[1:], start=1):
                            if axis < len(static) and isinstance(
                                static[axis], int
                            ):
                                stride = str(int(static[axis]))
                            else:
                                measured = module_extent_slot(public_id, axis)
                                address = f"%extent.addr.{tag}.{axis}"
                                register = f"%extent.{tag}.{axis}"
                                body.append(
                                    f"  {address} = getelementptr i32, "
                                    f"ptr %extents, i64 {measured}"
                                )
                                body.append(
                                    f"  {register} = load i32, ptr {address}, "
                                    "align 4"
                                )
                                stride = register
                            scaled = f"%address.{tag}.scale.{axis}"
                            body.append(
                                f"  {scaled} = mul i32 {offset}, {stride}"
                            )
                            summed = f"%address.{tag}.sum.{axis}"
                            body.append(
                                f"  {summed} = add i32 {scaled}, {index}"
                            )
                            offset = summed
                        element_type = _value_llvm_type(base)
                        if element_type == "ptr":
                            element_type = "i64"
                        computed = f"%address.{tag}"
                        body.append(
                            f"  {computed} = getelementptr {element_type}, "
                            f"ptr {pointer(base)}, i32 {offset}"
                        )
                        # The addressed element *is* the storage; a later Load
                        # aliases it rather than copying, exactly as the
                        # aggregate-member path already does.
                        pointers[result_id] = computed
                        span_addresses[result_id] = computed
                        span_address_types[result_id] = element_type
                        continue
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation,
                        f"{len(indices)}-axis address for %t{int(base.id)} "
                        f"has rank {rank} and no public span origin",
                    ))
                    continue

            if operation in {"Load", "load"} and result is not None and instruction.args:
                addressed = span_addresses.get(int(instruction.args[0].id))
                if addressed is not None:
                    loaded_type = _value_llvm_type(result)
                    if loaded_type in {"double", "i32", "i1"}:
                        # Pin the loaded scalar to its own slot.
                        #
                        # Aliasing the result to the ADDRESS instead (what
                        # this did) makes the value mean "whatever that
                        # element holds when a use is reached" rather than
                        # "what it held here". Array addresses are
                        # deliberately never register-cached, so every use
                        # re-emitted its own load; with an intervening store
                        # to the same element, later uses read the NEW
                        # contents. Reading a pair, combining them, and
                        # writing both back -- every plane rotation, every
                        # in-place swap -- then silently computed with the
                        # already-overwritten operand, while emitting
                        # cleanly. See tests/test_llvm_inplace_store_aliasing.
                        #
                        # Restricted to scalars on purpose: an i64 may be a
                        # reference and a wider load may be an aggregate
                        # base that later addressing still needs to reach
                        # through, and both are handled below.
                        pinned = f"%pinned.load.{tag}"
                        body.append(
                            f"  {pinned} = load {loaded_type}, ptr "
                            f"{addressed}, align 8"
                        )
                        body.append(
                            f"  store {loaded_type} {pinned}, ptr "
                            f"{pointer(result)}, align 8"
                        )
                        continue
                    pointers[result_id] = addressed
                    continue
                member = address_members.get(int(instruction.args[0].id))
                if member is not None:
                    pointers[result_id] = member
                    continue
                slot = address_slots.get(int(instruction.args[0].id))
                if slot is not None:
                    if _value_llvm_type(result) == "i64":
                        loaded_value = f"%reference.load.{tag}"
                        body.append(
                            f"  {loaded_value} = load i64, ptr {slot}, align 8"
                        )
                        body.append(
                            f"  store i64 {loaded_value}, ptr {pointer(result)}, align 8"
                        )
                        continue
                    loaded_pointer = f"%aggregate.load.{tag}"
                    body.append(
                        f"  {loaded_pointer} = load ptr, ptr {slot}, align 8"
                    )
                    pointers[result_id] = loaded_pointer
                    continue

            if operation in {"Store", "store"} and len(instruction.args) == 2:
                source, address = instruction.args
                # A scatter writes through a computed address; every cached
                # register must yield to the ordering of real memory. The
                # source's own read below may still be served from cache
                # first -- reading precedes the write.
                destination = address_slots.get(int(address.id), pointer(address))
                # An indexed destination has the element type of its base
                # span, not necessarily the scalar expression stored into
                # it.  Python commonly writes an integer loop/index value to
                # a default float work array.  Loading/storing in the source
                # type writes only four bytes of an eight-byte element and is
                # both numerically wrong and ABI-corrupting.  Reconcile at
                # the store boundary exactly as scalar result cells do.
                destination_type = span_address_types.get(
                    int(address.id), _value_llvm_type(source),
                )
                loaded_value = load_as(
                    source, destination_type, f"store.{tag}",
                )
                body.append(
                    f"  store {destination_type} {loaded_value}, "
                    f"ptr {destination}, align 8"
                )
                register_cache.clear()
                continue

            callee = instruction.attributes.get("callee")
            if callee is not None:
                # A callee may write caller storage through the pointer ABI
                # (aliased IO is a design feature); registers read before the
                # call must not serve reads after it.
                register_cache.clear()
                symbol = str(callee)
                tensor_operation = instruction.attributes.get("tensor_operation")
                if (
                    instruction.res is not None
                    and tensor_operation is not None
                    and symbol in {
                        "binary_double", "binary_scalar_double", "unary_double",
                    }
                ):
                    # Repository SSA keeps tensor calls semantic: operands and
                    # result are SSA values, while element count, C opcode and
                    # scalar direction remain attributes/shape facts. Expand
                    # that boundary here instead of requiring precompile SSA
                    # to carry one backend's private C ABI as fake operands.
                    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
                        c_tensor_opcode,
                    )

                    opcode = c_tensor_opcode(str(tensor_operation))
                    result_count = _value_element_count(instruction.res)
                    if opcode is None:
                        shortfalls.append(LLVMEmissionShortfall(
                            name,
                            symbol,
                            f"no C tensor opcode for {tensor_operation!r}",
                        ))
                        continue
                    opcode_kind, opcode_value = opcode
                    destination = pointer(instruction.res)
                    if symbol == "unary_double":
                        if opcode_kind != "unary" or len(instruction.args) != 1:
                            shortfalls.append(LLVMEmissionShortfall(
                                name,
                                symbol,
                                "semantic unary call has an invalid operand layout",
                            ))
                            continue
                        source = pointer(instruction.args[0])
                        body.append(
                            f"  call void @unary_double(ptr {source}, "
                            f"ptr {destination}, i32 {result_count}, "
                            f"i32 {opcode_value})"
                        )
                        kernels_used.add(symbol)
                        continue
                    if opcode_kind != "binary":
                        shortfalls.append(LLVMEmissionShortfall(
                            name,
                            symbol,
                            "semantic binary call resolved to a non-binary opcode",
                        ))
                        continue
                    if symbol == "binary_double":
                        if len(instruction.args) != 2:
                            shortfalls.append(LLVMEmissionShortfall(
                                name,
                                symbol,
                                "semantic binary call requires two operands",
                            ))
                            continue
                        left = pointer(instruction.args[0])
                        right = pointer(instruction.args[1])
                        body.append(
                            f"  call void @binary_double(ptr {left}, ptr {right}, "
                            f"ptr {destination}, i32 {result_count}, "
                            f"i32 {opcode_value})"
                        )
                        kernels_used.add(symbol)
                        continue

                    reverse = bool(instruction.attributes.get("reverse", False))
                    scalar_literal = instruction.attributes.get("right_scalar")
                    array_argument = None
                    scalar_rendering = None
                    if scalar_literal is not None and len(instruction.args) == 1:
                        array_argument = instruction.args[0]
                        scalar_rendering = _double_literal(float(scalar_literal))
                    elif len(instruction.args) == 2:
                        left, right = instruction.args
                        left_count = _value_element_count(left)
                        right_count = _value_element_count(right)
                        if left_count == result_count and right_count == 1:
                            array_argument, scalar_argument = left, right
                        elif left_count == 1 and right_count == result_count:
                            array_argument, scalar_argument = right, left
                            reverse = not reverse
                        else:
                            scalar_argument = None
                        if array_argument is not None and scalar_argument is not None:
                            scalar_rendering = load_as(
                                scalar_argument,
                                "double",
                                f"semantic.scalar.{result_id}",
                            )
                    if array_argument is None or scalar_rendering is None:
                        shortfalls.append(LLVMEmissionShortfall(
                            name,
                            symbol,
                            "semantic scalar binary call has an invalid operand layout",
                        ))
                        continue
                    source = pointer(array_argument)
                    body.append(
                        f"  call void @binary_scalar_double(ptr {source}, "
                        f"double {scalar_rendering}, ptr {destination}, "
                        f"i32 {result_count}, i32 {opcode_value}, "
                        f"i32 {int(reverse)})"
                    )
                    kernels_used.add(symbol)
                    continue
                try:
                    returns, argument_types = _kernel_signature(symbol)
                except (KeyError, ValueError):
                    returns = ""
                    argument_types = ()
                if argument_types or returns:
                    kernels_used.add(symbol)
                    arguments = list(instruction.args)
                    output_argument = instruction.attributes.get("ssa_output_argument")
                    if output_argument is not None and len(arguments) < len(argument_types):
                        arguments.insert(int(output_argument), result)
                    rendered: list[str] = []
                    for position, (argument_type, argument) in enumerate(zip(argument_types, arguments)):
                        rendered.append(
                            f"ptr {pointer(argument)}"
                            if argument_type == "ptr"
                            else f"{argument_type} {load_as(argument, argument_type, f'{tag}.{position}')}"
                        )
                    if len(rendered) != len(argument_types):
                        shortfalls.append(LLVMEmissionShortfall(
                            name, symbol, "authored call arity does not match its definition",
                        ))
                        continue
                    joined = ", ".join(rendered)
                    if returns == "void":
                        body.append(f"  call void @{symbol}({joined})")
                    elif result is not None:
                        call_result = f"%call.{tag}"
                        body.append(f"  {call_result} = call {returns} @{symbol}({joined})")
                        body.append(
                            f"  store {returns} {call_result}, ptr {pointer(result)}, "
                            f"align {_align(returns)}"
                        )
                    continue

                if symbol in reachable:
                    callee_outputs = function_outputs[symbol]
                    declared_ids = tuple(map(
                        int, instruction.attributes.get("output_ids", ())
                    ))
                    selected = aggregate_positions.get(
                        (name, result_id), tuple(range(len(callee_outputs)))
                    )
                    projections = projection_values.get(result_id, {})
                    forwarded = forwarded_aggregate_calls.get(name)
                    if (
                        forwarded is not None
                        and int(forwarded[0]) == result_id
                        and str(forwarded[1]) == symbol
                    ):
                        # Forwarding maps by OUTPUT ID -- output_pointer is
                        # that map.  Positional %out.i handed the wrapper's
                        # 7 selected out params the callee's FIRST seven
                        # outputs (velocity_x into the wave slot) and wrote
                        # the rest past the parameter list.
                        result_ptrs = []
                        for output_index, value in enumerate(callee_outputs):
                            known = output_pointer.get(int(value.id))
                            if known is not None:
                                result_ptrs.append(known)
                                continue
                            llvm_type = _value_llvm_type(value)
                            count = _value_element_count(value)
                            temporary = (
                                f"%forward.output.{tag}.{output_index}"
                            )
                            body.append(
                                f"  {temporary} = alloca {llvm_type}, "
                                f"i64 {count}, align 8"
                            )
                            result_ptrs.append(temporary)
                    elif declared_ids:
                        if len(selected) != len(callee_outputs):
                            shortfalls.append(LLVMEmissionShortfall(
                                name,
                                symbol,
                                "aggregate positions "
                                f"{selected!r} do not match "
                                f"{len(callee_outputs)} callee outputs; "
                                f"declared={declared_ids!r}",
                            ))
                            continue
                        result_ptrs = []
                        for output_index, position in enumerate(selected):
                            projected = projections.get(position)
                            if projected is not None:
                                result_ptrs.append(pointer(projected))
                            else:
                                value = callee_outputs[output_index]
                                llvm_type = _value_llvm_type(value)
                                count = _value_element_count(value)
                                temporary = (
                                    f"%call.output.{tag}.{output_index}"
                                )
                                body.append(
                                    f"  {temporary} = alloca {llvm_type}, "
                                    f"i64 {count}, align 8"
                                )
                                result_ptrs.append(temporary)
                    else:
                        if len(callee_outputs) == 1 and result is not None:
                            result_ptrs = [pointer(result)]
                        else:
                            result_ptrs = []
                            for output_index, value in enumerate(callee_outputs):
                                llvm_type = _value_llvm_type(value)
                                count = _value_element_count(value)
                                temporary = f"%call.output.{tag}.{output_index}"
                                body.append(
                                    f"  {temporary} = alloca {llvm_type}, i64 {count}, align 8"
                                )
                                result_ptrs.append(temporary)
                    semantic_arguments = list(instruction.args)
                    expected_argument_count = len(
                        module.functions[symbol].args
                    )
                    if len(semantic_arguments) > expected_argument_count:
                        surplus = semantic_arguments[expected_argument_count:]
                        if all(
                            (value.accounting or {}).get(
                                "linked_call_frame_storage"
                            )
                            for value in surplus
                        ):
                            # Source linking can retain caller-owned storage
                            # placeholders after the callee has internalized
                            # those temporaries.  They are neither semantic
                            # operands nor callee outputs.  Passing them
                            # positionally shifts the real output pointers and
                            # silently writes results into the caller frame.
                            semantic_arguments = semantic_arguments[
                                :expected_argument_count
                            ]
                        else:
                            shortfalls.append(LLVMEmissionShortfall(
                                name,
                                symbol,
                                "repository call has surplus non-frame "
                                f"arguments: actual={len(semantic_arguments)}, "
                                f"expected={expected_argument_count}",
                            ))
                            continue
                    if len(semantic_arguments) != expected_argument_count:
                        shortfalls.append(LLVMEmissionShortfall(
                            name,
                            symbol,
                            "repository call argument count does not match "
                            f"callee ABI: actual={len(semantic_arguments)}, "
                            f"expected={expected_argument_count}",
                        ))
                        continue
                    required_positions = callee_aggregate_parameter_positions.get(
                        symbol, ()
                    )
                    if required_positions:
                        # The callee will GetElementPtr past this argument's
                        # first pointer-sized slot, so it must already be a
                        # materialized `%aggregate.NN` table (built earlier
                        # in THIS emission when its producer's aggregate
                        # result escapes whole -- see above). Anything else
                        # -- most dangerously, an ordinary scalar/tensor
                        # buffer that merely happens to be the semantically
                        # "right" argument -- has no second pointer-sized
                        # slot to read: the callee loads whatever bytes
                        # follow that one buffer's allocation, dereferences
                        # them as a pointer, and writes through it. That is
                        # heap corruption with no Python-visible cause, and
                        # it is exactly what a native access violation deep
                        # inside compiled code, with no informative
                        # traceback, looks like from here.
                        unresolved = [
                            (position, int(argument.id))
                            for position, argument in enumerate(semantic_arguments)
                            if position in required_positions
                            and not str(
                                pointers.get(int(argument.id), "")
                            ).startswith("%aggregate.")
                        ]
                        if unresolved:
                            shortfalls.append(LLVMEmissionShortfall(
                                name,
                                symbol,
                                "call argument position(s) "
                                f"{[position for position, _ in unresolved]!r} "
                                f"feed {symbol}, whose body unpacks that "
                                "parameter as a pointer table, but the "
                                "supplied value id(s) "
                                f"{[value_id for _, value_id in unresolved]!r} "
                                "were never materialized as one in this "
                                "caller -- the producer that should have "
                                "built this aggregate is missing or its "
                                "result was not linked through",
                            ))
                            continue
                    call_args = [
                        pointer(argument) for argument in semantic_arguments
                    ]
                    if len(result_ptrs) != len(callee_outputs):
                        shortfalls.append(LLVMEmissionShortfall(
                            name, symbol,
                            "live aggregate projections do not match callee "
                            f"outputs: pointers={len(result_ptrs)}, "
                            f"outputs={len(callee_outputs)}, "
                            f"selected={selected!r}, declared={declared_ids!r}",
                        ))
                        continue
                    internal_call_records.append((
                        internal_symbols.get(name),
                        internal_symbols[symbol],
                        (*call_args, *result_ptrs),
                    ))
                    body.append(
                        f"  call void @{internal_symbols[symbol]}("
                        + ", ".join(f"ptr {value}" for value in (
                            *call_args, *result_ptrs,
                            *(("%extents",) if symbol in extent_users else ()),
                        ))
                        + ")"
                    )
                    if result is not None:
                        if forwarded is not None and int(forwarded[0]) == result_id:
                            # The call already wrote the wrapper's public
                            # outputs.  Its aggregate result has no independent
                            # storage and must not be reconstructed from local
                            # pointers.
                            pass
                        elif len(callee_outputs) == 1 and not declared_ids:
                            pointers[result_id] = result_ptrs[0]
                        else:
                            aggregate_members[result_id] = {
                                original_position: result_ptrs[index]
                                for index, original_position in enumerate(selected)
                                if index < len(result_ptrs)
                            }
                            # An in/out-aliased output has no distinct out
                            # parameter -- the callee writes through the
                            # argument's own storage.  Its aggregate position
                            # must still resolve, or the unpack walks past
                            # the alloca and reads garbage (the fluid wave
                            # chain read exactly that).
                            declared_output_ids = tuple(
                                int(output_id) for output_id in (
                                    instruction.attributes.get("output_ids")
                                    or ()
                                )
                            )
                            for original_position, output_id in enumerate(
                                declared_output_ids
                            ):
                                known = pointers.get(int(output_id))
                                if known is not None:
                                    aggregate_members[result_id].setdefault(
                                        original_position, known,
                                    )
                            # The projection map above IS the aggregate: a
                            # runtime pointer table restates compile-time
                            # knowledge as memory indirection, and every
                            # projection that missed the map read past the
                            # table -- undefined behavior varying run to
                            # run.  Materialize the table ONLY when the
                            # aggregate value itself escapes whole.
                            if (name, result_id) in aggregate_escapes_whole:
                                aggregate = f"%aggregate.{tag}"
                                body.append(
                                    f"  {aggregate} = alloca ptr, i64 "
                                    f"{len(result_ptrs)}, align 8"
                                )
                                for output_index, result_pointer in enumerate(
                                    result_ptrs
                                ):
                                    slot = (
                                        f"%aggregate.output.slot.{tag}."
                                        f"{output_index}"
                                    )
                                    body.append(
                                        f"  {slot} = getelementptr ptr, ptr "
                                        f"{aggregate}, i64 {output_index}"
                                    )
                                    body.append(
                                        f"  store ptr {result_pointer}, ptr "
                                        f"{slot}, align 8"
                                    )
                                pointers[result_id] = aggregate
                    continue

            if operation in {"Cast", "CastLike", "cast_like"} and result is not None and instruction.args:
                result_type = _value_llvm_type(result)
                rendered = load_as(
                    instruction.args[0], result_type, f"cast.{tag}"
                )
                cast_target = pointer(result)
                body.append(
                    f"  store {result_type} {rendered}, ptr {cast_target}, "
                    f"align {_align(result_type)}"
                )
                if _reuse_registers and cast_target.startswith(_CACHEABLE_SLOT):
                    register_cache[cast_target] = (rendered, result_type)
                else:
                    register_cache.clear()
                continue

            if (
                operation in {"Select", "where"}
                and result is not None
                and len(instruction.args) == 3
            ):
                # Select(mask, when_true, when_false); the mask reaches i1
                # through the same truthiness conversion every other target
                # applies, so a numeric mask needs no separate opcode.
                result_type = _value_llvm_type(result)
                mask = load_as(instruction.args[0], "i1", f"select.{tag}")
                when_true = load_as(
                    instruction.args[1], result_type, f"select.{tag}.true",
                )
                when_false = load_as(
                    instruction.args[2], result_type, f"select.{tag}.false",
                )
                register = f"%select.{tag}"
                body.append(
                    f"  {register} = select i1 {mask}, "
                    f"{result_type} {when_true}, {result_type} {when_false}"
                )
                body.append(
                    f"  store {result_type} {register}, "
                    f"ptr {pointer(result)}, align 8"
                )
                continue

            template = scalar_likeness(operation)
            if (
                template is not None
                and result is not None
                and _declared_span_rank(result) > 0
                and operation not in PREDICATE_OPERATIONS
            ):
                # An elementwise operation whose RESULT is a span is an array
                # operation, and rendering it as one scalar load/op/store
                # writes exactly one element of it. Fortran gets this right
                # from the same SSA because its declaration carries the rank
                # (`t122 = t120 + 0.0` over `t122(e1, e2)` is a whole-array
                # assignment); the rank lives in accounting, not in `.shape`,
                # so a backend that sizes from `.shape` alone silently
                # degrades a whole-array assignment to its first cell. That
                # is what left `state.height = state.next_height + 0.0`
                # updating 1 of 16 cells.
                element_type = _value_llvm_type(result)
                total = _span_element_count(
                    name, result, tag, body,
                    public_span_value, module_extent_slot,
                )
                if total is None:
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation,
                        f"elementwise {operation} over span %t{result_id} "
                        "needs its extents, and no public origin declares "
                        "them; refusing to emit a one-element stand-in",
                    ))
                    continue
                entry_label = block_exit_label.get(
                    active_block or "", active_block or "entry",
                )
                head = f"ew.head.{tag}"
                loop_body = f"ew.body.{tag}"
                done = f"ew.done.{tag}"
                index = f"%ew.i.{tag}"
                nxt = f"%ew.next.{tag}"
                body.append(f"  br label %{head}")
                body.append(f"{head}:")
                body.append(
                    f"  {index} = phi i32 [ 0, %{entry_label} ], "
                    f"[ {nxt}, %{loop_body} ]"
                )
                body.append(
                    f"  %ew.more.{tag} = icmp slt i32 {index}, {total}"
                )
                body.append(
                    f"  br i1 %ew.more.{tag}, label %{loop_body}, "
                    f"label %{done}"
                )
                body.append(f"{loop_body}:")
                operands = []
                for position, argument in enumerate(instruction.args):
                    slot = f"%ew.op.{tag}.{position}"
                    if _declared_span_rank(argument) > 0:
                        body.append(
                            f"  {slot}.addr = getelementptr {element_type}, "
                            f"ptr {pointer(argument)}, i32 {index}"
                        )
                        body.append(
                            f"  {slot} = load {element_type}, "
                            f"ptr {slot}.addr, align 8"
                        )
                    else:
                        # A rank-0 operand broadcasts, which is what `+ 0.0`
                        # over an array means.
                        body.append(
                            f"  {slot} = load {element_type}, "
                            f"ptr {pointer(argument)}, align 8"
                        )
                    operands.append(slot)
                computed = f"%ew.val.{tag}"
                for rendered_line in template.format(
                    *operands, out=computed
                ).splitlines():
                    body.append(f"  {rendered_line}")
                body.append(
                    f"  %ew.dst.{tag} = getelementptr {element_type}, "
                    f"ptr {pointer(result)}, i32 {index}"
                )
                body.append(
                    f"  store {element_type} {computed}, "
                    f"ptr %ew.dst.{tag}, align 8"
                )
                body.append(f"  {nxt} = add i32 {index}, 1")
                body.append(f"  br label %{head}")
                body.append(f"{done}:")
                if active_block is not None:
                    block_exit_label[active_block] = done
                continue

            if template is not None and result is not None:
                result_type = _value_llvm_type(result)
                operand_argument_types = [
                    _value_llvm_type(argument)
                    for argument in instruction.args
                ]
                # Promote, never demote. Taking the domain from args[0] alone
                # made a mixed operation run in whatever domain its FIRST
                # operand happened to have, so `Mul(int -1, float64 x)` --
                # how the lowering spells negation -- ran as an integer
                # multiply and `load_as` reached it by fptosi. Every
                # fractional operand truncated to zero, which silently
                # deleted whole terms from an expression rather than failing:
                # the compiled fluid step ignored six of its inputs and
                # never saw a tracer go negative.
                #
                # If any operand is floating point the operation is floating
                # point. That is the ordinary promotion rule, and it is the
                # safe direction: widening an integer operand is exact, while
                # narrowing a float discards its value. Where the DECLARED
                # result is integral the reconciliation below still converts
                # the result back.
                operand_type = (
                    "double" if "double" in operand_argument_types
                    else operand_argument_types[0]
                    if operand_argument_types else result_type
                )
                # The logical templates are spelled over i1; an operand that
                # arrives in its double storage type must be coerced to a
                # boolean, not passed through in whatever width it was
                # stored (`or i1 <double>` fails LLVM verification).
                if operation in {"LAnd", "LOr", "LNot"}:
                    operand_type = "i1"
                operands = [
                    load_as(argument, operand_type, f"{tag}.{position}")
                    for position, argument in enumerate(instruction.args)
                ]
                register = f"%scalar.{tag}"
                integer_cast = {
                    "SiToFp": "sitofp",
                    "UiToFp": "uitofp",
                    "SExt": "sext",
                    "ZExt": "zext",
                }
                float_cast = {"FpToSi": "fptosi", "FpToUi": "fptoui"}
                if operation in integer_cast and len(operands) == 1:
                    body.append(
                        f"  {register} = {integer_cast[operation]} "
                        f"{operand_type} {operands[0]} to {result_type}"
                    )
                elif operation in float_cast and len(operands) == 1:
                    body.append(
                        f"  {register} = {float_cast[operation]} "
                        f"{operand_type} {operands[0]} to {result_type}"
                    )
                elif operand_type in {"i1", "i32", "i64"}:
                    integer = integer_scalar_lines(
                        operation, operand_type, operands, register,
                    )
                    if integer is None:
                        shortfalls.append(LLVMEmissionShortfall(
                            name, operation,
                            f"integer scalar operation has no LLVM emission for {operand_type}",
                        ))
                        continue
                    integer_lines, result_type = integer
                    body.extend(f"  {line}" for line in integer_lines)
                else:
                    for rendered_line in template.format(
                        *operands, out=register
                    ).splitlines():
                        body.append(f"  {rendered_line}")
                    # Scalar likeness templates operate in the promoted
                    # operand domain.  Keep that actual register type until
                    # the declared-result reconciliation below.  Retaining
                    # the result's declared integer type here emitted, for
                    # example, ``store i32 %double_register`` when an integer
                    # compiler index was gated by a floating predicate.
                    result_type = (
                        "i1" if operation in PREDICATE_OPERATIONS
                        else operand_type
                    )
                # The cell was alloca'd with the value's DECLARED type; a
                # narrower computed result stored raw left garbage bytes that
                # a later declared-type load reinterpreted (an i32 mul result
                # read back as double turned row*4 into a denormal ~0).
                declared_type = _value_llvm_type(result)
                if declared_type != result_type:
                    converted = f"%declared.{tag}"
                    if declared_type == "double" and result_type in {
                        "i32", "i64",
                    }:
                        body.append(
                            f"  {converted} = sitofp {result_type} "
                            f"{register} to double"
                        )
                        register, result_type = converted, "double"
                    elif declared_type == "double" and result_type == "i1":
                        body.append(
                            f"  {converted} = uitofp i1 {register} to double"
                        )
                        register, result_type = converted, "double"
                    elif declared_type in {"i32", "i64"} and result_type == (
                        "double"
                    ):
                        body.append(
                            f"  {converted} = fptosi double {register} "
                            f"to {declared_type}"
                        )
                        register, result_type = converted, declared_type
                    elif (
                        declared_type in {"i32", "i64"}
                        and result_type in {"i32", "i64"}
                    ):
                        opcode_word = (
                            "sext" if declared_type == "i64" else "trunc"
                        )
                        body.append(
                            f"  {converted} = {opcode_word} {result_type} "
                            f"{register} to {declared_type}"
                        )
                        register, result_type = converted, declared_type
                scalar_target = pointer(result)
                body.append(
                    f"  store {result_type} {register}, ptr {scalar_target}, "
                    f"align {_align(result_type)}"
                )
                if _reuse_registers and scalar_target.startswith(_CACHEABLE_SLOT):
                    register_cache[scalar_target] = (register, result_type)
                else:
                    register_cache.clear()
                continue

            if operation in {"Deploy", "Join"}:
                # A deployment boundary describes scheduling around the
                # numerical program, not a computation inside it. One native
                # module executes that program serially, so the marker has no
                # instruction of its own; it stays as a comment so the
                # structural boundary remains visible in the emitted IR. This
                # matches the Fortran lane exactly.
                body.append(f"  ; {operation} deployment boundary")
                continue

            # Descriptor getters can be present as dead planned outputs after
            # call specialization. They are omitted only when no selected ABI
            # result or live instruction consumes them.
            if operation == "getattr" and result is not None and result_id not in output_pointer:
                continue
            shortfalls.append(LLVMEmissionShortfall(
                name, operation, "operation has no repository LLVM emission",
            ))

        if not emitted_return:
            emit_return_values()
            body.append("  ret void")
        if not any(line.endswith(":") for line in body):
            body.insert(0, "entry:")
        entry_label_index = next(
            (index for index, line in enumerate(body) if line.endswith(":")),
            0,
        )
        body[entry_label_index + 1:entry_label_index + 1] = entry_allocas
        emitted_functions.append("\n".join((
            f"define void @{internal_symbols[name]}({', '.join(parameters)}) {{",
            *body,
            "}",
        )))

    root = module.functions[function_name]
    root_outputs = function_outputs[function_name]
    # Storage introduced while linking a nested repository call belongs to
    # the root function's native frame, not to the authored program ABI.  It
    # remains an ordinary pointer argument of the internal root function so
    # call-frame linkage stays uniform, but the public wrapper owns and
    # allocates it locally.
    root_public_args = tuple(
        value for value in root.args
        if not (value.accounting or {}).get("linked_call_frame_storage")
    )
    root_internal_storage = tuple(
        value for value in root.args
        if (value.accounting or {}).get("linked_call_frame_storage")
    )
    public_values = [*root_public_args, *root_outputs]
    buffer_order: list[int] = []
    buffer_shapes: list[tuple[_Any, ...]] = []
    buffer_dtypes: list[str] = []
    public_pointer: dict[int, str] = {}
    wrapper: list[str] = ["entry:"]
    for value in public_values:
        value_id = int(value.id)
        if value_id in public_pointer:
            continue
        slot = len(buffer_order)
        buffer_order.append(value_id)
        buffer_shapes.append(tuple(value.shape or ()))
        buffer_dtypes.append(_value_llvm_type(value))
        address = f"%public.addr.{slot}"
        loaded = f"%public.{slot}"
        wrapper.append(f"  {address} = getelementptr ptr, ptr %buffers, i64 {slot}")
        wrapper.append(f"  {loaded} = load ptr, ptr {address}, align 8")
        public_pointer[value_id] = loaded
    for storage_index, value in enumerate(root_internal_storage):
        llvm_type = _value_llvm_type(value)
        count = _value_element_count(value)
        local = f"%root.frame.{storage_index}"
        wrapper.append(
            f"  {local} = alloca {llvm_type}, i64 {count}, align 8"
        )
        public_pointer[int(value.id)] = local
    internal_call_records.append((
        None,
        internal_symbols[function_name],
        tuple(
            public_pointer[int(value.id)]
            for value in (*root.args, *root_outputs)
        ),
    ))
    wrapper.append(
        f"  call void @{internal_symbols[function_name]}("
        + ", ".join((
            *(
                f"ptr {public_pointer[int(value.id)]}"
                for value in (*root.args, *root_outputs)
            ),
            *(("ptr %extents",) if function_name in extent_users else ()),
        ))
        + ")"
    )
    wrapper.append("  ret void")

    definitions: dict[str, str] = {}
    declarations: dict[str, str] = {}
    unresolved: set[str] = set()
    # The scalar tables reach target intrinsics from the emitted bodies and the
    # wrapper, not only from authored kernels, so the closure starts at every
    # symbol this module actually references.
    module_text = "\n".join((*emitted_functions, *wrapper))
    pending_kernels = set(kernels_used) | set(_re.findall(
        r"@([A-Za-z_$.-][\w$.-]*)\s*\(", module_text,
    ))
    # Functions this module defines itself are not external references.
    pending_kernels -= set(_re.findall(
        r"define\s+[^@]*@([A-Za-z_$.-][\w$.-]*)\s*\(", module_text,
    ))
    pending_kernels.discard(entry_name)
    while pending_kernels:
        symbol = pending_kernels.pop()
        if symbol in definitions or symbol in declarations:
            continue
        if symbol in _LLVM_INTRINSIC_DECLARATIONS:
            declarations[symbol] = _LLVM_INTRINSIC_DECLARATIONS[symbol]
            continue
        try:
            definition = extract_llvm_function(symbol)
        except KeyError:
            try:
                declarations[symbol] = extract_llvm_declaration(symbol)
            except KeyError:
                unresolved.add(symbol)
                shortfalls.append(LLVMEmissionShortfall(
                    function_name, symbol,
                    "referenced LLVM symbol has no authored definition or declaration",
                ))
            continue
        definitions[symbol] = definition
        for dependency in _re.findall(r"@([A-Za-z_$.-][\w$.-]*)\s*\(", definition):
            if dependency != symbol:
                pending_kernels.add(dependency)

    llvm_ir = "\n\n".join(part for part in (
        f'source_filename = "turing.ssa-llvm.{entry_name}"',
        "\n".join(_host_target_lines()),
        "\n".join(declarations[symbol] for symbol in sorted(declarations)),
        "\n\n".join(definitions[symbol] for symbol in sorted(definitions)),
        "\n\n".join(_annotate_noalias(
            emitted_functions, internal_call_records,
        )),
        "\n".join((
            f"define void @{entry_name}(ptr %buffers, ptr %extents) {{",
            *wrapper,
            "}",
        )),
    ) if part)
    publications = function_output_publications(module.functions[function_name])
    return LLVMFunctionArtifact(
        name=entry_name,
        llvm_ir=llvm_ir + "\n",
        buffer_order=tuple(buffer_order),
        buffer_shapes=tuple(buffer_shapes),
        extent_order=tuple(module_extent_order),
        shortfalls=tuple(shortfalls),
        buffer_dtypes=tuple(buffer_dtypes),
        needs_text_sink=bool(text_sink),
        output_publications=publications,
        output_surfaces=publication_surface_plan(publications, target="llvm"),
        watched=tuple(watched_ids),
        watch_shortfalls=tuple(watch_shortfalls),
    )


@_dataclass
class LLVMFunctionArtifact:
    """One SSA function emitted through the likeness table."""

    name: str
    llvm_ir: str
    buffer_order: tuple[int, ...]
    buffer_shapes: tuple[tuple[_Any, ...], ...]
    extent_order: tuple[tuple[int, str, int | None], ...]
    shortfalls: tuple[LLVMEmissionShortfall, ...]
    buffer_dtypes: tuple[str, ...] = ()
    needs_text_sink: bool = False
    output_publications: tuple[_Mapping[str, _Any], ...] = ()
    output_surfaces: _Mapping[str, _Any] | None = None
    library_path: _Path | None = None
    training_steps_value_id: int | None = None
    learning_rate_value_id: int | None = None
    #: Value ids additionally exposed because a caller asked to watch them.
    #: Diagnostics only: a watch appends an output slot and one copy of a
    #: value that was already computed, so it cannot change what the program
    #: computes. Empty unless asked for, and absent by default.
    watched: tuple[int, ...] = ()
    #: Watch requests that could not be honoured, with the reason. Never
    #: silently dropped -- a watch that vanishes reads as "this value is
    #: fine", which is the failure mode this whole mechanism exists to end.
    watch_shortfalls: tuple[tuple[int, str], ...] = ()
    _entry: _Any = _field(default=None, repr=False)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def entry(self):
        if self.library_path is None:
            raise RuntimeError("artifact was not compiled")
        if self._entry is None:
            library = _ctypes.CDLL(str(self.library_path))
            function = getattr(library, self.name)
            function.restype = None
            function.argtypes = [
                _ctypes.POINTER(_ctypes.c_void_p),
                _ctypes.POINTER(_ctypes.c_int32),
            ]
            self._entry = function
        return self._entry


@_dataclass
class LLVMExecution:
    """Allocated runtime state for one compiled LLVM artifact."""

    artifact: LLVMFunctionArtifact
    buffers: dict[int, _Any]
    pointers: _Any
    extents: _Any
    #: One contiguous array per scalar dtype, and each scalar's slot in it.
    scalar_arena: dict = _field(default_factory=dict)
    scalar_index: dict = _field(default_factory=dict)

    def run(self) -> "LLVMExecution":
        self.artifact.entry()(self.pointers, self.extents)
        return self


def prepare_artifact_execution(
    artifact: LLVMFunctionArtifact,
    feeds: _Any,
    *,
    shapes: _Any = None,
) -> LLVMExecution:
    """Allocate the public ABI and derive runtime extents from real buffers."""

    import numpy as np

    feed_values = {int(key): value for key, value in dict(feeds or {}).items()}
    shape_overrides = {
        int(key): tuple(value) for key, value in dict(shapes or {}).items()
    }
    llvm_dtypes = artifact.buffer_dtypes or tuple(
        "double" for _value_id in artifact.buffer_order
    )
    if len(llvm_dtypes) != len(artifact.buffer_order):
        raise ValueError("artifact buffer dtype metadata does not match its ABI")
    numpy_dtypes = {
        "double": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "i1": np.bool_,
        "ptr": np.uintp,
    }
    buffers: dict[int, _Any] = {}
    for value_id, authored_shape, llvm_dtype in zip(
        artifact.buffer_order, artifact.buffer_shapes, llvm_dtypes
    ):
        value_id = int(value_id)
        dtype = numpy_dtypes.get(str(llvm_dtype))
        if dtype is None:
            raise ValueError(f"unsupported LLVM ABI dtype {llvm_dtype!r}")
        if value_id in feed_values:
            value = np.asarray(feed_values[value_id], dtype=dtype)
            if value.ndim and not value.flags.c_contiguous:
                value = np.ascontiguousarray(value)
            expected = shape_overrides.get(value_id)
            if expected is not None and tuple(value.shape) != expected:
                raise ValueError(
                    f"feed {value_id} shape {value.shape!r} != {expected!r}"
                )
            buffers[value_id] = value
            continue
        runtime_shape = shape_overrides.get(value_id)
        if runtime_shape is None:
            try:
                runtime_shape = tuple(int(extent) for extent in authored_shape)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"buffer {value_id} has dynamic shape {authored_shape!r}; "
                    "provide a runtime shape"
                ) from error
        buffers[value_id] = np.zeros(runtime_shape or (), dtype=dtype)

    # Scalars are the overwhelming majority of a stencil ABI and were each an
    # independent allocation, so a caller could only fill them one element at a
    # time. Re-seat them as views into one contiguous arena per dtype: the
    # addresses in the pointer table are unchanged in kind, but a caller can
    # now write or read the whole set with a single indexed operation.
    scalar_arena: dict[_Any, _Any] = {}
    scalar_index: dict[int, int] = {}
    for wanted in {
        buffers[int(value_id)].dtype
        for value_id in artifact.buffer_order
        if buffers[int(value_id)].ndim == 0
    }:
        members = [
            int(value_id) for value_id in artifact.buffer_order
            if buffers[int(value_id)].ndim == 0
            and buffers[int(value_id)].dtype == wanted
        ]
        arena = np.zeros(len(members), dtype=wanted)
        for slot, value_id in enumerate(members):
            arena[slot] = buffers[value_id]
            buffers[value_id] = arena[slot:slot + 1].reshape(())
            scalar_index[value_id] = slot
        scalar_arena[wanted] = arena

    pointers = (_ctypes.c_void_p * len(artifact.buffer_order))(*(
        _ctypes.c_void_p(int(buffers[int(value_id)].ctypes.data))
        for value_id in artifact.buffer_order
    ))
    extent_values: list[int] = []
    for value_id, kind, axis in artifact.extent_order:
        shape = tuple(buffers[int(value_id)].shape)
        if kind in {"numel", "element_count"}:
            extent_values.append(int(buffers[int(value_id)].size))
        elif kind == "rank":
            extent_values.append(len(shape))
        elif kind in {"dim", "shape"} and axis is not None:
            extent_values.append(int(shape[int(axis)]))
        elif kind == "shape" and not shape:
            extent_values.append(0)
        else:
            raise ValueError(
                f"extent ({value_id}, {kind!r}, {axis!r}) cannot be measured"
            )
    extents = (_ctypes.c_int32 * len(extent_values))(*extent_values)
    return LLVMExecution(
        artifact=artifact,
        buffers=buffers,
        pointers=pointers,
        extents=extents,
        scalar_arena=scalar_arena,
        scalar_index=scalar_index,
    )


def with_native_sgd_loop(
    artifact: LLVMFunctionArtifact,
    *,
    parameter_gradient_pairs: _Any,
    entry_name: str | None = None,
) -> LLVMFunctionArtifact:
    """Wrap one native motion in a repeated in-process SGD update loop.

    The wrapped motion remains the compiled forward/loss/backward authority.
    This adds only the outer iteration and parameter update. Step count and
    learning rate are ordinary caller-owned ABI buffers, so the loop performs
    no Python callback and can be invoked repeatedly with new controls.
    """

    if artifact.shortfalls:
        raise ValueError("cannot wrap an incomplete LLVM artifact")
    if artifact.library_path is not None:
        raise ValueError("wrap the LLVM artifact before native compilation")
    pairs = tuple((int(parameter), int(gradient)) for parameter, gradient in (
        parameter_gradient_pairs or ()
    ))
    if not pairs:
        raise ValueError("native SGD loop requires parameter/gradient pairs")
    positions = {
        int(value_id): index
        for index, value_id in enumerate(artifact.buffer_order)
    }
    shapes = {
        int(value_id): tuple(shape or ())
        for value_id, shape in zip(artifact.buffer_order, artifact.buffer_shapes)
    }
    parameter_counts: dict[int, int] = {}
    for parameter, gradient in pairs:
        if parameter not in positions or gradient not in positions:
            raise ValueError(
                f"parameter/gradient pair ({parameter}, {gradient}) is not public"
            )
        if shapes[parameter] != shapes[gradient]:
            raise ValueError(
                f"parameter {parameter} shape {shapes[parameter]!r} does not "
                f"match gradient {gradient} shape {shapes[gradient]!r}"
            )
        count_value = 1
        try:
            for extent in shapes[parameter]:
                count_value *= int(extent)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "native SGD loop requires static parameter shapes"
            ) from error
        parameter_counts[parameter] = max(1, count_value)

    occupied = set(positions)
    steps_id = min((*occupied, 0)) - 1
    while steps_id in occupied:
        steps_id -= 1
    learning_rate_id = steps_id - 1
    while learning_rate_id in occupied:
        learning_rate_id -= 1
    steps_slot = len(artifact.buffer_order)
    learning_rate_slot = steps_slot + 1
    original_name = str(artifact.name)
    selected_name = str(entry_name or original_name)
    base_dtypes = artifact.buffer_dtypes or tuple(
        "double" for _value_id in artifact.buffer_order
    )
    once_name = "__" + _re.sub(r"[^A-Za-z0-9_$.-]", "_", selected_name) + "_motion_once"
    definition = _re.compile(
        r"define\s+void\s+@" + _re.escape(original_name)
        + r"\(ptr %buffers, ptr %extents\)\s*\{"
    )
    renamed, count = definition.subn(
        f"define internal void @{once_name}(ptr %buffers, ptr %extents) {{",
        artifact.llvm_ir,
        count=1,
    )
    if count != 1:
        raise ValueError(
            f"LLVM artifact has no unique public entry @{original_name}"
        )

    lines = [
        f"define void @{selected_name}(ptr %buffers, ptr %extents) {{",
        "entry:",
        f"  %steps.addr = getelementptr ptr, ptr %buffers, i64 {steps_slot}",
        "  %steps.ptr = load ptr, ptr %steps.addr, align 8",
        "  %steps = load i32, ptr %steps.ptr, align 4",
        f"  %lr.addr = getelementptr ptr, ptr %buffers, i64 {learning_rate_slot}",
        "  %lr.ptr = load ptr, ptr %lr.addr, align 8",
        "  %lr = load double, ptr %lr.ptr, align 8",
    ]
    for pair_index, (parameter, gradient) in enumerate(pairs):
        lines.extend((
            f"  %parameter.addr.{pair_index} = getelementptr ptr, ptr %buffers, i64 {positions[parameter]}",
            f"  %parameter.ptr.{pair_index} = load ptr, ptr %parameter.addr.{pair_index}, align 8",
            f"  %gradient.addr.{pair_index} = getelementptr ptr, ptr %buffers, i64 {positions[gradient]}",
            f"  %gradient.ptr.{pair_index} = load ptr, ptr %gradient.addr.{pair_index}, align 8",
        ))
    lines.extend((
        "  br label %training.header",
        "training.header:",
        "  %training.iteration = phi i32 [ 0, %entry ], [ %training.next, %training.latch ]",
        "  %training.active = icmp slt i32 %training.iteration, %steps",
        "  br i1 %training.active, label %training.motion, label %training.exit",
        "training.motion:",
        f"  call void @{once_name}(ptr %buffers, ptr %extents)",
        "  br label %update.0.header",
    ))
    for pair_index, (parameter, _gradient) in enumerate(pairs):
        count_value = parameter_counts[parameter]
        next_header = (
            f"update.{pair_index + 1}.header"
            if pair_index + 1 < len(pairs) else "training.latch"
        )
        lines.extend((
            f"update.{pair_index}.header:",
            f"  %update.index.{pair_index} = phi i64 [ 0, %training.motion ], "
            f"[ %update.next.{pair_index}, %update.{pair_index}.body ]"
            if pair_index == 0 else
            f"  %update.index.{pair_index} = phi i64 [ 0, %update.{pair_index - 1}.exit ], "
            f"[ %update.next.{pair_index}, %update.{pair_index}.body ]",
            f"  %update.active.{pair_index} = icmp ult i64 %update.index.{pair_index}, {count_value}",
            f"  br i1 %update.active.{pair_index}, label %update.{pair_index}.body, label %update.{pair_index}.exit",
            f"update.{pair_index}.body:",
            f"  %parameter.element.{pair_index} = getelementptr double, ptr %parameter.ptr.{pair_index}, i64 %update.index.{pair_index}",
            f"  %gradient.element.{pair_index} = getelementptr double, ptr %gradient.ptr.{pair_index}, i64 %update.index.{pair_index}",
            f"  %parameter.value.{pair_index} = load double, ptr %parameter.element.{pair_index}, align 8",
            f"  %gradient.value.{pair_index} = load double, ptr %gradient.element.{pair_index}, align 8",
            f"  %scaled.gradient.{pair_index} = fmul double %lr, %gradient.value.{pair_index}",
            f"  %updated.parameter.{pair_index} = fsub double %parameter.value.{pair_index}, %scaled.gradient.{pair_index}",
            f"  store double %updated.parameter.{pair_index}, ptr %parameter.element.{pair_index}, align 8",
            f"  %update.next.{pair_index} = add i64 %update.index.{pair_index}, 1",
            f"  br label %update.{pair_index}.header",
            f"update.{pair_index}.exit:",
            f"  br label %{next_header}",
        ))
    lines.extend((
        "training.latch:",
        "  %training.next = add i32 %training.iteration, 1",
        "  br label %training.header",
        "training.exit:",
        "  ret void",
        "}",
    ))
    return LLVMFunctionArtifact(
        name=selected_name,
        llvm_ir=renamed.rstrip() + "\n\n" + "\n".join(lines) + "\n",
        buffer_order=(*artifact.buffer_order, steps_id, learning_rate_id),
        buffer_shapes=(*artifact.buffer_shapes, (), ()),
        extent_order=artifact.extent_order,
        shortfalls=(),
        buffer_dtypes=(*base_dtypes, "i32", "double"),
        needs_text_sink=artifact.needs_text_sink,
        output_publications=artifact.output_publications,
        output_surfaces=artifact.output_surfaces,
        training_steps_value_id=steps_id,
        learning_rate_value_id=learning_rate_id,
    )


def emit_ssa_function_to_llvm(
    module: _IRModule, function_name: str, *, entry_name: str | None = None,
    text_sink: bool = False,
    pi_solver: str | None = None,
    pi_epsilon: float | None = None,
    watch: _Sequence[int] = (),
    history: int = 0,
) -> LLVMFunctionArtifact:
    """Render one SSA function of table-covered instructions as LLVM IR.

    ``text_sink`` states the target's publication capability. A shell-class
    target links ``turing_stream_buffer.c`` and takes publications as calls
    into it; a bare native artifact has no sink, so publications are elided
    -- they are never load-bearing for the numerics, and the same SSA runs
    either way.

    ``watch`` names value ids of ``function_name`` to expose in the public
    buffer ABI in addition to the program's own outputs, purely so a
    diagnostic can read them. Only the root function's own values can be
    watched, and only values that survive to its frame.

    This is deliberately the *non-perturbing* way to observe an internal
    value. It appends an output slot and copies a value the program already
    computed; it introduces no new arithmetic, reorders nothing, and renames
    nothing. The alternative people reach for -- adding an expression to the
    authored source to make a value reachable -- shifts value ids and can
    rebind the very thing being measured, which has produced false readings
    in this tree more than once.

    A watch is absent unless asked for: with ``watch=()`` the emitted module
    is byte-identical to what it would otherwise be, so nothing about a
    shipped artifact depends on this existing.
    """

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        extract_llvm_declaration,
        extract_llvm_function,
    )

    repository_closure, _authored_leaves = _internal_call_closure(
        module, function_name
    )
    if (
        len(repository_closure) > 1
        or len(module.functions[function_name].blocks) > 1
    ):
        return _emit_repository_call_module(
            module,
            function_name,
            entry_name=entry_name or function_name,
            text_sink=text_sink,
            watch=watch,
            history=history,
        )

    function = module.functions[function_name]
    name = entry_name or function_name
    shortfalls: list[LLVMEmissionShortfall] = []
    lines: list[str] = []
    globals_out: list[str] = []
    buffer_ids: list[int] = []
    buffer_index: dict[int, int] = {}
    extent_order: list[tuple[int, str, int | None]] = []
    extent_slots: dict[tuple[int, str, int | None], int] = {}
    scalars: dict[int, tuple[str, str]] = {}   # value id -> (rendering, type)
    kernels_used: set[str] = set()
    bounded_definitions: dict[str, str] = {}
    publishes_text = False
    value_shapes: dict[int, tuple[_Any, ...]] = {
        int(argument.id): tuple(argument.shape or ())
        for argument in function.args
    }
    value_llvm_types: dict[int, str] = {
        int(argument.id): _value_llvm_type(argument)
        for argument in function.args
    }
    for block in function.blocks.values():
        for instruction in block.instrs:
            for argument in instruction.args:
                value_shapes.setdefault(
                    int(argument.id), tuple(argument.shape or ()),
                )
            if instruction.res is not None:
                value_shapes[int(instruction.res.id)] = tuple(
                    instruction.res.shape or ()
                )
                value_llvm_types[int(instruction.res.id)] = _value_llvm_type(
                    instruction.res
                )
    returned_ids = {
        int(argument.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Ret", "ret", "Return", "return"}
        for argument in instruction.args
    }
    buffer_aliases: dict[int, int] = {}

    def buffer(value_id: int) -> str:
        # The instruction stream is already scheduled by the compiler; a
        # buffer pointer is loaded inline at its first use, in stream order,
        # never collected and reordered.
        value_id = int(value_id)
        seen = set()
        while value_id in buffer_aliases:
            if value_id in seen:
                raise ValueError("cyclic SSA buffer alias")
            seen.add(value_id)
            value_id = int(buffer_aliases[value_id])
        if value_id not in buffer_index:
            index = len(buffer_ids)
            buffer_index[value_id] = index
            buffer_ids.append(value_id)
            lines.append(
                f"  %buffer.addr.{value_id} = getelementptr ptr, ptr %buffers, "
                f"i64 {index}"
            )
            lines.append(
                f"  %buffer.{value_id} = load ptr, ptr %buffer.addr.{value_id}, "
                "align 8"
            )
        return f"%buffer.{value_id}"

    def runtime_extent(value_id: int, axis: int, tag: str) -> str:
        """An i32 register holding one axis length, measured at call time.

        The public ABI already carries an ``%extents`` vector that the executor
        fills from the real buffers, so a runtime axis length needs a slot in
        that vector rather than a compile-time constant. Slots are memoised so
        one axis is measured once per function.
        """

        key = (int(value_id), "dim", int(axis))
        slot = extent_slots.get(key)
        if slot is None:
            slot = len(extent_order)
            extent_order.append(key)
            extent_slots[key] = slot
        address = f"%extent.addr.{tag}.{slot}"
        register = f"%extent.{tag}.{slot}"
        lines.append(
            f"  {address} = getelementptr i32, ptr %extents, i64 {slot}"
        )
        lines.append(f"  {register} = load i32, ptr {address}, align 4")
        return register

    def as_type(value_id: int, wanted: str, tag: str) -> str | None:
        known = scalars.get(int(value_id))
        if known is None:
            return None
        rendering, kind = known
        if kind == wanted:
            return rendering
        if wanted == "double" and kind in {"i1", "i32", "i64"}:
            if not rendering.startswith("%"):
                return _double_literal(float(int(rendering)))
            register = f"%conv.{tag}"
            opcode = "uitofp" if kind == "i1" else "sitofp"
            lines.append(f"  {register} = {opcode} {kind} {rendering} to double")
            return register
        if wanted in {"i32", "i64"} and kind == "double":
            if not rendering.startswith("%"):
                return str(int(float.fromhex(rendering)))
            register = f"%conv.{tag}"
            lines.append(f"  {register} = fptosi double {rendering} to {wanted}")
            return register
        if wanted == "i1" and kind != "ptr":
            zero = "0.0" if kind == "double" else "0"
            comparison = "fcmp one" if kind == "double" else "icmp ne"
            register = f"%conv.{tag}"
            lines.append(
                f"  {register} = {comparison} {kind} {rendering}, {zero}"
            )
            return register
        return None

    def emit_semantic_tensor_call(instruction, symbol: str) -> str | None:
        """Expand a semantic tensor Call into the private C-kernel ABI.

        Returns ``None`` when this is not one of the elementwise kernels,
        an empty string on success, or a diagnostic string on malformed
        semantic operands.
        """

        tensor_operation = instruction.attributes.get("tensor_operation")
        if (
            instruction.res is None
            or tensor_operation is None
            or symbol not in {
                "binary_double", "binary_scalar_double", "unary_double",
            }
        ):
            return None
        from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
            c_tensor_opcode,
        )

        opcode = c_tensor_opcode(str(tensor_operation))
        if opcode is None:
            return f"no C tensor opcode for {tensor_operation!r}"
        opcode_kind, opcode_value = opcode
        result_id = int(instruction.res.id)
        result_count = _value_element_count(instruction.res)
        destination = buffer(result_id)
        if symbol == "unary_double":
            if opcode_kind != "unary" or len(instruction.args) != 1:
                return "semantic unary call has an invalid operand layout"
            source = buffer(int(instruction.args[0].id))
            lines.append(
                f"  call void @unary_double(ptr {source}, ptr {destination}, "
                f"i32 {result_count}, i32 {opcode_value})"
            )
            kernels_used.add(symbol)
            return ""
        if opcode_kind != "binary":
            return "semantic binary call resolved to a non-binary opcode"
        if symbol == "binary_double":
            if len(instruction.args) != 2:
                return "semantic binary call requires two operands"
            left = buffer(int(instruction.args[0].id))
            right = buffer(int(instruction.args[1].id))
            lines.append(
                f"  call void @binary_double(ptr {left}, ptr {right}, "
                f"ptr {destination}, i32 {result_count}, i32 {opcode_value})"
            )
            kernels_used.add(symbol)
            return ""

        reverse = bool(instruction.attributes.get("reverse", False))
        scalar_literal = instruction.attributes.get("right_scalar")
        array_argument = None
        scalar_rendering = None
        if scalar_literal is not None and len(instruction.args) == 1:
            array_argument = instruction.args[0]
            scalar_rendering = _double_literal(float(scalar_literal))
        elif len(instruction.args) == 2:
            left, right = instruction.args
            left_count = _value_element_count(left)
            right_count = _value_element_count(right)
            if left_count == result_count and right_count == 1:
                array_argument, scalar_argument = left, right
            elif left_count == 1 and right_count == result_count:
                array_argument, scalar_argument = right, left
                reverse = not reverse
            else:
                scalar_argument = None
            if array_argument is not None and scalar_argument is not None:
                scalar_rendering = as_type(
                    int(scalar_argument.id),
                    "double",
                    f"semantic.scalar.{result_id}",
                )
                if scalar_rendering is None:
                    scalar_pointer = buffer(int(scalar_argument.id))
                    scalar_rendering = f"%semantic.scalar.{result_id}"
                    lines.append(
                        f"  {scalar_rendering} = load double, "
                        f"ptr {scalar_pointer}, align 8"
                    )
        if array_argument is None or scalar_rendering is None:
            return "semantic scalar binary call has an invalid operand layout"
        source = buffer(int(array_argument.id))
        lines.append(
            f"  call void @binary_scalar_double(ptr {source}, "
            f"double {scalar_rendering}, ptr {destination}, "
            f"i32 {result_count}, i32 {opcode_value}, i32 {int(reverse)})"
        )
        kernels_used.add(symbol)
        return ""

    # Scalar public arguments are ordinary resident buffers just like tensor
    # arguments. Load them once at entry so scalar SSA can use the same
    # likeness table instead of requiring a Python-side scalar evaluator.
    for argument in function.args:
        argument_pointer = buffer(int(argument.id))
        if tuple(argument.shape or ()):
            continue
        llvm_type = _value_llvm_type(argument)
        register = f"%argument.{int(argument.id)}"
        lines.append(
            f"  {register} = load {llvm_type}, ptr {argument_pointer}, "
            f"align {_align(llvm_type)}"
        )
        scalars[int(argument.id)] = (register, llvm_type)

    for block in function.blocks.values():
        for instruction in block.instrs:
            operation = instruction.op
            result_id = int(instruction.res.id) if instruction.res is not None else None

            if operation == "extent":
                kind = str(instruction.attributes.get("extent_kind"))
                axis = instruction.attributes.get("axis")
                slot = len(extent_order)
                source_id = int(instruction.args[0].id)
                if kind == "shape":
                    rank = len(value_shapes.get(source_id, ()))
                    extent_order.extend(
                        (source_id, "shape", shape_axis)
                        for shape_axis in range(rank)
                    )
                    if rank == 0:
                        extent_order.append((source_id, "shape", None))
                else:
                    extent_order.append((
                        source_id, kind,
                        int(axis) if axis is not None else None,
                    ))
                address = f"%extent.addr.{slot}"
                lines.append(
                    f"  {address} = getelementptr i32, ptr %extents, i64 {slot}"
                )
                if kind == "shape":
                    scalars[result_id] = (address, "ptr")
                else:
                    register = f"%extent.{slot}"
                    lines.append(f"  {register} = load i32, ptr {address}, align 4")
                    scalars[result_id] = (register, "i32")
                continue

            if operation == "Pi":
                from .bounded_constants import materialize_pi

                selected_solver = (
                    pi_solver
                    or instruction.attributes.get("constant_solver")
                    or "literal"
                )
                selected_epsilon = (
                    pi_epsilon
                    if pi_epsilon is not None
                    else instruction.attributes.get("requested_epsilon")
                )
                materialization = materialize_pi(
                    selected_solver, selected_epsilon,
                )
                if materialization.value is None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, "Pi",
                        "pi materialization rejected by caller policy",
                    ))
                elif materialization.llvm_symbol is None:
                    scalars[result_id] = (
                        _double_literal(materialization.value), "double",
                    )
                else:
                    register = f"%pi.{result_id}"
                    lines.append(
                        f"  {register} = call double "
                        f"@{materialization.llvm_symbol}()"
                    )
                    scalars[result_id] = (register, "double")
                    bounded_definitions[materialization.llvm_symbol] = (
                        materialization.llvm_function
                    )
                continue

            if operation == "Const":
                payload = instruction.attributes.get("constant")
                if payload is None:
                    payload = instruction.attributes.get("values")
                if payload is None and "value" in instruction.attributes:
                    payload = instruction.attributes.get("value")
                if isinstance(payload, (tuple, list)):
                    symbol = f"@const.vec.{result_id}"
                    elements = ", ".join(f"i32 {int(item)}" for item in payload)
                    globals_out.append(
                        f"{symbol} = private constant [{len(payload)} x i32] [{elements}]"
                    )
                    scalars[result_id] = (symbol, "ptr")
                elif isinstance(payload, float) and not payload.is_integer():
                    scalars[result_id] = (_double_literal(payload), "double")
                elif payload is not None:
                    scalars[result_id] = (str(int(payload)), "i32")
                else:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, "Const", "constant without payload",
                    ))
                continue

            if operation == "StaticRef":
                scalars[result_id] = (
                    str(int(instruction.attributes["reference_handle"])),
                    "i64",
                )
                continue

            if operation in {"GetElementPtr", "getelementptr"} and (
                instruction.res is not None and len(instruction.args) >= 2
            ):
                base = instruction.args[0]
                base_value = scalars.get(int(base.id))
                base_pointer = (
                    base_value[0]
                    if base_value is not None and base_value[1] == "ptr"
                    else buffer(int(base.id))
                )
                indices = []
                trouble = None
                for position, argument in enumerate(instruction.args[1:]):
                    rendered = as_type(
                        int(argument.id), "i32",
                        f"gep.{result_id}.{position}",
                    )
                    if rendered is None:
                        trouble = (
                            "address index is not an emitted integer scalar"
                        )
                        break
                    indices.append(rendered)
                if trouble is not None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, operation, trouble,
                    ))
                    continue
                # The element type is the span's own dtype. A fixed i64 stride
                # silently addresses the wrong element for any span that is
                # not eight bytes wide.
                element_type = _value_llvm_type(base)
                if element_type == "ptr":
                    element_type = "i64"
                extents = value_shapes.get(int(base.id), ())
                if not extents:
                    # A span whose extents are only known at call time still
                    # declares its rank through its record-field identity.
                    declared_rank = int(
                        (getattr(base, "accounting", None) or {}).get(
                            "program_abi_rank"
                        ) or 0
                    )
                    if declared_rank:
                        extents = (None,) * declared_rank
                if len(indices) == 1:
                    offset = indices[0]
                elif len(extents) == len(indices):
                    # Row-major linearisation. A declared integer axis folds to
                    # a constant; a symbolic one is measured from the real
                    # buffer through the extents vector, so a runtime grid
                    # size needs no compile-time specialisation.
                    offset = indices[0]
                    for axis, index in enumerate(indices[1:], start=1):
                        extent = extents[axis]
                        stride = (
                            str(int(extent)) if isinstance(extent, int)
                            else runtime_extent(
                                int(base.id), axis, f"gep{result_id}",
                            )
                        )
                        scaled = f"%address.{result_id}.scale.{axis}"
                        lines.append(
                            f"  {scaled} = mul i32 {offset}, {stride}"
                        )
                        summed = f"%address.{result_id}.sum.{axis}"
                        lines.append(
                            f"  {summed} = add i32 {scaled}, {index}"
                        )
                        offset = summed
                else:
                    # Multi-axis addressing needs this span's declared extents.
                    # Folding the trailing indices away would silently read the
                    # wrong element, so the storage identity is named instead.
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, operation,
                        f"{len(indices)}-axis address needs declared extents "
                        f"for %t{int(base.id)}; its storage carries {extents!r}",
                    ))
                    continue
                address = f"%address.{result_id}"
                lines.append(
                    f"  {address} = getelementptr {element_type}, "
                    f"ptr {base_pointer}, i32 {offset}"
                )
                scalars[result_id] = (address, "ptr")
                continue

            if operation in {"Store", "store"} and len(instruction.args) == 2:
                source, address = instruction.args
                stored = scalars.get(int(source.id))
                destination = scalars.get(int(address.id))
                if stored is None or destination is None or destination[1] != "ptr":
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, operation,
                        "store source or destination address has no emitted producer",
                    ))
                    continue
                lines.append(
                    f"  store {stored[1]} {stored[0]}, ptr {destination[0]}, "
                    f"align {_align(stored[1])}"
                )
                continue

            if operation in {"Load", "load"} and (
                instruction.res is not None and len(instruction.args) == 1
            ):
                address = scalars.get(int(instruction.args[0].id))
                if address is None or address[1] != "ptr":
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, operation,
                        "load address has no emitted pointer producer",
                    ))
                    continue
                llvm_type = _value_llvm_type(instruction.res)
                register = f"%load.{result_id}"
                lines.append(
                    f"  {register} = load {llvm_type}, ptr {address[0]}, "
                    f"align {_align(llvm_type)}"
                )
                scalars[result_id] = (register, llvm_type)
                continue

            if operation in {"Ret", "ret", "Return", "return"}:
                for position, argument in enumerate(instruction.args):
                    value_id = int(argument.id)
                    known = scalars.get(value_id)
                    if known is None or known[1] == "ptr":
                        # Tensor-producing kernels already write to this
                        # buffer. Calling buffer() here also exposes an output
                        # that otherwise had no downstream consumer.
                        buffer(value_id)
                        continue
                    # The public buffer for this output is allocated with the
                    # value's declared dtype, so the store must use that same
                    # type. Widening to double here would write eight bytes
                    # into a four-byte slot and read back as noise.
                    declared = _value_llvm_type(argument)
                    rendering = as_type(
                        value_id, declared, f"return.{position}"
                    )
                    if rendering is None:
                        shortfalls.append(LLVMEmissionShortfall(
                            function_name, "return",
                            f"output %t{value_id} cannot render as {declared}",
                        ))
                        continue
                    destination = buffer(value_id)
                    lines.append(
                        f"  store {declared} {rendering}, ptr {destination}, "
                        "align 8"
                    )
                continue

            if (
                operation in _SHAPE_ONLY
                and instruction.res is not None
                and instruction.args
            ):
                source_id = int(instruction.args[0].id)
                if result_id in returned_ids:
                    source_pointer = buffer(source_id)
                    destination = buffer(result_id)
                    if destination != source_pointer:
                        lines.append(
                            "  call void @llvm.memcpy.p0.p0.i64("
                            f"ptr {destination}, ptr {source_pointer}, "
                            f"i64 {_value_element_count(instruction.res) * 8}, "
                            "i1 false)"
                        )
                    scalars[result_id] = (destination, "ptr")
                else:
                    buffer_aliases[result_id] = source_id
                    scalars[result_id] = (buffer(source_id), "ptr")
                continue

            callee = instruction.attributes.get("callee")
            if (
                operation == "stream_publish"
                or callee == "turing_stream_publish"
            ):
                if not text_sink:
                    continue        # no sink on this target: elide
                payload = instruction.args[0] if instruction.args else None
                if payload is None:
                    continue
                value_id = int(payload.id)
                known = scalars.get(value_id)
                if known is not None and known[1] == "double":
                    rendered = known[0]
                else:
                    pointer = buffer(value_id)
                    rendered = f"%publish.{value_id}.{len(lines)}"
                    lines.append(
                        f"  {rendered} = load double, ptr {pointer}, align 8"
                    )
                stream_id = int(instruction.attributes.get("stream_id", 0))
                final = 1 if instruction.attributes.get("final") else 0
                lines.append(
                    f"  call void @turing_stream_publish_double("
                    f"i32 {stream_id}, double {rendered}, i32 {final})"
                )
                publishes_text = True
                continue
            if callee is not None:
                symbol = str(callee)
                semantic_diagnostic = emit_semantic_tensor_call(
                    instruction, symbol,
                )
                if semantic_diagnostic is not None:
                    if semantic_diagnostic:
                        shortfalls.append(LLVMEmissionShortfall(
                            function_name, symbol, semantic_diagnostic,
                        ))
                    continue
                try:
                    returns, argument_types = _kernel_signature(symbol)
                except ValueError as error:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, symbol, str(error),
                    ))
                    continue
                kernels_used.add(symbol)
                arguments = list(instruction.args)
                output_argument = instruction.attributes.get("ssa_output_argument")
                if output_argument is not None and len(arguments) < len(argument_types):
                    arguments.insert(int(output_argument), instruction.res)
                if len(arguments) != len(argument_types):
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, symbol,
                        f"call has {len(arguments)} operands for "
                        f"{len(argument_types)} parameters",
                    ))
                    continue
                rendered: list[str] = []
                trouble: str | None = None
                for position, (argument_type, argument) in enumerate(
                    zip(argument_types, arguments)
                ):
                    value_id = int(argument.id)
                    if argument_type == "ptr":
                        known = scalars.get(value_id)
                        if known is not None and known[1] == "ptr":
                            rendered.append(f"ptr {known[0]}")
                        else:
                            rendered.append(f"ptr {buffer(value_id)}")
                    else:
                        rendering = as_type(
                            value_id, argument_type,
                            f"{result_id}.{position}",
                        )
                        if rendering is None and argument_type == "double":
                            pointer = buffer(value_id)
                            register = f"%load.{value_id}.{len(lines)}"
                            lines.append(
                                f"  {register} = load double, ptr {pointer}, align 8"
                            )
                            rendering = register
                        if rendering is None:
                            trouble = (
                                f"operand %t{value_id} cannot render as "
                                f"{argument_type}"
                            )
                            break
                        rendered.append(f"{argument_type} {rendering}")
                if trouble is not None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, symbol, trouble,
                    ))
                    continue
                joined = ", ".join(rendered)
                if returns == "void":
                    lines.append(f"  call void @{symbol}({joined})")
                else:
                    register = f"%call.{result_id}"
                    lines.append(
                        f"  {register} = call {returns} @{symbol}({joined})"
                    )
                    scalars[result_id] = (register, returns)
                    destination = buffer(result_id)
                    lines.append(
                        f"  store {returns} {register}, ptr {destination}, "
                        f"align {_align(returns)}"
                    )
                continue

            if operation in {"Cast", "CastLike", "cast_like"} and instruction.res is not None and instruction.args:
                target_type = _value_llvm_type(instruction.res)
                rendering = as_type(
                    int(instruction.args[0].id), target_type,
                    f"cast.{result_id}",
                )
                if rendering is None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, str(operation),
                        f"operand %t{int(instruction.args[0].id)} cannot cast to {target_type}",
                    ))
                    continue
                scalars[result_id] = (rendering, target_type)
                continue

            if operation in {"Deploy", "Join"}:
                # See the whole-module emitter: a deployment boundary is
                # scheduling around the program, not an instruction in it.
                lines.append(f"  ; {operation} deployment boundary")
                continue

            if (
                operation in {"Select", "where"}
                and instruction.res is not None
                and len(instruction.args) == 3
            ):
                # Select(mask, when_true, when_false). The mask carries its own
                # type; `as_type(..., "i1", ...)` is the same truthiness rule
                # the other targets apply, so a numeric or reference mask does
                # not need a separate opcode.
                target_type = _value_llvm_type(instruction.res)
                mask = as_type(
                    int(instruction.args[0].id), "i1", f"select.{result_id}",
                )
                when_true = as_type(
                    int(instruction.args[1].id), target_type,
                    f"select.{result_id}.true",
                )
                when_false = as_type(
                    int(instruction.args[2].id), target_type,
                    f"select.{result_id}.false",
                )
                if mask is None or when_true is None or when_false is None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, str(operation),
                        f"select operands cannot render as {target_type}",
                    ))
                    continue
                register = f"%select.{result_id}"
                lines.append(
                    f"  {register} = select i1 {mask}, "
                    f"{target_type} {when_true}, {target_type} {when_false}"
                )
                scalars[result_id] = (register, target_type)
                continue

            template = scalar_likeness(str(operation))
            if template is not None:
                # The declared result type is the authority on the evaluation
                # domain. Widening an integer result to double and storing it
                # back into the narrower declared slot is a real type/rank
                # defect, not a harmless promotion, so the integer column is
                # used whenever the result is an integer. Comparisons produce
                # i1 and follow their operands instead.
                declared = (
                    _value_llvm_type(instruction.res)
                    if instruction.res is not None else "double"
                )
                operand_types = [
                    (scalars.get(int(argument.id)) or (None, "double"))[1]
                    for argument in instruction.args
                ]
                if declared in {"i32", "i64"}:
                    domain = declared
                elif (
                    declared == "i1"
                    and operand_types
                    and all(
                        kind in {"i1", "i32", "i64"} for kind in operand_types
                    )
                ):
                    domain = "i64" if "i64" in operand_types else (
                        "i32" if "i32" in operand_types else "i1"
                    )
                else:
                    domain = "double"

                operands = []
                trouble = None
                for position, argument in enumerate(instruction.args):
                    rendering = as_type(
                        int(argument.id), domain, f"{result_id}.{position}"
                    )
                    if rendering is None:
                        trouble = (
                            f"scalar operand %t{int(argument.id)} unavailable"
                        )
                        break
                    operands.append(rendering)
                if trouble is not None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, str(operation), trouble,
                    ))
                    continue
                register = f"%scalar.{result_id}"
                integer = (
                    None if domain == "double"
                    else integer_scalar_lines(
                        str(operation), domain, operands, register,
                    )
                )
                if integer is not None:
                    integer_lines, produced = integer
                    lines.extend(f"  {line}" for line in integer_lines)
                    scalars[result_id] = (register, produced)
                    continue
                # What the TEMPLATE yields, which is not always the declared
                # type. `Lt` renders as `fcmp`, and fcmp yields i1 no matter
                # what the value was declared as. Recording "double" below
                # made the register disagree with its own type, and the first
                # consumer of it emitted `fcmp one double` against an i1 --
                # rejected by the verifier. Nothing consumed a comparison in
                # this path until Piecewise did, so it sat unnoticed.
                if str(operation) in PREDICATE_OPERATIONS:
                    for rendered in template.format(
                        *operands, out=register
                    ).splitlines():
                        lines.append(f"  {rendered}")
                    scalars[result_id] = (register, "i1")
                    continue
                if domain != "double":
                    # No exact integer spelling. Evaluate in the double column
                    # and convert back, so the value still reaches its declared
                    # slot with the declared type rather than a wider one.
                    operands = []
                    for position, argument in enumerate(instruction.args):
                        rendering = as_type(
                            int(argument.id), "double",
                            f"{result_id}.wide.{position}",
                        )
                        if rendering is None:
                            trouble = (
                                f"scalar operand %t{int(argument.id)} unavailable"
                            )
                            break
                        operands.append(rendering)
                    if trouble is not None:
                        shortfalls.append(LLVMEmissionShortfall(
                            function_name, str(operation), trouble,
                        ))
                        continue
                    wide = f"%scalar.wide.{result_id}"
                    for line in template.format(*operands, out=wide).splitlines():
                        lines.append(f"  {line}")
                    lines.append(
                        f"  {register} = fptosi double {wide} to {domain}"
                    )
                    scalars[result_id] = (register, domain)
                    continue
                for line in template.format(
                    *operands, out=register
                ).splitlines():
                    lines.append(f"  {line}")
                scalars[result_id] = (register, "double")
                continue

            shortfalls.append(LLVMEmissionShortfall(
                function_name, str(operation),
                "operation has no likeness-table entry",
            ))

    # Authored kernels can call other authored helpers as well as external
    # math/intrinsic symbols.  Carry their definition closure and exact
    # canonical declarations into this otherwise standalone module.
    definitions = dict(bounded_definitions)
    definitions.update({
        symbol: extract_llvm_function(symbol) for symbol in kernels_used
    })
    external_declarations: dict[str, str] = {}
    unresolved_symbols: set[str] = set()
    while True:
        dependency_text = "\n".join((*definitions.values(), *lines))
        referenced = set(_re.findall(
            r"@([A-Za-z_$.-][\w$.-]*)\s*\(", dependency_text,
        ))
        pending = referenced - set(definitions) - set(external_declarations)
        pending.discard("turing_stream_publish_double")
        pending -= unresolved_symbols
        if not pending:
            break
        for symbol in sorted(pending):
            if symbol in _LLVM_INTRINSIC_DECLARATIONS:
                external_declarations[symbol] = _LLVM_INTRINSIC_DECLARATIONS[
                    symbol
                ]
                continue
            try:
                definitions[symbol] = extract_llvm_function(symbol)
                continue
            except KeyError:
                pass
            try:
                external_declarations[symbol] = extract_llvm_declaration(symbol)
            except KeyError:
                unresolved_symbols.add(symbol)
                shortfalls.append(LLVMEmissionShortfall(
                    function_name, symbol,
                    "referenced LLVM symbol has no authored definition or declaration",
                ))

    kernel_texts = "\n\n".join(
        definitions[symbol] for symbol in sorted(definitions)
    )
    declarations = [
        external_declarations[symbol]
        for symbol in sorted(external_declarations)
    ]
    if publishes_text:
        # Resolved by linking turing_stream_buffer.c -- the shell-class sink.
        declarations.append(
            "declare void @turing_stream_publish_double(i32, double, i32)"
        )
    llvm_ir = "\n".join((
        f'source_filename = "turing.ssa-llvm.{name}"',
        *globals_out,
        *declarations,
        "",
        kernel_texts,
        "",
        f"define void @{name}(ptr %buffers, ptr %extents) {{",
        "entry:",
        *lines,
        "  ret void",
        "}",
        "",
    ))
    publications = function_output_publications(function)
    return LLVMFunctionArtifact(
        name=name,
        llvm_ir=llvm_ir,
        buffer_order=tuple(buffer_ids),
        buffer_shapes=tuple(value_shapes.get(value_id, ()) for value_id in buffer_ids),
        extent_order=tuple(extent_order),
        shortfalls=tuple(shortfalls),
        buffer_dtypes=tuple(
            value_llvm_types.get(value_id, "double") for value_id in buffer_ids
        ),
        needs_text_sink=publishes_text,
        output_publications=publications,
        output_surfaces=publication_surface_plan(publications, target="llvm"),
        watched=tuple(
            int(item) for item in watch if int(item) in set(buffer_ids)
        ),
        # This is the single-block path, whose buffer set is already every
        # value the function has. A watch for anything else is reported
        # rather than dropped: a request that quietly disappears reads as
        # "nothing to see here", which is exactly the false reassurance the
        # watch mechanism exists to prevent.
        watch_shortfalls=tuple(
            (
                int(item),
                "single-block emission publishes its own value set; this id "
                "is not among them",
            )
            for item in watch if int(item) not in set(buffer_ids)
        ),
    )


def compile_artifact(
    artifact: LLVMFunctionArtifact, *, directory: _Path | None = None,
) -> LLVMFunctionArtifact:
    """Build the emitted module with the LLVM compiler, ahead of time."""

    if not artifact.complete:
        raise ValueError(
            "artifact has shortfalls: "
            + "; ".join(s.reason for s in artifact.shortfalls[:5])
        )
    build_dir = _Path(directory) if directory is not None else _Path(
        _tempfile.mkdtemp(prefix=f"ssa_llvm_{artifact.name}_")
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / f"{artifact.name}.ll"
    source.write_text(artifact.llvm_ir, encoding="utf-8")
    library = build_dir / f"{artifact.name}.dll"
    # Same LLVM toolchain resolution the C backend uses: the ziglang package
    # bundles clang, invoked through the interpreter, no PATH assumptions.
    import sys as _sys
    command = [_sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
               "-o", str(library), str(source)]
    if _fma_contract_enabled():
        # The module names no target, so contraction permission alone reaches
        # no FMA unit; name the host. Same switch as the `contract` flag.
        command.insert(command.index("-O2") + 1, "-march=native")
    from .work_contract import active_contract as _active_work_contract

    for extra_flag in _active_work_contract().compiler_flags:
        command.insert(command.index("-O2") + 1, str(extra_flag))
    if artifact.needs_text_sink:
        command.append(str(
            _Path(__file__).resolve().parents[1]
            / "common" / "tensors" / "accelerator_backends" / "c_backend"
            / "turing_stream_buffer.c"
        ))
    completed = _subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0 or not library.is_file():
        raise RuntimeError(
            f"LLVM compile failed ({completed.returncode}):\n"
            + completed.stderr[-2000:]
        )
    artifact.library_path = library
    return artifact
