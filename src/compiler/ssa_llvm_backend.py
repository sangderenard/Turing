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
    "Mod": "{out} = frem double {0}, {1}",
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
    "Invert": "{out} = xor i64 {0}, -1",
    "SIToFP": "{out} = sitofp i32 {0} to double",
    "FPToSI": "{out} = fptosi double {0} to i32",
}

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
    "tanh": "unary_double",
    "sin": "unary_double",
    "cos": "unary_double",
    "sign": "sign_double",
    "matmul": "matmul_double",
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
    "fill": "fill_double",
    "zeros": "fill_double",
    "ones": "fill_double",
    "full": "fill_double",
    "float": "cast_double_to_float_values",
    "double": "cast_double_to_float_values",
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
    return frozenset(_TENSOR) | _SHAPE_ONLY


def scalar_likeness(operation: str) -> str | None:
    return _BINARY.get(operation) or _UNARY.get(operation)


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
from typing import Any as _Any

from ..transmogrifier.ssa import IRModule as _IRModule


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


@_dataclass
class LLVMFunctionArtifact:
    """One SSA function emitted through the likeness table."""

    name: str
    llvm_ir: str
    buffer_order: tuple[int, ...]
    buffer_shapes: tuple[tuple[_Any, ...], ...]
    extent_order: tuple[tuple[int, str, int | None], ...]
    shortfalls: tuple[LLVMEmissionShortfall, ...]
    needs_text_sink: bool = False
    library_path: _Path | None = None
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


def emit_ssa_function_to_llvm(
    module: _IRModule, function_name: str, *, entry_name: str | None = None,
    text_sink: bool = False,
) -> LLVMFunctionArtifact:
    """Render one SSA function of table-covered instructions as LLVM IR.

    ``text_sink`` states the target's publication capability. A shell-class
    target links ``turing_stream_buffer.c`` and takes publications as calls
    into it; a bare native artifact has no sink, so publications are elided
    -- they are never load-bearing for the numerics, and the same SSA runs
    either way.
    """

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        extract_llvm_declaration,
        extract_llvm_function,
    )

    function = module.functions[function_name]
    name = entry_name or function_name
    shortfalls: list[LLVMEmissionShortfall] = []
    lines: list[str] = []
    globals_out: list[str] = []
    buffer_ids: list[int] = []
    buffer_index: dict[int, int] = {}
    extent_order: list[tuple[int, str, int | None]] = []
    scalars: dict[int, tuple[str, str]] = {}   # value id -> (rendering, type)
    kernels_used: set[str] = set()
    publishes_text = False
    value_shapes: dict[int, tuple[_Any, ...]] = {
        int(argument.id): tuple(argument.shape or ())
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

    def buffer(value_id: int) -> str:
        # The instruction stream is already scheduled by the compiler; a
        # buffer pointer is loaded inline at its first use, in stream order,
        # never collected and reordered.
        value_id = int(value_id)
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

    def as_type(value_id: int, wanted: str, tag: str) -> str | None:
        known = scalars.get(int(value_id))
        if known is None:
            return None
        rendering, kind = known
        if kind == wanted:
            return rendering
        if wanted == "double" and kind == "i32":
            if not rendering.startswith("%"):
                return _double_literal(float(int(rendering)))
            register = f"%conv.{tag}"
            lines.append(f"  {register} = sitofp i32 {rendering} to double")
            return register
        if wanted == "i32" and kind == "double" and not rendering.startswith("%"):
            return str(int(float.fromhex(rendering)))
        return None

    for block in function.blocks.values():
        for instruction in block.instrs:
            operation = instruction.op
            result_id = int(instruction.res.id) if instruction.res is not None else None

            if operation == "extent":
                kind = str(instruction.attributes.get("extent_kind"))
                axis = instruction.attributes.get("axis")
                slot = len(extent_order)
                extent_order.append((
                    int(instruction.args[0].id), kind,
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

            if operation == "Const":
                payload = instruction.attributes.get("constant")
                if payload is None:
                    payload = instruction.attributes.get("values")
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
                    rendering = as_type(
                        value_id, "double", f"return.{position}"
                    )
                    if rendering is None:
                        shortfalls.append(LLVMEmissionShortfall(
                            function_name, "return",
                            f"output %t{value_id} cannot render as double",
                        ))
                        continue
                    destination = buffer(value_id)
                    lines.append(
                        f"  store double {rendering}, ptr {destination}, align 8"
                    )
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
                        f"  store {returns} {register}, ptr {destination}, align 8"
                    )
                continue

            template = scalar_likeness(str(operation))
            if template is not None:
                operands = []
                trouble = None
                for position, argument in enumerate(instruction.args):
                    rendering = as_type(
                        int(argument.id), "double", f"{result_id}.{position}"
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
    definitions = {
        symbol: extract_llvm_function(symbol) for symbol in kernels_used
    }
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
    return LLVMFunctionArtifact(
        name=name,
        llvm_ir=llvm_ir,
        buffer_order=tuple(buffer_ids),
        buffer_shapes=tuple(value_shapes.get(value_id, ()) for value_id in buffer_ids),
        extent_order=tuple(extent_order),
        shortfalls=tuple(shortfalls),
        needs_text_sink=publishes_text,
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
