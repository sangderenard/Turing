"""Whole-tape Fortran JIT lowering through ``ssa_fortran_backend``.

Bridges an AbstractTensor torture-case capture (the same autograd tape
``c_jit_backend``/``llvm_jit_backend`` walk) into the transmogrifier SSA
``Function`` that :mod:`compiler.ssa_fortran_backend` emits Fortran from,
compiles it with gfortran, and links in a thin C shell-ABI shim so the
result plugs into the same ``profiled_c_shell`` launch boundary the C and
LLVM backends use -- their ``shell_ns``/``device_ns`` numbers stay comparable.

The Fortran emitter declares each array dummy over its own per-dimension
extents (see ``ssa_fortran_backend.dimension_extents``), so shape-changing
ops like matmul are expressible.  What is still not lowered here -- permute,
cumsum/sum along a specific axis, stack, cat -- has no per-op Fortran
codegen written yet and is reported as a :class:`FortranJITShortfall`
rather than guessed at.

The compiled artifact is loaded with ``ctypes.CDLL``, not cffi.  cffi's
``ffi.verify`` builds a full CPython extension per artifact -- the
10x-slower path this repository's own numbers rule out (see
``docs/BACKEND_PERFORMANCE_HANDOFF.md``, "native compile vs cffi").  Fortran
already produces a plain shared library; loading it as one costs nothing
extra.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from ....compiler.ssa_fortran_backend import (
    FortranEmissionError,
    dimension_extents,
    emit_module,
    fortran_compiler,
)
from ....transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
from ..fused_ir import canonical_elementwise_op
from .artifact_cache import (
    CachedArtifact,
    RepositoryArtifactCache,
    implementation_digest,
    repository_cache_root,
)
from .c_jit_backend import _flatten, _required_nodes
from .profiled_c_shell import CLaunchProfile, ShellLanguage, profiled_c_shell
from .tensor_torture import CapturedTortureCase


class FortranJITShortfall(RuntimeError):
    pass


_COMPARISON_OPS = frozenset(
    {
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "equal",
        "not_equal",
        "logical_and",
        "logical_or",
    }
)

_REDUCTION_OPS = frozenset({"sum", "prod", "max", "min", "mean", "all", "any"})


@dataclass(frozen=True)
class FortranOutputSpec:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class FortranJITExecution:
    outputs: Mapping[str, np.ndarray]
    profile: CLaunchProfile
    cache_hit: bool


def _count(value: Any) -> int:
    total = 1
    for size in tuple(value.shape):
        total *= int(size)
    return total


def _fortran_implementation_digest() -> str:
    directory = Path(__file__).resolve().parent
    return implementation_digest(
        (
            directory / "fortran_jit_backend.py",
            directory.parents[2] / "compiler" / "ssa_fortran_backend.py",
        )
    )


def _lower_tape_to_fortran_function(
    captured: CapturedTortureCase,
    *,
    function_name: str,
) -> tuple[
    Function,
    tuple[str, ...],
    tuple[FortranOutputSpec, ...],
    tuple[str, ...],
    tuple[SSAValue, ...],
]:
    nodes = _required_nodes(captured)
    produced = {id(node.ctx["result"]) for node in nodes}
    feeds = []
    seen: set[int] = set()
    for node in nodes:
        for operand in node.ctx.get("inputs", ()):
            identity = id(operand)
            if (
                identity not in produced
                and identity not in seen
                and hasattr(operand, "shape")
            ):
                feeds.append(operand)
                seen.add(identity)
    names_by_id = {id(value): name for name, value in captured.inputs.items()}
    try:
        feed_names = tuple(names_by_id[id(value)] for value in feeds)
    except KeyError as error:
        raise FortranJITShortfall(
            "Fortran lowering exposed an undeclared torture-case feed "
            f"{error.args[0]}"
        ) from error

    output_specs = tuple(
        FortranOutputSpec(name, tuple(value.shape))
        for name, value in captured.outputs.items()
    )

    next_id = 0

    def new_value(*, shape: tuple[int, ...] = (), dtype: str = "float64") -> SSAValue:
        nonlocal next_id
        next_id += 1
        return SSAValue(id=next_id, dtype=dtype, shape=shape)

    ssa_by_id: dict[int, SSAValue] = {}
    instrs: list[Instr] = []
    args: list[SSAValue] = []
    for value in feeds:
        ssa = new_value(shape=tuple(value.shape))
        ssa_by_id[id(value)] = ssa
        args.append(ssa)

    for node in nodes:
        original_operation = str(node.op)
        result = node.ctx["result"]
        result_id = id(result)
        if result_id in ssa_by_id:
            continue

        if original_operation == "tensor_from_list":
            values = tuple(_flatten(node.ctx["params"]["data"]))
            if len(values) != 1:
                raise FortranJITShortfall(
                    "Fortran lowering only supports scalar tape constants"
                )
            ssa = new_value(shape=())
            instrs.append(
                Instr(
                    op="Const",
                    args=[],
                    res=ssa,
                    attributes={"constant": float(values[0])},
                )
            )
            ssa_by_id[result_id] = ssa
            continue

        operands = tuple(node.ctx.get("inputs", ()))
        missing = [id(value) for value in operands if id(value) not in ssa_by_id]
        if missing:
            raise FortranJITShortfall(
                f"{original_operation} operands have no Fortran SSA value: {missing}"
            )
        operand_values = [ssa_by_id[id(value)] for value in operands]

        if original_operation in {"reshape", "view"}:
            if len(operands) != 1 or _count(operands[0]) != _count(result):
                raise FortranJITShortfall(
                    f"{original_operation} is not a compatible zero-copy view"
                )
            if tuple(result.shape) == tuple(operands[0].shape):
                ssa_by_id[result_id] = operand_values[0]
                continue
            # Aliasing the operand would leave the value declared with its
            # old shape. That is harmless only while nothing looks at the
            # shape; a Fortran dummy is explicit-shape, so any later op --
            # permute, a section, a dimension-wise reduction -- would be
            # emitted against the wrong rank.
            ssa = new_value(shape=tuple(result.shape))
            instrs.append(
                Instr(op="reshape", args=operand_values, res=ssa)
            )
            ssa_by_id[result_id] = ssa
            continue

        try:
            operation, reverse = canonical_elementwise_op(original_operation)
        except KeyError:
            operation, reverse = original_operation, False

        if original_operation in {"matmul", "rmatmul", "imatmul"}:
            if len(operand_values) != 2:
                raise FortranJITShortfall("matmul requires two operands")
            left, right = operands
            left_ssa, right_ssa = operand_values
            if original_operation == "rmatmul":
                left, right = right, left
                left_ssa, right_ssa = right_ssa, left_ssa
            if len(left.shape) != 2 or len(right.shape) != 2:
                raise FortranJITShortfall("matmul requires rank-two operands")
            rows, inner = (int(size) for size in left.shape)
            inner_right, columns = (int(size) for size in right.shape)
            if inner != inner_right:
                raise FortranJITShortfall("matmul dimensions do not agree")
            ssa = new_value(shape=(rows, columns))
            instrs.append(
                Instr(op="matmul", args=[left_ssa, right_ssa], res=ssa)
            )
            ssa_by_id[result_id] = ssa
            continue

        if operation == "where":
            if len(operand_values) != 3:
                raise FortranJITShortfall("where requires three operands")
            ssa = new_value(shape=tuple(result.shape))
            instrs.append(Instr(op="where", args=operand_values, res=ssa))
            ssa_by_id[result_id] = ssa
            continue

        # Shape ops are defined by a parameter -- which axis, which
        # permutation -- so it must reach the instruction. Dropping it and
        # letting the emitter assume a default does not produce a shortfall,
        # it produces the wrong answer silently.
        if operation in {"stack", "concat", "cat", "cumsum", "permute"}:
            parameters = dict(node.ctx.get("params") or {})
            attributes: dict[str, Any] = {}
            if operation == "permute":
                perm = parameters.get("perm", parameters.get("dims"))
                if perm is None:
                    raise FortranJITShortfall(
                        "permute has no recorded permutation"
                    )
                attributes["dims"] = [int(entry) for entry in perm]
            else:
                dim = parameters.get("dim", parameters.get("axis"))
                if dim is None:
                    raise FortranJITShortfall(
                        f"{operation} has no recorded dimension"
                    )
                attributes["dim"] = int(dim)
            canonical = "concat" if operation in {"cat", "concat"} else operation
            ssa = new_value(shape=tuple(result.shape))
            instrs.append(
                Instr(
                    op=canonical,
                    args=operand_values,
                    res=ssa,
                    attributes=attributes,
                )
            )
            ssa_by_id[result_id] = ssa
            continue

        if operation in _REDUCTION_OPS:
            parameters = dict(node.ctx.get("params") or {})
            axis = parameters.get("axis", parameters.get("dim"))
            if len(operand_values) != 1:
                raise FortranJITShortfall(f"{operation} requires one operand")
            if axis is not None:
                # Fortran reduces along one dimension natively --
                # sum(a, dim=k) -- so an axis reduction is expressible; it is
                # only the result rank that differs, and keepdim restores it.
                ssa = new_value(shape=tuple(result.shape))
                instrs.append(
                    Instr(
                        op=operation,
                        args=operand_values,
                        res=ssa,
                        attributes={
                            "dim": int(axis),
                            "keepdim": bool(parameters.get("keepdim")),
                        },
                    )
                )
                ssa_by_id[result_id] = ssa
                continue
            ssa = new_value(shape=())
            instrs.append(Instr(op=operation, args=operand_values, res=ssa))
            ssa_by_id[result_id] = ssa
            continue

        if len(operand_values) == 1:
            dtype = "bool" if operation in _COMPARISON_OPS else "float64"
            ssa = new_value(shape=tuple(result.shape), dtype=dtype)
            instrs.append(Instr(op=operation, args=operand_values, res=ssa))
            ssa_by_id[result_id] = ssa
            continue

        if len(operand_values) == 2:
            left_count = _count(operands[0])
            right_count = _count(operands[1])
            if left_count != right_count and _count(result) not in (
                left_count,
                right_count,
            ):
                raise FortranJITShortfall(
                    f"{original_operation} has unsupported Fortran operand extents"
                )
            dtype = "bool" if operation in _COMPARISON_OPS else "float64"
            ssa = new_value(shape=tuple(result.shape), dtype=dtype)
            instrs.append(
                Instr(
                    op=operation,
                    args=operand_values,
                    res=ssa,
                    attributes={"reverse": True} if reverse else {},
                )
            )
            ssa_by_id[result_id] = ssa
            continue

        raise FortranJITShortfall(
            f"{original_operation} has no Fortran tape binding"
        )

    outputs: list[SSAValue] = []
    for value in captured.outputs.values():
        identity = id(value)
        if identity not in ssa_by_id:
            raise FortranJITShortfall(
                "Fortran wrapper does not produce all torture-case outputs"
            )
        output_ssa = ssa_by_id[identity]
        if output_ssa.dtype == "bool":
            # A boolean leaving through a real(c_double) buffer needs an
            # explicit conversion: the Python side always allocates float64
            # storage for outputs, and a LOGICAL dummy over that buffer is a
            # layout mismatch (1 byte per element written into an 8-byte
            # slot), not a type coercion.
            converted = new_value(shape=output_ssa.shape, dtype="float64")
            instrs.append(
                Instr(op="bool_to_float64", args=[output_ssa], res=converted)
            )
            output_ssa = converted
        outputs.append(output_ssa)

    block = BasicBlock(name="entry", instrs=instrs)
    function = Function(name=function_name, args=args, blocks={"entry": block})
    extent_names = tuple(sorted(dimension_extents((*args, *outputs)).values()))
    return function, feed_names, output_specs, extent_names, tuple(outputs)


def _shim_source(
    *,
    fortran_name: str,
    shell_name: str,
    arg_count: int,
    output_count: int,
    extent_names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
    kernel_module: str,
) -> str:
    """The launch boundary, in Fortran.

    This used to be a separate C translation unit, which meant the Fortran
    backend needed a C compiler as well as gfortran to produce anything.  It
    does not: ``iso_c_binding`` is how Fortran speaks the C ABI, and the
    emitted subroutine already uses it.  ``c_ptr``/``c_f_pointer`` unpack the
    shell's ``void **`` context, and ``c_funloc`` yields the entry address the
    loader asks for, so the whole launch boundary is one more ``bind(C)``
    procedure beside the kernel in the same source file.
    """

    total = arg_count + output_count
    # Each extent parameter is literally named "extent_<size>" (see
    # ssa_fortran_backend.dimension_extents), so its value is the integer
    # embedded in the name.
    extent_values = [name.rsplit("_", 1)[1] for name in extent_names]
    call_arguments = ", ".join(
        [f"{value}_c_int" for value in extent_values]
        + [f"argument_{index}" for index in range(total)]
    )
    def element_count(shape: tuple[int, ...]) -> int:
        total_elements = 1
        for size in shape:
            total_elements *= int(size)
        return total_elements

    # A reduction's result is a scalar, not a one-element array, and its
    # dummy is declared scalar to match; binding it as rank-1 is a rank
    # mismatch the compiler rejects outright. Arrays bind as rank-1 over
    # their element count -- sequence association then passes a contiguous
    # rank-1 actual to an explicit-shape dummy of any rank.
    pointer_declarations = [
        f"    real(c_double), pointer :: argument_{index}"
        + ("" if not shape else "(:)")
        for index, shape in enumerate(shapes)
    ]
    pointer_bindings = [
        f"    call c_f_pointer(handles({index + 1}), argument_{index})"
        if not shape
        else (
            f"    call c_f_pointer(handles({index + 1}), argument_{index}, "
            f"[{max(element_count(shape), 1)}])"
        )
        for index, shape in enumerate(shapes)
    ]
    return "\n".join(
        (
            f"module {shell_name}_module",
            "  use, intrinsic :: iso_c_binding",
            f"  use {kernel_module}",
            "  implicit none",
            "contains",
            "",
            f"  function {shell_name}(context, device_ns) result(status) &",
            f"      bind(C, name=\"{shell_name}\")",
            "    implicit none",
            "    type(c_ptr), value :: context",
            "    integer(c_long_long), intent(out) :: device_ns",
            "    integer(c_int) :: status",
            "    type(c_ptr), pointer :: handles(:)",
            *pointer_declarations,
            "",
            f"    call c_f_pointer(context, handles, [{total}])",
            *pointer_bindings,
            "    ! A CPU kernel reports no separate device duration.",
            "    device_ns = 0_c_long_long",
            f"    call {fortran_name}({call_arguments})",
            "    status = 1_c_int",
            f"  end function {shell_name}",
            "",
            f"  function {shell_name}_address() result(address) &",
            f"      bind(C, name=\"{shell_name}_address\")",
            "    implicit none",
            "    integer(c_intptr_t) :: address",
            f"    address = transfer(c_funloc({shell_name}), 0_c_intptr_t)",
            f"  end function {shell_name}_address",
            f"end module {shell_name}_module",
            "",
        )
    )


class FortranJITProgram:
    def __init__(
        self,
        *,
        library_path: Path,
        function_name: str,
        shell_name: str,
        feed_names: tuple[str, ...],
        outputs: tuple[FortranOutputSpec, ...],
        source_artifact: CachedArtifact,
    ):
        self.library_path = library_path
        self.function_name = function_name
        self.shell_name = shell_name
        self.feed_names = feed_names
        self.output_specs = outputs
        self.source_artifact = source_artifact
        # A library using a Fortran intrinsic implemented in libgfortran's
        # runtime (matmul, at least) needs that DLL found at *load* time, not
        # just compile time -- PATH alone no longer satisfies Windows' DLL
        # search since Python 3.8 hardened it; add_dll_directory is required.
        compiler = fortran_compiler()
        if compiler is not None and sys.platform == "win32":
            os.add_dll_directory(str(Path(compiler).parent))
        # ctypes.CDLL, not cffi: the library is already a compiled shared
        # object, so there is nothing for cffi's extension-building step to
        # buy here -- only its cost.
        self._handle = ctypes.CDLL(str(library_path))
        address_symbol = getattr(self._handle, f"{shell_name}_address")
        address_symbol.restype = ctypes.c_size_t
        self._shell_address = int(address_symbol())

    def execute(self, inputs: Mapping[str, Any]) -> FortranJITExecution:
        missing = set(self.feed_names) - set(inputs)
        if missing:
            raise ValueError(f"missing Fortran JIT feeds: {sorted(missing)}")
        # Fortran arrays are column-major. For a 1-D array this is identical
        # to C order (nothing changes for the existing elementwise cases);
        # for a real N-D array (matmul's operands/result) it is not, and
        # asfortranarray is what makes the flat buffer the emitted
        # `array(rows, cols)` declaration reads mean the same (row, col)
        # values C-order numpy indexing on this same array already gives.
        # A 0-d (scalar) array has no dimension to order -- and
        # np.asfortranarray on one promotes it to shape (1,), a numpy quirk
        # that silently breaks scalar outputs (flat_sum) -- so it is left
        # alone.
        def _ordered(array: np.ndarray) -> np.ndarray:
            return np.asfortranarray(array) if array.ndim else array

        feed_arrays = [
            _ordered(np.asarray(inputs[name], dtype=np.float64))
            for name in self.feed_names
        ]
        output_arrays = [
            _ordered(np.empty(spec.shape or (), dtype=np.float64))
            for spec in self.output_specs
        ]
        arrays = [*feed_arrays, *output_arrays]
        shell = profiled_c_shell()
        context = shell.ffi.new(
            "void *[]",
            [
                shell.ffi.cast("void *", int(array.ctypes.data))
                for array in arrays
            ],
        )
        profile = shell.launch(
            self._shell_address, context, language=ShellLanguage.FORTRAN
        )
        if profile.status != 1:
            raise RuntimeError(
                f"Fortran compute closure returned status {profile.status}"
            )
        return FortranJITExecution(
            outputs={
                spec.name: array
                for spec, array in zip(self.output_specs, output_arrays)
            },
            profile=profile,
            cache_hit=self.source_artifact.hit,
        )


def compile_torture_case_to_fortran(
    captured: CapturedTortureCase,
    *,
    cache: RepositoryArtifactCache | None = None,
) -> FortranJITProgram:
    cache = cache or RepositoryArtifactCache()
    function_name = "turing_torture_compute"
    shell_name = "turing_torture_fortran_shell_entry"
    record = {
        "case": captured.case.semantic_record(),
        "compiler": "turing-ssa-fortran",
        "implementation": _fortran_implementation_digest(),
    }
    artifact = cache.load("fortran", record, suffix=".f90")
    if artifact is None:
        function, feed_names, outputs, extent_names, output_values = (
            _lower_tape_to_fortran_function(captured, function_name=function_name)
        )
        module = emit_module(
            {function_name: function},
            name="turing_torture_fortran",
            outputs={function_name: output_values},
        )
        if not module.complete:
            raise FortranJITShortfall(
                "; ".join(shortfall.format() for shortfall in module.shortfalls)
            )
        shim = _shim_source(
            fortran_name=function_name,
            shell_name=shell_name,
            arg_count=len(feed_names),
            output_count=len(outputs),
            extent_names=extent_names,
            shapes=(
                *(tuple(value.shape) for value in function.args),
                *(tuple(value.shape) for value in output_values),
            ),
            kernel_module="turing_torture_fortran",
        )
        artifact = cache.store(
            "fortran",
            record,
            module.source,
            suffix=".f90",
            metadata={
                "feed_names": list(feed_names),
                "outputs": [
                    {"name": item.name, "shape": list(item.shape)}
                    for item in outputs
                ],
                "extent_names": list(extent_names),
                "shim_source": shim,
            },
        )
    else:
        metadata = artifact.manifest["metadata"]
        feed_names = tuple(metadata["feed_names"])
        outputs = tuple(
            FortranOutputSpec(item["name"], tuple(item["shape"]))
            for item in metadata["outputs"]
        )
        shim = metadata["shim_source"]

    compiler = fortran_compiler()
    if compiler is None:
        raise FortranEmissionError(
            "no Fortran compiler found; install gfortran to run the fortran "
            "torture backend"
        )

    build_directory = (
        repository_cache_root() / "fortran" / "jit" / artifact.identity[:16]
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if sys.platform == "win32" else ".so"
    library_path = build_directory / f"turing_torture_fortran{suffix}"
    if not library_path.exists():
        fortran_source_path = build_directory / "turing_torture_fortran.f90"
        fortran_source_path.write_text(artifact.source, encoding="utf-8")
        # The shell is Fortran too, so gfortran alone builds the whole
        # library; the backend no longer needs a C compiler as well. It uses
        # the kernel's module, so it must be compiled after it.
        shim_source_path = build_directory / "turing_torture_fortran_shell.f90"
        shim_source_path.write_text(shim, encoding="utf-8")
        command = [
            compiler,
            "-shared",
            *(() if sys.platform == "win32" else ("-fPIC",)),
            "-O3",
            "-march=native",
            "-funroll-loops",
            str(fortran_source_path),
            str(shim_source_path),
            "-o",
            str(library_path),
        ]
        # gfortran spawns f951, which loads support DLLs from the toolchain's
        # own bin directory. Invoked by absolute path with that directory
        # missing from PATH, it exits non-zero with no diagnostic at all.
        environment = dict(os.environ)
        environment["PATH"] = (
            str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(build_directory),
            env=environment,
        )
        if completed.returncode != 0 or not library_path.exists():
            raise FortranEmissionError(
                f"Fortran compilation failed:\n{completed.stderr or completed.stdout}"
            )

    return FortranJITProgram(
        library_path=library_path,
        function_name=function_name,
        shell_name=shell_name,
        feed_names=feed_names,
        outputs=outputs,
        source_artifact=artifact,
    )


__all__ = [
    "FortranJITExecution",
    "FortranJITProgram",
    "FortranJITShortfall",
    "FortranOutputSpec",
    "compile_torture_case_to_fortran",
]
