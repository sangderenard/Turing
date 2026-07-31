"""Whole-tape C JIT lowering through the real ``ctensor_ops.c`` kernels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import prod
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..fused_ir import canonical_elementwise_op
from .artifact_cache import (
    CachedArtifact,
    RepositoryArtifactCache,
    implementation_digest,
    repository_cache_root,
)
from .c_primitive_program import _OP_NAMES
from .profiled_c_shell import CLaunchProfile, profiled_c_shell
from .tensor_torture import CapturedTortureCase


class CJITShortfall(RuntimeError):
    pass


@dataclass(frozen=True)
class COutputSpec:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class CJITExecution:
    outputs: Mapping[str, np.ndarray]
    profile: CLaunchProfile
    cache_hit: bool


def _count(value: Any) -> int:
    return int(prod(tuple(value.shape))) if tuple(value.shape) else 1


def _c_double(value: Any) -> str:
    value = float(value)
    if np.isnan(value):
        return "NAN"
    if np.isposinf(value):
        return "INFINITY"
    if np.isneginf(value):
        return "-INFINITY"
    literal = format(value, ".17g")
    return literal if any(mark in literal for mark in ".eE") else literal + ".0"


def _flatten(value: Any):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _c_implementation_digest() -> str:
    directory = Path(__file__).resolve().parent
    return implementation_digest(
        (
            directory / "c_jit_backend.py",
            directory / "c_backend" / "ctensor_ops.c",
            directory / "c_backend" / "ctensor_ops.h",
        )
    )


def _required_nodes(captured: CapturedTortureCase):
    nodes = dict(captured.tape._nodes)
    required: set[int] = set()

    def visit(value):
        identity = id(value)
        if identity in required:
            return
        node = nodes.get(identity)
        if node is None:
            return
        required.add(identity)
        for operand in node.ctx.get("inputs", ()):
            visit(operand)

    for output in captured.outputs.values():
        visit(output)
    return tuple(
        node for identity, node in nodes.items() if identity in required
    )


def _emit_c_tape(
    captured: CapturedTortureCase,
    *,
    function_name: str,
    shell_name: str,
) -> tuple[str, tuple[str, ...], tuple[COutputSpec, ...]]:
    nodes = _required_nodes(captured)
    produced = {id(node.ctx["result"]) for node in nodes}
    feeds = []
    seen = set()
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
    names_by_id = {
        id(value): name for name, value in captured.inputs.items()
    }
    try:
        feed_names = tuple(names_by_id[id(value)] for value in feeds)
    except KeyError as error:
        raise CJITShortfall(
            "C lowering exposed an undeclared torture-case feed "
            f"{error.args[0]}"
        ) from error
    output_specs = tuple(
        COutputSpec(name, tuple(value.shape))
        for name, value in captured.outputs.items()
    )
    parameters = [
        *(f"double *feed{index}" for index in range(len(feeds))),
        *(f"double *output{index}" for index in range(len(output_specs))),
    ]
    pointers = {
        id(value): f"feed{index}" for index, value in enumerate(feeds)
    }
    output_pointers = {
        id(value): f"output{index}"
        for index, value in enumerate(captured.outputs.values())
    }
    global_declarations = []
    local_declarations = []
    workspace_allocations = []
    setup_statements = []
    workspace_cleanup = []
    statements = []
    written_outputs: set[int] = set()

    for constant_index, node in enumerate(
        node for node in nodes if str(node.op) == "tensor_from_list"
    ):
        result = node.ctx["result"]
        values = tuple(_flatten(node.ctx["params"]["data"]))
        symbol = f"tape_constant_{constant_index}"
        global_declarations.append(
            f"static const double {symbol}[{len(values)}] = "
            "{"
            + ", ".join(_c_double(value) for value in values)
            + "};"
        )
        pointers[id(result)] = symbol

    for node_index, node in enumerate(nodes):
        original_operation = str(node.op)
        result = node.ctx["result"]
        result_id = id(result)
        if original_operation == "tensor_from_list":
            continue
        operands = tuple(node.ctx.get("inputs", ()))
        missing = [id(value) for value in operands if id(value) not in pointers]
        if missing:
            raise CJITShortfall(
                f"{original_operation} operands have no C storage: {missing}"
            )

        if original_operation in {"reshape", "view"}:
            if len(operands) != 1 or _count(operands[0]) != _count(result):
                raise CJITShortfall(
                    f"{original_operation} is not a compatible zero-copy view"
                )
            source = pointers[id(operands[0])]
            if result_id in output_pointers:
                destination = output_pointers[result_id]
                statements.append(
                    f"    memcpy({destination}, {source}, "
                    f"{_count(result)} * sizeof(double));"
                )
                written_outputs.add(result_id)
                pointers[result_id] = destination
            else:
                pointers[result_id] = source
            continue

        destination = output_pointers.get(result_id)
        if destination is None:
            destination = f"temporary_{node_index}"
            local_declarations.append(
                f"double *{destination} = NULL;"
            )
            workspace_allocations.extend((
                f"{destination} = (double *)malloc("
                f"{_count(result)} * sizeof(double));",
                f"if ({destination} == NULL) goto cleanup;",
            ))
            workspace_cleanup.insert(
                0,
                f"free({destination});",
            )
        else:
            written_outputs.add(result_id)
        pointers[result_id] = destination
        parameters_ = dict(node.ctx.get("params") or {})

        try:
            operation, reverse = canonical_elementwise_op(original_operation)
        except KeyError:
            operation, reverse = original_operation, False

        if operation in _OP_NAMES:
            opcode = _OP_NAMES[operation]
            if len(operands) == 1:
                statements.append(
                    f"    unary_double({pointers[id(operands[0])]}, "
                    f"{destination}, {_count(result)}, {opcode});"
                )
                continue
            if len(operands) == 2:
                left, right = operands
                left_count, right_count = _count(left), _count(right)
                if left_count == _count(result) and right_count == _count(result):
                    left_pointer = pointers[id(left)]
                    right_pointer = pointers[id(right)]
                    if reverse:
                        left_pointer, right_pointer = right_pointer, left_pointer
                    statements.append(
                        f"    binary_double({left_pointer}, {right_pointer}, "
                        f"{destination}, {_count(result)}, {opcode});"
                    )
                    continue
                if left_count == _count(result) and right_count == 1:
                    statements.append(
                        f"    binary_scalar_double({pointers[id(left)]}, "
                        f"{pointers[id(right)]}[0], {destination}, "
                        f"{_count(result)}, {opcode}, {int(reverse)});"
                    )
                    continue
                if left_count == 1 and right_count == _count(result):
                    statements.append(
                        f"    binary_scalar_double({pointers[id(right)]}, "
                        f"{pointers[id(left)]}[0], {destination}, "
                        f"{_count(result)}, {opcode}, {int(not reverse)});"
                    )
                    continue
            raise CJITShortfall(
                f"{original_operation} has unsupported C operand extents"
            )

        if operation == "where":
            if len(operands) != 3 or any(
                _count(value) != _count(result) for value in operands
            ):
                raise CJITShortfall(
                    "where_double requires three equal extents"
                )
            statements.append(
                "    where_double("
                + ", ".join(pointers[id(value)] for value in operands)
                + f", {destination}, {_count(result)});"
            )
            continue

        if operation in {"matmul", "rmatmul", "imatmul"}:
            if len(operands) != 2:
                raise CJITShortfall("matmul requires two operands")
            left, right = operands
            if operation == "rmatmul":
                left, right = right, left
            if len(left.shape) != 2 or len(right.shape) != 2:
                raise CJITShortfall("matmul_double requires rank-two operands")
            rows, inner = tuple(left.shape)
            inner_right, columns = tuple(right.shape)
            if inner != inner_right:
                raise CJITShortfall("matmul dimensions do not agree")
            statements.append(
                f"    matmul_double({pointers[id(left)]}, "
                f"{pointers[id(right)]}, {destination}, "
                f"{rows}, {inner}, {columns});"
            )
            continue

        if operation == "permute":
            source = operands[0]
            shape = tuple(int(size) for size in source.shape)
            axes = tuple(
                int(axis) for axis in parameters_.get(
                    "perm", parameters_.get("dims")
                )
            )
            global_declarations.extend(
                (
                    f"static const int shape_{node_index}[] = "
                    "{" + ", ".join(map(str, shape)) + "};",
                    f"static const int axes_{node_index}[] = "
                    "{" + ", ".join(map(str, axes)) + "};",
                )
            )
            statements.append(
                f"    transpose_double({pointers[id(source)]}, {destination}, "
                f"shape_{node_index}, axes_{node_index}, {len(shape)});"
            )
            continue

        if operation == "cumsum":
            source = operands[0]
            shape = tuple(int(size) for size in source.shape)
            dim = int(parameters_.get("dim", 0)) % len(shape)
            global_declarations.append(
                f"static const int shape_{node_index}[] = "
                "{" + ", ".join(map(str, shape)) + "};"
            )
            statements.append(
                f"    cumsum_dim_double({pointers[id(source)]}, {destination}, "
                f"shape_{node_index}, {len(shape)}, {dim});"
            )
            continue

        if operation == "sum":
            source = operands[0]
            axis = parameters_.get("axis")
            if axis is None:
                statements.append(
                    f"    {destination}[0] = sum_double("
                    f"{pointers[id(source)]}, {_count(source)});"
                )
                continue
            shape = tuple(int(size) for size in source.shape)
            dim = int(axis) % len(shape)
            global_declarations.append(
                f"static const int shape_{node_index}[] = "
                "{" + ", ".join(map(str, shape)) + "};"
            )
            statements.append(
                f"    reduce_dim_double({pointers[id(source)]}, {destination}, "
                f"shape_{node_index}, {len(shape)}, {dim}, 0);"
            )
            continue

        if operation == "stack":
            if not operands:
                raise CJITShortfall("stack requires tensor inputs")
            shape = tuple(int(size) for size in operands[0].shape)
            dim = int(parameters_.get("dim", 0)) % (len(shape) + 1)
            global_declarations.append(
                f"static const int shape_{node_index}[] = "
                "{" + ", ".join(map(str, shape)) + "};"
            )
            local_declarations.append(
                f"const double *inputs_{node_index}[{len(operands)}];"
            )
            setup_statements.extend(
                f"inputs_{node_index}[{index}] = {pointers[id(value)]};"
                for index, value in enumerate(operands)
            )
            statements.append(
                f"    stack_double(inputs_{node_index}, {len(operands)}, "
                f"shape_{node_index}, {len(shape)}, {dim}, {destination});"
            )
            continue

        if operation in {"cat", "concat"}:
            if not operands:
                raise CJITShortfall("cat requires tensor inputs")
            shape = tuple(int(size) for size in operands[0].shape)
            dim = int(parameters_.get("dim", 0)) % len(shape)
            sizes = tuple(int(value.shape[dim]) for value in operands)
            global_declarations.extend(
                (
                    f"static const int shape_{node_index}[] = "
                    "{" + ", ".join(map(str, shape)) + "};",
                    f"static const int sizes_{node_index}[] = "
                    "{" + ", ".join(map(str, sizes)) + "};",
                )
            )
            local_declarations.append(
                f"const double *inputs_{node_index}[{len(operands)}];"
            )
            setup_statements.extend(
                f"inputs_{node_index}[{index}] = {pointers[id(value)]};"
                for index, value in enumerate(operands)
            )
            statements.append(
                f"    cat_double(inputs_{node_index}, sizes_{node_index}, "
                f"{len(operands)}, shape_{node_index}, {len(shape)}, {dim}, "
                f"{destination});"
            )
            continue

        raise CJITShortfall(
            f"{original_operation} has no exact ctensor_ops.c tape binding"
        )

    missing_outputs = set(output_pointers) - written_outputs
    if missing_outputs:
        raise CJITShortfall(
            f"C wrapper does not write outputs {sorted(missing_outputs)}"
        )
    arguments = [
        *(f"(double *)arguments[{index}]" for index in range(len(parameters)))
    ]
    source = "\n".join(
        (
            "#include <math.h>",
            "#include <stdint.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            '#include "ctensor_ops.c"',
            "",
            *global_declarations,
            "",
            f"int {function_name}({', '.join(parameters)}) {{",
            "    int status = 0;",
            *(f"    {line}" for line in local_declarations),
            *(f"    {line}" for line in workspace_allocations),
            *(f"    {line}" for line in setup_statements),
            *statements,
            "    status = 1;",
            "cleanup:",
            *(f"    {line}" for line in workspace_cleanup),
            "    return status;",
            "}",
            "",
            f"int {shell_name}(void *context, unsigned long long *device_ns) {{",
            "    void **arguments = (void **)context;",
            "    *device_ns = 0;",
            f"    return {function_name}({', '.join(arguments)});",
            "}",
            "",
            f"size_t {shell_name}_address(void) {{",
            f"    return (size_t)&{shell_name};",
            "}",
            "",
        )
    )
    return source, feed_names, output_specs


class CJITProgram:
    def __init__(
        self,
        *,
        ffi: Any,
        library: Any,
        function_name: str,
        shell_name: str,
        feed_names: tuple[str, ...],
        outputs: tuple[COutputSpec, ...],
        source_artifact: CachedArtifact,
    ):
        self.ffi = ffi
        self.library = library
        self.function_name = function_name
        self.shell_name = shell_name
        self.feed_names = feed_names
        self.output_specs = outputs
        self.source_artifact = source_artifact
        self._shell_address = int(
            getattr(library, f"{shell_name}_address")()
        )

    def execute(
        self,
        inputs: Mapping[str, Any],
        *,
        profiler: Any | None = None,
        profile_path: str = "torture/c",
    ) -> CJITExecution:
        missing = set(self.feed_names) - set(inputs)
        if missing:
            raise ValueError(f"missing C JIT feeds: {sorted(missing)}")
        feed_arrays = [
            np.ascontiguousarray(inputs[name], dtype=np.float64)
            for name in self.feed_names
        ]
        output_arrays = [
            np.empty(spec.shape or (), dtype=np.float64)
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
        token = (
            profiler.begin_shell(profile_path)
            if profiler is not None
            else None
        )
        profile = shell.launch(self._shell_address, context)
        if profiler is not None:
            shell.record(
                profiler,
                profile,
                path=profile_path,
                label=self.function_name,
            )
            profiler.end_shell(profile_path, token)
        if profile.status != 1:
            raise RuntimeError(
                f"C compute closure returned status {profile.status}"
            )
        return CJITExecution(
            outputs={
                spec.name: array
                for spec, array in zip(self.output_specs, output_arrays)
            },
            profile=profile,
            cache_hit=self.source_artifact.hit,
        )


def compile_torture_case_to_c(
    captured: CapturedTortureCase,
    *,
    cache: RepositoryArtifactCache | None = None,
) -> CJITProgram:
    cache = cache or RepositoryArtifactCache()
    function_name = "turing_torture_compute"
    shell_name = "turing_torture_c_shell_entry"
    record = {
        "case": captured.case.semantic_record(),
        "compiler": "turing-direct-ctensor-c",
        "implementation": _c_implementation_digest(),
        "optimization": False,
    }
    artifact = cache.load("c", record, suffix=".c")
    if artifact is None:
        source, feed_names, outputs = _emit_c_tape(
            captured,
            function_name=function_name,
            shell_name=shell_name,
        )
        artifact = cache.store(
            "c",
            record,
            source,
            suffix=".c",
            metadata={
                "feed_names": list(feed_names),
                "outputs": [
                    {"name": item.name, "shape": list(item.shape)}
                    for item in outputs
                ],
            },
        )
    else:
        metadata = artifact.manifest["metadata"]
        feed_names = tuple(metadata["feed_names"])
        outputs = tuple(
            COutputSpec(item["name"], tuple(item["shape"]))
            for item in metadata["outputs"]
        )
    from cffi import FFI

    ffi = FFI()
    ffi.cdef(
        f"int {shell_name}(void *, unsigned long long *);"
        f"size_t {shell_name}_address(void);"
    )
    # CFFI derives its extension name from the source but writes through a
    # shared temporary directory.  On Windows, one loaded extension prevents
    # the linker from replacing that DLL.  Give each immutable source identity
    # its own build directory so distinct torture programs never contend for a
    # loaded module path.
    build_directory = (
        repository_cache_root()
        / "c"
        / "cffi"
        / artifact.identity[:16]
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    library = ffi.verify(
        artifact.source,
        include_dirs=[
            str(Path(__file__).resolve().parent / "c_backend")
        ],
        tmpdir=str(build_directory),
    )
    return CJITProgram(
        ffi=ffi,
        library=library,
        function_name=function_name,
        shell_name=shell_name,
        feed_names=feed_names,
        outputs=outputs,
        source_artifact=artifact,
    )


__all__ = [
    "CJITExecution",
    "CJITProgram",
    "CJITShortfall",
    "COutputSpec",
    "compile_torture_case_to_c",
]
