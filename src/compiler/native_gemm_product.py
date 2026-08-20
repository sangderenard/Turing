"""Compile a prebaked GEMM launch matrix into one native pooled product.

The tiling strategy already records every source offset, source stride,
packed stride, core binding, lane and pool claim.  This module is the product
consumer of that artifact: it emits the context-bearing C lane which packs
the recorded windows, calls the admitted specialized LLVM core, publishes
each disjoint C tile, and joins the lanes through ``turing_pool.c``.

The finished shared library contains the core LLVM IR and the pool runtime.
It has no Python runtime dependency; Python is involved only at build and
measurement time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


PRODUCT_SCHEMA = "turing.native-gemm-product.v1"


class NativeGemmProductError(RuntimeError):
    """The launch matrix cannot be represented or the product did not build."""


@dataclass(frozen=True)
class NativeGemmProductSource:
    function_name: str
    source: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class NativeGemmProduct:
    function_name: str
    library_path: Path
    source_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _identifier(value: str) -> str:
    if not value.isidentifier():
        raise NativeGemmProductError(f"invalid C identifier {value!r}")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise NativeGemmProductError(f"{label} must be an integer") from error
    if result < minimum:
        raise NativeGemmProductError(
            f"{label} must be at least {minimum}, got {result}"
        )
    return result


def _mapping_fields(mapping: Mapping[str, Any], prefix: str) -> tuple[int, ...]:
    shape = tuple(mapping.get("source_shape") or ())
    strides = tuple(mapping.get("source_strides") or ())
    packed_shape = tuple(mapping.get("packed_shape") or ())
    packed_strides = tuple(mapping.get("packed_strides") or ())
    if len(shape) != 2 or len(strides) != 2:
        raise NativeGemmProductError(f"{prefix} must be a rank-two source window")
    if len(packed_shape) != 2 or len(packed_strides) != 2:
        raise NativeGemmProductError(f"{prefix} must be a rank-two packed window")
    return (
        _integer(mapping.get("source_offset"), f"{prefix}.source_offset"),
        _integer(shape[0], f"{prefix}.source_rows", minimum=1),
        _integer(shape[1], f"{prefix}.source_columns", minimum=1),
        _integer(strides[0], f"{prefix}.source_row_stride", minimum=1),
        _integer(strides[1], f"{prefix}.source_column_stride", minimum=1),
        _integer(packed_strides[0], f"{prefix}.packed_row_stride", minimum=1),
        _integer(packed_strides[1], f"{prefix}.packed_column_stride", minimum=1),
    )


def render_prebaked_gemm_product_c(
    core: Any,
    launch_matrix: Mapping[str, Any],
    *,
    function_name: str = "turing_gemm_product",
) -> NativeGemmProductSource:
    """Render one fully specialized, correctness-admitted GEMM product."""

    function_name = _identifier(str(function_name))
    matrix = dict(launch_matrix)
    if matrix.get("schema") != "turing.prebaked-gemm-launch-matrix.v1":
        raise NativeGemmProductError("unsupported prebaked GEMM matrix schema")
    if str(matrix.get("module_key")) != str(core.key):
        raise NativeGemmProductError(
            "launch matrix module key does not name the supplied core"
        )
    recorded_bindings = {
        str(name): int(identifier)
        for name, identifier in dict(
            matrix.get("module_binding_by_name") or {}
        ).items()
    }
    live_bindings = {
        str(name): int(identifier)
        for name, identifier in dict(core.id_by_name).items()
    }
    if recorded_bindings != live_bindings:
        raise NativeGemmProductError(
            "launch matrix deterministic bindings do not match the core: "
            f"matrix={recorded_bindings!r}, core={live_bindings!r}"
        )
    required = {"A", "B", "C", "alpha", "beta"}
    if set(live_bindings) != required:
        raise NativeGemmProductError(
            f"native GEMM product requires exactly {sorted(required)!r}; "
            f"core exposes {sorted(live_bindings)!r}"
        )
    artifact = core.native
    if tuple(getattr(artifact, "extent_order", ()) or ()):
        raise NativeGemmProductError(
            "native GEMM product requires a fully size-specialized core"
        )
    buffer_order = tuple(map(int, artifact.buffer_order))
    if set(buffer_order) != set(live_bindings.values()):
        raise NativeGemmProductError(
            "core public buffer order is not the deterministic parameter set"
        )
    if any(str(dtype) != "double" for dtype in artifact.buffer_dtypes):
        raise NativeGemmProductError("native GEMM product currently requires f64 ABI slots")

    problem = tuple(matrix.get("problem_shape") or ())
    tile_shape = tuple(matrix.get("tile_shape") or ())
    if len(problem) != 3 or len(tile_shape) != 3:
        raise NativeGemmProductError("problem_shape and tile_shape must be rank three")
    m, n, k = (
        _integer(value, f"problem_shape[{axis}]", minimum=1)
        for axis, value in enumerate(problem)
    )
    tm, tn, tk = (
        _integer(value, f"tile_shape[{axis}]", minimum=1)
        for axis, value in enumerate(tile_shape)
    )
    if (tm, tn, tk) != (tm, tm, tm):
        raise NativeGemmProductError("the admitted GEMM core must be square")
    tile = tm
    launch = dict(matrix.get("launch") or {})
    workers = _integer(launch.get("workers"), "launch.workers")
    chunk = _integer(launch.get("chunk_size"), "launch.chunk_size", minimum=1)
    lanes = tuple(matrix.get("lanes") or ())
    lane_count = _integer(launch.get("lane_count"), "launch.lane_count", minimum=1)
    if lane_count != len(lanes):
        raise NativeGemmProductError("launch lane_count does not match lane records")

    call_rows: list[tuple[int, ...]] = []
    lane_rows: list[tuple[int, ...]] = []
    pack_rows: list[tuple[int, ...]] = []
    pack_indices: dict[tuple[int, ...], int] = {}

    def intern_pack(source_kind: int, fields: tuple[int, ...]) -> int:
        key = (source_kind, *fields)
        if key not in pack_indices:
            pack_indices[key] = len(pack_rows)
            pack_rows.append(key)
        return pack_indices[key]

    for lane_index, lane_record in enumerate(lanes):
        if _integer(lane_record.get("lane"), "lane index") != lane_index:
            raise NativeGemmProductError("lane records must be dense and ordered")
        calls = tuple(lane_record.get("calls") or ())
        if not calls:
            raise NativeGemmProductError(f"lane {lane_index} has no core calls")
        first_call = len(call_rows)
        for call_index, call in enumerate(calls):
            if str(call.get("module_key")) != str(core.key):
                raise NativeGemmProductError(
                    f"lane {lane_index} call {call_index} names another core"
                )
            parameters = dict(call.get("parameters_by_name") or {})
            a = _mapping_fields(parameters.get("A") or {}, "A")
            b = _mapping_fields(parameters.get("B") or {}, "B")
            c = _mapping_fields(parameters.get("C") or {}, "C")
            if any(value > tile for value in (a[1], a[2], b[1], b[2], c[1], c[2])):
                raise NativeGemmProductError("a source window exceeds the packed core")
            beta = parameters.get("beta")
            beta_mode = 0 if beta == "caller_beta" else 1 if float(beta) == 1.0 else -1
            if beta_mode < 0:
                raise NativeGemmProductError(
                    "prebaked GEMM product supports caller beta then unit accumulation"
                )
            a_pack = intern_pack(0, a)
            b_pack = intern_pack(1, b)
            call_rows.append((*a, *b, *c, beta_mode, a_pack, b_pack))
        c_fields = call_rows[-1][14:21]
        lane_rows.append((first_call, len(calls), *c_fields))

    def c_rows(rows: list[tuple[int, ...]]) -> str:
        return ",\n".join(
            "    { " + ", ".join(map(str, row)) + " }" for row in rows
        )

    slot_lines = []
    slot_value = {
        live_bindings["A"]: "a_tile",
        live_bindings["B"]: "b_tile",
        live_bindings["C"]: "c_tile",
        live_bindings["alpha"]: "&context->alpha",
        live_bindings["beta"]: "&call_beta",
    }
    for ordinal, value_id in enumerate(buffer_order):
        slot_lines.append(f"        buffers[{ordinal}] = {slot_value[value_id]};")

    core_symbol = _identifier(str(artifact.name))
    source = f"""#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef void (*turing_span_fn)(void*, long, long);
extern int turing_pool_start(int workers);
extern int turing_pool_deploy_span(turing_span_fn fn, void* context,
                                   long item_count, long chunk_size);
extern void turing_pool_stop(void);
extern void {core_symbol}(void **buffers, int32_t *extents);

typedef struct {{
    long a_offset, a_rows, a_columns, a_row_stride, a_column_stride;
    long a_packed_row_stride, a_packed_column_stride;
    long b_offset, b_rows, b_columns, b_row_stride, b_column_stride;
    long b_packed_row_stride, b_packed_column_stride;
    long c_offset, c_rows, c_columns, c_row_stride, c_column_stride;
    long c_packed_row_stride, c_packed_column_stride;
    int beta_mode;
    long a_pack, b_pack;
}} TuringGemmCall;

typedef struct {{
    int source_kind;
    long offset, rows, columns, row_stride, column_stride;
    long packed_row_stride, packed_column_stride;
}} TuringGemmPack;

typedef struct {{
    long first_call, call_count;
    long c_offset, c_rows, c_columns, c_row_stride, c_column_stride;
    long c_packed_row_stride, c_packed_column_stride;
}} TuringGemmLane;

typedef struct {{
    const double *a;
    const double *b;
    double *c;
    double alpha;
    double beta;
    double *scratch;
}} TuringGemmContext;

static const TuringGemmCall turing_gemm_calls[] = {{
{c_rows(call_rows)}
}};
static const TuringGemmPack turing_gemm_packs[] = {{
{c_rows(pack_rows)}
}};
static const TuringGemmLane turing_gemm_lanes[] = {{
{c_rows(lane_rows)}
}};

static void turing_pack(double *packed, const double *source,
                        long offset, long rows, long columns,
                        long source_row_stride, long source_column_stride,
                        long packed_row_stride, long packed_column_stride) {{
    long row, column;
    /* Full windows overwrite every packed element. Zeroing them first is
       pure bandwidth; only padded edge windows need a clean margin. */
    if (rows != {tile} || columns != {tile})
        memset(packed, 0, sizeof(double) * {tile * tile});
    if (source_column_stride == 1 && packed_column_stride == 1) {{
        for (row = 0; row < rows; ++row) {{
            memcpy(packed + row * packed_row_stride,
                   source + offset + row * source_row_stride,
                   sizeof(double) * (size_t)columns);
        }}
        return;
    }}
    for (row = 0; row < rows; ++row)
        for (column = 0; column < columns; ++column)
            packed[row * packed_row_stride + column * packed_column_stride] =
                source[offset + row * source_row_stride
                              + column * source_column_stride];
}}

static void turing_publish(double *destination, const double *packed,
                           long offset, long rows, long columns,
                           long destination_row_stride,
                           long destination_column_stride,
                           long packed_row_stride,
                           long packed_column_stride) {{
    long row, column;
    if (destination_column_stride == 1 && packed_column_stride == 1) {{
        for (row = 0; row < rows; ++row) {{
            memcpy(destination + offset + row * destination_row_stride,
                   packed + row * packed_row_stride,
                   sizeof(double) * (size_t)columns);
        }}
        return;
    }}
    for (row = 0; row < rows; ++row)
        for (column = 0; column < columns; ++column)
            destination[offset + row * destination_row_stride
                               + column * destination_column_stride] =
                packed[row * packed_row_stride + column * packed_column_stride];
}}

static void turing_gemm_lane_span(void *raw, long start, long stop) {{
    TuringGemmContext *context = (TuringGemmContext *)raw;
    long lane_index;
    for (lane_index = start; lane_index < stop; ++lane_index) {{
        const TuringGemmLane *lane = &turing_gemm_lanes[lane_index];
        double *c_tile = context->scratch
            + (long)({len(pack_rows)} + lane_index) * {tile * tile};
        long call_index;
        turing_pack(c_tile, context->c, lane->c_offset,
                    lane->c_rows, lane->c_columns,
                    lane->c_row_stride, lane->c_column_stride,
                    lane->c_packed_row_stride, lane->c_packed_column_stride);
        for (call_index = lane->first_call;
             call_index < lane->first_call + lane->call_count;
             ++call_index) {{
            const TuringGemmCall *call = &turing_gemm_calls[call_index];
            double *a_tile = context->scratch
                + call->a_pack * {tile * tile};
            double *b_tile = context->scratch
                + call->b_pack * {tile * tile};
            double call_beta = call->beta_mode ? 1.0 : context->beta;
            void *buffers[{len(buffer_order)}];
{chr(10).join(slot_lines)}
            {core_symbol}(buffers, 0);
        }}
        turing_publish(context->c, c_tile, lane->c_offset,
                       lane->c_rows, lane->c_columns,
                       lane->c_row_stride, lane->c_column_stride,
                       lane->c_packed_row_stride,
                       lane->c_packed_column_stride);
    }}
}}

static void turing_gemm_pack_span(void *raw, long start, long stop) {{
    TuringGemmContext *context = (TuringGemmContext *)raw;
    long index;
    for (index = start; index < stop; ++index) {{
        const TuringGemmPack *pack = &turing_gemm_packs[index];
        const double *source = pack->source_kind ? context->b : context->a;
        double *destination = context->scratch + index * {tile * tile};
        turing_pack(destination, source, pack->offset,
                    pack->rows, pack->columns,
                    pack->row_stride, pack->column_stride,
                    pack->packed_row_stride, pack->packed_column_stride);
    }}
}}

static int turing_gemm_execute(const double *a, const double *b, double *c,
                               double alpha, double beta, int use_pool) {{
    TuringGemmContext context;
    int status;
    if (a == 0 || b == 0 || c == 0) return -1;
    context.a = a; context.b = b; context.c = c;
    context.alpha = alpha; context.beta = beta;
    context.scratch = (double *)malloc(
        sizeof(double) * (size_t){(len(pack_rows) + lane_count) * tile * tile});
    if (context.scratch == 0) return -2;
    if (!use_pool) {{
        turing_gemm_pack_span(&context, 0, {len(pack_rows)});
        turing_gemm_lane_span(&context, 0, {lane_count});
        status = 0;
    }} else {{
        status = turing_pool_start({workers});
        if (status < 0) {{
            turing_gemm_pack_span(&context, 0, {len(pack_rows)});
            turing_gemm_lane_span(&context, 0, {lane_count});
            status = 1;
        }} else {{
            status = turing_pool_deploy_span(
                turing_gemm_pack_span, &context, {len(pack_rows)}, 1);
            if (status != 0) {{
                turing_gemm_pack_span(&context, 0, {len(pack_rows)});
            }}
            if (turing_pool_deploy_span(
                    turing_gemm_lane_span, &context, {lane_count}, {chunk}) != 0) {{
                turing_gemm_lane_span(&context, 0, {lane_count});
                status = 1;
            }} else if (status != 0) {{
                status = 1;
            }}
        }}
    }}
    free(context.scratch);
    return status;
}}

int {function_name}(const double *a, const double *b, double *c,
                    double alpha, double beta) {{
    return turing_gemm_execute(a, b, c, alpha, beta, 1);
}}

int {function_name}_serial(const double *a, const double *b, double *c,
                           double alpha, double beta) {{
    return turing_gemm_execute(a, b, c, alpha, beta, 0);
}}

int {function_name}_m(void) {{ return {m}; }}
int {function_name}_n(void) {{ return {n}; }}
int {function_name}_k(void) {{ return {k}; }}
int {function_name}_tile(void) {{ return {tile}; }}
int {function_name}_workers(void) {{ return {workers}; }}
int {function_name}_chunk(void) {{ return {chunk}; }}
int {function_name}_lanes(void) {{ return {lane_count}; }}
void {function_name}_shutdown(void) {{ turing_pool_stop(); }}
"""
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    manifest = {
        "schema": PRODUCT_SCHEMA,
        "function": function_name,
        "problem_shape": [m, n, k],
        "tile_shape": [tile, tile, tile],
        "core": {
            "key": str(core.key),
            "symbol": core_symbol,
            "binding_by_name": live_bindings,
            "buffer_order": list(buffer_order),
        },
        "launch": {
            "workers": workers,
            "chunk_size": chunk,
            "lane_count": lane_count,
            "join": "barrier",
        },
        "packing": {
            "unique_a_windows": sum(1 for row in pack_rows if row[0] == 0),
            "unique_b_windows": sum(1 for row in pack_rows if row[0] == 1),
            "prepacked_once_per_product_call": True,
        },
        "source_sha256": source_sha,
        "dependencies": ["embedded LLVM core", "turing_pool.c"],
        "python_runtime_dependency": False,
        "fallback": "serial execution of the same prebaked lane span",
        "serial_control": f"{function_name}_serial",
    }
    return NativeGemmProductSource(function_name, source, manifest)


def compile_prebaked_gemm_product(
    core: Any,
    launch_matrix: Mapping[str, Any],
    directory: str | Path,
    *,
    function_name: str = "turing_gemm_product",
) -> NativeGemmProduct:
    """Build the generated executor, LLVM core and pool into one library."""

    rendered = render_prebaked_gemm_product_c(
        core, launch_matrix, function_name=function_name,
    )
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / f"{rendered.function_name}.c"
    source_path.write_text(rendered.source, encoding="utf-8")
    core_source = Path(core.native.library_path).with_suffix(".ll")
    if not core_source.is_file():
        raise NativeGemmProductError(f"core LLVM source is missing: {core_source}")
    pool_source = (
        Path(__file__).resolve().parents[1]
        / "common" / "tensors" / "accelerator_backends" / "c_backend"
        / "turing_pool.c"
    )
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    library_path = output / f"{rendered.function_name}{suffix}"
    command = [
        sys.executable, "-m", "ziglang", "cc", "-shared", "-O3",
        "-march=native", str(source_path), str(pool_source),
        str(core_source), "-o", str(library_path),
    ]
    if os.name != "nt":
        command.append("-pthread")
    completed = subprocess.run(
        command, cwd=str(output), capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0 or not library_path.is_file():
        raise NativeGemmProductError(
            f"native GEMM product build failed ({completed.returncode}):\n"
            + (completed.stderr or completed.stdout)[-4000:]
        )
    manifest = dict(rendered.manifest)
    manifest["build"] = {
        "command": command,
        "core_llvm": str(core_source),
        "pool_source": str(pool_source),
        "library": str(library_path),
    }
    manifest_path = output / f"{rendered.function_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return NativeGemmProduct(
        rendered.function_name, library_path, source_path, manifest_path,
        manifest,
    )


def compile_native_gemm_product(
    bank: Any,
    sizes: Mapping[str, int],
    directory: str | Path,
    *,
    contract: str | None = "fast",
    cores: int | None = None,
    candidate_sizes: tuple[int, ...] = (32, 64, 128, 256),
    function_name: str | None = None,
) -> NativeGemmProduct:
    """Compile ordinary bank GEMM source into a chosen pooled native product.

    This is the canonical orchestration seam: build and admit the candidate
    specializations, rank their composed critical paths, select the native
    worker/chunk geometry, prebake every layout permutation, then compile the
    resulting matrix. No tile or launch fact is selected by the caller.
    """

    from .deployment_lowering import select_deployment_strategy
    from .kernel_bank import parameter_layout_permutation
    from .tiling_strategy import (
        build_gemm_tile_plan,
        decide_tiling,
        prebake_gemm_launch_matrix,
    )

    shape = {
        axis: _integer(sizes.get(axis), axis, minimum=1)
        for axis in ("m", "n", "k")
    }
    for candidate in sorted(set(map(int, candidate_sizes))):
        if candidate <= min(shape.values()):
            bank.get(
                "gemm", contract=contract,
                specialized={axis: candidate for axis in ("m", "n", "k")},
            )
    decision = decide_tiling(
        bank, "gemm", shape, contract=contract, cores=cores,
        must_divide=False,
    )
    if not decision.tiled or decision.tile is None:
        raise NativeGemmProductError(
            "native pooled GEMM composition was not selected: "
            + "; ".join(decision.reasons)
        )
    tile = int(decision.tile)
    core = bank.get(
        "gemm", contract=contract,
        specialized={axis: tile for axis in ("m", "n", "k")},
    )
    lane_count = (
        ((shape["m"] + tile - 1) // tile)
        * ((shape["n"] + tile - 1) // tile)
    )
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers", join_mode="barrier",
        work=lane_count, cores=cores,
    )
    if choice.strategy != "pool" or not choice.workers:
        raise NativeGemmProductError(
            "native C deployment did not select a usable pool: "
            + "; ".join(choice.reasons)
        )
    plan = build_gemm_tile_plan(
        shape["m"], shape["n"], shape["k"], tile,
        worker_budget=int(choice.workers),
        reasons=(*decision.reasons, *choice.reasons),
    )
    matrix = prebake_gemm_launch_matrix(
        plan,
        variant_key=core.key,
        parameter_ids=core.id_by_name,
        total_layout=parameter_layout_permutation(core.spec, shape),
        core_layout=parameter_layout_permutation(
            core.spec, {axis: tile for axis in ("m", "n", "k")},
        ),
        chunk_size=int(choice.chunk or 1),
    )
    matrix_payload = json.dumps(
        matrix, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    matrix_sha = hashlib.sha256(matrix_payload).hexdigest()
    product_name = function_name or (
        f"turing_gemm_{shape['m']}_{shape['n']}_{shape['k']}"
    )
    product_directory = (
        Path(directory).resolve()
        / f"gemm_{shape['m']}_{shape['n']}_{shape['k']}_{tile}_{matrix_sha[:12]}"
    )
    product = compile_prebaked_gemm_product(
        core, matrix, product_directory, function_name=product_name,
    )
    manifest = dict(product.manifest)
    manifest["compiler_decision"] = {
        "tiled": decision.tiled,
        "tile": decision.tile,
        "worker_budget": decision.worker_budget,
        "candidates": [list(candidate) for candidate in decision.candidates],
        "reasons": list(decision.reasons),
    }
    manifest["deployment_choice"] = {
        "backend": choice.backend,
        "strategy": choice.strategy,
        "workers": choice.workers,
        "chunk": choice.chunk,
        "reasons": list(choice.reasons),
    }
    manifest["launch_matrix_sha256"] = matrix_sha
    product.manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return NativeGemmProduct(
        product.function_name, product.library_path, product.source_path,
        product.manifest_path, manifest,
    )


__all__ = [
    "NativeGemmProduct",
    "NativeGemmProductError",
    "NativeGemmProductSource",
    "PRODUCT_SCHEMA",
    "compile_prebaked_gemm_product",
    "compile_native_gemm_product",
    "render_prebaked_gemm_product_c",
]
