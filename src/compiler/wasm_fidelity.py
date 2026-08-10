"""Run the emitted WebAssembly and record numerical parity with its source IR.

This is the Wasm twin of :mod:`fortran_fidelity`: a verification boundary, not
another lowering path.  It executes the exact ``.wasm`` binary a site bundle
publishes and compares its observable outputs with the NumPy rendering of the
same :class:`FusedProgram` -- the very NumPy source the bundle already exposes
under its ``NumPy`` backend tab.  A mismatch raises ``AssertionError`` so a
bundle can never claim successful verification while shipping numerically
different WebAssembly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


WASM_FIDELITY_SCHEMA = "turing-wasm-fidelity-v1"

_NUMPY_DTYPES = {
    "f32": "float32",
    "f64": "float64",
    # Integer working types. The oracle, the feed layout, and the JS memory
    # view all key off this so a bitwise/integer program is verified against an
    # integer NumPy reference and integer linear memory, not silently coerced
    # to float (which would make ``np.bitwise_and`` raise, and would round a
    # shift result). The compiler's own integer results default to int64, so
    # i64 is the primary integer working type here; i32 is supported for
    # narrower programs.
    "i32": "int32",
    "i64": "int64",
}

# Working types whose linear-memory values are integers, not IEEE floats. i64
# additionally needs BigInt on the JS side (a 64-bit integer has no exact
# IEEE-double JSON encoding), handled in the run script and re-parsed from
# decimal strings on the Python side.
_INTEGER_VALUE_TYPES = {"i32", "i64"}
_BIGINT_VALUE_TYPES = {"i64"}

# The reference-case generator and the NumPy oracle both already exist for the
# Fortran path; reuse them rather than minting parallel copies.
from .fortran_fidelity import deterministic_verification_cases


def node_runtime() -> str | None:
    """``node`` if it is installed.  Emission never needs it."""

    return shutil.which("node")


def _element_count(program: Any, value_id: int) -> int:
    from .fused_program_wasm_backend import _shape_product

    meta = (getattr(program, "meta", None) or {}).get(int(value_id))
    shape = getattr(meta, "shape", None) if meta is not None else None
    total = _shape_product(shape)
    return int(total) if total else 1


def _invocation_extent(program: Any, output_ids: Sequence[int]) -> int:
    """The outer lane length ``run(count, ...)`` iterates.

    For an axis reduction (rank change N*K -> N) it is the surviving output
    extent; otherwise it is the flat element extent the elementwise / whole
    tensor loop walks, which the largest feed reports.
    """

    from .fused_program_wasm_backend import (
        _plan_axis_reductions,
        program_feed_order,
        required_steps,
    )

    live = required_steps(program)
    plan = _plan_axis_reductions(program, live, program_feed_order(program), [])
    if plan is not None and plan.ok:
        return _element_count(program, output_ids[0])
    feed_counts = [
        _element_count(program, feed_id) for feed_id in program_feed_order(program)
    ]
    output_counts = [_element_count(program, value_id) for value_id in output_ids]
    return max(feed_counts + output_counts + [1])


_RUN_SCRIPT = """
import { readFileSync } from "node:fs";
const [wasmPath, planPath] = process.argv.slice(2);
const plan = JSON.parse(readFileSync(planPath, "utf-8"));
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const memory = instance.exports.memory;
if (plan.required_bytes > memory.buffer.byteLength) {
  memory.grow(Math.ceil((plan.required_bytes - memory.buffer.byteLength) / 65536));
}
const isI64 = plan.value_type === "i64";
const View =
  plan.value_type === "f32" ? Float32Array :
  plan.value_type === "i32" ? Int32Array :
  plan.value_type === "i64" ? BigInt64Array :
  Float64Array;
for (const feed of plan.feeds) {
  // A 64-bit integer view stores BigInt elements; the plan carries them as
  // plain JSON numbers (small, exact) and they are lifted to BigInt here.
  const data = isI64 ? feed.data.map(BigInt) : feed.data;
  new View(memory.buffer, feed.offset, feed.data.length).set(data);
}
instance.exports.run(plan.count, ...plan.run_offsets);
const outputs = plan.outputs.map(o => {
  const window = Array.from(new View(memory.buffer, o.offset, o.length));
  // BigInt has no JSON encoding, so a 64-bit result is emitted as decimal
  // strings and re-parsed to integers on the Python side.
  return isI64 ? window.map(String) : window;
});
console.log(JSON.stringify(outputs));
"""


def _json_value(array: Any) -> Any:
    return array.tolist()


def verify_wasm_module(
    module: Any,
    program: Any,
    feeds: Mapping[int, Any],
    directory: str | Path,
    *,
    entrypoint: str,
    cases: Sequence[tuple[str, Mapping[int, Any]]] | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Instantiate ``module`` in Node, run cases, return a proof record.

    ``module`` is the :class:`WasmModule` a site bundle publishes; it must own
    its exported memory (the top-level numeric module does) and carry an
    assembled ``binary``.  A mismatch with the NumPy oracle raises
    ``AssertionError``.
    """

    import numpy as np

    from ..common.tensors.fused_ir import ordered_feed_ids
    from .fused_program_python_backend import compile_single_region_python
    from .fused_program_wasm_backend import program_feed_order

    node = node_runtime()
    if node is None:
        raise RuntimeError("node is required for wasm fidelity verification")
    if getattr(module, "binary", None) is None:
        raise RuntimeError("wasm fidelity needs an assembled binary to run")
    metadata = dict(module.api.metadata)
    if metadata.get("shared_memory_import"):
        raise RuntimeError(
            "wasm fidelity verifies exported-memory modules; this one imports "
            "its memory and is driven by the control path"
        )

    value_type = str(metadata.get("value_type", "f64"))
    element_bytes = int(metadata.get("element_bytes", 8))
    np_dtype = _NUMPY_DTYPES.get(value_type, "float64")

    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    module_path = output_directory / f"{entrypoint}.wasm"
    module_path.write_bytes(module.binary)
    script_path = output_directory / f"{entrypoint}.run.mjs"
    script_path.write_text(_RUN_SCRIPT, encoding="utf-8")

    reference_feed_ids = ordered_feed_ids(program)
    abi_feed_ids = program_feed_order(program)
    output_ids = list(program.outputs.values())
    feed_names = {
        feed_id: f"feed{index}" for index, feed_id in enumerate(reference_feed_ids)
    }
    reference = compile_single_region_python(
        program,
        feed_names,
        dialect="numpy",
        function_name=f"{entrypoint}_reference",
    ).callable

    feed_origins = dict(
        (getattr(program, "extras", None) or {}).get("capture_feed_origins", {})
    )
    count = _invocation_extent(program, output_ids)

    selected_cases = tuple(cases or deterministic_verification_cases(feeds))
    case_records: list[dict[str, Any]] = []

    for case_name, case_feeds in selected_cases:
        missing = set(reference_feed_ids) - set(case_feeds)
        if missing:
            raise ValueError(f"{case_name} is missing feed ids {sorted(missing)}")
        inputs = {
            feed_id: np.asarray(case_feeds[feed_id], dtype=np_dtype)
            for feed_id in reference_feed_ids
        }
        expected_raw = reference(*(inputs[feed_id] for feed_id in reference_feed_ids))
        expected_values = (
            tuple(expected_raw) if len(program.outputs) > 1 else (expected_raw,)
        )
        expected_by_id = {
            value_id: np.asarray(value, dtype=np_dtype)
            for value_id, value in zip(program.outputs.values(), expected_values)
        }

        # Lay feeds and output windows into the exported memory after the
        # baked-table region, element-aligned, in the module's own ABI order.
        cursor = ((int(metadata.get("reserved_bytes", 0)) + element_bytes - 1)
                  // element_bytes) * element_bytes
        # An integer working type lays integer elements into memory (read back
        # through an Int32Array in the run script); a float type lays floats.
        cast = int if value_type in _INTEGER_VALUE_TYPES else float
        feed_layout: list[dict[str, Any]] = []
        feed_offsets: dict[int, int] = {}
        for feed_id in abi_feed_ids:
            flat = np.asarray(inputs[feed_id], dtype=np_dtype).ravel(order="C")
            feed_offsets[feed_id] = cursor
            feed_layout.append({
                "offset": cursor,
                "data": [cast(v) for v in flat.tolist()],
            })
            cursor += int(flat.size) * element_bytes
        output_layout: list[dict[str, Any]] = []
        output_offsets: list[int] = []
        for value_id in output_ids:
            length = _element_count(program, value_id)
            output_offsets.append(cursor)
            output_layout.append({"offset": cursor, "length": length})
            cursor += length * element_bytes

        run_offsets = [feed_offsets[feed_id] for feed_id in abi_feed_ids]
        run_offsets += output_offsets
        plan = {
            "value_type": value_type,
            "count": int(count),
            "required_bytes": int(cursor),
            "feeds": feed_layout,
            "outputs": output_layout,
            "run_offsets": run_offsets,
        }
        plan_path = output_directory / f"{entrypoint}.{case_name}.plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        completed = subprocess.run(
            [node, str(script_path), str(module_path), str(plan_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        actual_outputs = json.loads(completed.stdout)

        outputs = []
        case_passed = True
        for position, (output_name, value_id) in enumerate(program.outputs.items()):
            expected = expected_by_id[value_id].ravel(order="C")
            actual = np.asarray(actual_outputs[position], dtype=np_dtype)
            passed = bool(np.allclose(
                actual, expected, rtol=rtol, atol=atol, equal_nan=True
            ))
            case_passed = case_passed and passed
            finite = np.isfinite(actual) & np.isfinite(expected)
            absolute_error = (
                float(np.max(np.abs(actual[finite] - expected[finite])))
                if np.any(finite) else 0.0
            )
            outputs.append({
                "name": output_name,
                "value_id": int(value_id),
                "length": int(actual.size),
                "reference": _json_value(expected),
                "wasm": _json_value(actual),
                "max_absolute_error": absolute_error,
                "passed": passed,
            })
        record = {
            "name": case_name,
            "inputs": [
                {
                    "name": feed_origins.get(
                        feed_id, feed_origins.get(str(feed_id), {})
                    ).get("binding_name", f"feed_{feed_id}"),
                    "value_id": int(feed_id),
                    "value": _json_value(inputs[feed_id]),
                }
                for feed_id in reference_feed_ids
            ],
            "outputs": outputs,
            "passed": case_passed,
        }
        case_records.append(record)
        if not case_passed:
            raise AssertionError(
                f"emitted WebAssembly disagrees with the reference in {case_name}: "
                + json.dumps(outputs, separators=(",", ":"))
            )

    source_bytes = module.binary
    return {
        "schema": WASM_FIDELITY_SCHEMA,
        "passed": all(case["passed"] for case in case_records),
        "entrypoint": entrypoint,
        "binary_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "wasm_module": module_path.name,
        "runtime": {"path": str(Path(node).resolve())},
        "tolerances": {"relative": rtol, "absolute": atol},
        "case_count": len(case_records),
        "cases": case_records,
    }


def verify_wasm_source(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    directory: str | Path,
    *,
    dtype: str = "float64",
    cases: Sequence[tuple[str, Mapping[int, Any]]] | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Compile one operator/expression from source and verify its WebAssembly.

    The single-call path for unit-testing operators as extensively as wanted:
    it runs the real AOT front end so ``feeds`` are named exactly as the
    function signature, projects the public numeric program, emits the module,
    and compares it in Node against the NumPy rendering of the same program.
    A step WebAssembly cannot yet express raises ``WasmEmissionError`` naming
    the operator, rather than silently skipping it.
    """

    from ..common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
        project_public_numerical_program,
    )
    from ..common.tensors.fused_ir import ordered_feed_ids
    from .fused_program_wasm_backend import WasmEmissionError, emit_wasm_module

    aot = compile_ast_aot(
        source, entrypoint, dict(feeds), precompile_only=True, remove_loops=True
    )
    program = project_public_numerical_program(aot)
    module = emit_wasm_module(program, name=entrypoint, dtype=dtype)
    if not module.complete:
        raise WasmEmissionError(module.shortfall_report())

    origins = program.extras["capture_feed_origins"]
    feed_values = {
        feed_id: feeds[origins[feed_id]["binding_name"]]
        for feed_id in ordered_feed_ids(program)
    }
    return verify_wasm_module(
        module,
        program,
        feed_values,
        directory,
        entrypoint=entrypoint,
        cases=cases,
        rtol=rtol,
        atol=atol,
    )


__all__ = [
    "WASM_FIDELITY_SCHEMA",
    "node_runtime",
    "verify_wasm_module",
    "verify_wasm_source",
]
