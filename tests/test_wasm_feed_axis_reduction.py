"""A trailing-axis reduction over a raw dense feed (not a derived grid).

``_plan_axis_reductions`` in the WASM backend only ever sees reductions whose
operand is *computed in-program* from row/kaxis feeds (see the ``_REDUCE_CASES``
in ``test_wasm_fidelity.py``); it explicitly refuses a feed classified as
grid-shaped (N*K), since the count-based ABI cannot size it.

``unroll_feed_axis_reductions`` (``fused_ir.py``) covers the gap: a raw (N, K)
feed reduced over its trailing axis is rewritten, before any other emission
logic runs, into K strided *views* of that same buffer (via the Meta view
descriptor) folded together with ordinary elementwise steps. This is a
genuinely different, non-overlapping case from the existing K-loop machinery
and does not replace it.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.fused_program_wasm_backend import emit_wasm_module

_RUN_SCRIPT = """
import { readFileSync } from "node:fs";
const [wasmPath, countArg, offsetsArg, dataArg, lengthsArg] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const memory = instance.exports.memory;
const offsets = JSON.parse(offsetsArg);
const data = JSON.parse(dataArg);
const lengths = JSON.parse(lengthsArg);
for (let i = 0; i < offsets.length - 1; i++) {
  new Float64Array(memory.buffer, offsets[i], data[i].length).set(data[i]);
}
instance.exports.run(Number(countArg), ...offsets);
const out = Array.from(
  new Float64Array(memory.buffer, offsets[offsets.length - 1], lengths[lengths.length - 1])
);
console.log(JSON.stringify(out));
"""


def _run_reduction(op_name: str, data: np.ndarray, tmp_path) -> np.ndarray:
    n, k = data.shape
    feed_id, result_id = 1, 2
    program = FusedProgram(
        version=1,
        feeds={feed_id},
        steps=[
            OpStep(
                step_id=0, op_name=op_name, input_ids=[feed_id], attrs={"axis": -1},
                result_id=result_id,
            )
        ],
        outputs={"out": result_id},
        meta={
            feed_id: Meta(shape=(n, k), dtype="float64"),
            result_id: Meta(shape=(n,), dtype="float64"),
        },
    )
    module = emit_wasm_module(program, name=f"reduce_{op_name}", dtype="float64")
    assert not module.shortfalls, module.shortfall_report()
    assert module.binary is not None

    wasm_path = tmp_path / f"{op_name}.wasm"
    wasm_path.write_bytes(module.binary)
    script_path = tmp_path / f"{op_name}.run.mjs"
    script_path.write_text(_RUN_SCRIPT, encoding="utf-8")

    feed_offset, out_offset = 8, 8 + n * k * 8
    result = subprocess.run(
        [
            shutil.which("node"), str(script_path), str(wasm_path), str(n),
            json.dumps([feed_offset, out_offset]),
            json.dumps([data.ravel(order="C").tolist()]),
            json.dumps([n]),
        ],
        capture_output=True, text=True, check=True,
    )
    return np.asarray(json.loads(result.stdout))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("op_name", ["sum", "mean", "prod", "min", "max"])
def test_raw_feed_axis_reduction_matches_numpy(tmp_path, op_name):
    data = np.arange(12, dtype=np.float64).reshape(4, 3) + 1.0
    expected = getattr(data, op_name)(axis=-1)
    actual = _run_reduction(op_name, data, tmp_path)
    assert np.allclose(actual, expected)
