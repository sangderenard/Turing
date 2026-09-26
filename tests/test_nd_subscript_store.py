"""A full-rank N-D subscript store (``out[i, j] = value``) flattens against the
output tensor's row-major strides.

The 1-D scatter addresses elements directly (stride 1). A 2-D store must fold
each subscript by its dimension's stride: element ``[i, j]`` of a ``(R, C)``
buffer sits at ``(i*C + j)`` elements. This is what the decoder's token
multigraph stores need; a *partial*-rank index (fewer subscripts than
dimensions) stays an honest shortfall until a block copy is lowered.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep, Meta
from src.compiler.fused_program_wasm_backend import emit_wasm_module


def _store_2d(shape=(3, 4)):
    """``data[i, j] = value`` with runtime i, j into a row-major buffer."""
    return FusedProgram(
        version=1, feeds={0, 1, 2, 3},
        steps=[OpStep(0, "IndexedStore", [0, 1, 2, 3],
                      {"source_type": "SubscriptStore"}, 4)],
        outputs={"data": 4},
        meta={4: Meta(shape=shape, dtype="int64")},
        extras={"capture_feed_origins": {
            0: {"binding_name": "data"},
            1: {"binding_name": "i"},
            2: {"binding_name": "j"},
            3: {"binding_name": "value"}}},
    )


def test_2d_subscript_store_emits_complete():
    module = emit_wasm_module(_store_2d(), name="s2d", dtype="int64")
    assert module.complete, module.shortfall_report()
    # The inner axis scales by width (8) and the outer axis by C*width (32).
    assert "i32.const 32" in module.source  # stride 4 elements * 8 bytes
    assert "i64.store" in module.source


def test_partial_rank_store_is_an_honest_shortfall():
    # Two subscripts into a 3-D buffer would write a whole row, not one cell.
    program = FusedProgram(
        version=1, feeds={0, 1, 2, 3},
        steps=[OpStep(0, "IndexedStore", [0, 1, 2, 3],
                      {"source_type": "SubscriptStore"}, 4)],
        outputs={"data": 4},
        meta={4: Meta(shape=(2, 3, 4), dtype="int64")},
        extras={"capture_feed_origins": {
            0: {"binding_name": "data"}, 1: {"binding_name": "i"},
            2: {"binding_name": "j"}, 3: {"binding_name": "value"}}},
    )
    module = emit_wasm_module(program, name="partial", dtype="int64")
    assert not module.complete
    assert "full-rank" in module.shortfall_report()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_2d_subscript_store_runs_correctly(tmp_path):
    module = emit_wasm_module(_store_2d((3, 4)), name="s2d", dtype="int64")
    assert module.complete, module.shortfall_report()
    wasm = tmp_path / "s2d.wasm"
    wasm.write_bytes(module.binary)
    # A 3x4 row-major buffer; write value 77 at [1, 2] -> flat index 1*4+2 = 6.
    script = tmp_path / "run.mjs"
    script.write_text(
        """
        import {readFileSync} from "node:fs";
        const mod = await WebAssembly.instantiate(readFileSync(process.argv[2]), {});
        const {run, memory} = mod.instance.exports;
        const mem = new BigInt64Array(memory.buffer);
        const N = 12;
        for (let k=0;k<N;k++) mem[0+k]=BigInt(k);   // data  @ byte 0   (12 cells)
        for (let k=0;k<N;k++) mem[12+k]=1n;          // i     @ byte 96
        for (let k=0;k<N;k++) mem[24+k]=2n;          // j     @ byte 192
        for (let k=0;k<N;k++) mem[36+k]=77n;         // value @ byte 288
        for (let k=0;k<N;k++) mem[48+k]=0n;          // out   @ byte 384
        run(N, 0, 96, 192, 288, 384);
        const out=[]; for (let k=0;k<N;k++) out.push(Number(mem[48+k]));
        console.log(JSON.stringify(out));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)], capture_output=True, text=True, check=True,
    )
    # Copy of 0..11 with index 6 (row 1, col 2) overwritten by 77.
    assert completed.stdout.strip().endswith(
        "[0,1,2,3,4,5,77,7,8,9,10,11]"
    ), completed.stdout + completed.stderr
