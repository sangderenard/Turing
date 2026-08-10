"""A class field write (``self.data[index] = value``) lowers to a WebAssembly
scatter -- the write counterpart of ``gather``'s address+load.

This is the first reference-operator lowering toward classes-as-state-machines:
the fused-IR ``index_set`` operator (what subscript assignment compiles to) is
emitted as a per-cell copy of the source buffer followed by a single
address+store of the value at its subscript. The store is symmetric to how a
subscript *read* already lowers (``gather`` -> address+load), rather than being
reserved as a high-level whole-array select.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.compiler.fused_program_wasm_backend import emit_wasm_module


_SRC = textwrap.dedent(
    """
    class Buffer:
        def __init__(self, data):
            self.data = data
        def store(self, index, value):
            self.data[index] = value
            return self.data
    """
)


class Buffer:
    def __init__(self, data):
        self.data = data

    def store(self, index, value):
        self.data[index] = value
        return self.data


def _store_region():
    buffer = Buffer(AT.get_tensor([10, 20, 30, 40], dtype="int64"))
    with contextlib.redirect_stdout(io.StringIO()):
        compilation = compile_ast_aot(
            _SRC, "store",
            {
                "self": buffer,
                "index": AT.get_tensor(2, dtype="int64"),
                "value": AT.get_tensor(99, dtype="int64"),
            },
            precompile_only=True,
            python_bindings={"Buffer": Buffer},
        )
    region = next(iter(compilation.region_programs.values()))
    return getattr(region, "program", region)


def test_index_set_emits_a_complete_scatter_module():
    module = emit_wasm_module(_store_region(), name="store_r0", dtype="int64")
    assert module.complete, module.shortfall_report()
    assert module.binary[:4] == b"\x00asm"
    # index 2 * 8 bytes = a store at out + 16, after the elementwise copy walk.
    assert "i32.const 16" in module.source
    assert "i64.store" in module.source


def test_index_set_output_aliases_its_data_slot_in_place():
    """The field write is in place: the region's output resolves onto the same
    resident field slot as its ``data`` input, so the coordinator hands both the
    one buffer and the scatter mutates the live field rather than a copy."""

    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.control_source import ControlProgram, StatementBlock
    from src.compiler.wasm_class_modules import emit_control_region_modules
    from src.compiler.wasm_class_coordinator import build_class_inventory

    program = FusedProgram(
        version=1, feeds={0, 1},
        steps=[OpStep(0, "index_set", [0, 1],
                      {"slices": AT.get_tensor(2, dtype="int64")}, 2)],
        outputs={"data": 2},
        extras={"capture_feed_origins": {
            0: {"binding_name": "data"}, 1: {"binding_name": "value"}}},
    )
    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__",)), region_indices=(0,)
    )
    modules, manifest = emit_control_region_modules(
        control, {0: program}, owner_name="buf", module_dir=".", dtype="int64",
    )
    assert modules[0].complete
    assert manifest["storage_redirects"] == {"out::buf_region_0::data": "in::data"}
    inventory = build_class_inventory(manifest)
    method = inventory.methods[0]
    # data flows in on some slot; the update flows out on that *same* slot.
    assert method.output_slots[0] == method.input_slots[0]


def _indexed_store_runtime_region():
    """A reference-path ``IndexedStore`` with a RUNTIME index operand."""
    from src.common.tensors.fused_ir import FusedProgram, OpStep
    return FusedProgram(
        version=1, feeds={0, 1, 2},
        steps=[OpStep(0, "IndexedStore", [0, 1, 2],
                      {"source_type": "SubscriptStore"}, 3)],
        outputs={"data": 3},
        extras={"capture_feed_origins": {
            0: {"binding_name": "data"},
            1: {"binding_name": "index"},
            2: {"binding_name": "value"}}},
    )


def test_indexed_store_runtime_index_emits_a_complete_scatter():
    module = emit_wasm_module(
        _indexed_store_runtime_region(), name="istore", dtype="int64",
    )
    assert module.complete, module.shortfall_report()
    # A runtime subscript narrows to an i32 address, unlike the constant fold.
    assert "i32.wrap_i64" in module.source
    assert "i64.store" in module.source


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_indexed_store_runtime_scatter_runs_correctly(tmp_path):
    module = emit_wasm_module(
        _indexed_store_runtime_region(), name="istore", dtype="int64",
    )
    wasm = tmp_path / "istore.wasm"
    wasm.write_bytes(module.binary)
    script = tmp_path / "run.mjs"
    script.write_text(
        """
        import {readFileSync} from "node:fs";
        const mod = await WebAssembly.instantiate(readFileSync(process.argv[2]), {});
        const {run, memory} = mod.instance.exports;
        const mem = new BigInt64Array(memory.buffer);
        [10n,20n,30n,40n].forEach((v,i)=>mem[0+i]=v);    // data @ 0
        [2n,2n,2n,2n].forEach((v,i)=>mem[8+i]=v);        // index @ 64
        [99n,99n,99n,99n].forEach((v,i)=>mem[16+i]=v);   // value @ 128
        for (let i=0;i<4;i++) mem[24+i]=0n;              // out @ 192
        run(4, 0, 64, 128, 192);
        const out=[]; for (let i=0;i<4;i++) out.push(Number(mem[24+i]));
        console.log(JSON.stringify(out));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip().endswith("[10,20,99,40]"), completed.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_index_set_scatter_runs_correctly(tmp_path):
    module = emit_wasm_module(_store_region(), name="store_r0", dtype="int64")
    wasm = tmp_path / "store_r0.wasm"
    wasm.write_bytes(module.binary)
    script = tmp_path / "run.mjs"
    script.write_text(
        """
        import {readFileSync} from "node:fs";
        const bytes = readFileSync(process.argv[2]);
        const mod = await WebAssembly.instantiate(bytes, {});
        const {run, memory} = mod.instance.exports;
        const mem = new BigInt64Array(memory.buffer);
        [10n,20n,30n,40n].forEach((v,i)=>mem[0+i]=v);   // data @ byte 0
        [99n,99n,99n,99n].forEach((v,i)=>mem[8+i]=v);   // value @ byte 64
        for (let i=0;i<4;i++) mem[16+i]=0n;             // out @ byte 128
        run(4, 0, 64, 128);
        const out=[]; for (let i=0;i<4;i++) out.push(Number(mem[16+i]));
        console.log(JSON.stringify(out));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip().endswith("[10,20,99,40]"), completed.stdout
