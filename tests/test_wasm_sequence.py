"""The null-terminated-prefix hash primitive matches the compile-time string hash.

The decoder's ``data[offset:offset+8].split(b"\\x00", 1)[0]`` (PE section names)
collapses to an FNV-1a hash of the bytes before the first null. This must equal
the compile-time hash used for string dict keys, so a runtime-derived name and a
constant name collide iff they are the same bytes.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from src.compiler.wasm_binary import CodeBuilder, build_module
from src.compiler.wasm_sequence import emit_hash_delimited_prefix
from src.compiler.fused_program_wasm_backend import _fnv1a_64


def _build_hash_module() -> bytes:
    """``run(buf, start, maxlen, delim, out_addr)`` -> stores the i64 hash."""
    body = CodeBuilder(value_type="i64", parameter_count=5)
    result = body.declare_local("i64")
    index = body.declare_local("i32")
    byte = body.declare_local("i32")
    emit_hash_delimited_prefix(
        body, buf_addr_local=0, start_local=1, maxlen_local=2, delim_local=3,
        result_local=result, index_local=index, byte_local=byte,
    )
    body.local_get(4).local_get(result).i64_store()
    return build_module(
        function_name="run",
        parameter_types=["i32", "i32", "i32", "i32", "i32"],
        body=body, memory_pages=1,
    )


def test_hash_module_is_valid_wasm():
    assert _build_hash_module()[:4] == b"\x00asm"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("name", ["text", ".data", "UPX0", ""])
def test_null_terminated_prefix_hash_matches_compile_time(name, tmp_path):
    binary = _build_hash_module()
    wasm = tmp_path / "seq.wasm"
    wasm.write_bytes(binary)
    # An 8-byte PE-style field: the name, null-padded to 8 bytes.
    field = name.encode("ascii").ljust(8, b"\x00")
    expected = _fnv1a_64(name)  # hash of the bytes before the first null
    script = tmp_path / "run.mjs"
    script.write_text(
        f"""
        import {{readFileSync}} from "node:fs";
        const mod = await WebAssembly.instantiate(readFileSync(process.argv[2]), {{}});
        const {{run, memory}} = mod.instance.exports;
        const bytes = new Uint8Array(memory.buffer);
        const field = {list(field)};
        const BUF = 64, OUT = 32;
        field.forEach((b, i) => bytes[BUF + i] = b);
        run(BUF, 0, 8, 0, OUT);          // start 0, maxlen 8, delim 0x00
        const out = new BigInt64Array(memory.buffer)[OUT / 8];
        console.log(out.toString());
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)], capture_output=True, text=True, check=True,
    )
    assert int(completed.stdout.strip()) == expected, (
        f"name={name!r} got {completed.stdout.strip()} expected {expected}"
    )
