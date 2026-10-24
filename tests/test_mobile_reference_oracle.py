"""The mobile reference products return Python's exact value.

Receipt (2026-10-24, extra frontends/backends lane): the mobile commits
31e0fdf7/8fb85e7 checked the Wasm entry with ``pytest.approx`` and checked
the ARM64 products only for shape.  Nothing ran the shipped page binding
(``turing-wasm-shell.js`` with the builder's fixed buffer offsets) or the
cross-target LLVM module against ``program.py`` itself.

These tests compare bit patterns, never a tolerance:

* the built PWA directory is loaded in node through the page's own
  ``TuringWasmProgram`` binding and invoked row by row;
* each ARM64 module is the host module plus its target triple.  The test
  asserts that, then runs the host-triple text natively.  An ARM64 runner is
  not needed for the value; the triple is the only target-owned line.
"""

from __future__ import annotations

import ctypes
import json
import random
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "examples/mobile_reference_app"
NODE = shutil.which("node") or (
    "/opt/node22/bin/node" if Path("/opt/node22/bin/node").exists() else None
)


def _app_modules():
    for path in (str(ROOT), str(APP)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import build_mobile
    from program import mobile_main
    return build_mobile, mobile_main


def _rows(seed: int, count: int) -> list[list[float]]:
    generator = random.Random(seed)
    rows = [[generator.uniform(-2.0, 2.0) for _ in range(3)] for _ in range(count)]
    return rows + [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-0.0, 5e-324, 1e308]]


def _bits(values) -> list[str]:
    return [struct.pack("<d", float(value)).hex() for value in values]


_RUNNER = """
import fs from 'fs';
import {pathToFileURL} from 'url';
const dir = process.argv[2];
const {TuringWasmProgram} = await import(pathToFileURL(dir + '/turing-wasm-shell.js'));
const contract = JSON.parse(fs.readFileSync(dir + '/program.abi.json'));
const {instance} = await WebAssembly.instantiate(fs.readFileSync(dir + '/mobile_main.wasm'), {});
const program = new TuringWasmProgram(instance, contract);
const bits = x => Buffer.from(new Float64Array([x]).buffer).toString('hex');
const rows = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(rows.map(([phase, touch_x, touch_y]) => {
  const out = program.invoke({phase, touch_x, touch_y});
  return contract.outputs.map(name => bits(out[name]));
})));
"""


@pytest.mark.skipif(NODE is None, reason="node is required to run the page binding")
def test_built_pwa_binding_returns_python_bits(tmp_path):
    build_mobile, mobile_main = _app_modules()
    web = tmp_path / "web"
    build_mobile.build_web(web)
    runner = tmp_path / "run.mjs"
    runner.write_text(_RUNNER, encoding="utf-8")
    rows = _rows(1, 300)

    completed = subprocess.run(
        [NODE, str(runner), str(web)], input=json.dumps(rows),
        capture_output=True, text=True, check=True, timeout=120,
    )
    native = json.loads(completed.stdout)

    assert len(native) == len(rows)
    mismatches = [
        (row, _bits(mobile_main(*row)), got)
        for row, got in zip(rows, native) if _bits(mobile_main(*row)) != got
    ]
    assert mismatches == []


_TRIPLE = re.compile(r'^target triple = "[^"]*"$', re.MULTILINE)


@pytest.mark.parametrize("target", ["android-arm64-v8a", "ios-arm64"])
def test_arm64_module_is_host_module_plus_triple_and_returns_python_bits(
    tmp_path, target,
):
    build_mobile, mobile_main = _app_modules()
    from src.compiler.native_code_retention import resolve_build_target
    from src.compiler.ssa_llvm_backend import _host_target_lines

    module, outputs, entry, artifact = build_mobile.compile_source(target)
    triple = resolve_build_target(target).llvm_triple
    assert _TRIPLE.findall(artifact.llvm_ir) == [f'target triple = "{triple}"']

    host_ir = _TRIPLE.sub(
        lambda _match: "\n".join(_host_target_lines()), artifact.llvm_ir,
    )
    source = tmp_path / f"{entry}.ll"
    source.write_text(host_ir, encoding="utf-8")
    library = tmp_path / f"{entry}{'.dll' if sys.platform == 'win32' else '.so'}"
    subprocess.run(
        [sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
         "-o", str(library), str(source)],
        check=True, capture_output=True, timeout=300,
    )
    function = getattr(ctypes.CDLL(str(library)), entry)
    slots = len(artifact.buffer_order)
    assert artifact.buffer_dtypes == ("double",) * slots

    # The same slot reading the generated shell header performs.
    slot = {int(value_id): index for index, value_id in enumerate(artifact.buffer_order)}
    inputs = [slot[int(value.id)] for value in module.functions[entry].args]
    results = [slot[int(value.id)] for value in outputs[entry]]

    mismatches = []
    for row in _rows(2, 500):
        cells = [ctypes.c_double() for _ in range(slots)]
        for index, value in zip(inputs, row):
            cells[index].value = value
        pointers = (ctypes.c_void_p * slots)(
            *[ctypes.cast(ctypes.pointer(cell), ctypes.c_void_p) for cell in cells]
        )
        extents = (ctypes.c_int32 * slots)(*([1] * slots))
        function(pointers, extents)
        got = _bits(cells[index].value for index in results)
        if got != _bits(mobile_main(*row)):
            mismatches.append((row, got))
    assert mismatches == []
