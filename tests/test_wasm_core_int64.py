"""The independent Wasm core carries Python int as i64 (Lane X2).

Receipt (2026-10-24): the core typed every integer formal as i32, so
``a > b`` with a = 2**32 + 1 compared 1 > 3 and returned the other branch,
silently.  Corpus c961 is that program; c962 holds the loop-header Phis'
parallel copy.  (c960's unproven ``a - b`` is refused by name: the core has
no checked int64 spelling.)  These run in node and compare exactly.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_wasm_backend import (
    emit_ssa_module_to_wasm_core,
    prepare_wasm_core_execution,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/native_semantics_corpus"
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node runs the Wasm core",
)


@pytest.mark.parametrize("program", [
    "c961_wide_int_formal_selects_branch", "c962_loop_swaps_float_pair",
])
def test_wasm_core_returns_python_values(program):
    source = (CORPUS / f"{program}.py").read_text(encoding="utf-8")
    rows = ast.literal_eval(source.split("# INPUTS", 1)[1].strip())
    namespace = {"__name__": f"wasm_{program}"}
    exec(compile(source, program, "exec"), namespace)
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"wasm_{program}",
        extraction_contract=ExtractionContract(
            ROOT / "extraction_contracts/program_extraction.yaml"
        ),
        progress=lambda _: None,
    )
    entry = exports[0]
    artifact = emit_ssa_module_to_wasm_core(module, entry, entry_name=entry)
    assert artifact.complete, artifact.shortfalls
    kinds = dict((value_id, kind) for kind, value_id in artifact.parameters)
    names = dict(module.functions[entry].metadata["parameter_names"])
    int_formals = [name for name in names if isinstance(rows[0][name], int)]
    assert {kinds[int(names[name])] for name in int_formals} <= {"i64"}
    (result,) = outputs[entry]

    for row in rows:
        execution = prepare_wasm_core_execution(artifact, {
            int(names[name]): value for name, value in row.items()
        })
        try:
            got = execution.run().buffers[int(result.id)]
        finally:
            execution.close()
        want = namespace["root"](**row)
        assert got.dtype == np.float64
        assert float(got[0]).hex() == float(want).hex(), row
