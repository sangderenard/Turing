"""``lowered_system(backend="wasm")``: the dt system's round through the
direct Wasm core, compared against the Python lane on the same piece.

The core has no linker, so the law is part of the program
(``piece_mode="inline"``); the emitted module is its own executable and
runs in node.  The round's state field is read back through the same
``program_abi_field_slot`` rows the C and LLVM lanes post, so
``NativeSystem`` drives this lane unchanged.

The piece is built from an authored AbstractTensor law (the route
``examples/meowdoku`` takes) rather than ``compile_sympy_equations``, which
refuses on current main (``topological_reducer``: "has no lexical read scope
for its identity row") before any backend is reached.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from src.common.tensors import AbstractTensor  # noqa: E402
from src.compiler.native_law_kernels import LLVMPiece, batch_contract  # noqa: E402
from llvm_dt_system import dt_system, lowered_system  # noqa: E402

BATCH = 4
LAW = '''
def drift(x, v, dt):
    x_next = x + dt * v
    dt_limit = x * 0.0 + 0.01
    max_vel = (v * v).sqrt()
    return x_next, dt_limit, max_vel
'''
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node runs the Wasm core",
)


def drift_piece(directory):
    """An ``LLVMPiece`` of the authored drift law, saved to ``directory``."""
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    names = ("x", "v", "dt")
    outputs = ("x_next", "dt_limit", "max_vel")
    module, lowered_outputs, exports = lower_ast_source_to_ssa(
        LAW, "drift", python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name="drift_batched", runtime_closure_only=True,
        extraction_contract=batch_contract("drift", names, BATCH),
        progress_bars=False,
    )
    entry = exports[0]
    function = module.functions[entry]
    named = {str(name): int(value_id)
             for name, value_id in tuple(function.metadata.get("named_outputs") or ())}
    missing = [name for name in outputs if name not in named]
    assert not missing, (missing, named)
    artifact = compile_artifact(emit_ssa_function_to_llvm(module, entry), directory=directory)
    piece = LLVMPiece(
        artifact, names, tuple(int(value.id) for value in function.args),
        outputs, {name: named[name] for name in outputs}, {}, BATCH,
        module=module, entry=entry, outputs=lowered_outputs, source=LAW,
    )
    path = Path(directory) / "drift.piece"
    piece.save(path)
    return path


@pytest.fixture(scope="module")
def piece_file(tmp_path_factory):
    return drift_piece(tmp_path_factory.mktemp("drift"))


def test_wasm_backend_requires_inlined_laws(piece_file, tmp_path):
    with pytest.raises(ValueError, match="piece_mode='inline'"):
        lowered_system([piece_file], backend="wasm", directory=tmp_path,
                       progress=lambda message: None)


def test_wasm_round_matches_the_python_lane(piece_file, tmp_path):
    x0 = np.array([0.0, 1.0, -2.0, 10.0])
    velocity = np.array([1.0, 2.0, 3.0, -4.0])

    # The Python lane: one round over the same piece.
    reference = {"x": x0.copy(), "v": velocity.copy()}
    _state, _ctrl, results = dt_system(
        [piece_file], reference, rounds=1, round_dt=0.05, dx=1.0)
    (total, _dt_next, _telemetry), = results
    assert abs(float(total) - 0.05) < 1e-12
    np.testing.assert_allclose(reference["x"], x0 + 0.05 * velocity, rtol=0, atol=1e-12)

    # The Wasm lane: the same round, lowered as one program with the law
    # inlined, run in node, read back through the ABI slot rows.
    try:
        system = lowered_system(
            [piece_file], backend="wasm", piece_mode="inline",
            directory=tmp_path / "wasm", progress=lambda message: None)
    except Exception as error:  # the lowering of dt_system_over, not the backend
        if "MissingResourceControlTopology" not in str(error):
            raise
        # Recorded compiler-lane refusal (docs/hazards/20261102T145430Z_main.md
        # and later): the exception graph wants concorded entry/exit/cleanup
        # edges for dt_system_over's native resource scope.  backend="c"
        # refuses identically on this piece, so the backend is not reached.
        pytest.xfail("lower_ast_source_to_ssa(dt_system_over) refuses: "
                     "MissingResourceControlTopology (compiler lane)")
    assert system.artifact.complete
    assert system.artifact.emission is not None
    state, controller, _nothing = dt_system(
        [piece_file], {"x": x0.copy(), "v": velocity.copy()},
        rounds=0, round_dt=0.05, dx=1.0)
    execution = system.prepare(state, state.targets, controller, 0.05, 0.05, 1.0)
    try:
        execution.run()
        x_id = system.state_field_ids()["x"]
        native_x = np.asarray(execution.buffers[x_id], dtype=np.float64)
    finally:
        execution.close()
    np.testing.assert_array_equal(native_x, reference["x"])
    layout = system.layout(execution=None)
    assert layout["batch"] == BATCH and "x" in layout["state_fields"]
