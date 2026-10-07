"""The dt system with an LLVM piece as its step: one law whose answer is known.

    x_next = x + dt * v        dt_limit = 0.01        max_vel = |v|

After three rounds of 0.05 s, x must be x0 + 0.15 v for every column.  The
first round has no ceiling and is taken in one step; after it the piece's
published dt_limit = 0.01 is the engine's causal ceiling, so the dt system
must refuse every larger attempt: rounds 2 and 3 take >= 5 substeps each and
propose dt_next <= 0.01.  Either all of that holds or the piece is not
running under the dt system's controller.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from src.compiler.native_package import piece_from_law  # noqa: E402
from src.compiler.symbolic_equation_compiler import compile_sympy_equations  # noqa: E402
from llvm_dt_system import NativeSystem, dt_system  # noqa: E402

BATCH = 4


def test_native_system_reads_back_authoritative_written_state_slot():
    """The written slot is the field's resident, whatever the argument order:
    the rule is applied once, where the ``program_abi_field_slot`` rows are
    posted, and ``NativeSystem`` reads the rows."""
    from src.compiler.concordance_declarations import (
        EMISSION_C, Backend, PROGRAM_ABI_FIELD_SLOT, ProgramAbiSlotRole,
    )
    from src.compiler.emission_concordance import post_program_abi_field_slots
    from src.compiler.identity_concordance import IdentityBook
    from src.transmogrifier.ssa import Function, SSAValue

    def argument(value_id, *, written=False, callsite_id=None):
        accounting = {
            "program_abi_parameter": "state",
            "program_abi_field": "x",
            "program_abi_storage": "span",
            "program_abi_field_written": written,
        }
        if callsite_id is not None:
            accounting["callsite_id"] = callsite_id
        return SSAValue(value_id, "float64", (1,), accounting=accounting)

    root = Function("entry", [
        argument(10),
        argument(11, written=True),
        # This alias comes later on purpose: argument order must not replace
        # the buffer that the compiler marks as the field's mutable output.
        argument(12, callsite_id=99),
    ], {})
    book = IdentityBook()
    posted = post_program_abi_field_slots(
        book, root, "entry", Backend.C_MODULE, buffer_order=(10, 11, 12),
        buffer_dtypes=("float64",) * 3, buffer_shapes=((1,),) * 3,
        buffer_order_cell=None, stage=EMISSION_C,
    )
    key = ("state", "x", ProgramAbiSlotRole.PAYLOAD)
    slot = book.pages[PROGRAM_ABI_FIELD_SLOT.name].latest(
        ("entry", Backend.C_MODULE, *key))
    assert list(posted) == [key]
    assert (slot.value_id, slot.buffer_index, slot.aliases) == (11, 1, (10, 12))

    module = SimpleNamespace(functions={"entry": root},
                             metadata={"identity_book": book})
    artifact = SimpleNamespace(
        name="entry", emission=SimpleNamespace(backend=Backend.C_MODULE))
    system = NativeSystem(artifact, module, "entry", (), columns=("x",), batch=1)

    assert system.state_field_ids() == {"x": 11}


def test_llvm_piece_steps_under_the_dt_system(tmp_path):
    x, v, dt = sp.symbols("x v dt")
    law = compile_sympy_equations([
        sp.Eq(sp.Symbol("x_next"), x + dt * v, evaluate=False),
        sp.Eq(sp.Symbol("dt_limit"), sp.Float(0.01) + 0 * x, evaluate=False),
        sp.Eq(sp.Symbol("max_vel"), sp.sqrt(v * v), evaluate=False),
    ], name="drift")
    piece_file = tmp_path / "drift.piece"
    piece_from_law(law, "drift", BATCH, directory=tmp_path).save(piece_file)

    x0 = np.array([0.0, 1.0, -2.0, 10.0])
    velocity = np.array([1.0, 2.0, 3.0, -4.0])
    columns = {"x": x0.copy(), "v": velocity.copy()}

    _state, _ctrl, results = dt_system([piece_file], columns, rounds=3, round_dt=0.05, dx=1.0)

    # run_superstep returns (total, dt_next, metrics).  Round 1 has no ceiling
    # yet and advances the whole window in one step; from then on the
    # controller must never propose more than the piece's published
    # dt_limit = 0.01.
    advanced = sum(float(total) for total, _dt_next, _metrics in results)
    assert abs(advanced - 0.15) < 1e-12, advanced
    for total, dt_next, _metrics in results:
        assert abs(float(total) - 0.05) < 1e-12, total
    for _total, dt_next, _metrics in results[1:]:
        assert float(dt_next) <= 0.01 + 1e-12, dt_next
    np.testing.assert_allclose(columns["x"], x0 + 0.15 * velocity, rtol=0, atol=1e-12)
