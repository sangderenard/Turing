"""Declared appended errors cross the existing piece/DT native ABI.

The channel is a conservation discrepancy, not stored energy or exchange power.
Its exact value and limit use the same declared layout at every boundary.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from llvm_dt_system import (  # noqa: E402
    NativeSystem, configure_publication_limits, dt_system, dt_system_contract, instantiate_state,
)
from src.common.dt_system.dt_controller import STController, Targets  # noqa: E402
from src.common.dt_system.dt_scaler import Metrics  # noqa: E402
from src.common.dt_system.error_channels import DT_CHANNEL_NAMES, channel_fields  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402

CONSERVATION_ERROR = "test_energy_conservation_error_j"
CHANNEL_NAMES = (*DT_CHANNEL_NAMES, CONSERVATION_ERROR)


def test_declared_conservation_error_reaches_native_proposal(tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id
    from src.compiler.identity_concordance import concordance_report
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    source = (
        "from src.common.dt_system.dt_controller import _propose_dt_pen\n"
        "def root(metrics, targets):\n"
        "    return _propose_dt_pen(metrics, targets, 1.0, None)\n"
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="declared_conservation_proposal",
        extraction_contract=dt_system_contract(
            "root", (), 1, 1, channel_names=CHANNEL_NAMES),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    print(concordance_report(module), flush=True)
    root = module.functions[exports[0]]
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path, optimization="O0")
    targets = Targets(1.0, 1.0, 1.0, **channel_fields(
        {CONSERVATION_ERROR: 2.0}, names=CHANNEL_NAMES, limits=True))
    output_id = int(root.metadata["named_outputs"][0][1])
    for route in ("aggregate", "participant"):
        for published, expected in (({}, 0.5), ({CONSERVATION_ERROR: 4.0}, 0.25),
                                    ({CONSERVATION_ERROR: 0.0}, 0.5)):
            fields = channel_fields(published if route == "aggregate" else {}, names=CHANNEL_NAMES)
            metrics = Metrics(2.0, 0.0, 0.0, 0.0, **fields)
            for field in ("pub_exchange_time", "pub_exchange_time_present", "pub_contract",
                          "pub_dt_limit", "pub_dt_limit_present"):
                setattr(metrics, field, AbstractTensor.zeros((1,)))
            for field in ("pub_values", "pub_present", "pub_limits", "pub_limits_present"):
                setattr(metrics, field, AbstractTensor.zeros((len(CHANNEL_NAMES),)))
            if route == "participant":
                row = channel_fields(published, names=CHANNEL_NAMES)
                metrics.pub_values[...] = row["error_channels"]
                metrics.pub_present[...] = row["error_present"]
                metrics.pub_limits[...] = targets.error_limits
                metrics.pub_limits_present[...] = targets.error_limits_present
            feeds = _managed_native_feeds_by_id(
                SimpleNamespace(module=module, root_name=root.name),
                {"metrics": metrics, "targets": targets},
            )
            execution = artifact.prepare_execution(feeds)
            execution.run()
            assert execution.buffers[output_id].reshape(-1)[0] == expected


def test_compiled_conservation_publication_rejects_restores_and_lands_window(tmp_path):
    from src.compiler.native_package import piece_from_law
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations

    x, dt = sp.symbols("x dt")
    law = compile_sympy_equations([
        sp.Eq(sp.Symbol("x_next"), x + dt, evaluate=False),
        sp.Eq(sp.Symbol(CONSERVATION_ERROR), dt * dt, evaluate=False),
        sp.Eq(sp.Symbol("undeclared_error"), 1000 * dt, evaluate=False),
        sp.Eq(sp.Symbol("max_vel"), sp.Integer(1), evaluate=False),
    ], name="conservation_publication")
    piece = piece_from_law(law, "conservation_publication", 1, directory=tmp_path)
    targets = Targets(0.5, 1.0, 1.0, **channel_fields(
        {CONSERVATION_ERROR: 0.01}, names=CHANNEL_NAMES, limits=True))
    columns = {"x": np.zeros(1)}
    state = instantiate_state([piece], columns, targets=targets, channel_names=CHANNEL_NAMES)
    owned_x = state.x
    with pytest.raises(ValueError, match="declared channel layout"):
        configure_publication_limits(state, Targets(0.5, 1.0, 1.0))
    assert np.shares_memory(owned_x, state.span)
    assert state.participants.declared() == (piece.entry,)
    attempts = []
    advance = state.program["advance_pieces"]

    def observe(actual_state, step_dt):
        before = actual_state.x.copy()
        ok, metrics = advance(actual_state, step_dt)
        attempts.append((float(step_dt), before, actual_state.x.copy()))
        assert metrics.error_channels.shape == (len(CHANNEL_NAMES),)
        assert metrics.error_channels[-1].item() == float(step_dt) ** 2
        assert metrics.error_present[-1].item() == 1.0
        assert actual_state.pub_values[-1].item() == float(step_dt) ** 2
        assert actual_state.pub_limits[-1].item() == 0.01
        return ok, metrics

    state.program["advance_pieces"] = observe
    _, _, results = dt_system(
        [piece], columns, rounds=1, round_dt=0.25, dt_initial=0.25, dx=0.1,
        state=state, targets=targets, controller=STController(dt_min=None), rollback=True,
    )
    assert [attempt[0] for attempt in attempts[:3]] == [0.25, 0.125, 0.0625]
    for _, before, _ in attempts[:3]:
        np.testing.assert_array_equal(before, [0.0])
    assert state.x is owned_x
    assert columns["x"] is owned_x
    assert results[0][0] == 0.25
    np.testing.assert_array_equal(state.x, [0.25])
    assert "undeclared_error" not in state.channel_names


def test_layout_declaration_preserves_prefix_and_checks_target_extent():
    with pytest.raises(ValueError, match="shared ABI prefix"):
        dt_system_contract("root", (), 1, channel_names=(CONSERVATION_ERROR,))
    with pytest.raises(ValueError, match="unique channel identities"):
        dt_system_contract("root", (), 1, channel_names=(*CHANNEL_NAMES, CONSERVATION_ERROR))
    with pytest.raises(ValueError, match="alias existing metric outputs"):
        dt_system_contract("root", (), 1, channel_names=(*DT_CHANNEL_NAMES, "residual"))
    abi = dt_system_contract("root", (), 1, 2, channel_names=CHANNEL_NAMES).program_abi.receipt()
    for field in ("error_channels", "error_present"):
        assert abi["records"]["Metrics"]["fields"][field]["shape"] == [len(CHANNEL_NAMES)]
    for field in ("error_limits", "error_limits_present"):
        assert abi["records"]["Targets"]["fields"][field]["shape"] == [len(CHANNEL_NAMES)]
    assert abi["records"]["PieceState"]["fields"]["pub_values"]["shape"] == [2 * len(CHANNEL_NAMES)]
    native = NativeSystem(None, None, "root", (), channel_names=CHANNEL_NAMES)
    with pytest.raises(ValueError, match="different declared channel layout"):
        native.feeds(SimpleNamespace(channel_names=DT_CHANNEL_NAMES), None, None, 1.0, 1.0, 1.0)
