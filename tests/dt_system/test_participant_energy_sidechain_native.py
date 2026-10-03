"""The declared participant's exchange time owns the energy scheduling law."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from llvm_dt_system import dt_system_contract, instantiate_state  # noqa: E402
from src.common.dt_system.dt_controller import Targets  # noqa: E402
from src.common.dt_system.time_contracts import BIND, HOLD, DILATE, SUBCYCLE  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402


def test_actual_piece_publication_owns_native_energy_sidechain(tmp_path):
    from src.compiler.native_package import piece_from_law
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id
    from src.compiler.identity_concordance import concordance_report
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    x, dt, power = sp.symbols("x dt power")
    law = compile_sympy_equations([
        sp.Eq(sp.Symbol("x_next"), x + dt, evaluate=False),
        sp.Eq(sp.Symbol("energy_j"), sp.Integer(1), evaluate=False),
        sp.Eq(sp.Symbol("exchangeable_energy_j"), sp.Integer(64), evaluate=False),
        sp.Eq(sp.Symbol("power_w"), power, evaluate=False),
        sp.Eq(sp.Symbol("max_vel"), sp.Integer(2), evaluate=False),
    ], name="participant_energy")
    piece = piece_from_law(law, "participant_energy", 1, directory=tmp_path / "law")
    source = (
        "from src.common.dt_system.dt_controller import _propose_dt_pen, _apply_energy_sidechain\n"
        "def root(metrics, targets):\n"
        "    proposal = _propose_dt_pen(metrics, targets, 1.0, None)\n"
        "    next_value = _apply_energy_sidechain(AbstractTensor.tensor(proposal), "
        "AbstractTensor.tensor(0.25), metrics, targets)\n"
        "    return proposal, next_value\n"
    )
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="participant_energy_consumer",
        extraction_contract=dt_system_contract("root", (), 1, 1),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda message: None,
    )
    print(concordance_report(module), flush=True)
    root = module.functions[exports[0]]
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "consumer", optimization="O0")
    output_ids = dict(root.metadata["named_outputs"])
    actual_rows = []
    expected_rows = []
    for measured_power in (128.0, 0.0):
        for contract in (BIND, HOLD, DILATE, SUBCYCLE):
            participant = copy.copy(piece)
            participant.contract = contract
            targets = Targets(1.0, 1.0, 1.0, energy_exchange_fraction=0.25)
            state = instantiate_state(
                [participant], {"x": np.zeros(1), "power": np.array([measured_power])},
                targets=targets,
            )
            _, metrics = state.program["advance_pieces"](state, 0.25)
            feeds = _managed_native_feeds_by_id(
                SimpleNamespace(module=module, root_name=root.name),
                {"metrics": metrics, "targets": targets},
            )
            execution = artifact.prepare_execution(feeds)
            execution.run()
            proposal = float(execution.buffers[output_ids["proposal"]].reshape(-1)[0])
            next_value = float(execution.buffers[output_ids["next_value"]].reshape(-1)[0])
            expected = (0.125 if contract == BIND and measured_power > 0.0
                        else 0.25 if contract == HOLD else 0.5)
            print(f"contract={contract} power={measured_power} stored_energy=1 "
                  f"exchange_time={metrics.pub_exchange_time.tolist()} "
                  f"exchange_present={metrics.pub_exchange_time_present.tolist()} "
                  f"proposal={proposal} next={next_value} expected={expected}", flush=True)
            actual_rows.append((proposal, next_value))
            expected_rows.append((0.5, expected))
    assert actual_rows == expected_rows
