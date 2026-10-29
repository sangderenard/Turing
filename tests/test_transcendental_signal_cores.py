"""Scalar float ``math`` transcendentals route through the signal cores.

* The numeric contract names the route (``transcendentals``); its only value
  is ``signal_cores`` and anything else refuses by name.
* ``ir_identities.lower_transcendentals_to_signal_cores`` rewrites each op
  with a ``TRANSCENDENTAL_SIGNAL_CORES`` row into that emitter method's
  instructions, binds the op's own result id to the core's value, and
  guards each fault predicate with ``CondBr`` to ``turing_validation_error``
  with the code the core names.  An op without a row stays and is named in
  the receipt.

The routing tests install a probe row over ``exp_expansion`` (whose scale is
still a host ``Pow``, so it is not a production row): what they check is the
plumbing -- the rewrite, the collapse, the result binding, the guard -- on
the C and LLVM lanes, for both answer shapes the cores use: ``(double,
((predicate, code, name), ...))`` and ``(expansion, bare predicate)`` with
the code on the route row.  ``INT_OVERFLOW`` stands in for a math range code
(both raise ``OverflowError``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
import math

def root(x: float) -> float:
    return 1.0 * math.exp(x)
'''


def test_contract_names_signal_cores_in_every_mode():
    from src.compiler import ssa_call_input_adapters as adapters

    for mode, contract in adapters._NUMERIC_CONVERSION_MODES.items():
        assert contract["transcendentals"] == "signal_cores", mode
        assert adapters.transcendental_policy(mode) == "signal_cores"
    assert adapters.transcendental_policy() == "signal_cores"


def test_contract_refuses_any_other_transcendental_route(monkeypatch):
    from src.compiler import ssa_call_input_adapters as adapters

    monkeypatch.setitem(
        adapters._NUMERIC_CONVERSION_MODES, "compute",
        {**adapters._NUMERIC_CONVERSION_MODES["compute"],
         "transcendentals": "host_libm"},
    )
    with pytest.raises(ValueError, match="transcendentals must be one of"):
        adapters.transcendental_policy("compute")


def _lower():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="signal_core_route",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _ops(module):
    return [
        str(instruction.op)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    ]


def test_op_without_a_core_stays_and_is_named():
    module, _export, _parameters, _outputs = _lower()
    assert "Exp" in _ops(module)
    receipts = [
        function.metadata.get("transcendental_pipeline")
        for function in module.functions.values()
        if function.metadata.get("transcendental_pipeline")
    ]
    assert receipts and all(
        receipt["policy"] == "signal_cores" and "Exp" in receipt["left_native"]
        for receipt in receipts
    )


@pytest.fixture(params=["rows", "bare"])
def probe_route(request, monkeypatch):
    from src.compiler import ir_identities
    from src.compiler.ir_identities import SignalCoreRoute
    from src.compiler.native_faults import INT_OVERFLOW
    from src.compiler.ssa_record_return_state import mint_ssa_value
    from src.transmogrifier.ssa import Instr, SSAValue

    def overflow(self, x):
        limit = self.scalar_constant(709.782712893384, x)
        predicate = SSAValue(mint_ssa_value(
            self.function, self._limb_kind, (x, limit),
            dtype="bool", shape=(), stage=self._stage,
        ), dtype="bool", shape=())
        self.emitted.append(Instr("Gt", [x, limit], predicate, attributes={}))
        return predicate

    def exp_rows(self, x):
        core = self.exp_expansion(self.expanded(x, 2), 2)
        result = self.put("Add", (core[0], core[1]), core[0])
        return result, ((overflow(self, x), INT_OVERFLOW, "probe_overflow"),)

    def exp_bare(self, x):
        return self.exp_expansion(self.expanded(x, 2), 2), overflow(self, x)

    rows = {
        "rows": SignalCoreRoute("exp_rows"),
        "bare": SignalCoreRoute(
            "exp_bare", fault=(INT_OVERFLOW, "probe_overflow"),
        ),
    }
    for name, method in (("exp_rows", exp_rows), ("exp_bare", exp_bare)):
        monkeypatch.setattr(
            ir_identities.WideExpansionEmitter, name, method, raising=False,
        )
    monkeypatch.setitem(
        ir_identities.TRANSCENDENTAL_SIGNAL_CORES, "Exp", rows[request.param],
    )


def _c_lane(tmp_path, module, export):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    return artifact.prepare_execution


def _llvm_lane(tmp_path, module, export):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    return lambda feeds: prepare_artifact_execution(artifact, feeds)


@pytest.mark.parametrize("lane", [_c_lane, _llvm_lane], ids=["c", "llvm"])
def test_routed_op_binds_its_result_and_guards_its_fault(
    probe_route, tmp_path, lane,
):
    module, export, parameters, outputs = _lower()
    ops = _ops(module)
    assert "Exp" not in ops and "Pow" in ops and "Fma" in ops
    receipt = next(
        function.metadata["transcendental_pipeline"]
        for function in module.functions.values()
        if function.metadata.get("transcendental_pipeline")
    )
    assert receipt["lowered"] == {"Exp": 1} and not receipt["left_native"]
    prepare = lane(tmp_path, module, export)

    def run(x):
        execution = prepare({parameters["x"]: np.array([x])}).run()
        return execution.buffers[outputs[0]].reshape(-1)[0].item()

    for x in (0.5, -3.25, 100.0, 709.0):
        assert run(x) == math.exp(x)
    with pytest.raises(OverflowError):
        run(710.0)
