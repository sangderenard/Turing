"""Real-source parity for effect-only control and canonical loop feeds."""

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)

EFFECT_SOURCE = '''
class Counter:
    def __init__(self):
        self.n = 0.0
    def tick(self):
        self.n = self.n + 1.0
        return self.n
    def bump(self):
        self.n = self.n + 2.0
        return self.n
def root(ctrl, rollback):
    while True:
        if not rollback:
            return ctrl.tick()
        return ctrl.bump()
'''
ACCUMULATOR_SOURCE = '''
from src.common.tensors import AbstractTensor
class Counter:
    def __init__(self):
        self.n = 0.0
    def tick(self):
        self.n = self.n + 1.0
        return self.n
def root(ctrl, amount):
    total = AbstractTensor.tensor(0.0)
    while (0.02 - total).item() > 1e-12:
        ctrl.n = ctrl.n + 1.0
        previous = float(total.item())
        total += AbstractTensor.tensor(amount)
        if float(total.item()) == previous:
            break
        if ctrl.n >= 3.0:
            break
    return total
'''

# A second authored binding can hold the same initial graph value without
# advancing with total. Its read must select the pre-loop generation.
HELD_SEED_SOURCE = ACCUMULATOR_SOURCE.replace(
    "    total = AbstractTensor.tensor(0.0)\n",
    "    total = AbstractTensor.tensor(0.0)\n    fixed = total\n",
).replace(
    "        ctrl.n = ctrl.n + 1.0",
    "        ctrl.n = ctrl.n + float(fixed.item()) + 1.0",
)


@pytest.mark.parametrize("source,parameter,cases", [
    (EFFECT_SOURCE, "rollback", (False, True)),
    (ACCUMULATOR_SOURCE, "amount", (.01, .037, .007)),
    (HELD_SEED_SOURCE, "amount", (.01, .037, .007)),
], ids=("effect-guard", "aliased-accumulator", "held-seed"))
@pytest.mark.parametrize("backend", ("llvm", "c", "fortran"))
def test_source_native_control_identity(source, parameter, cases, backend, tmp_path):
    namespace = {}
    exec(source, namespace)
    scalar_type = "bool" if parameter == "rollback" else "float64"
    policy = ExtractionContract("extraction_contracts/program_extraction.yaml").with_execution_file(
        "extraction_contracts/vehicle_full_native_execution.yaml"
    ).with_program_abi({
        "records": {"Counter": {"identity": "Counter", "fields": {
            "n": {"storage": "scalar", "dtype": "float64", "rank": 0, "mutable": True},
        }}},
        "bindings": [
            {"function": "*", "parameter": "ctrl", "record": "Counter"},
            {"function": "Counter.tick", "parameter": "self", "record": "Counter"},
        ],
        "values": [{"function": "root", "parameter": parameter, "storage": "scalar",
                    "dtype": scalar_type, "rank": 0,
                    "python_type": "builtins.bool" if parameter == "rollback" else "builtins.float"}],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="control_identity", extraction_contract=policy,
        runtime_closure_only=True, tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda message: None,
    )
    root = module.functions[next(name for name in exports if name.endswith("__root"))]
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report
    if backend == "llvm":
        artifact = emit_ssa_function_to_llvm(module, root.name)
        assert artifact.complete, artifact.shortfalls
        artifact = compile_artifact(artifact, directory=tmp_path, optimization="O3")
        prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)
    elif backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, root.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native", optimization="O3")
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        artifact = emit_module(module, name="control_identity", outputs=outputs)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(
            artifact, {}, tmp_path / "native", entrypoint=root.name, library=True,
        )
        native = FortranCoreNative(built.executable_path, artifact.api.entry_point(root.name), artifact.api)
        prepare = native.prepare_execution
    fields = [arg.id for arg in root.args if (arg.accounting or {}).get("program_abi_field") == "n"]
    assert fields
    argument = dict(root.metadata["parameter_names"])[parameter]
    returned = next(op.args[0].id for block in root.blocks.values() for op in block.instrs if op.op == "Ret")
    for value in cases:
        oracle = namespace["Counter"]()
        expected = namespace["root"](oracle, value)
        if hasattr(expected, "item"):
            expected = expected.item()
        feeds = {field: 0.0 for field in fields}
        feeds[argument] = value
        execution = prepare(feeds).run()
        assert execution.buffers[returned].reshape(-1)[0] == expected
        for field in fields:
            assert execution.buffers[field].reshape(-1)[0] == oracle.n


@pytest.mark.parametrize("backend", ("llvm", "c", "fortran"))
def test_written_span_returned_in_metrics_keeps_its_owned_storage(backend, tmp_path):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from src.common.tensors import AbstractTensor
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from llvm_dt_system import PUBLICATION_FIELDS, dt_system_contract
    fields = (*PUBLICATION_FIELDS, "pub_values", "pub_present", "pub_limits", "pub_limits_present")
    source = "from src.common.dt_system.dt_scaler import Metrics, coerce_metrics\n"
    source += (
        "def advance(state):\n"
        "    state.pub_contract[0] = state.pub_contract[0] + 1.0\n"
        "    return True, Metrics(0.0, 0.0, 0.0, 0.0, "
        + ", ".join(f"{field}=state.{field}" for field in fields) + ")\n"
        "def root(state):\n"
        "    ok, metrics = advance(state)\n"
        "    metrics = coerce_metrics(metrics)\n"
        "    return metrics.pub_contract[0] + metrics.pub_exchange_time[0]\n"
    )
    namespace = {}
    exec(source, namespace)
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="written_record", extraction_contract=dt_system_contract("root", (), 1, 1),
        python_bindings={"AbstractTensor": AbstractTensor}, runtime_closure_only=True,
        tensor_ssa_reference=c_backend_repository_ssa_reference(), progress=lambda _: None,
    )
    root = module.functions[exports[0]]
    assert not [arg for arg in root.args if not (arg.accounting or {}).get("program_abi_parameter")]
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report
    if backend == "llvm":
        artifact = emit_ssa_function_to_llvm(module, root.name)
        assert artifact.complete, artifact.shortfalls
        artifact = compile_artifact(artifact, directory=tmp_path, optimization="O3")
        prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)
    elif backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, root.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native", optimization="O3")
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        artifact = emit_module(module, name="written_record", outputs=outputs)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, tmp_path / "native", entrypoint=root.name, library=True)
        prepare = FortranCoreNative(built.executable_path, artifact.api.entry_point(root.name), artifact.api).prepare_execution

    def state():
        return SimpleNamespace(**{
            field: AbstractTensor.tensor(np.full((1 if field in PUBLICATION_FIELDS else 15,), 7.0))
            for field in fields
        })
    oracle, native_state = state(), state()
    feeds = _managed_native_feeds_by_id(SimpleNamespace(module=module, root_name=root.name), {"state": native_state})
    execution = prepare(feeds)
    output = outputs[root.name][0].id
    for _ in range(3):
        expected = namespace["root"](oracle)
        execution.run()
        assert execution.buffers[output].reshape(-1)[0] == float(expected.item())


def test_real_dt_step_runs_one_pi_update_and_preserves_publication(tmp_path):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from src.common.tensors import AbstractTensor
    from src.common.dt_system.dt_controller import STController, Targets
    from src.common.dt_system.error_channels import channel_fields, DT_CHANNEL_NAMES
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import llvm_dt_system as lds

    fields = (*lds.PUBLICATION_FIELDS, "pub_values", "pub_present", "pub_limits", "pub_limits_present")
    source = (
        "from src.common.tensors import AbstractTensor\n"
        "from src.common.dt_system.dt_scaler import Metrics\n"
        "from src.common.dt_system.dt_controller import step_with_dt_control_used\n"
        + lds.state_source(("x",), 1)
        + "\ndef advance(state, dt):\n"
        "    state.x[...] = state.x + dt\n"
        "    state.pub_contract[0] = 0.0\n"
        "    return True, Metrics(max_vel=1e-30, max_flux=0.0, div_inf=0.0, mass_err=0.0, "
        + ", ".join(f"{field}=state.{field}" for field in fields) + ")\n"
        "def root(state, dt_initial, dx, targets, ctrl):\n"
        "    metrics, next_dt, used = step_with_dt_control_used(state, dt_initial, dx, targets, ctrl, advance, ref=dt_initial, rollback=state.rollback, allow_unresolved=True)\n"
        "    return used, next_dt\n"
    )
    namespace = {}
    exec(source, namespace)
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="real_step_parity", extraction_contract=lds.dt_system_contract("root", ("x",), 1, 1),
        python_bindings={"AbstractTensor": AbstractTensor}, runtime_closure_only=True,
        tensor_ssa_reference=c_backend_repository_ssa_reference(), progress=lambda _: None,
    )
    root = module.functions[exports[0]]
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path, optimization="O3")

    def inputs(rollback):
        state = namespace["PieceState"](AbstractTensor.tensor([0.0]), AbstractTensor.tensor([.01]), AbstractTensor.zeros((8,)))
        state.rollback = rollback
        state.rollback_threshold_multiplier = 1.0
        targets = Targets(cfl=.5, div_max=1e9, mass_max=1e-3, **channel_fields({}, names=DT_CHANNEL_NAMES, limits=True))
        return state, targets, STController(dt_min=1e-9, dt_max=.1)

    control_ids = [arg.id for arg in root.args if (arg.accounting or {}).get("program_abi_parameter") == "ctrl" and arg.accounting.get("program_abi_field") == "acc"]
    assert control_ids
    for rollback in (False, True):
        for dt in (.01, .025):
            oracle, oracle_targets, oracle_ctrl = inputs(rollback)
            native_state, targets, ctrl = inputs(rollback)
            feeds = _managed_native_feeds_by_id(SimpleNamespace(module=module, root_name=root.name), {
                "state": native_state, "targets": targets, "ctrl": ctrl, "dt_initial": dt, "dx": 1.0,
            })
            execution = prepare_artifact_execution(artifact, feeds)
            for _ in range(2):
                expected = namespace["root"](oracle, dt, 1.0, oracle_targets, oracle_ctrl)
                execution.run()
                actual = tuple(float(execution.buffers[value.id].reshape(-1)[0]) for value in outputs[root.name])
                assert actual == tuple(float(value) for value in expected)
                for value_id in control_ids:
                    assert execution.buffers[value_id].reshape(-1)[0] == float(oracle_ctrl.acc.item())
                for argument in root.args:
                    accounting = argument.accounting or {}
                    if accounting.get("program_abi_parameter") == "state" and accounting.get("program_abi_field") in ("x", "pub_contract"):
                        np.testing.assert_array_equal(execution.buffers[argument.id].reshape(-1), np.asarray(getattr(oracle, accounting["program_abi_field"])).reshape(-1))
