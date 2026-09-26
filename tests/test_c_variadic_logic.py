import pickle
import subprocess
import sys

from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule


def test_native_three_input_logical_operations_match_all_boolean_inputs(tmp_path):
    inputs = [SSAValue(i, "bool") for i in range(3)]
    conjunction, disjunction = SSAValue(3, "bool"), SSAValue(4, "bool")
    function = Function("logical", inputs, {"entry": BasicBlock("entry", [
        Instr("LAnd", inputs, conjunction), Instr("LOr", inputs, disjunction),
        Instr("Ret", [conjunction, disjunction], None),
    ])}, metadata={"output_names": ("conjunction", "disjunction")})
    artifact = emit_ssa_module_to_c(IRModule({function.name: function}), function.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    saved = tmp_path / "artifact.pkl"
    saved.write_bytes(pickle.dumps(artifact))
    result = subprocess.run([sys.executable, "-c", '''
import itertools, pickle, sys
import numpy as np
artifact = pickle.load(open(sys.argv[1], 'rb'))
for values in itertools.product((False, True), repeat=3):
    execution = artifact.prepare_execution({i: np.array([value], dtype=np.bool_)
                                            for i, value in enumerate(values)}).run()
    assert bool(execution.buffers[3].item()) == all(values), values
    assert bool(execution.buffers[4].item()) == any(values), values
''', str(saved)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
