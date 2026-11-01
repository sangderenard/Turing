"""Python source becomes an executable C shell with checked native faults."""
import subprocess

import numpy as np
import pytest

from tests.test_source_ssa_synchronization import compile_source
from src.compiler.identity_concordance import concordance_report
from src.compiler.native_faults import INT_ZERO_DIVISION
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("denominator,frames,warmup", [(3, 2, 1), (0, 1, 0), (0, 0, 1)])
def test_standalone_c_shell_observes_checked_source_execution(tmp_path, denominator, frames, warmup):
    module = compile_source(
        "def root(numerator: int, denominator: int) -> int:\n"
        "    return numerator // denominator\n"
    )
    root = module.functions["root__root"]
    parameters = dict(root.metadata["parameter_names"])
    feeds = {parameters["numerator"]: np.array([17], dtype=np.int64),
             parameters["denominator"]: np.array([denominator], dtype=np.int64)}
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    assert artifact.needs_validation_runtime
    executable = artifact.compile_standalone(tmp_path / "program", feeds)
    result = subprocess.run([str(executable.executable_path), str(frames), str(warmup)],
                            cwd=executable.directory, capture_output=True, text=True, timeout=10)
    if denominator:
        assert result.returncode == 0, result.stderr
        assert "elapsed_s=" in result.stdout
        values = np.fromfile(executable.final_outputs_path, dtype=np.int64)
        output_position, = [i for i, role in enumerate(artifact.buffer_roles) if role == "output"]
        assert values[output_position] == 5
    else:
        assert result.returncode == 7, result.stderr
        assert f"native validation error code={INT_ZERO_DIVISION}" in result.stderr
        assert "elapsed_s=" not in result.stdout
        assert not executable.final_outputs_path.exists()
    assert "0 finding(s)" in concordance_report(module)
