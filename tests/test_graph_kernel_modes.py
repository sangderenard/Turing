"""Same authored Python, eager or an already accepted native compiler kernel."""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

spec = importlib.util.spec_from_file_location("graph_kernel_modes",
    Path(__file__).resolve().parents[1] / "tools/compiler_probes/probe_graph_kernel_modes.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("nodes", [2, 3, 17, 256])
def test_authored_python_handles_exact_asap_and_alap(nodes):
    result = probe.measure(nodes, repeats=2)
    assert result["mode"] == "eager"
    assert all(case["exact"] for case in result["cases"])
    assert len({case["output_sha256"] for case in result["cases"]}) == 1


def test_default_probe_does_not_compile(monkeypatch, capsys):
    from src.compiler import native_compiler_accelerators as accelerators
    def forbidden(*args, **kwargs):
        raise AssertionError("eager probe invoked compiler")
    monkeypatch.setattr(accelerators, "compile_and_register_dependency_levels_accelerator", forbidden)
    assert probe.main(["--nodes", "17", "--repeats", "1"]) == 0
    assert '"mode": "eager"' in capsys.readouterr().out


def test_same_authored_function_keeps_disconnected_sink_and_alap_slack():
    # 0 -> 1 -> 2 -> 4, 3 -> 4, isolated 5. The short root's
    # ALAP moves to 2; the isolated sink stays at its own ASAP 0.
    values = dict(n=6, preference=1, order=np.array([0, 1, 2, 3, 4, 5]),
                  parent_offsets=np.array([0, 0, 1, 2, 2, 4, 4]),
                  parents=np.array([0, 1, 2, 3]),
                  child_offsets=np.array([0, 1, 2, 3, 4, 4, 4]),
                  children=np.array([1, 2, 4, 4]),
                  asap=np.zeros(6, dtype=np.int64), levels=np.zeros(6, dtype=np.int64))
    assert probe.dependency_levels(*(values[name] for name in probe.NAMES)) == 6
    np.testing.assert_array_equal(values["asap"], [0, 1, 2, 0, 3, 0])
    np.testing.assert_array_equal(values["levels"], [0, 1, 2, 2, 3, 0])


def test_compile_reuse_two_owned_native_arenas_and_refuse_stale_source(tmp_path):
    import json
    import subprocess
    import sys
    script = str(Path(probe.__file__))
    compiled = subprocess.run([sys.executable, "-u", script, "--nodes", "17",
                               "--repeats", "2", "--compile", str(tmp_path)],
                              text=True, capture_output=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    assert "fresh public-source compile" in compiled.stdout
    payload = tmp_path / "dependency-levels.pkl"
    command = [sys.executable, "-u", script, "--nodes", "256", "--repeats", "2",
               "--workers", "2", "--native-payload", str(payload)]
    reused = subprocess.run(command, text=True, capture_output=True, timeout=15)
    assert reused.returncode == 0, reused.stdout + reused.stderr
    result = json.loads(next(line.split("=", 1)[1] for line in reused.stdout.splitlines()
                             if line.startswith("GRAPH_KERNEL_MODE=")))
    assert result["source_receipt"] == "source/payload/library digest matched"
    assert result["mode"] == "native" and all(case["exact"] for case in result["cases"])
    assert len({case["worker_thread"] for case in result["cases"]}) == 2
    receipt_path = payload.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["source_sha256"] = "stale"
    receipt_path.write_text(json.dumps(receipt))
    refused = subprocess.run(command, text=True, capture_output=True, timeout=15)
    assert refused.returncode != 0 and "receipt is stale" in refused.stderr
