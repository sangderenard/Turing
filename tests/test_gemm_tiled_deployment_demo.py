"""gemm runs on the expected tiling with a commensurate gain -- proven.

Runs the real demo (``tools/demo_gemm_tiled_deployment.py``) end to end in
a subprocess against a temporary bank: bank build + auto-profile, the
deployment strategy's workers/chunk choice, the source-derived per-item
partition, and pool execution of the C-block lanes -- then asserts the
three claims that make it a proof rather than a printout:

* the tiling is the CHOOSER'S tiling: no ``--tile`` override is supplied,
  every lane executes, and the lane count follows the measured winner;
* the result is exact against the oracle at every worker count (the demo
  itself hard-asserts worst |err| < 1e-9 before printing it);
* the gain is COMMENSURATE: the native pooled product beats its identical
  native serial control by a real margin, while the host-pool instrument
  independently checks the plan's lane semantics.

Subprocess on purpose: the demo's own entry point is what users run, so
the test proves that artifact, not a reimplementation that could drift.
A generous margin (1.3x for threads on an 8-core host that measured
2.9x; 1.2x for locality that measured 2.0x) keeps a loaded machine from
flaking the assertion while still catching a real regression to serial.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_demo_proves_tiling_shape_correctness_and_gain(tmp_path):
    plan_path = tmp_path / "gemm-prebake.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "demo_gemm_tiled_deployment.py"),
            "--size", "256", "--contract", "fast",
            "--root", str(tmp_path / "bank"),
            "--plan-output", str(plan_path),
        ],
        capture_output=True, text=True, timeout=900,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert plan_path.is_file(), output

    assert "FORCED by --tile" not in output
    chosen = re.search(r"core (\d+)\^3 chosen", output)
    assert chosen, output
    tile = int(chosen.group(1))
    lanes = re.search(r"execute 256\^3 as (\d+) C-block lanes", output)
    assert lanes and int(lanes.group(1)) == (256 // tile) ** 2, output
    for line in re.findall(r"workers=\d+: .*", output):
        assert "worst |err|" in line, output

    # Correctness was hard-asserted inside the demo; re-read it here so a
    # softened demo cannot silently stop checking.
    errors = [
        float(value) for value in re.findall(r"worst \|err\| ([\d.e+-]+)", output)
    ]
    assert errors and all(err < 1e-9 for err in errors), output

    ratio = re.search(r"pool vs serial :\s*([\d.]+)x", output)
    assert ratio, output
    assert float(ratio.group(1)) > 1.3, (
        "pooled lanes should meaningfully beat the identical serial code "
        f"path; got {ratio.group(1)}x\n{output}"
    )

    single = re.search(r"pool vs single :\s*([\d.]+)x", output)
    assert single and float(single.group(1)) > 1.2, output

    import json
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == "turing.prebaked-gemm-launch-matrix.v1"
    assert plan["launch"]["lane_count"] == int(lanes.group(1))
    assert plan["module_binding_by_name"]["A"] == 0
    assert all(
        call["module_key"] == plan["module_key"]
        for lane in plan["lanes"] for call in lane["calls"]
    )
    assert "built native pooled product" in output
    assert "Python=false" in output
    native_error = re.search(
        r"native product: .*worst \|err\| ([\d.e+-]+)", output,
    )
    assert native_error and float(native_error.group(1)) < 1e-9, output
    native_ratio = re.search(
        r"native pool vs native serial:\s*([\d.]+)x", output,
    )
    assert native_ratio and float(native_ratio.group(1)) > 1.3, output
