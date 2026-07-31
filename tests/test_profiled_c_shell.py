from __future__ import annotations

from src.common.tensors.accelerator_backends.profiled_c_shell import (
    profiled_c_shell,
)
from src.compiler.glsl_deployment_strategy import DeploymentProfiler


def test_profiled_c_shell_separates_shell_and_supplied_device_time():
    shell = profiled_c_shell()

    @shell.callback
    def compute(_context, device_ns):
        device_ns[0] = 123456
        return 1

    result = shell.launch(compute)

    assert result.status == 1
    assert result.shell_ns > 0
    assert result.device_ns == 123456


def test_profiled_c_shell_uses_existing_deployment_profiler_rows():
    shell = profiled_c_shell()
    profiler = DeploymentProfiler(enabled=True)

    @shell.callback
    def compute(_context, device_ns):
        device_ns[0] = 2000
        return 1

    token = profiler.begin_shell("torture/c")
    result = shell.launch(compute)
    shell.record(
        profiler,
        result,
        path="torture/c",
        label="operator-grab-bag",
    )
    profiler.end_shell("torture/c", token)

    row = next(
        row
        for row in profiler.report()["rows"]
        if row["section"] == "compiled-c-shell"
    )
    assert row["cpu_ms"] > 0.0
    assert row["gpu_ms"] == 0.002
    assert row["dispatches"] == 1
