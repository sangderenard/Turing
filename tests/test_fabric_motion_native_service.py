"""The REST-owned fabric process applies row controls before publishing motion."""

import os
from pathlib import Path
import socket
import time

import pytest

from examples.meowdoku.fabric_motion import FabricMotion


def _configuration():
    fields, count = 38, 8 * 256
    data = [0.0] * (fields * count)
    for node in range(4):
        at = node * fields
        x, y = node % 2, node // 2
        data[at + 0] = float(x)
        data[at + 1] = 0.5 + y
        data[at + 6] = 1
        data[at + 7] = float(y == 1)
        data[at + 8] = float(x)
        data[at + 9] = 0.5 + y
        data[at + 11] = 1
        data[at + 12] = 10
        data[at + 13] = 0.35
        data[at + 15] = -9.8
    return data


@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_native_fabric_service_controls_and_complete_frames(tmp_path, backend):
    host = os.environ.get("TURING_FABRIC_ROW_HOST")
    plugin = os.environ.get("TURING_FABRIC_ROW_PLUGIN")
    core = os.environ.get("TURING_FABRIC_ROW_CORE")
    if not all(value and Path(value).is_file() for value in (host, plugin)) or not core:
        pytest.skip("set TURING_FABRIC_ROW_HOST/PLUGIN/CORE to built Nodus products")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    motion = FabricMotion(Path(host), Path(plugin), Path(core), tmp_path, port=port)
    try:
        started = motion.start(
            "owner", _configuration(), 2, backend=backend, substeps=2,
            gpu_target_ms=2 if backend == "vulkan" else 0,
            floor_y=0, table_contact=[2, 0.25, -1, 2, -1, 1],
        )
        deadline = time.monotonic() + 3
        state = None
        while time.monotonic() < deadline:
            state = motion.state("owner", started["token"])
            if state["frame"] is not None and state["frame"]["frame"] >= 10:
                break
            time.sleep(0.025)
        assert state is not None and state["running"], state
        assert state["frame"] is not None, state
        assert len(state["frame"]["positions"]) == 4
        assert state["frame"]["positions"][0][1] < 0.5
        assert min(point[1] for point in state["frame"]["positions"]) >= 0.25 - 1e-6
        assert state["substeps"] == 2 and state["backend"] == backend
        assert motion.command("owner", started["token"], "pause")["frozen"]
        assert motion.command("owner", started["token"], "resume")["running"]
        assert not motion.command("owner", started["token"], "stop")["running"]
    finally:
        motion.close()
