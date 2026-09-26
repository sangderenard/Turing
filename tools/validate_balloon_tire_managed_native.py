"""Validate the compiler-owned native balloon tire plus dt controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _read_buffers(path: Path, manifest: dict) -> dict[str, np.ndarray]:
    payload = path.read_bytes()
    offset = 0
    result: dict[str, np.ndarray] = {}
    for buffer in manifest["buffers"]:
        dtype = np.dtype(buffer["dtype"])
        count = int(buffer["element_count"])
        size = count * dtype.itemsize
        values = np.frombuffer(payload[offset:offset + size], dtype=dtype).copy()
        if values.size != count:
            raise ValueError(f"truncated buffer {buffer['name']!r}")
        result[f"{buffer['index']}:{buffer['name']}"] = values
        offset += size
    if offset != len(payload):
        raise ValueError(f"unclaimed native payload bytes: {len(payload) - offset}")
    return result


def _named(buffers: dict[str, np.ndarray], name: str) -> np.ndarray:
    matches = [value for key, value in buffers.items() if key.endswith(f":{name}")]
    if not matches:
        raise KeyError(name)
    return matches[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path,
        nargs="?", default=ROOT / "build" / "balloon-tire-managed-integrated-o0",
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads(
        (directory / "balloon_tire_managed.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    executable = directory / (
        f"{manifest['entrypoint']}.exe" if sys.platform == "win32"
        else manifest["entrypoint"]
    )
    initial = _read_buffers(directory / "initial-state.bin", manifest)

    completed_one = subprocess.run(
        [str(executable), "1"], cwd=directory, capture_output=True, text=True,
    )
    if completed_one.returncode:
        raise RuntimeError(completed_one.stderr or completed_one.stdout)
    one = _read_buffers(directory / "final-outputs.bin", manifest)

    completed_two = subprocess.run(
        [str(executable), "2"], cwd=directory, capture_output=True, text=True,
    )
    if completed_two.returncode:
        raise RuntimeError(completed_two.stderr or completed_two.stdout)
    two = _read_buffers(directory / "final-outputs.bin", manifest)

    for frame_name, buffers in (("one", one), ("two", two)):
        for key, values in buffers.items():
            if values.dtype.kind == "f" and not np.isfinite(values).all():
                raise AssertionError(f"{frame_name}-window {key} is non-finite")

    state_initial = _named(initial, "material.state")
    state_one = _named(one, "material.state")
    state_two = _named(two, "material.state")
    output_initial = _named(initial, "material.output")
    output_one = _named(one, "material.output")
    output_two = _named(two, "material.output")
    state_first_changed = int(np.count_nonzero(state_one != state_initial))
    state_second_changed = int(np.count_nonzero(state_two != state_one))
    output_first_changed = int(np.count_nonzero(output_one != output_initial))
    output_second_changed = int(np.count_nonzero(output_two != output_one))
    if min(state_first_changed, state_second_changed) <= 0:
        raise AssertionError("resident tire state did not evolve in both windows")
    if min(output_first_changed, output_second_changed) <= 0:
        raise AssertionError("tire observation did not evolve in both windows")

    dt_max = float(_named(two, "controller.dt_max")[0])
    maximum_displacement = float(
        _named(two, "material.last_maximum_displacement_m")[0]
    )
    maximum_velocity = float(
        _named(two, "material.last_maximum_velocity_m_s")[0]
    )
    if not (np.isfinite(dt_max) and dt_max > 0.0):
        raise AssertionError(f"invalid controller dt_max {dt_max!r}")
    if not (np.isfinite(maximum_displacement) and maximum_displacement > 0.0):
        raise AssertionError(
            f"invalid maximum displacement {maximum_displacement!r}"
        )
    if not (np.isfinite(maximum_velocity) and maximum_velocity > 0.0):
        raise AssertionError(f"invalid maximum velocity {maximum_velocity!r}")

    print(completed_two.stdout, end="")
    print(f"finite_float_buffers={sum(v.dtype.kind == 'f' for v in two.values())}")
    print(
        f"state_changed=({state_first_changed},{state_second_changed}) "
        f"output_changed=({output_first_changed},{output_second_changed})"
    )
    print(
        f"controller_dt_max={dt_max:.17g} "
        f"maximum_displacement_m={maximum_displacement:.17g} "
        f"maximum_velocity_m_s={maximum_velocity:.17g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
