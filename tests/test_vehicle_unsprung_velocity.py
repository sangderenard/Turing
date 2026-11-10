"""Bounded geometry derivative and tire-boundary checks; no compiled vehicle build."""
import ast
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_solved_linkage_velocity_reaches_tire_contact_and_sweep():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the authored worker geometry")
    result = subprocess.run(
        [node, str(ROOT / "tests/vehicle_unsprung_velocity_check.cjs")],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"result":"PASS"' in result.stdout


def test_gpu_geometry_transports_existing_compression_rate_selector():
    source = (ROOT / "src/compiler/abstract_ui_vehicles.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selectors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "compression_velocity" and isinstance(value, ast.JoinedStr):
                selectors.append(value)
    assert len(selectors) == 1
    # Evaluate the existing exact-index gather expression, not an assumed
    # contiguous ABI base. Deliberately use noncontiguous sentinel indices.
    indices = dict(zip(("front_left", "front_right", "rear_left", "rear_right"), (7, 19, 31, 43)))
    gathered = eval(compile(ast.Expression(selectors[0]), "<compression-rate-selector>", "eval"),
                    {"vehicle": lambda name: str(indices[name.removeprefix("compression_velocity_")])})
    python_selector = gathered.replace("3u", "3").replace("1u", "1").replace("2u", "2")
    for lane, expected in enumerate(indices.values()):
        assert eval(python_selector, {"select": lambda a, b, condition: b if condition else a, "lane": lane}) == expected
    assert '"@@COMPRESSION_VELOCITY@@": expressions["compression_velocity"]' in source
    assert "vec3<f32>(0.0f, @@COMPRESSION_VELOCITY@@, 0.0f)" in source
    assert "let hub_velocity = velocity + relative_hub_velocity + cross(angular, hub_offset);" in source
    assert "let point_velocity = velocity + relative_hub_velocity + cross(angular, attachment);" in source
    assert "let next_ring_support_point = ring_support_point + hub_velocity * @@FIXED_DT@@;" in source
