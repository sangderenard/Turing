"""Shared canonical solid geometry; bounded CPU tests, not GPU parity claims."""
from pathlib import Path
import shutil
import subprocess

import pytest

from src.compiler.closed_solid_geometry import (
    CLOSED_SOLID_OFFSETS, CLOSED_SOLID_STRIDE, emit_closed_solid_wgsl,
)


def test_cpu_oriented_queries_and_real_cage_wrench():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for authored worker queries")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests/closed_solid_geometry_check.cjs")],
                            cwd=root, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"result":"PASS"' in result.stdout


def test_shared_wgsl_layout_emission():
    source = emit_closed_solid_wgsl()
    assert CLOSED_SOLID_STRIDE == 32
    assert set(CLOSED_SOLID_OFFSETS.values()) == set(range(0, 32, 4))
    assert "index * 32u" in source
    assert "@@" not in source
    assert "fn closed_solid_closest" in source
    assert "fn closed_solid_segment" in source
