"""Math lab: whitelist parser, per-target product records, NumPy parity, job timeout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "examples" / "meowdoku")):
    if path not in sys.path:
        sys.path.insert(0, path)

import math_lab  # noqa: E402
from math_lab import MathSyntaxError, parse_math  # noqa: E402

PROBE = "r = sqrt(x*x + y*y)\ne = x*y + 2*z"


def test_parser_accepts_probe_laws():
    equations = parse_math(PROBE)
    assert [str(eq.lhs) for eq in equations] == ["r", "e"]
    x, y = sympy.symbols("x y")
    assert equations[0].rhs == sympy.sqrt(x ** 2 + y ** 2)
    assert str(parse_math("s = sin(x)*exp(y) + atan2(x, y) + max(x, 2) + x**-2")[0].lhs) == "s"


@pytest.mark.parametrize("text, construct", [
    ("r = __import__('os')", "Call"),
    ("r = x.real", "Attribute"),
    ("r = (lambda q: q)(x)", "Lambda"),
    ("\n".join(f"o{i} = x + {i}" for i in range(40)), None),
    ("x = x + 1", "Name"),
    ("r = x // 2", "FloorDiv"),
    ("r = x ** 17", "Pow"),
    ("r = x ** y", "Name"),
    ("r = tan(x)", "Call"),
    ("r = x if y else z", "IfExp"),
    ("r = t1 + 1", "Name"),
    ("a = x; b = y", "Assign"),
])
def test_parser_rejects(text, construct):
    with pytest.raises(MathSyntaxError) as raised:
        parse_math(text)
    if construct is not None:
        assert construct in (raised.value.construct or "") or construct in str(raised.value)


def test_parser_limits():
    with pytest.raises(MathSyntaxError):
        parse_math("r = " + "x + " * 2000 + "x")
    with pytest.raises(MathSyntaxError):
        parse_math("r = " + "(" * 70 + "x" + ")" * 70 + " + " + "-" * 70 + "x")


def test_compile_math_records_each_target(tmp_path):
    products = math_lab.compile_math("r = sqrt(x*x + y*y)", tmp_path / "a", batch=1024)
    record = json.loads((tmp_path / "a" / "products.json").read_text(encoding="utf-8"))
    assert record["schema"] == "turing.math-lab-products.v1"
    assert record["inputs"] == ["x", "y"] and record["outputs"] == ["r"]
    for target in ("numpy", "wasm", "wgsl", "native"):
        state = record["targets"][target]
        assert state["state"] == "complete", (target, state)
        assert state["sha256"] and state["bytes"] > 0 and state["seconds"] >= 0
    assert (tmp_path / "a" / "law.wasm").read_bytes()[:4] == b"\0asm"
    assert (tmp_path / "a" / "law.at.py").exists() and (tmp_path / "a" / "law.api.yaml").exists()
    wgsl_abi = json.loads((tmp_path / "a" / "law.wgsl.abi.json").read_text(encoding="utf-8"))
    assert wgsl_abi["arguments"] == ["x", "y"] and "r" in wgsl_abi["output_slots"]
    # NumPy product vs sympy lambdify on 1e5 rows
    function = math_lab._load_numpy_product(products.path("numpy"), "user_law")
    columns = math_lab._sample(["x", "y"], 100_000, 7)
    got = math_lab._as_tuple(function(columns["x"], columns["y"]))
    reference = math_lab._reference(products.equations, ["x", "y"], ["r"])(columns["x"], columns["y"])
    assert math_lab._parity(got, reference)["max_abs_err"] <= 1e-12


def test_refusal_is_recorded_not_raised(tmp_path):
    # An output that is a bare input has no equation structure the compiler accepts on every
    # target; whatever each target does, it must end as a state record and never raise.
    products = math_lab.compile_math("r = x", tmp_path / "b", batch=256, targets=("numpy", "wgsl"))
    for state in products.record["targets"].values():
        assert state["state"] in ("complete", "refused")
        if state["state"] == "refused":
            assert state["error_type"] and state["error"] and state["stage"]


def test_run_job_timeout_and_rejection(tmp_path):
    assert math_lab.run_job("r = __import__('os')", tmp_path / "c")["state"] == "rejected"
    assert math_lab.run_job(PROBE, tmp_path / "d", timeout_s=0.5) == {"state": "timeout", "timeout_s": 0.5}
