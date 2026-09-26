from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reuse_only_ladder_retains_a_missing_variant_as_a_row(tmp_path):
    report = tmp_path / "ladder.json"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "tools" / "kernel_bank_probe.py"),
            "--root", str(tmp_path / "bank"),
            "--ladder", "32", "64",
            "--kernel", "gemm", "--contract", "fast",
            "--rebuild-policy", "reuse-only",
            "--output", str(report),
        ],
        capture_output=True, text=True, timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = json.loads(report.read_text(encoding="utf-8"))
    assert [row["specialized"]["m"] for row in rows] == [32, 64]
    assert all(row["status"] == "REFUSED" for row in rows)
    assert all("not in the bank" in row["reason"] for row in rows)
    assert all(row["compiler_fingerprint"] for row in rows)
    assert all(row["source_sha256"] for row in rows)

