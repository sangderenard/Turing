from pathlib import Path

from src.common.tensors.backend_capability_audit import audit_catalog


def test_shared_catalog_audit_reports_each_target_without_hiding_gaps():
    catalog = (
        Path(__file__).resolve().parents[2]
        / "nodus"
        / "ops"
        / "canonical_ops.json"
    )
    rows = audit_catalog(catalog)
    by_name = {row["name"]: row for row in rows}

    assert len(rows) == 66
    assert by_name["add"] == {
        "name": "add",
        "class": "binary",
        "c_native": True,
        "glsl": True,
        "nodus_kernel": True,
        "complete": True,
    }
    assert by_name["sin"]["c_native"]
    assert by_name["sin"]["glsl"]
    assert by_name["sin"]["nodus_kernel"]
    assert not by_name["matmul"]["nodus_kernel"]
