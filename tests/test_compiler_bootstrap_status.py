from __future__ import annotations

import hashlib
import json

from tools.compiler_bootstrap_status import build_status


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_status_distinguishes_registry_acceptance_from_loose_receipts(tmp_path):
    source = tmp_path / "compiler" / "stage.py"
    source.parent.mkdir()
    source.write_text(
        "def accepted(): return 1\ndef frontier(): return 2\n",
        encoding="utf-8",
    )
    product = tmp_path / "product"
    manifest = product / "manifest.json"
    _write_json(manifest, {
        "source": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    receipt = product / "native" / "native-verification.json"
    _write_json(receipt, {"status": "verified"})
    registry = tmp_path / "registry.json"
    _write_json(registry, {
        "schema": "turing.compiler-bootstrap-registry.v1",
        "products": [{
            "product": product.as_posix(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "installable": [{
                "qualified_name": "accepted",
                "activation_adapter": "test-v1",
                "verification_receipt": "native/native-verification.json",
            }],
        }],
    })
    root = tmp_path / "bootstrap"
    _write_json(root / "bootstrap-state.json", {
        "status": "running",
        "generation": 3,
        "cursor": 0,
        "sources": [{
            "source": source.as_posix(),
            "batch_index": 0,
            "attempts": 1,
            "entries": ["accepted", "frontier"],
        }, {
            "source": source.as_posix(),
            "batch_index": 1,
            "attempts": 0,
            "entries": ["pending"],
        }],
        "waves": [{"status": "complete"}],
        "compiler_usage": {"generation": 0, "records": [{
            "source": source.as_posix(),
            "qualified_name": "pending",
            "call_count": 7,
            "inclusive_seconds": 2.0,
        }]},
    })
    _write_json(
        root / "loose" / "native-verification.json",
        {"status": "verified"},
    )

    report = build_status(root, registry)

    assert report["graph"]["call_counts"] == {
        "accepted_native": 1,
        "attempted_frontier": 1,
        "pending": 1,
    }
    assert report["native"]["accepted"][0]["qualified_name"] == "accepted"
    assert report["native"]["verified_receipts_under_bootstrap_root"] == 1
    assert report["graph"]["batches"][0]["status"] == "mixed"
    assert report["graph"]["batches"][1]["entries"][0][
        "observed_usage"
    ]["call_count"] == 7


def test_status_refuses_a_stale_registered_source(tmp_path):
    source = tmp_path / "stage.py"
    source.write_text("def stage(): return 1\n", encoding="utf-8")
    product = tmp_path / "product"
    manifest = product / "manifest.json"
    _write_json(manifest, {
        "source": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    receipt = product / "native" / "native-verification.json"
    _write_json(receipt, {"status": "verified"})
    registry = tmp_path / "registry.json"
    _write_json(registry, {
        "schema": "turing.compiler-bootstrap-registry.v1",
        "products": [{
            "product": product.as_posix(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "installable": [{
                "qualified_name": "stage",
                "verification_receipt": "native/native-verification.json",
            }],
        }],
    })
    root = tmp_path / "bootstrap"
    _write_json(root / "bootstrap-state.json", {
        "generation": 0,
        "sources": [],
        "waves": [],
    })
    source.write_text("def stage(): return 2\n", encoding="utf-8")

    report = build_status(root, registry)

    assert report["native"]["accepted"] == []
    assert "source digest differs" in report["native"]["registry_refusals"][0][
        "reason"
    ]
