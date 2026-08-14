import json
import threading
import time
from types import SimpleNamespace

from src.compiler.translation_growth_cascade import (
    BoundaryRestartCascade,
    GrowthCascadeState,
)


def test_flagged_growth_restarts_after_precise_boundary_edit(tmp_path):
    language = tmp_path / "python"
    language.mkdir()
    messages = []
    cascade = BoundaryRestartCascade(
        tmp_path,
        wait_seconds=2.0,
        poll_seconds=0.01,
        status_sink=messages.append,
    )
    fingerprint = cascade.observing()
    flag = cascade.flag(
        SimpleNamespace(
            owner="scope Model.forward",
            boundary_hint="python/Model/forward",
            node_count=900,
            depth=80,
            height=42,
            stages={"ssa": 900},
        ),
        fingerprint,
    )
    result = []
    waiter = threading.Thread(
        target=lambda: result.append(cascade.wait_for_change(fingerprint))
    )
    waiter.start()
    time.sleep(0.05)
    rule_dir = language / "Model" / "forward"
    rule_dir.mkdir(parents=True)
    (rule_dir / "opaque.node.json").write_text(
        json.dumps({
            "version": 1,
            "id": "python.Model.forward.opaque",
            "action": "spoof",
            "node_type": "Call",
            "match": {"func.id": "opaque"},
            "result": {"type": "opaque_boundary"},
        }),
        encoding="utf-8",
    )
    waiter.join(timeout=2.0)

    assert result == [True]
    assert cascade.state is GrowthCascadeState.RESTARTING
    receipt = json.loads(flag.read_text(encoding="utf-8"))
    assert receipt["state"] == "restarting"
    assert any("RESTARTING" in message for message in messages)
