from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from tools import bootstrap_compiler_exponentially as exponential


def test_catalogue_discovery_is_deterministic_and_cheap_first(tmp_path):
    large = tmp_path / "large.py"
    large.write_text(
        "def large(value):\n    return value + 1\n" + "\n" * 20,
        encoding="utf-8",
    )
    small = tmp_path / "small.py"
    small.write_text(
        "def small(value):\n    return value\n",
        encoding="utf-8",
    )
    (tmp_path / "empty.py").write_text("VALUE = 1\n", encoding="utf-8")

    catalogues = exponential.discover_compiler_catalogues(tmp_path)

    assert [Path(record["source"]).name for record in catalogues] == [
        "small.py", "large.py",
    ]
    assert all(record["authored_call_count"] == 1 for record in catalogues)


def test_supervisor_joins_then_restarts_until_a_stable_sweep(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "compiler"
    source_root.mkdir()
    source = source_root / "one.py"
    source.write_text(
        "def one(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    output = tmp_path / "bootstrap"
    launches = []

    def run(command, **_kwargs):
        launches.append(tuple(command))
        generation = int(command[command.index("--generation") + 1])
        wave_root = Path(command[command.index("--output") + 1])
        result = {
            "schema": exponential.WAVE_SCHEMA,
            "status": "complete",
            "generation": generation,
            "process_id": 1000 + generation,
            "source": source.as_posix(),
            "workers_joined": True,
            "registry_changed": generation == 0,
            "outcome_sha256": "stable-outcome",
            "seed_product": (wave_root / "product" / "round_000").as_posix(),
        }
        wave_root.mkdir(parents=True)
        (wave_root / "wave-result.json").write_text(
            json.dumps(result), encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(exponential.subprocess, "run", run)
    arguments = argparse.Namespace(
        source_root=source_root,
        output=output,
        jobs=2,
        max_total_gb=16.0,
        worker_reserve_gb=4.0,
        max_worker_gb=6.0,
        unit_timeout_seconds=60.0,
        max_generations=8,
        max_sweeps=4,
        extraction_contract=contract,
    )

    return_code = exponential._supervise(arguments)

    assert return_code == 0
    assert len(launches) == 2
    assert all("--wave-worker" in command for command in launches)
    assert "--seed-product" not in launches[0]
    assert "--seed-product" in launches[1]
    state = json.loads(
        (output / "bootstrap-state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "fixed-point"
    assert [wave["process_id"] for wave in state["waves"]] == [1000, 1001]
    assert all(wave["workers_joined"] for wave in state["waves"])
    assert not (output / "supervisor.lock").exists()
