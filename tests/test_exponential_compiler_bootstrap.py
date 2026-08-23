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


def test_interrupted_generation_is_preserved_under_a_new_attempt_path(tmp_path):
    interrupted = tmp_path / "waves" / "generation_00005"
    interrupted.mkdir(parents=True)

    assert exponential._unused_wave_root(tmp_path, 5) == (
        tmp_path / "waves" / "generation_00005_attempt_001"
    )

    (tmp_path / "waves" / "generation_00005_attempt_001").mkdir()
    assert exponential._unused_wave_root(tmp_path, 5) == (
        tmp_path / "waves" / "generation_00005_attempt_002"
    )


def test_work_batches_are_smallest_ready_and_dependency_first(tmp_path):
    source = tmp_path / "compiler_part.py"
    source.write_text(
        "def large(value):\n"
        "    first = value + 1\n"
        "    second = first + 2\n"
        "    third = second + 3\n"
        "    return third\n\n"
        "def leaf(value):\n"
        "    return value + 1\n\n"
        "def caller(value):\n"
        "    return leaf(value)\n",
        encoding="utf-8",
    )

    batches = exponential.discover_compiler_work_batches(
        tmp_path, batch_size=1,
    )
    entries = [record["entries"][0] for record in batches]

    assert sorted(entries) == ["caller", "large", "leaf"]
    assert entries.index("leaf") < entries.index("caller")
    assert all(record["authored_call_count"] == 1 for record in batches)


def test_usage_priority_puts_hot_compiler_chain_first_without_reordering_it():
    records = [
        {"source": "cold.py", "entries": ["cold"], "batch_index": 0,
         "dependency_batch_indices": [],
         "estimated_authored_bytes": 1, "estimated_ast_nodes": 1},
        {"source": "hot.py", "entries": ["hot_leaf"], "batch_index": 0,
         "dependency_batch_indices": [],
         "estimated_authored_bytes": 20, "estimated_ast_nodes": 4},
        {"source": "hot.py", "entries": ["hot_parent"], "batch_index": 1,
         "dependency_batch_indices": [0],
         "estimated_authored_bytes": 30, "estimated_ast_nodes": 6},
    ]
    usage = [{
        "source": "hot.py", "qualified_name": "hot_parent",
        "call_count": 100, "inclusive_seconds": 4.0,
    }]

    ordered = exponential.prioritize_compiler_work_batches(records, usage)

    assert [item["entries"][0] for item in ordered] == [
        "hot_leaf", "hot_parent", "cold",
    ]


def test_legacy_state_migration_rebuilds_call_batches_and_preserves_seed(
    tmp_path,
):
    source_root = tmp_path / "compiler"
    source_root.mkdir()
    source = source_root / "one.py"
    source.write_text("def one():\n    return 1\n", encoding="utf-8")
    seed = tmp_path / "seed"
    arguments = argparse.Namespace(source_root=source_root, jobs=1)
    legacy = {
        "schema": exponential.LEGACY_STATE_SCHEMA,
        "sources": [{
            "source": source.resolve().as_posix(),
            "seed_product": seed.as_posix(),
        }],
        "waves": [{"generation": 0}],
    }

    migrated = exponential._migrate_legacy_state(legacy, arguments)

    assert migrated["schema"] == exponential.STATE_SCHEMA
    assert migrated["sources"][0]["entries"] == ["one"]
    assert migrated["sources"][0]["seed_product"] == seed.as_posix()
    assert migrated["migration"]["preserved_wave_count"] == 1


def test_catalogue_revision_check_finds_noncurrent_changed_source(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("def first():\n    return 1\n", encoding="utf-8")
    second.write_text("def second():\n    return 2\n", encoding="utf-8")
    state = {"sources": exponential.discover_compiler_work_batches(
        tmp_path, batch_size=1,
    )}
    second.write_text("def second():\n    return 3\n", encoding="utf-8")

    assert exponential._changed_catalogue_source(state) == second.resolve()


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
        minimum_compile_seconds_before_widening=30.0,
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


def test_supervisor_stops_on_native_installation_hard_failure(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "compiler"
    source_root.mkdir()
    source = source_root / "one.py"
    source.write_text("def one(value):\n    return value + 1\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    output = tmp_path / "bootstrap"
    launches = []

    def run(command, **_kwargs):
        launches.append(tuple(command))
        wave_root = Path(command[command.index("--output") + 1])
        wave_root.mkdir(parents=True)
        (wave_root / "wave-result.json").write_text(json.dumps({
            "schema": exponential.WAVE_SCHEMA,
            "status": "failed",
            "generation": 0,
            "process_id": 3000,
            "source": source.as_posix(),
            "workers_joined": True,
            "hard_failure": True,
            "error_type": "NativeInstallationRequiredError",
            "error": "one remained Python",
            "failures": [{"qualified_name": "one"}],
        }), encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(exponential.subprocess, "run", run)
    arguments = argparse.Namespace(
        source_root=source_root,
        output=output,
        jobs=2,
        max_total_gb=16.0,
        worker_reserve_gb=4.0,
        max_worker_gb=6.0,
        unit_timeout_seconds=60.0,
        minimum_compile_seconds_before_widening=30.0,
        max_generations=8,
        max_sweeps=4,
        extraction_contract=contract,
    )

    assert exponential._supervise(arguments) == 1
    assert len(launches) == 1
    state = json.loads(
        (output / "bootstrap-state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "hard-failed"
    assert state["cursor"] == 0
    assert state["hard_failure"]["failures"] == [{"qualified_name": "one"}]
    assert not (output / "supervisor.lock").exists()


def test_supervisor_stops_on_any_failed_wave(tmp_path, monkeypatch):
    source_root = tmp_path / "compiler"
    source_root.mkdir()
    source = source_root / "one.py"
    source.write_text("def one(value):\n    return value + 1\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    output = tmp_path / "bootstrap"

    def run(command, **_kwargs):
        wave_root = Path(command[command.index("--output") + 1])
        wave_root.mkdir(parents=True)
        (wave_root / "wave-result.json").write_text(json.dumps({
            "schema": exponential.WAVE_SCHEMA,
            "status": "failed",
            "generation": 0,
            "process_id": 3001,
            "source": source.as_posix(),
            "workers_joined": True,
            "error_type": "ValueError",
            "error": "lowering failed",
        }), encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(exponential.subprocess, "run", run)
    arguments = argparse.Namespace(
        source_root=source_root,
        output=output,
        jobs=2,
        max_total_gb=16.0,
        worker_reserve_gb=4.0,
        max_worker_gb=6.0,
        unit_timeout_seconds=60.0,
        minimum_compile_seconds_before_widening=30.0,
        max_generations=8,
        max_sweeps=4,
        extraction_contract=contract,
    )

    assert exponential._supervise(arguments) == 1
    state = json.loads(
        (output / "bootstrap-state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "hard-failed"
    assert state["cursor"] == 0
    assert state["hard_failure"]["error_type"] == "ValueError"
    assert state["hard_failure"]["error"] == "lowering failed"


def test_startup_hard_fails_oldest_current_unsuccessful_archive_and_prioritizes_it(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "compiler"
    source_root.mkdir()
    cold = source_root / "cold.py"
    failed = source_root / "failed.py"
    cold.write_text("def cold():\n    return 0\n", encoding="utf-8")
    failed.write_text("def failed():\n    return 1\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    output = tmp_path / "bootstrap"
    output.mkdir()
    sources = exponential.discover_compiler_work_batches(
        source_root, batch_size=1,
    )
    sources.sort(key=lambda item: item["source"] == failed.resolve().as_posix())
    wave_root = output / "waves" / "generation_00000"
    wave_root.mkdir(parents=True)
    result_path = wave_root / "wave-result.json"
    result_path.write_text(json.dumps({
        "schema": exponential.WAVE_SCHEMA,
        "status": "complete",
        "generation": 0,
        "source": failed.resolve().as_posix(),
        "source_sha256": exponential._sha256(failed),
        "entries": ["failed"],
        "outcome": {
            "status": "frontier",
            "installed_qualified_names": [],
            "creep_frontier": [{
                "qualified_name": "failed",
                "status": "partial",
                "action": "materialize-required-source-values",
            }],
            "native_verification_frontier": [],
            "unit_counts": {"partial": 1},
        },
    }), encoding="utf-8")
    (output / "bootstrap-state.json").write_text(json.dumps({
        "schema": exponential.STATE_SCHEMA,
        "status": "running",
        "source_root": source_root.as_posix(),
        "generation": 1,
        "sweep": 0,
        "cursor": 1,
        "batch_size": 1,
        "sweep_progress": True,
        "sources": sources,
        "waves": [{
            "generation": 0,
            "source": failed.resolve().as_posix(),
            "result": result_path.resolve().as_posix(),
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(
        exponential.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("startup archive failure must precede a new worker")
        ),
    )
    arguments = argparse.Namespace(
        source_root=source_root, output=output, jobs=1,
        max_total_gb=16.0, worker_reserve_gb=4.0,
        max_worker_gb=6.0, unit_timeout_seconds=60.0,
        minimum_compile_seconds_before_widening=30.0,
        max_generations=8, max_sweeps=4, extraction_contract=contract,
    )

    assert exponential._supervise(arguments) == 1
    state = json.loads(
        (output / "bootstrap-state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "hard-failed"
    assert state["hard_failure"]["archive_generation"] == 0
    assert state["sources"][0]["entries"] == ["failed"]
    assert state["cursor"] == 0


def test_revised_source_resumes_a_persisted_hard_failure(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "compiler"
    source_root.mkdir()
    source = source_root / "one.py"
    source.write_text("def one():\n    return 1\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    output = tmp_path / "bootstrap"
    output.mkdir()
    old_sources = exponential.discover_compiler_work_batches(
        source_root, batch_size=1,
    )
    (output / "bootstrap-state.json").write_text(json.dumps({
        "schema": exponential.STATE_SCHEMA,
        "status": "hard-failed",
        "source_root": source_root.as_posix(),
        "generation": 4,
        "sweep": 0,
        "cursor": 0,
        "batch_size": 1,
        "sweep_progress": False,
        "sources": old_sources,
        "waves": [],
        "hard_failure": {"error": "old failure"},
    }), encoding="utf-8")
    source.write_text("def one():\n    return 2\n", encoding="utf-8")
    launches = []

    def run(command, **_kwargs):
        launches.append(tuple(command))
        wave_root = Path(command[command.index("--output") + 1])
        wave_root.mkdir(parents=True)
        (wave_root / "wave-result.json").write_text(json.dumps({
            "schema": exponential.WAVE_SCHEMA,
            "status": "complete",
            "generation": 4,
            "process_id": 4000,
            "source": source.as_posix(),
            "workers_joined": True,
            "registry_changed": False,
            "outcome_sha256": "stable",
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(exponential.subprocess, "run", run)
    arguments = argparse.Namespace(
        source_root=source_root,
        output=output,
        jobs=1,
        max_total_gb=16.0,
        worker_reserve_gb=4.0,
        max_worker_gb=6.0,
        unit_timeout_seconds=60.0,
        minimum_compile_seconds_before_widening=30.0,
        max_generations=5,
        max_sweeps=4,
        extraction_contract=contract,
    )

    assert exponential._supervise(arguments) == 1
    assert len(launches) == 1
    state = json.loads(
        (output / "bootstrap-state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "frontier"
    assert "hard_failure" not in state
    assert state["resolved_hard_failures"][0]["error"] == "old failure"


def test_timed_out_unit_gets_one_fresh_process_unbounded_retry(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "compiler"
    source_root.mkdir()
    source = source_root / "one.py"
    source.write_text("def one(value):\n    return value + 1\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    output = tmp_path / "bootstrap"
    launches = []

    def run(command, **_kwargs):
        launches.append(tuple(command))
        generation = int(command[command.index("--generation") + 1])
        wave_root = Path(command[command.index("--output") + 1])
        deep_retry = "--deep-retry" in command
        result = {
            "schema": exponential.WAVE_SCHEMA,
            "status": "complete",
            "generation": generation,
            "process_id": 2000 + generation,
            "source": source.as_posix(),
            "mode": "deep-retry" if deep_retry else "bounded",
            "workers_joined": True,
            "registry_changed": False,
            "registry_before_sha256": "registry-r1",
            "registry_after_sha256": "registry-r1",
            "elapsed_seconds": 1.0,
            "outcome_sha256": "deep" if deep_retry else "bounded",
            "timed_out_entries": [] if deep_retry else ["one"],
            "terminal_timed_out_entries": (
                [] if deep_retry else ["one"]
            ),
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
        minimum_compile_seconds_before_widening=30.0,
        max_generations=8,
        max_sweeps=4,
        extraction_contract=contract,
    )

    assert exponential._supervise(arguments) == 0

    assert len(launches) == 3
    assert "--deep-retry" not in launches[0]
    assert "--deep-retry" in launches[1]
    assert launches[1][launches[1].index("--unit-timeout-seconds") + 1] == "0"
    assert launches[1][launches[1].index("--entry") + 1] == "one"
    assert "--deep-retry" not in launches[2]
    state = json.loads(
        (output / "bootstrap-state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "fixed-point"
    assert [wave["mode"] for wave in state["waves"]] == [
        "bounded", "deep-retry", "bounded",
    ]
    assert state["waves"][0]["scheduled_deep_retry"] is True
    assert state["waves"][2]["scheduled_deep_retry"] is False
