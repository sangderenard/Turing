"""Small regressions for supplied schedules and in-compile deployment work."""
import dataclasses
import io
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture(autouse=True)
def bounded_compiler_memory():
    from src.compiler.work_contract import active_contract, scoped_work_contract
    with scoped_work_contract(dataclasses.replace(active_contract(),
            memory_budget_bytes=1 << 30, memory_elastic_available_percent=0)):
        yield


def test_auto_does_not_call_the_prepartitioner(monkeypatch):
    from src.compiler import fortran_c_shell as shell, project_compilation_product as products
    observed = []
    monkeypatch.setattr(products, "compile_partitioned_source",
                        lambda *a, **k: pytest.fail("auto prepartitioned source"))
    monkeypatch.setattr(shell, "_run_source_lowering", lambda *a, **k: observed.append((a, k)))
    shell.lower_ast_source_to_ssa("def root(x):\n    return x + 1\n", "root", source_compilation="auto")
    assert observed[0][1]["source_compilation"] == "auto"
    with pytest.raises(ValueError, match="source list"):
        shell.lower_ast_source_to_ssa("def root():\n    return 1\n", "root", source_compilation="parallel")


def test_keyboard_interrupt_cancels_admitted_compiler_threads():
    from src.compiler.graph_work import (
        CompilerWorkPolicy, bounded_compiler_work,
        raise_if_compiler_work_cancelled,
    )

    entered = threading.Barrier(2)
    cancelled = threading.Event()

    def work(index):
        entered.wait(timeout=2)
        if index == 0:
            raise KeyboardInterrupt()
        try:
            while True:
                raise_if_compiler_work_cancelled()
                time.sleep(0.001)
        except KeyboardInterrupt:
            cancelled.set()
            raise

    policy = CompilerWorkPolicy(
        mode="parallel", jobs=2, minimum_jobs=2,
        memory_bytes=None, reserve_bytes=0, working_bytes=1,
    )
    with pytest.raises(KeyboardInterrupt):
        bounded_compiler_work((0, 1), work, lambda *_args: None, policy=policy)
    assert cancelled.wait(timeout=2)


@pytest.mark.parametrize("inner_jobs", (3, None))
def test_source_schedule_inner_limit_is_scoped_separately(monkeypatch, inner_jobs):
    from types import SimpleNamespace
    from src.compiler import fortran_c_shell as shell
    from src.compiler.graph_work import compiler_work_settings
    observed = []
    def lower(_implementation, *args, **kwargs):
        policy = compiler_work_settings()
        observed.append((policy.jobs, policy.mode, kwargs["source_compilation"]))
        return SimpleNamespace(metadata={}), {}, ()
    monkeypatch.setattr(shell, "_run_source_lowering", lower)
    prior = compiler_work_settings()
    options = {"jobs": 1, "worker_resident_reservation_bytes": 1 << 20}
    if inner_jobs is not None:
        options["inner_jobs"] = inner_jobs
    shell.lower_ast_source_to_ssa([
        {"label": "whole", "source": "def whole():\n    return 7\n", "entrypoint": "whole"},
    ], source_compilation="parallel", partitioned_options=options, progress_bars=False)
    assert observed == [(inner_jobs, "auto", "auto")]
    assert compiler_work_settings() == prior


def test_parallel_ingestion_never_restores_a_closed_worker_stdout():
    from tools.compiler_probes.probe_parallel_stdout_lifecycle import observe_parallel_ingestion_stdout
    receipt = observe_parallel_ingestion_stdout()
    assert receipt["sources_completed"] == 2
    assert receipt["stdout_preserved"]
    assert not receipt["stdout_closed"]


def test_parallel_book_posts_are_one_atomic_concordance_transaction():
    from src.compiler.identity_concordance import (IdentityBook, Mode, Novel,
                                                   concurrent_identity_publication)
    from src.compiler.concordance_declarations import COMPILE_POLICY, COMPILE_ENTRY, POLICY_DECLARATION
    book = IdentityBook()
    def post(index):
        return book.post(COMPILE_POLICY, (str(index),), str(index), stage=COMPILE_ENTRY,
                         provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD)
    with concurrent_identity_publication(book), ThreadPoolExecutor(max_workers=2) as pool:
        refs = tuple(pool.map(post, range(32)))
    assert type(book) is IdentityBook
    assert book.clock[0] == 32
    assert sorted(book.stamp_of(ref) for ref in refs) == list(range(32))
    assert all(book.mint_of(ref) is not None for ref in refs)


def test_workers_consume_existing_region_closures_without_recutting():
    from src.compiler.hierarchical_plan import PlanClosure, PlanLine, expand_plan_regions
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book
    from src.compiler.graph_work import CompilerWorkPolicy, compiler_work_policy
    regions = tuple(PlanClosure(f"region_{i}", (0, 1, 2), (
        PlanLine.create("Max", inputs=(0, 1, 2), outputs=(3 + i,), input_roles=("operand", "operand", "operand")),
    ), closure_id=i, value_shapes=tuple((j, (), "float64") for j in range(5))) for i in range(2))
    _book, token = begin_identity_book()
    policy = CompilerWorkPolicy(memory_bytes=1 << 30, working_bytes=1 << 20)
    try:
        with compiler_work_policy(policy):
            expanded = expand_plan_regions(PlanClosure("root", (), regions), resident_limit=2)
        assert tuple(expanded) == tuple((region.name, region.closure_id) for region in regions)
        temporaries = [int(instructions[0].res.id) for instructions in expanded.values()]
        assert len(set(temporaries)) == 2
        assert policy.stats["outstanding_peak"] == 2
    finally:
        end_identity_book(token)


def test_parallel_admission_reopens_while_first_job_is_running():
    from src.compiler.graph_work import bounded_compiler_work

    second_started = threading.Event()
    active = 0
    peak = 0
    lock = threading.Lock()

    class OpeningPolicy:
        mode = "auto"
        stats = {}
        calls = 0

        def capacity(self, ready):
            self.calls += 1
            return min(ready, 1 if self.calls < 3 else 2)

    def work(job):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if job == 0:
                assert second_started.wait(2), "admission never reopened"
            else:
                second_started.set()
            return job
        finally:
            with lock:
                active -= 1

    published = []
    bounded_compiler_work((0, 1), work,
        lambda ordinal, job, result: published.append((ordinal, job, result)),
        policy=OpeningPolicy())
    assert peak == 2
    assert published == [(0, 0, 0), (1, 1, 1)]


def test_minimum_workers_floor_memory_admission(monkeypatch):
    from src.compiler import shell_telemetry
    from src.compiler.graph_work import CompilerWorkPolicy

    monkeypatch.setattr(shell_telemetry, "process_rss_bytes", lambda: 10 << 30)
    monkeypatch.setattr(shell_telemetry, "system_available_memory_bytes", lambda: 2 << 30)
    policy = CompilerWorkPolicy(minimum_jobs=12, memory_bytes=10 << 30,
                                reserve_bytes=2 << 30, working_bytes=2 << 30)
    assert policy.capacity(20) == 12
    assert policy.capacity(7) == 7


def test_live_standard_stream_checkpoint_preserves_module_slot():
    import sys
    from rich.console import Console
    from rich.live import Live
    from src.compiler.project_compilation_product import _dump_resolved_process_graph
    with Live("status", console=Console(file=io.StringIO(), force_terminal=True), auto_refresh=False):
        stream = io.BytesIO()
        _dump_resolved_process_graph({"source_stdout": sys.stdout}, stream)
        assert pickle.loads(stream.getvalue())["source_stdout"] is sys.stdout
    with pytest.raises(pickle.PicklingError, match="module slot"):
        _dump_resolved_process_graph(io.StringIO(), io.BytesIO())


def test_terminal_queue_keeps_full_log_and_summarizes_numeric_runs(tmp_path, monkeypatch):
    from src.compiler.shell_telemetry import CompileProgress
    monkeypatch.chdir(tmp_path)
    stream = io.StringIO()
    display = CompileProgress(enabled=True, stream=stream, renderer="rich", heartbeat=None)
    with display:
        for index in range(32):
            display.log(f"region node={index} prepared")
    from pathlib import Path
    saved = Path(display.log_path).read_text(encoding="utf-8")
    assert saved.splitlines() == [f"region node={index} prepared" for index in range(32)]
    assert "related lines" in stream.getvalue()


def test_two_real_sources_share_scheduler_and_eleven_stage_status():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.extraction_contract import ExtractionContract
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": name, "parameter": "x", "storage": "scalar", "dtype": "float64",
             "rank": 0, "shape": [], "python_type": "builtins.float"} for name in ("left", "right")],
    })
    results = lower_ast_source_to_ssa([
        {"label": name, "source": f"def {name}(x):\n    return x * {factor}.0 + 1.0\n",
         "entrypoint": name, "options": {"extraction_contract": contract}}
        for name, factor in (("left", 2), ("right", 3))], source_compilation="parallel",
        partitioned_options={"max_total_resident_bytes": 1 << 30,
                             "max_worker_memory_bytes": 1 << 30,
                             "worker_resident_reservation_bytes": 1 << 20}, progress_bars=False)
    assert len(results) == 2
    for module, outputs, exports in results:
        assert outputs[exports[0]]
        assert module.metadata["source_compilation"]["partition_point"] is None
        stages = module.metadata["compile_profile"]["stages"]
        assert len(stages) == 11
        assert all(stage["state"] == "done" for stage in stages)


def test_every_source_still_reports_to_the_shared_terminal_queue(tmp_path, monkeypatch):
    from pathlib import Path
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.shell_telemetry import CompileProgress
    from src.compiler.extraction_contract import ExtractionContract
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_execution_file(
        "extraction_contracts/vehicle_full_native_execution.yaml")
    monkeypatch.chdir(tmp_path)
    display = CompileProgress(enabled=True, stream=io.StringIO(), renderer="rich", heartbeat=None)
    with display:
        lower_ast_source_to_ssa([
            {"label": name, "source": f"def {name}():\n    return 7.0\n", "entrypoint": name,
             "options": {"extraction_contract": contract}}
            for name in ("left", "right")], source_compilation="parallel", progress_bars=False,
            partitioned_options={"jobs": 2, "max_total_resident_bytes": 1 << 30,
                "max_worker_memory_bytes": 1 << 30, "worker_resident_reservation_bytes": 1 << 20})
    saved = Path(display.log_path).read_text(encoding="utf-8")
    for name in ("left", "right"):
        assert f"[{name}] ssa-source: building complete ProcessGraph source closure" in saved
        assert f"[{name}] compiler storage: completed planning state released" in saved
        for phase in (
            "topology: extract lexical function graphs",
            "instantiate shared function catalogue",
            "release local dispatch graphs",
            "record forwarding fixed point",
            "link source call records",
            "post-frame: reconcile record and sequence captures",
            "apply precision pipeline",
            "pre-native: full native link gate",
            "final concordance detector",
            "compiler finalization: release planning caches",
        ):
            assert f"[{name}] compiler phase: {phase}: begin" in saved
        worker = next(lane for lane in display._workers.values() if lane["label"] == name)
        assert worker["status"] == "complete"
        assert len(worker["stages"]) == 11
        assert all(stage["state"] == "done" for stage in worker["stages"])


def test_final_detector_skips_an_absent_partial_module(monkeypatch):
    from src.compiler.fortran_c_shell import _run_source_lowering
    from src.compiler.identity_concordance import CorrelationTable

    def detector_must_not_run(_module):
        raise AssertionError("the detector received an absent module")

    monkeypatch.setattr(CorrelationTable, "build", detector_must_not_run)
    messages = []
    result = _run_source_lowering(
        lambda **_kwargs: (None, {}, ()),
        progress=messages.append,
        progress_bars=False,
        compile_profile=False,
    )

    assert result == (None, {}, ())
    assert any(
        "detector skipped: no repository SSA module" in message
        for message in messages
    )
    assert all("detector failed" not in message for message in messages)


@pytest.mark.parametrize("fail_release", (False, True))
def test_finalization_is_visible_until_storage_release_finishes(tmp_path, monkeypatch, fail_release):
    from pathlib import Path
    from src.compiler import memory_regulation
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.shell_telemetry import CompileProgress, active_compile_progress
    contract = Path("extraction_contracts/program_extraction.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    display = CompileProgress(enabled=True, stream=io.StringIO(), renderer="rich", heartbeat=None)
    release = memory_regulation.end_regulation
    observed = []

    def observed_release(*args, **kwargs):
        layer = active_compile_progress()
        observed.append(layer.summary())
        assert layer is display and layer._depth > 0
        assert layer.stages[layer._stage_index].key == "compiler-finalization"
        release(*args, **kwargs)
        if fail_release:
            raise RuntimeError("storage release fixture")

    monkeypatch.setattr(memory_regulation, "end_regulation", observed_release)
    if fail_release:
        with pytest.raises(RuntimeError, match="storage release fixture"), display:
            lower_ast_source_to_ssa("def root():\n    return 7.0\n", "root", extraction_contract=contract)
    else:
        with display:
            lower_ast_source_to_ssa("def root():\n    return 7.0\n", "root", extraction_contract=contract)
    assert observed[0]["stages"][-1]["state"] == "running"
    assert display.summary()["stages"][-1]["state"] == ("failed" if fail_release else "done")
    saved = Path(display.log_path).read_text(encoding="utf-8")
    assert "stage 11/11 compiler finalization: begin" in saved
    assert "compiler phase: identity log checkpoint: begin" in saved


def test_one_failed_source_does_not_stop_independent_work():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import SourceListCompilationError
    from src.compiler.shell_telemetry import CompileProgress
    display = CompileProgress(enabled=False, collect_profile=True)
    with display, pytest.raises(SourceListCompilationError) as caught:
        lower_ast_source_to_ssa([
            {"label": "bad", "source": "def bad(:\n", "entrypoint": "bad",
             "options": {"extraction_contract": "extraction_contracts/program_extraction.yaml"}},
            {"label": "good", "source": "def good():\n    return 7\n", "entrypoint": "good",
             "options": {"extraction_contract": "extraction_contracts/program_extraction.yaml"}},
        ], source_compilation="parallel", progress_bars=False,
            partitioned_options={"max_total_resident_bytes": 1 << 30,
                "max_worker_memory_bytes": 1 << 30, "worker_resident_reservation_bytes": 1 << 20})
    assert caught.value.failures[0]["label"] == "bad"
    assert caught.value.results[1][2]
    assert any(lane["label"] == "bad" and lane["status"] == "failed" for lane in display._workers.values())
    assert any(lane["label"] == "good" and lane["status"] == "complete" for lane in display._workers.values())


def test_rule_summary_survives_background_redraws(tmp_path, monkeypatch):
    from src.compiler.shell_telemetry import CompileProgress, compile_rule_batch
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "160")
    display = CompileProgress(enabled=True, stream=io.StringIO(), renderer="rich", heartbeat=None)
    with display, compile_rule_batch(2, names=("left", "right")) as batch:
        batch.complete("left", 2.0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            line = pool.submit(lambda: display.bars._frame().renderables[0].plain).result()
        assert "Rules 1/2" in line and "average" in line and "projected total" in line


def test_single_source_stage_advances_after_inner_workers_join(tmp_path, monkeypatch):
    from src.compiler.shell_telemetry import CompileProgress
    from rich.console import Console
    monkeypatch.chdir(tmp_path)
    display = CompileProgress(enabled=True, stream=io.StringIO(), renderer="rich", heartbeat=None)
    with display:
        display.stage("ssa-lowering")
        display.worker("region-thread", label="SSA region", status="done", stage_number=9, stage_count=11,
            stages=[{"key": stage.key, "label": stage.label, "state": "done" if index < 9 else "pending"}
                    for index, stage in enumerate(display.stages)])
        display.stage("pre-native-repairs")
        output = io.StringIO()
        display.bars.console.width = 160
        display.bars.console.height = 30
        Console(file=output, width=160, height=30).print(display.bars._frame())
        frame = output.getvalue()
        assert "source compiler" in frame and "pre-native repairs" in frame and "running" in frame


def test_completed_source_column_survives_more_recent_inner_lanes(tmp_path, monkeypatch):
    from src.compiler.shell_telemetry import CompileProgress
    from rich.console import Console
    monkeypatch.chdir(tmp_path)
    display = CompileProgress(enabled=True, stream=io.StringIO(), renderer="rich", heartbeat=None)
    done = [{"key": stage.key, "label": stage.label, "state": "done"}
            for stage in display.stages]
    with display:
        display.worker("source", label="whole source", status="complete", stage_number=11,
                       stage_count=11, stages=done)
        for index in range(4):
            display.worker(f"inner-{index}", label=f"SSA region {index}", status="done",
                           stage_number=9, stage_count=11,
                           stages=[{**row, "state": "pending" if number == 9 else "done"}
                                   for number, row in enumerate(done)])
        display.bars.console.width = 80
        display.bars.console.height = 30
        output = io.StringIO()
        Console(file=output, width=80, height=30).print(display.bars._frame())
        assert "whole source" in output.getvalue()
        assert "pre-native repairs" in output.getvalue()


def test_post_native_reporting_error_is_not_a_source_normalisation_failure(tmp_path, monkeypatch):
    from src.compiler.shell_telemetry import CompileProgress
    monkeypatch.chdir(tmp_path)
    stream = io.StringIO()
    display = CompileProgress(enabled=True, stream=stream, renderer="rich", heartbeat=None)
    with pytest.raises(ValueError, match="post-native report"), display:
        display.worker("source", label="whole source", status="complete", stage_number=11,
                       stage_count=11, stages=[
                           {"key": stage.key, "label": stage.label, "state": "done"}
                           for stage in display.stages])
        raise ValueError("post-native report")
    assert "source schedule FAILED" in stream.getvalue()
    assert "source normalisation: FAILED" not in stream.getvalue()
    assert not display._stage_profiles
    assert display._workers["source"]["status"] == "complete"


def test_source_planning_cache_keeps_other_running_books():
    from src.compiler.identity_concordance import IdentityBook
    from src.compiler.glsl_deployment_strategy import _child_signature_results, release_compile_planning_state
    first, second = IdentityBook(), IdentityBook()
    first_answers = _child_signature_results(first)
    first_answers["answer"] = 42
    _child_signature_results(second)["answer"] = 7
    assert _child_signature_results(first) is first_answers
    release_compile_planning_state(second)
    assert _child_signature_results(first)["answer"] == 42
    release_compile_planning_state(first)


def test_supplied_law_list_produces_exact_native_pieces(tmp_path):
    import numpy as np
    import sympy as sp
    from src.compiler.native_package import pieces_from_equations
    x = sp.Symbol("x")
    pieces = pieces_from_equations({
        "left_law": (sp.Eq(sp.Symbol("result"), x + 2, evaluate=False),),
        "right_law": (sp.Eq(sp.Symbol("result"), x * 3, evaluate=False),),
    }, 1, directory=tmp_path, source_compilation="parallel",
        partitioned_options={"max_total_resident_bytes": 1 << 30,
                             "max_worker_memory_bytes": 1 << 30,
                             "worker_resident_reservation_bytes": 1 << 20},
        progress=lambda _message: None)
    for name, expected in (("left_law", 9.0), ("right_law", 21.0)):
        result, = pieces[name](np.array([7.0]))
        np.testing.assert_array_equal(result, np.array([expected]))
        assert len(pieces[name].module.metadata["compile_profile"]["stages"]) == 11
