"""The compile's stage progress bars (``shell_telemetry``'s progress layer).

None of this runs the compiler.  The layer is driven with a fake stage
sequence and fake log lines, and the call sites it is wired into are read as
source -- the way to check that a stage key or a planner fact the layer
consumes is one the compiler really emits.
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import pytest

from src.compiler import shell_telemetry as telemetry
from src.compiler.shell_telemetry import (
    COMPILE_STAGES,
    LOWERING_STAGES,
    CompileProgress,
    CompileStage,
    active_compile_progress,
    compile_count,
    compile_iter,
    compile_progress,
    compile_stage,
    declare_progress_bars_policy,
    progress_bars_default,
)

COMPILER = Path(__file__).resolve().parents[1] / "src" / "compiler"

FAKE_STAGES = (
    CompileStage("alpha", "first stage"),
    CompileStage("beta", "second stage"),
    CompileStage("gamma", "third stage"),
)
GIB = 1024 ** 3


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_fake_tqdm():
    """A tqdm stand-in that records everything, in order, in one event list."""

    events: list[tuple] = []
    bars: list["Bar"] = []

    class Bar:
        def __init__(self, total=None, desc=None, leave=True, **kwargs):
            self.index = len(bars)
            bars.append(self)
            self.total = total
            self.desc = desc
            self.leave = leave
            self.kwargs = kwargs
            self.n = 0
            self.postfix = ""
            self.closed = False
            events.append(("open", self.index, desc, leave, total))

        def set_description_str(self, desc, refresh=True):
            self.desc = desc

        def set_postfix_str(self, text, refresh=True):
            self.postfix = text

        def refresh(self):
            pass

        def update(self, delta=1):
            self.n += delta

        def close(self):
            self.closed = True
            events.append(("close", self.index, self.desc, self.n))

        @classmethod
        def write(cls, text, file=None, end="\n"):
            events.append(("write", text))

    return Bar, events, bars


def make_layer(stages=FAKE_STAGES, **options):
    tqdm_class, events, bars = make_fake_tqdm()
    clock = options.pop("clock", Clock())
    options.setdefault("rss", lambda: int(1.5 * GIB))
    options.setdefault("heartbeat", None)
    options.setdefault("min_interval", 0.0)
    layer = CompileProgress(
        enabled=True, stages=stages, tqdm_class=tqdm_class,
        stream=io.StringIO(), clock=clock, **options,
    )
    return layer, events, bars, clock


@pytest.fixture(autouse=True)
def no_leaked_layer():
    yield
    assert active_compile_progress() is None


# -- the stage bar -----------------------------------------------------------


def test_stage_bar_opens_first_advances_in_order_and_closes_last():
    layer, events, bars, clock = make_layer()
    with layer:
        main = bars[0]
        assert main.total == 3 and main.leave is True
        assert main.n == 0 and "first stage" in main.desc
        clock.now += 5
        compile_stage("beta")
        assert main.n == 1 and "2/3" in main.desc and "second stage" in main.desc
        compile_stage("gamma")
        assert main.n == 2 and "third stage" in main.desc
    assert main.n == 3 and main.closed and "done" in main.desc
    opens = [e for e in events if e[0] == "open"]
    closes = [e for e in events if e[0] == "close"]
    assert opens[0][1] == 0 and closes[-1][1] == 0  # first opened, last closed


def test_stage_completion_lines_carry_time_and_rss():
    layer, events, bars, clock = make_layer()
    with layer:
        clock.now += 12
        compile_stage("beta")
    written = [e[1] for e in events if e[0] == "write"]
    assert "[compiler] stage 1/3 first stage: done in 12s (rss 1.50GB)" in written


def test_unknown_stage_is_refused_not_guessed():
    layer, _events, _bars, _clock = make_layer()
    with layer:
        with pytest.raises(ValueError, match="undeclared compile stage"):
            compile_stage("not-a-stage")


def test_postfix_carries_stage_elapsed_and_rss():
    rss = [int(1.5 * GIB)]
    layer, _events, bars, clock = make_layer(rss=lambda: rss[0])
    with layer:
        main = bars[0]
        assert "rss 1.50GB" in main.postfix
        clock.now += 75
        rss[0] = int(4.25 * GIB)
        layer.bars.heartbeat()
        assert "rss 4.25GB" in main.postfix
        assert "stage 1m15s" in main.postfix
        layer.substage("sub phase")
        assert main.postfix.startswith("sub phase, stage")


def test_final_postfix_reports_peak_rss():
    rss = [int(2 * GIB)]
    layer, _events, bars, clock = make_layer(rss=lambda: rss[0])
    with layer:
        clock.now += 1
        layer.bars.heartbeat()
        rss[0] = int(1 * GIB)
        clock.now += 1
        layer.bars.heartbeat()
    assert "peak 2.00GB" in bars[0].postfix


# -- nested bars ---------------------------------------------------------------


def test_nested_bars_are_not_left_and_close_with_their_stage():
    layer, events, bars, _clock = make_layer()
    with layer:
        compile_count("items", 1, total=4, label="items", detail="first")
        nested = bars[1]
        assert nested.leave is False and nested.total == 4 and nested.n == 1
        compile_count("items")
        assert nested.n == 2 and "rss 1.50GB" in nested.postfix
        assert not nested.closed
        compile_stage("beta")
        assert nested.closed
        # closed before the main bar moved on
        close_at = events.index(("close", 1, "items", 2))
        assert close_at < len(events)
        assert bars[0].n == 1


def test_iter_drives_a_nested_bar_and_closes_it_at_exhaustion():
    layer, _events, bars, _clock = make_layer()
    with layer:
        seen = []
        for item in compile_iter(iter([10, 20, 30]), "walk", label="walk", total=3):
            seen.append(item)
            assert not bars[1].closed
        assert seen == [10, 20, 30]
        assert bars[1].closed and bars[1].n == 3 and bars[1].total == 3


def test_iter_is_the_iterable_itself_when_no_layer_is_running():
    data = [1, 2, 3]
    assert compile_iter(data, "walk") is data
    compile_count("items", 1)  # a no-op, not an error


def test_unknown_total_is_an_open_ended_bar():
    layer, _events, bars, _clock = make_layer()
    with layer:
        compile_count("rounds", 1)
        compile_count("rounds", 5, detail="changed=3")
        assert bars[1].total is None and bars[1].n == 5
        assert bars[1].postfix.startswith("changed=3")


def test_updates_are_throttled_but_the_final_one_is_not():
    layer, _events, bars, clock = make_layer(min_interval=1.0)
    with layer:
        compile_count("items", 0, total=10)
        for _ in range(9):
            clock.now += 0.01
            compile_count("items")
        assert bars[1].n == 0  # all throttled
        clock.now += 0.01
        compile_count("items")  # reaches the total
        assert bars[1].n == 10


def test_heartbeat_thread_refreshes_clock_and_rss_between_events_and_stops():
    import threading
    import time as real_time

    rss = [GIB]
    layer, _events, bars, clock = make_layer(
        rss=lambda: rss[0], heartbeat=0.01,
    )
    before = threading.active_count()
    with layer:
        assert threading.active_count() == before + 1
        rss[0] = 3 * GIB
        clock.now += 30
        deadline = real_time.monotonic() + 5.0
        while "rss 3.00GB" not in bars[0].postfix:
            assert real_time.monotonic() < deadline, bars[0].postfix
            real_time.sleep(0.01)
        assert "stage 30s" in bars[0].postfix
    assert threading.active_count() == before


# -- the logger inside the bars -------------------------------------------------


def test_log_lines_use_tqdm_write_while_a_bar_is_open(capsys):
    layer, events, bars, _clock = make_layer()
    with layer:
        callback = layer.progress_callback()
        callback("first line")
        callback("second line")
        assert not bars[0].closed
    writes = [e[1] for e in events if e[0] == "write"]
    assert writes[:2] == ["[compiler] first line", "[compiler] second line"]
    first_write = next(i for i, e in enumerate(events) if e[0] == "write")
    assert events[0][0] == "open" and first_write > 0  # bar opened before
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""  # nothing went around tqdm.write


def test_a_callers_own_callback_is_kept_and_its_prints_go_above_the_bars(capsys):
    layer, events, _bars, _clock = make_layer()
    received = []

    def theirs(message):
        received.append(message)
        print(f"theirs: {message}")
        print(f"theirs err: {message}", file=sys.stderr)

    with layer:
        layer.progress_callback(theirs)("hello")
    assert received == ["hello"]
    writes = [e[1] for e in events if e[0] == "write"]
    assert "theirs: hello" in writes and "theirs err: hello" in writes
    assert capsys.readouterr().out == ""
    # and the streams are restored
    assert sys.stdout is not None and not isinstance(
        sys.stdout, telemetry._TqdmLineFile
    )


def test_log_lines_are_also_records_on_the_channel():
    layer, _events, _bars, _clock = make_layer()
    with layer:
        layer.progress_callback()("a line")
        compile_stage("beta")
    kinds = [(r.kind, r.message) for r in layer.channel.records]
    assert ("log", "a line") in kinds
    progress_paths = [r.path for r in layer.channel.of_kind("progress")]
    assert "compile/start" in progress_paths and "compile/stage" in progress_paths


# -- disabled: today's behaviour -------------------------------------------------


def reference_default_line(message: str, buffer: io.StringIO) -> None:
    """The line the entry printed before the layer existed, verbatim."""

    print(f"[compiler] {message}", file=buffer, flush=True)


def test_disabled_mode_prints_todays_plain_lines_byte_for_byte():
    messages = [
        "ssa-source: reducing source topology",
        "ssa-program +1.2s (phase +0.3s): start x",
        "unicode µ and % signs",
    ]
    expected = io.StringIO()
    for message in messages:
        reference_default_line(message, expected)
    actual = io.StringIO()
    layer = CompileProgress(enabled=False)
    callback = layer.progress_callback()
    with contextlib.redirect_stderr(actual):
        for message in messages:
            callback(message)
        layer.log("via log")
    reference_default_line("via log", expected)
    assert actual.getvalue() == expected.getvalue()


def test_disabled_mode_builds_nothing_and_leaves_callbacks_alone():
    tqdm_class, events, bars = make_fake_tqdm()
    layer = CompileProgress(enabled=False, tqdm_class=tqdm_class)

    def theirs(message):
        pass

    assert layer.progress_callback(theirs) is theirs
    layer.open()
    assert active_compile_progress() is None  # not even registered
    compile_stage("anything")
    compile_count("anything", 3)
    layer.stage("alpha")
    layer.substage("x")
    layer.close()
    assert layer.channel is None and layer.bars is None
    assert events == [] and bars == []


def test_default_is_off_unless_stdout_is_a_terminal(monkeypatch):
    class Stream(io.StringIO):
        def __init__(self, tty):
            super().__init__()
            self._tty = tty

        def isatty(self):
            return self._tty

    monkeypatch.setattr(sys, "stdout", Stream(False))
    monkeypatch.setattr(sys, "stderr", Stream(True))
    assert progress_bars_default() is False
    assert compile_progress().enabled is False
    monkeypatch.setattr(sys, "stdout", Stream(True))
    assert progress_bars_default() is True
    # the stream the bars draw on must be a terminal too
    monkeypatch.setattr(sys, "stderr", Stream(False))
    assert progress_bars_default() is False
    # an explicit choice wins either way
    assert compile_progress(False).enabled is False
    forced = compile_progress(True, tqdm_class=make_fake_tqdm()[0])
    assert forced.enabled is True


# -- failure ----------------------------------------------------------------------


def test_an_exception_inside_a_stage_closes_every_bar_cleanly():
    layer, events, bars, _clock = make_layer()
    with pytest.raises(RuntimeError, match="boom"):
        with layer:
            compile_stage("beta")
            compile_count("items", 1, total=9)
            compile_count("rounds", 2)
            raise RuntimeError("boom")
    assert all(bar.closed for bar in bars)
    assert bars[0].desc == "compile FAILED in second stage"
    assert bars[0].n == 1  # stayed on the stage that failed
    assert [e for e in events if e[0] == "close"][-1][1] == 0  # main last
    writes = [e[1] for e in events if e[0] == "write"]
    assert any("second stage: FAILED after" in line for line in writes)
    assert active_compile_progress() is None


def test_a_drawing_fault_falls_back_to_plain_lines_and_says_so():
    layer, _events, bars, _clock = make_layer()
    stream = layer.stream
    with layer:
        def broken(*_args, **_kwargs):
            raise OSError("terminal went away")

        bars[0].refresh = broken
        compile_stage("beta")  # the redraw fails
        assert layer.bars.failure.startswith("OSError")
        layer.progress_callback()("still logged")
    text = stream.getvalue()
    assert "progress bars disabled: OSError" in text
    assert "[compiler] still logged" in text


# -- sharing and the declared sequences ------------------------------------------------


def test_a_nested_compile_joins_the_running_layer():
    layer, _events, bars, _clock = make_layer()
    with layer:
        inner = compile_progress(False)  # display wins; nothing new is built
        assert inner is layer
        inner.open()
        inner.close()
        assert not bars[0].closed
        assert active_compile_progress() is layer
    assert bars[0].closed


def test_lowering_stage_sequence_is_the_compiler_order():
    keys = [stage.key for stage in LOWERING_STAGES]
    assert keys == [
        "source-normalisation", "source-closure", "topology-reduction",
        "compilation-units", "deployment-select", "deployment-instantiate",
        "call-topology", "graph-planning", "ssa-lowering",
        "pre-native-repairs",
    ]
    assert [s.key for s in COMPILE_STAGES][-2:] == ["emission", "build"]
    assert len({stage.key for stage in COMPILE_STAGES}) == len(COMPILE_STAGES)


def test_every_stage_key_the_compiler_names_is_declared():
    declared = {stage.key for stage in LOWERING_STAGES}
    named = []
    for name in ("fortran_c_shell.py", "glsl_deployment_strategy.py"):
        source = (COMPILER / name).read_text(encoding="utf-8")
        named += re.findall(r'compile_stage\("([^"]+)"\)', source)
    assert named, "no compile_stage call sites found"
    assert set(named) <= declared
    # in the order the compiler runs them: the entry's earlier stages, then the
    # deployment's, then the repairs after it returns
    assert {"source-closure", "topology-reduction", "compilation-units",
            "deployment-select", "deployment-instantiate", "call-topology",
            "graph-planning", "ssa-lowering", "pre-native-repairs"} <= set(named)


def test_the_entry_is_wired_to_the_layer_and_keeps_the_plain_default():
    source = (COMPILER / "fortran_c_shell.py").read_text(encoding="utf-8")
    entry = source[source.index("def lower_ast_source_to_ssa("):]
    entry = entry[: entry.index("lower_ast_source_to_ssa.__canonical_source_compiler__")]
    assert 'kwargs.pop("progress_bars", None)' in entry
    assert "_bars.progress_callback(kwargs.get(\"progress\"))" in entry
    assert "_bars.close(failed=not ok)" in entry
    assert "declare_progress_bars_policy(" in entry
    # the plain default line lives in one place and is the one it always was
    plain = telemetry._plain_compiler_line
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        plain("x")
    assert buffer.getvalue() == "[compiler] x\n"


def test_planner_facts_the_layer_reads_are_facts_the_planner_emits():
    planner = (COMPILER / "glsl_deployment_strategy.py").read_text(encoding="utf-8")
    fusion = (COMPILER / "process_graph_fusion.py").read_text(encoding="utf-8")
    for literal in (
        '"specialization_state": "round-begin"',
        '"specialization_state": "prefold-end"',
        '"specialization_state": "prefold-fixed-point"',
        '"specialization_state": "callsite-progress"',
        '"specialization_state": "round-end"',
        '"caller_index"', '"graphs"', '"callsites"',
        "selection_phase(\n            \"propagate-callsite-tensor-specializations\"",
        '"extract-dispatch-subgraphs"',
        '"reduce-scheduled-shader-regions"',
    ):
        assert literal in planner, literal
    for literal in ('"reduction_state"', '"iteration"', '"merges"', '"regions"'):
        assert literal in fusion, literal
    assert '"fold_iteration"' in planner


# -- the deployment planner's events -------------------------------------------------------


def test_selection_events_name_the_substage_and_drive_the_callsite_bars():
    layer, _events, bars, _clock = make_layer()
    with layer:
        main = bars[0]
        layer.selection("lower-python-scalar-intrinsics", "begin", 0, "<root>", {})
        assert main.postfix.startswith("lower-python-scalar-intrinsics")
        phase = "propagate-callsite-tensor-specializations"
        layer.selection(phase, "begin", 0, "<root>", {})
        layer.selection(
            "callsite-tensor-specialization", "progress", 0, "<root>",
            {"specialization_state": "round-begin", "round": 1, "graphs": 3,
             "graph_nodes": 99},
        )
        callers = bars[1]
        assert callers.total == 3 and "round 1" in callers.desc
        layer.selection(
            "callsite-tensor-specialization", "progress", 0, "<root>",
            {"specialization_state": "prefold-end", "caller_index": 1,
             "caller": "f"},
        )
        assert callers.n == 2 and callers.postfix.startswith("f")
        layer.selection(
            "callsite-tensor-specialization", "progress", 0, "<root>",
            {"specialization_state": "prefold-fixed-point", "fold_iteration": 2,
             "caller": "f", "changed": True},
        )
        fold = bars[2]
        assert fold.n == 2 and fold.leave is False
        layer.selection(
            "callsite-tensor-specialization", "progress", 0, "<root>",
            {"specialization_state": "callsite-progress", "callsites": 16,
             "caller": "f", "callee": "g"},
        )
        assert bars[3].n == 16 and "f -> g" in bars[3].postfix
        layer.selection(phase, "end", 0, "<root>", {})
        assert all(bar.closed for bar in bars[1:])
        assert not main.closed


def test_function_shell_bar_survives_the_roots_per_function_phases():
    layer, _events, bars, _clock = make_layer()
    with layer:
        compile_count("function-shells", 0, total=5, label="function shells")
        shells = bars[1]
        layer.selection("function-shell", "begin", 0, "<root>",
                        {"child_function": "mod.f"})
        assert shells.postfix.startswith("mod.f")
        # the root's own per-function bookkeeping phase does not rename the
        # sub-stage nor close the bar
        layer.selection("extract-function-subgraph", "begin", 0, "<root>", {})
        assert not shells.closed
        assert not bars[0].postfix.startswith("extract-function-subgraph")
        # a child shell's phases name what the bar is on
        layer.selection("fold-callsite-structural-values-before-tensors",
                        "begin", 1, "mod.f", {})
        assert shells.postfix.startswith("mod.f fold-callsite")


def test_region_reduction_rounds_are_a_bar_that_ends_with_the_phase():
    layer, _events, bars, _clock = make_layer()
    with layer:
        layer.selection("reduce-scheduled-shader-regions", "begin", 1, "f", {})
        layer.selection(
            "shader-region-fixed-point", "progress", 1, "f",
            {"reduction_state": "merge", "iteration": 3, "regions": 7,
             "merges": 2},
        )
        assert bars[1].n == 3 and "regions=7" in bars[1].postfix
        layer.selection("reduce-scheduled-shader-regions", "end", 1, "f", {})
        assert bars[1].closed


def test_every_guarded_fixed_point_loop_is_a_nested_round_bar():
    from src.compiler.bounded_fixed_point import BoundedFixedPoint
    from src.compiler.identity_concordance import IdentityBook

    layer, _events, bars, _clock = make_layer()
    guard = BoundedFixedPoint(
        "toy-fixed-point", 50, scope=("whole-program",), book=IdentityBook(),
    )
    with layer:
        guard.round(True, state="a")
        guard.round(True, state="b")
        nested = bars[1]
        assert nested.total == 50 and nested.n == 2 and not nested.closed
        assert "toy-fixed-point rounds (bound 50)" in nested.desc
        guard.round(False, state="c")
        assert nested.closed
    # and with no layer running the guard is exactly what it was
    quiet = BoundedFixedPoint(
        "toy-fixed-point", 50, scope=("whole-program",), book=IdentityBook(),
    )
    assert quiet.round(False, state="x") is False


# -- the book receipt -----------------------------------------------------------------------


def test_the_choice_is_a_compile_policy_row_on_the_book():
    from src.compiler.concordance_declarations import COMPILE_POLICY
    from src.compiler.identity_concordance import IdentityBook

    row = ("progress_bars",)
    on = IdentityBook()
    declare_progress_bars_policy(True, on)
    assert on.page(COMPILE_POLICY).latest(row) == "on"
    off = IdentityBook()
    declare_progress_bars_policy(False, off)
    assert off.page(COMPILE_POLICY).latest(row) == "off"
    # a resumed book keeps its first choice; a later compile does not
    # contradict it
    declare_progress_bars_policy(False, on)
    assert on.page(COMPILE_POLICY).latest(row) == "on"


# -- the real tqdm ------------------------------------------------------------------------------


def test_with_real_tqdm_lines_scroll_above_the_bars(monkeypatch):
    tqdm_module = pytest.importorskip("tqdm")
    written = []
    real_write = tqdm_module.tqdm.write.__func__

    def spy(cls, s, file=None, end="\n", nolock=False):
        written.append(s)
        return real_write(cls, s, file=file, end=end, nolock=nolock)

    monkeypatch.setattr(tqdm_module.tqdm, "write", classmethod(spy))
    stream = io.StringIO()
    layer = CompileProgress(
        enabled=True, stages=FAKE_STAGES, stream=stream, heartbeat=None,
        rss=lambda: GIB, nested_delay=0.0, min_interval=0.0,
    )
    with layer:
        layer.progress_callback()("a compiler line")
        compile_stage("beta")
        for _ in compile_iter(range(3), "walk", label="walk", total=3):
            layer.progress_callback()("inside the loop")
    text = stream.getvalue()
    assert "[compiler] a compiler line" in text
    assert "[compiler] inside the loop" in text
    assert "compile 2/3 second stage" in text
    assert "rss 1.00GB" in text
    assert "[compiler] a compiler line" in written
    assert "compile 3/3 done" in text
