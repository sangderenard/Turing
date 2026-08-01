import pytest

from src.compiler.shell_telemetry import (
    ERROR,
    KINDS,
    LOG,
    PROFILE,
    PROGRESS,
    TelemetryChannel,
    attach_error_buffer,
    attach_profiler,
    record_launch_profile,
    record_shortfalls,
    summarize_process_graph,
)


def test_one_stream_carries_every_kind_in_order():
    channel = TelemetryChannel()
    channel.log("a")
    channel.error("b")
    channel.profile("c", nanoseconds=5)
    channel.progress("d", done=1, total=2)

    kinds = [r.kind for r in channel.records]
    assert kinds == [LOG, ERROR, PROFILE, PROGRESS]
    assert [r.sequence for r in channel.records] == [1, 2, 3, 4]
    assert set(KINDS) >= set(kinds)


def test_an_unknown_kind_is_refused_rather_than_invented():
    with pytest.raises(ValueError, match="unknown record kind"):
        TelemetryChannel().emit("chatter", "hello")


def test_timed_records_the_duration_even_when_the_block_raises():
    """A phase that failed still took time; losing that makes a slow failure
    look like a fast one."""

    channel = TelemetryChannel()
    with pytest.raises(RuntimeError):
        with channel.timed("doomed", path="p"):
            raise RuntimeError("nope")

    profiles = channel.of_kind(PROFILE)
    assert len(profiles) == 1
    assert profiles[0].detail["failed"] is True
    assert profiles[0].detail["nanoseconds"] > 0
    errors = channel.of_kind(ERROR)
    assert errors and "RuntimeError" in errors[0].message


def test_progress_is_a_record_so_a_bar_cannot_drift_from_the_log():
    channel = TelemetryChannel()
    with channel.stepped("work", 3) as advance:
        advance("one")
        advance("two")
        advance("three")

    progress = channel.of_kind(PROGRESS)
    assert [(r.detail["done"], r.detail["total"]) for r in progress] == [
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]


def test_an_abandoned_scope_says_so_instead_of_leaving_the_bar_stuck():
    channel = TelemetryChannel()
    with pytest.raises(RuntimeError):
        with channel.stepped("work", 5) as advance:
            advance("one")
            raise RuntimeError("stop")

    last = channel.of_kind(PROGRESS)[-1]
    assert last.detail["incomplete"] is True
    assert last.detail["done"] == 1 and last.detail["total"] == 5


def test_a_broken_subscriber_cannot_take_the_channel_down():
    """Telemetry that can fail the thing it observes is worse than none."""

    channel = TelemetryChannel()
    channel.subscribe(lambda record: 1 / 0)
    seen = []
    channel.subscribe(seen.append)
    channel.log("still fine")
    assert len(seen) == 1


def test_attaching_an_error_buffer_does_not_change_what_it_returns():
    """These wrappers sit on load-bearing objects; their existing consumers
    must not be able to tell."""

    class Buffer:
        def __init__(self):
            self.pushed = []

        def push(self, exception, *, path, phase, node_id=None, handled=False):
            record = {"path": path, "phase": phase}
            self.pushed.append(record)
            return record

    buffer, channel = Buffer(), TelemetryChannel()
    detach = attach_error_buffer(buffer, channel)
    returned = buffer.push(ValueError("bad"), path="p", phase="f")

    assert returned == {"path": "p", "phase": "f"}
    assert len(buffer.pushed) == 1
    assert channel.of_kind(ERROR)[0].message == "ValueError: bad"

    detach()
    buffer.push(ValueError("after"), path="p", phase="f")
    assert len(channel.of_kind(ERROR)) == 1


def test_attaching_a_profiler_mirrors_without_replacing():
    class Profiler:
        def __init__(self):
            self.calls = 0

        def record(self, path, **kwargs):
            self.calls += 1
            return "kept"

    profiler, channel = Profiler(), TelemetryChannel()
    detach = attach_profiler(profiler, channel)
    assert profiler.record("shell/a", label="phase", elapsed_ns=1234) == "kept"
    assert profiler.calls == 1

    profile = channel.of_kind(PROFILE)[0]
    assert profile.path == "shell/a"
    assert profile.detail["nanoseconds"] == 1234
    detach()


def test_a_launch_profile_lands_on_the_same_timeline():
    class Launch:
        shell_ns, device_ns, host_ns, status, language = 500, 200, 300, 0, "wasm"

    channel = TelemetryChannel()
    record_launch_profile(channel, Launch(), path="root")
    detail = channel.of_kind(PROFILE)[0].detail
    assert detail["nanoseconds"] == 500 and detail["device_ns"] == 200


def test_shortfalls_are_errors_not_a_separate_report():
    class Shortfall:
        def format(self):
            return "step 0 (exp): no instruction"

    channel = TelemetryChannel()
    record_shortfalls(channel, [Shortfall()], path="wasm")
    record = channel.of_kind(ERROR)[0]
    assert record.detail["shortfall"] is True
    assert "exp" in record.message


def test_the_channel_serializes_for_a_page():
    channel = TelemetryChannel(name="demo")
    channel.log("hello", where="there")
    mapping = channel.to_mapping()
    assert mapping["name"] == "demo"
    assert mapping["records"][0]["detail"] == {"where": "there"}
    assert "schema" in mapping


def test_a_process_graph_summary_is_capped_and_says_so():
    import networkx as nx

    class Graph:
        def __init__(self, n):
            self.G = nx.DiGraph()
            for i in range(n):
                self.G.add_node(i, type="Load", label=f"n{i}", parents=[])
            for i in range(n - 1):
                self.G.add_edge(i, i + 1)

    summary = summarize_process_graph(Graph(10), limit=4)
    assert summary["nodes"] == 10 and summary["edges"] == 9
    assert summary["truncated"] is True
    assert len(summary["table"]) == 4
    assert summary["histogram"]["Load"] == 10
