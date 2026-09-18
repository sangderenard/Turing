"""Execute the authored viewer loop with controlled publication interleavings."""

import ast
from pathlib import Path
from types import SimpleNamespace
import threading


def test_viewer_does_not_acknowledge_revision_published_during_draw():
    source = Path(__file__).resolve().parents[1] / "tools/run_vehicle_native_assembly.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    entry = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_run_dually_python_profile")
    loops = [n for n in ast.walk(entry) if isinstance(n, ast.While)
             and isinstance(n.test, ast.Name) and n.test.id == "running"]
    assert len(loops) == 1
    # Run the actual authored loop, excluding expensive model/worker startup.
    code = compile(ast.Module(body=loops, type_ignores=[]), str(source), "exec")
    revision, displayed, attempted = [0], [0], [0]
    frames = []
    acknowledgments = []
    live = dict.fromkeys((
        "stage", "progress", "sim_time", "status", "accepted_time",
        "substep_dt", "substep_index", "accepted_substeps", "rejected_substeps",
        "error_max", "error_rms", "error_p95", "error_per_wheel",
        "error_location", "rule_violation", "finished",
    ), 0)

    class Viewer:
        def events(self):
            return len(frames) == 0

        def draw(self, snapshot, **metrics):
            acknowledgments.append(displayed[0])
            frames.append(snapshot)
            if len(frames) == 1:
                # Worker publishes the initialized mesh after this frame was
                # acquired but before drawing/presentation returns.
                revision[0] = 1

    exec(code, {
        "running": True, "viewer": Viewer(), "status_lock": threading.Condition(),
        "live": live, "visual_revision": revision,
        "displayed_visual_revision": displayed, "displayed_attempt": attempted,
        "material": SimpleNamespace(visual_snapshot=lambda **kwargs: revision[0]),
        "args": SimpleNamespace(headless_frame=None, python_viewer=True),
    })
    assert frames == [0, 1]
    assert acknowledgments == [0, 0], "unpresented revision was acknowledged"
    assert displayed == [1]
